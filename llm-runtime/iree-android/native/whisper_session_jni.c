/* whisper_session_jni.c — Whisper speech-to-text over the IREE runtime (Vulkan or CPU).
 *
 * Runs the three graphs of `WhisperExportCli` (`:llm-inference:whisper:exportWhisper`, whisper contract
 * version 2) for one utterance: encoder over a log-mel window, masked prefill of the decode prompt, then
 * greedy KV steps until end-of-text. Prefill and step share one parameter archive, bound once under the
 * contract's scope. Nothing about the checkpoint is compiled in: layers, width, audio context, mel
 * geometry, KV length and vocabulary come from the contract (manifest.json) through nativeCreate.
 *
 * Exposes to sk.ainet.transformers.iree.android.IreeWhisperSession:
 *   long   nativeCreate(String device, String enc, String prefill, String step, String params, String scope,
 *                       String encFn, String prefillFn, String stepFn, int[] geometry)
 *            geometry = { nLayers, dModel, audioCtx, nMels, melFrames, maxPositions, vocab }
 *   int[]  nativeDecode(long h, float[] mel, int[] promptIds, int eot, int maxNewTokens, int repeatNgram)
 *            mel = [nMels * melFrames] row-major; returns the generated ids, end-of-text excluded
 *   String nativeStats(long h)      // the last decode's costs as key=value pairs
 *   void   nativeDestroy(long h)
 *
 * The step contract (fixed masked KV): self K/V are [1, maxPositions, dModel] and rewritten by the graph at
 * the one-hot write vector `wf`; `addMask` is additive (0 attends, -1e4 blocks) — no boolean tensors, so the
 * graphs stay valid SPIR-V. K/V stay on the device between steps; only the logits row comes back.
 *
 * Repetition guard: greedy decoding over a short window can transcribe the utterance and then start it again
 * instead of ending (end-of-text is the runner-up, the same tokens come out of the PyTorch reference). With
 * repeatNgram = n > 0 the decode stops as soon as its last n tokens already occurred earlier, and those n
 * tokens are dropped; 0 decodes until end-of-text, maxNewTokens or a full KV cache.
 *
 * One session = one decode at a time; calls are single-threaded (the host serializes). Every failure is
 * thrown as IllegalStateException with the formatted IREE status (also logged under `skainet_whisper`).
 */
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef __ANDROID__
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "skainet_whisper", __VA_ARGS__)
#else
#define LOGE(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#endif
#include "iree/runtime/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/vm/api.h"

#define NEG (-1.0e4f)
#define MAX_LAYERS 64

typedef struct {
  iree_runtime_instance_t* inst;
  iree_hal_device_t* dev;
  iree_hal_allocator_t* alloc;
  iree_runtime_session_t* enc, *pre, *step;
  iree_vm_module_t* params;
  char fn_enc[160], fn_pre[160], fn_step[160];
  int n_layers, d_model, audio_ctx, n_mels, mel_frames, max_pos, vocab;
  float* logits;                  /* one row, [vocab] */
  char stats[256];
} Session;

static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void throw_msg(JNIEnv* env, const char* msg) {
  LOGE("%s", msg);
  jclass cls = (*env)->FindClass(env, "java/lang/IllegalStateException");
  if (cls) (*env)->ThrowNew(env, cls, msg);
}

/* Consumes [st]. */
static void throw_status(JNIEnv* env, const char* what, iree_status_t st) {
  char msg[1024];
  char* buf = NULL; iree_host_size_t len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  if (iree_status_to_string(st, &alloc, &buf, &len) && buf) {
    snprintf(msg, sizeof msg, "%s: %.*s", what, (int)(len < 900 ? len : 900), buf);
    iree_allocator_free(alloc, buf);
  } else {
    snprintf(msg, sizeof msg, "%s: status code %d", what, (int)iree_status_code(st));
  }
  iree_status_ignore(st);
  throw_msg(env, msg);
}

/* The archive as an io_parameters VM module under [scope]; appended before every module that imports it. */
static iree_status_t load_params(Session* s, const char* path, const char* scope) {
  iree_allocator_t host = iree_allocator_system();
  iree_io_parameter_index_t* index = NULL;
  iree_io_file_handle_t* file = NULL;
  iree_io_parameter_provider_t* provider = NULL;
  iree_status_t st = iree_io_parameter_index_create(host, &index);
  if (iree_status_is_ok(st)) st = iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, iree_make_cstring_view(path), host, &file);
  if (iree_status_is_ok(st)) st = iree_io_parse_irpa_index(file, index, host);
  if (iree_status_is_ok(st)) st = iree_io_parameter_index_provider_create(iree_make_cstring_view(scope), index,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS, host, &provider);
  if (iree_status_is_ok(st)) st = iree_io_parameters_module_create(iree_runtime_instance_vm_instance(s->inst), 1,
      &provider, host, &s->params);
  if (provider) iree_io_parameter_provider_release(provider);
  if (index) iree_io_parameter_index_release(index);
  if (file) iree_io_file_handle_release(file);
  return st;
}

static iree_status_t load_session(Session* s, const char* path, int with_params, iree_runtime_session_t** out) {
  iree_runtime_session_options_t o; iree_runtime_session_options_initialize(&o);
  iree_status_t st = iree_runtime_session_create_with_device(s->inst, &o, s->dev,
      iree_runtime_instance_host_allocator(s->inst), out);
  if (iree_status_is_ok(st) && with_params) st = iree_runtime_session_append_module(*out, s->params);
  if (iree_status_is_ok(st)) st = iree_runtime_session_append_bytecode_module_from_file(*out, path);
  return st;
}

static iree_status_t mkview(Session* s, const iree_hal_dim_t* shape, iree_host_size_t rank,
    iree_hal_element_type_t et, const void* data, size_t bytes, iree_hal_buffer_view_t** out) {
  return iree_hal_buffer_view_allocate_buffer_copy(s->dev, s->alloc, rank, shape, et,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
      iree_make_const_byte_span(data, bytes), out);
}

/* argmax over the logits row at [row] of [bv] ([1, rows, vocab]). */
static iree_status_t argmax_row(Session* s, iree_hal_buffer_view_t* bv, int row, int* out) {
  const size_t bytes = (size_t)s->vocab * sizeof(float);
  if (iree_hal_buffer_view_byte_length(bv) < (iree_device_size_t)((row + 1) * bytes))
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "logits have %d bytes, row %d of vocab %d does not fit",
        (int)iree_hal_buffer_view_byte_length(bv), row, s->vocab);
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(s->dev, iree_hal_buffer_view_buffer(bv), (iree_device_size_t)row * bytes,
      s->logits, bytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  int best = 0; float b = s->logits[0];
  for (int i = 1; i < s->vocab; ++i) if (s->logits[i] > b) { b = s->logits[i]; best = i; }
  *out = best;
  return iree_ok_status();
}

static void release_all(iree_hal_buffer_view_t** v, int n) {
  for (int i = 0; i < n; ++i) if (v[i]) { iree_hal_buffer_view_release(v[i]); v[i] = NULL; }
}

static void destroy(Session* s) {
  if (!s) return;
  if (s->enc) iree_runtime_session_release(s->enc);
  if (s->pre) iree_runtime_session_release(s->pre);
  if (s->step) iree_runtime_session_release(s->step);
  if (s->params) iree_vm_module_release(s->params);
  if (s->dev) iree_hal_device_release(s->dev);
  if (s->inst) iree_runtime_instance_release(s->inst);
  free(s->logits);
  free(s);
}

#define JNIFN(name) Java_sk_ainet_transformers_iree_android_IreeWhisperSession_##name

JNIEXPORT jlong JNICALL JNIFN(nativeCreate)(JNIEnv* env, jobject thiz,
    jstring jdev, jstring jenc, jstring jpre, jstring jstep, jstring jparams, jstring jscope,
    jstring jfenc, jstring jfpre, jstring jfstep, jintArray jgeom) {
  if ((*env)->GetArrayLength(env, jgeom) != 7) { throw_msg(env, "whisper: geometry must have 7 entries"); return 0; }
  jint g[7]; (*env)->GetIntArrayRegion(env, jgeom, 0, 7, g);
  if (g[0] < 1 || g[0] > MAX_LAYERS || g[1] < 1 || g[2] < 1 || g[3] < 1 || g[4] < 1 || g[5] < 2 || g[6] < 2) {
    throw_msg(env, "whisper: geometry out of range"); return 0;
  }
  Session* s = calloc(1, sizeof(Session));
  if (!s) { throw_msg(env, "whisper: out of memory"); return 0; }
  s->n_layers = g[0]; s->d_model = g[1]; s->audio_ctx = g[2]; s->n_mels = g[3]; s->mel_frames = g[4];
  s->max_pos = g[5]; s->vocab = g[6];
  s->logits = malloc((size_t)s->vocab * sizeof(float));

  const char* dev = (*env)->GetStringUTFChars(env, jdev, 0);
  const char* enc = (*env)->GetStringUTFChars(env, jenc, 0);
  const char* pre = (*env)->GetStringUTFChars(env, jpre, 0);
  const char* stp = (*env)->GetStringUTFChars(env, jstep, 0);
  const char* par = (*env)->GetStringUTFChars(env, jparams, 0);
  const char* scope = (*env)->GetStringUTFChars(env, jscope, 0);
  const char* fe = (*env)->GetStringUTFChars(env, jfenc, 0);
  const char* fp = (*env)->GetStringUTFChars(env, jfpre, 0);
  const char* fs = (*env)->GetStringUTFChars(env, jfstep, 0);
  /* the export's modules are anonymous, which IREE names `module` */
  snprintf(s->fn_enc, sizeof s->fn_enc, "module.%s", fe);
  snprintf(s->fn_pre, sizeof s->fn_pre, "module.%s", fp);
  snprintf(s->fn_step, sizeof s->fn_step, "module.%s", fs);

  const char* what = "instance";
  iree_runtime_instance_options_t io; iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  iree_status_t st = s->logits ? iree_runtime_instance_create(&io, iree_allocator_system(), &s->inst)
                               : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "logits row");
  if (iree_status_is_ok(st)) { what = "device"; st = iree_runtime_instance_try_create_default_device(s->inst, iree_make_cstring_view(dev), &s->dev); }
  if (iree_status_is_ok(st)) { what = "parameters"; st = load_params(s, par, scope); }
  if (iree_status_is_ok(st)) { what = "encoder module"; st = load_session(s, enc, 0, &s->enc); }
  if (iree_status_is_ok(st)) { s->alloc = iree_runtime_session_device_allocator(s->enc); what = "prefill module"; st = load_session(s, pre, 1, &s->pre); }
  if (iree_status_is_ok(st)) { what = "step module"; st = load_session(s, stp, 1, &s->step); }

  (*env)->ReleaseStringUTFChars(env, jdev, dev); (*env)->ReleaseStringUTFChars(env, jenc, enc);
  (*env)->ReleaseStringUTFChars(env, jpre, pre); (*env)->ReleaseStringUTFChars(env, jstep, stp);
  (*env)->ReleaseStringUTFChars(env, jparams, par); (*env)->ReleaseStringUTFChars(env, jscope, scope);
  (*env)->ReleaseStringUTFChars(env, jfenc, fe); (*env)->ReleaseStringUTFChars(env, jfpre, fp);
  (*env)->ReleaseStringUTFChars(env, jfstep, fs);
  if (!iree_status_is_ok(st)) {
    char w[64]; snprintf(w, sizeof w, "whisper: %s", what);
    throw_status(env, w, st);
    destroy(s);
    return 0;
  }
  return (jlong)(intptr_t)s;
}

/* encoder → prefill → greedy steps; [out] gets the generated ids (end-of-text excluded). */
/* 1 when the last [k] of out[0..n) already occurred earlier in out. */
static int repeats(const int32_t* out, int n, int k) {
  if (k < 1 || n <= k) return 0;
  for (int s = 0; s + k < n; ++s) {
    int j = 0;
    while (j < k && out[s + j] == out[n - k + j]) ++j;
    if (j == k) return 1;
  }
  return 0;
}

static iree_status_t decode(Session* s, const float* mel, const int32_t* prompt, int n_prompt, int eot, int max_new,
                            int repeat_ngram, int32_t* out, int* n_out) {
  const int nl = s->n_layers;
  iree_hal_buffer_view_t* mel_bv = NULL, *feat = NULL, *prompt_bv = NULL, *logits = NULL;
  iree_hal_buffer_view_t* sk[MAX_LAYERS] = {0}, *sv[MAX_LAYERS] = {0}, *ck[MAX_LAYERS] = {0}, *cv[MAX_LAYERS] = {0};
  iree_runtime_call_t c;
  double t0 = now_ms(), t_enc = 0, t_pre = 0;
  int steps = 0, repeated = 0;
  *n_out = 0;

  /* encoder: mel [1, nMels, melFrames] → features [1, audioCtx, dModel] */
  iree_status_t st = mkview(s, (iree_hal_dim_t[]){1, s->n_mels, s->mel_frames}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      mel, (size_t)s->n_mels * s->mel_frames * sizeof(float), &mel_bv);
  if (iree_status_is_ok(st)) st = iree_runtime_call_initialize_by_name(s->enc, iree_make_cstring_view(s->fn_enc), &c);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_call_inputs_push_back_buffer_view(&c, mel_bv);
    if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
    if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &feat);
    iree_runtime_call_deinitialize(&c);
  }
  t_enc = now_ms() - t0;

  /* prefill: prompt [n] i32, features → logits [1, n, vocab] + per layer selfK, selfV, crossK, crossV */
  if (iree_status_is_ok(st)) st = mkview(s, (iree_hal_dim_t[]){(iree_hal_dim_t)n_prompt}, 1, IREE_HAL_ELEMENT_TYPE_INT_32,
      prompt, (size_t)n_prompt * sizeof(int32_t), &prompt_bv);
  if (iree_status_is_ok(st)) st = iree_runtime_call_initialize_by_name(s->pre, iree_make_cstring_view(s->fn_pre), &c);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_call_inputs_push_back_buffer_view(&c, prompt_bv);
    if (iree_status_is_ok(st)) st = iree_runtime_call_inputs_push_back_buffer_view(&c, feat);
    if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
    if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &logits);
    for (int l = 0; l < nl && iree_status_is_ok(st); ++l) {
      st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &sk[l]);
      if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &sv[l]);
      if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &ck[l]);
      if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &cv[l]);
    }
    iree_runtime_call_deinitialize(&c);
  }
  int tok = eot;
  if (iree_status_is_ok(st)) st = argmax_row(s, logits, n_prompt - 1, &tok);
  if (logits) { iree_hal_buffer_view_release(logits); logits = NULL; }
  t_pre = now_ms() - t0 - t_enc;

  /* greedy steps: tok [1], pos [1], addMask [1,1,1,maxP], wf [1,maxP,1], per layer self/cross K/V
   *            → logits [1,1,vocab], per layer updated selfK, selfV */
  float* add_mask = malloc((size_t)s->max_pos * sizeof(float));
  float* wf = malloc((size_t)s->max_pos * sizeof(float));
  if (iree_status_is_ok(st) && (!add_mask || !wf)) st = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "step masks");
  int pos = n_prompt;
  while (iree_status_is_ok(st) && tok != eot && *n_out < max_new) {
    out[(*n_out)++] = tok;
    if (repeat_ngram > 0 && repeats(out, *n_out, repeat_ngram)) { *n_out -= repeat_ngram; repeated = 1; break; }
    if (pos >= s->max_pos) break;             /* the KV cache is full: the generated ids so far are the transcript */
    for (int i = 0; i < s->max_pos; ++i) { add_mask[i] = i <= pos ? 0.0f : NEG; wf[i] = i == pos ? 1.0f : 0.0f; }
    int32_t tok32 = tok, pos32 = pos;
    iree_hal_buffer_view_t* in[4] = {0};
    st = mkview(s, (iree_hal_dim_t[]){1}, 1, IREE_HAL_ELEMENT_TYPE_INT_32, &tok32, sizeof tok32, &in[0]);
    if (iree_status_is_ok(st)) st = mkview(s, (iree_hal_dim_t[]){1}, 1, IREE_HAL_ELEMENT_TYPE_INT_32, &pos32, sizeof pos32, &in[1]);
    if (iree_status_is_ok(st)) st = mkview(s, (iree_hal_dim_t[]){1, 1, 1, s->max_pos}, 4, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        add_mask, (size_t)s->max_pos * sizeof(float), &in[2]);
    if (iree_status_is_ok(st)) st = mkview(s, (iree_hal_dim_t[]){1, s->max_pos, 1}, 3, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        wf, (size_t)s->max_pos * sizeof(float), &in[3]);
    if (iree_status_is_ok(st)) st = iree_runtime_call_initialize_by_name(s->step, iree_make_cstring_view(s->fn_step), &c);
    if (iree_status_is_ok(st)) {
      for (int i = 0; i < 4 && iree_status_is_ok(st); ++i) st = iree_runtime_call_inputs_push_back_buffer_view(&c, in[i]);
      for (int l = 0; l < nl && iree_status_is_ok(st); ++l) {
        st = iree_runtime_call_inputs_push_back_buffer_view(&c, sk[l]);
        if (iree_status_is_ok(st)) st = iree_runtime_call_inputs_push_back_buffer_view(&c, sv[l]);
        if (iree_status_is_ok(st)) st = iree_runtime_call_inputs_push_back_buffer_view(&c, ck[l]);
        if (iree_status_is_ok(st)) st = iree_runtime_call_inputs_push_back_buffer_view(&c, cv[l]);
      }
      if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
      if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &logits);
      /* the call holds its own references to the inputs: swap in the updated self K/V */
      for (int l = 0; l < nl && iree_status_is_ok(st); ++l) {
        iree_hal_buffer_view_release(sk[l]); sk[l] = NULL;
        iree_hal_buffer_view_release(sv[l]); sv[l] = NULL;
        st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &sk[l]);
        if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &sv[l]);
      }
      iree_runtime_call_deinitialize(&c);
    }
    release_all(in, 4);
    if (iree_status_is_ok(st)) st = argmax_row(s, logits, 0, &tok);
    if (logits) { iree_hal_buffer_view_release(logits); logits = NULL; }
    pos++; steps++;
  }
  free(add_mask); free(wf);
  release_all(sk, nl); release_all(sv, nl); release_all(ck, nl); release_all(cv, nl);
  if (mel_bv) iree_hal_buffer_view_release(mel_bv);
  if (feat) iree_hal_buffer_view_release(feat);
  if (prompt_bv) iree_hal_buffer_view_release(prompt_bv);
  double total = now_ms() - t0;
  snprintf(s->stats, sizeof s->stats, "encoderMs=%.0f,prefillMs=%.0f,stepMs=%.0f,steps=%d,tokens=%d,totalMs=%.0f,endedOnEot=%d,stoppedOnRepeat=%d",
      t_enc, t_pre, total - t_enc - t_pre, steps, *n_out, total, !repeated && tok == eot ? 1 : 0, repeated);
  return st;
}

JNIEXPORT jintArray JNICALL JNIFN(nativeDecode)(JNIEnv* env, jobject thiz,
    jlong handle, jfloatArray jmel, jintArray jprompt, jint eot, jint max_new, jint repeat_ngram) {
  Session* s = (Session*)(intptr_t)handle;
  if (!s) { throw_msg(env, "whisper: session is closed"); return NULL; }
  const jsize nmel = (*env)->GetArrayLength(env, jmel);
  if (nmel != s->n_mels * s->mel_frames) {
    char m[160]; snprintf(m, sizeof m, "whisper: mel has %d values, the encoder takes %d x %d", (int)nmel, s->n_mels, s->mel_frames);
    throw_msg(env, m); return NULL;
  }
  const jsize nprompt = (*env)->GetArrayLength(env, jprompt);
  if (nprompt < 1 || nprompt >= s->max_pos) { throw_msg(env, "whisper: prompt length out of range"); return NULL; }
  if (max_new < 1) max_new = s->max_pos;
  int32_t* out = malloc((size_t)max_new * sizeof(int32_t));
  int32_t* prompt = malloc((size_t)nprompt * sizeof(int32_t));
  if (!out || !prompt) { free(out); free(prompt); throw_msg(env, "whisper: out of memory"); return NULL; }
  (*env)->GetIntArrayRegion(env, jprompt, 0, nprompt, (jint*)prompt);
  jfloat* mel = (*env)->GetFloatArrayElements(env, jmel, 0);
  int n = 0;
  iree_status_t st = decode(s, mel, prompt, nprompt, eot, max_new, repeat_ngram, out, &n);
  (*env)->ReleaseFloatArrayElements(env, jmel, mel, JNI_ABORT);
  free(prompt);
  jintArray result = NULL;
  if (!iree_status_is_ok(st)) {
    throw_status(env, "whisper: decode", st);
  } else {
    result = (*env)->NewIntArray(env, n);
    if (result) (*env)->SetIntArrayRegion(env, result, 0, n, (const jint*)out);
  }
  free(out);
  return result;
}

JNIEXPORT jstring JNICALL JNIFN(nativeStats)(JNIEnv* env, jobject thiz, jlong handle) {
  Session* s = (Session*)(intptr_t)handle;
  return s ? (*env)->NewStringUTF(env, s->stats) : NULL;
}

JNIEXPORT void JNICALL JNIFN(nativeDestroy)(JNIEnv* env, jobject thiz, jlong handle) {
  destroy((Session*)(intptr_t)handle);
}
