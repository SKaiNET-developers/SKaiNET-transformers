/*
 * Stateful KV-cache session for the FunctionGemma contract v1 + addendum (see
 * FunctionGemmaContract.kt): the tool catalog is prefilled once, its per-layer K/V stays on the
 * device, a snapshot of that prefix is restored per turn, the utterance goes in as ONE chunk call
 * (`gemma_prefill_with_past`), and each generated token is one `gemma_with_past` step.
 *
 * Three graphs, three IREE sessions on one device (each graph has its own parameter archive with
 * its own key numbering, so they cannot share a parameter scope):
 *   prefill  : gemma_prefill_at(tokens SEQ i32, emb SEQx{hidden} f32, select 1xSEQ f32)
 *              -> per-layer K,V [1,nKV,SEQ,headDim] ..., token 1xi32          (released after use)
 *   chunk    : gemma_prefill_with_past(tokens C, emb Cx{hidden}, per-base cos/sin [C,headDim],
 *              per-layer K,V (dynamic), per-type masks [1,nHeads,C,past+C], select 1xC)
 *              -> per-layer K,V extended by C ..., token
 *   withPast : gemma_with_past(token 1, emb 1x{hidden}, per-base cos/sin [1,headDim], per-layer K,V)
 *              -> per-layer K,V extended by 1 ..., token
 * All three are the host-gather variants: the token embedding rows are read from the parameter
 * archive here (bf16 -> f32) and passed as an input, because IREE 3.11's SPIR-V backend cannot
 * lower the in-graph gather (iree-org/iree#24035 and its successor). This keeps the Kotlin API
 * token-ids only.
 *
 * Contract facts encoded here (all measured on a MagentaTV One, see SKaiNET-transformers#410):
 *  - the with-past graphs carry NO sliding-window mask: the 15 sliding layers must only ever see
 *    the last `slidingWindow` (512) cache positions -> tail views, zero-copy;
 *  - position enters only through host-built split-half RoPE tables (sign folded into the first
 *    half, as GemmaKvDecoder.splitHalfCosSin) and, for the chunk graph, through additive masks;
 *  - argument order is the tracer's first-use order (FunctionGemmaContract.withPastArgs /
 *    prefillWithPastArgs), with `emb` right after the tokens.
 * Every failure is thrown as an IllegalStateException carrying the formatted iree_status_t.
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "iree/runtime/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/modules/io/parameters/module.h"
#include <android/log.h>

#define TAG "skainet_iree_kv"
#define PARAM_SCOPE "model"
#define MAX_CONCURRENT_PARAM_OPS 16
#define JNIFN(name) Java_sk_ainet_transformers_iree_android_IreeKvSession_##name
#define MASK_NEG (-1.0e30f)

typedef struct {
  iree_runtime_session_t* sess;
  iree_io_parameter_provider_t* provider;
  char* fn;
} Graph;

typedef struct {
  iree_runtime_instance_t* inst;
  iree_hal_device_t* dev;
  Graph withPast, chunk, prefill;
  int hasPrefill;
  /* architecture (from the manifest) */
  int nLayers, headDim, nKV, nHeads, hidden, vocab, window, period, chunkC;
  float baseS, baseG;
  /* device-resident cache: per layer [1, nKV, len[l], headDim] f32 */
  iree_hal_buffer_view_t** kv;   /* 2*nLayers, K then V */
  int* len;                      /* rows per layer (sliding layers stay <= window+1) */
  int pos;                       /* absolute position of the next token */
  /* embedding table: bf16 rows inside the mmapped with-past archive */
  uint8_t* embMap; size_t embMapLen; size_t embOff; int embFound;
} Kv;

typedef struct {
  iree_hal_buffer_view_t** kv;
  int* len;
  int pos;
  int n;
} Snap;

/* ---------- errors ---------- */
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
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg);
  jclass cls = (*env)->FindClass(env, "java/lang/IllegalStateException");
  if (cls) (*env)->ThrowNew(env, cls, msg);
}
static void throw_msg(JNIEnv* env, const char* msg) {
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg);
  jclass cls = (*env)->FindClass(env, "java/lang/IllegalStateException");
  if (cls) (*env)->ThrowNew(env, cls, msg);
}

/* ---------- graphs ---------- */
static iree_status_t graph_open(Kv* k, Graph* g, const char* vmfb, const char* irpa, const char* fn,
                                iree_io_parameter_index_t** out_index) {
  iree_allocator_t alloc = iree_allocator_system();
  iree_runtime_session_options_t o; iree_runtime_session_options_initialize(&o);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      k->inst, &o, k->dev, iree_runtime_instance_host_allocator(k->inst), &g->sess));

  iree_io_file_handle_t* file = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, iree_make_cstring_view(irpa), alloc, &file));
  iree_io_parameter_index_t* index = NULL;
  iree_status_t st = iree_io_parameter_index_create(alloc, &index);
  if (iree_status_is_ok(st)) st = iree_io_parse_irpa_index(file, index, alloc);
  iree_io_file_handle_release(file);
  if (!iree_status_is_ok(st)) { if (index) iree_io_parameter_index_release(index); return st; }

  iree_io_parameter_provider_t* provider = NULL;
  st = iree_io_parameter_index_provider_create(iree_make_cstring_view(PARAM_SCOPE), index, MAX_CONCURRENT_PARAM_OPS, alloc, &provider);
  if (!iree_status_is_ok(st)) { iree_io_parameter_index_release(index); return st; }
  iree_vm_module_t* pm = NULL;
  st = iree_io_parameters_module_create(iree_runtime_instance_vm_instance(k->inst), 1, &provider, alloc, &pm);
  if (iree_status_is_ok(st)) st = iree_runtime_session_append_module(g->sess, pm);
  iree_vm_module_release(pm);
  if (!iree_status_is_ok(st)) { iree_io_parameter_provider_release(provider); iree_io_parameter_index_release(index); return st; }
  g->provider = provider;
  st = iree_runtime_session_append_bytecode_module_from_file(g->sess, vmfb);
  if (!iree_status_is_ok(st)) { iree_io_parameter_index_release(index); return st; }
  g->fn = strdup(fn);
  if (out_index) *out_index = index; else iree_io_parameter_index_release(index);
  return iree_ok_status();
}
static void graph_close(Graph* g) {
  if (g->sess) iree_runtime_session_release(g->sess);
  if (g->provider) iree_io_parameter_provider_release(g->provider);
  free(g->fn);
  memset(g, 0, sizeof *g);
}

/* Find the token-embedding entry (vocab x hidden bf16) in the archive index and mmap the file. */
static int locate_embedding(Kv* k, const char* irpa, iree_io_parameter_index_t* index) {
  uint64_t want = (uint64_t)k->vocab * (uint64_t)k->hidden * 2u;
  iree_host_size_t n = iree_io_parameter_index_count(index);
  for (iree_host_size_t i = 0; i < n; ++i) {
    const iree_io_parameter_index_entry_t* e = NULL;
    if (!iree_status_is_ok(iree_io_parameter_index_get(index, i, &e)) || !e) continue;
    if (e->length == want && e->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
      k->embOff = (size_t)e->storage.file.offset;
      int fd = open(irpa, O_RDONLY);
      if (fd < 0) return 0;
      struct stat stt; if (fstat(fd, &stt) != 0) { close(fd); return 0; }
      k->embMapLen = (size_t)stt.st_size;
      k->embMap = mmap(NULL, k->embMapLen, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (k->embMap == MAP_FAILED) { k->embMap = NULL; return 0; }
      k->embFound = 1;
      return 1;
    }
  }
  return 0;
}
static void embed_rows(const Kv* k, const int32_t* toks, int n, int rows, float* out) {
  memset(out, 0, sizeof(float) * (size_t)rows * (size_t)k->hidden);
  for (int i = 0; i < n; ++i) {
    int32_t t = toks[i]; if (t < 0 || t >= k->vocab) t = 0;
    const uint16_t* src = (const uint16_t*)(k->embMap + k->embOff + (size_t)t * (size_t)k->hidden * 2u);
    float* dst = out + (size_t)i * (size_t)k->hidden;
    for (int j = 0; j < k->hidden; ++j) { uint32_t b = ((uint32_t)src[j]) << 16; memcpy(&dst[j], &b, 4); }
  }
}

/* ---------- buffer views ---------- */
static iree_status_t view_f32(Kv* k, const float* data, int rank, const iree_hal_dim_t* dims, iree_hal_buffer_view_t** out) {
  iree_hal_allocator_t* alloc = iree_hal_device_allocator(k->dev);
  size_t count = 1; for (int i = 0; i < rank; ++i) count *= (size_t)dims[i];
  return iree_hal_buffer_view_allocate_buffer_copy(k->dev, alloc, rank, dims, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
      iree_make_const_byte_span(data, count * 4), out);
}
static iree_status_t view_i32(Kv* k, const int32_t* data, int rank, const iree_hal_dim_t* dims, iree_hal_buffer_view_t** out) {
  iree_hal_allocator_t* alloc = iree_hal_device_allocator(k->dev);
  size_t count = 1; for (int i = 0; i < rank; ++i) count *= (size_t)dims[i];
  return iree_hal_buffer_view_allocate_buffer_copy(k->dev, alloc, rank, dims, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
      iree_make_const_byte_span(data, count * 4), out);
}
/* Zero-copy [1, nKV, rows, headDim] view over rows [rowStart, rowStart+rows) of a cache view (nKV == 1: rows are contiguous). */
static iree_status_t sub_rows(Kv* k, iree_hal_buffer_view_t* v, int rowStart, int rows, iree_hal_buffer_view_t** out) {
  iree_device_size_t rowBytes = (iree_device_size_t)k->nKV * (iree_device_size_t)k->headDim * 4;
  iree_hal_buffer_t* sub = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(iree_hal_buffer_view_buffer(v), (iree_device_size_t)rowStart * rowBytes,
                                               (iree_device_size_t)rows * rowBytes, iree_allocator_system(), &sub));
  iree_hal_dim_t dims[4] = {1, (iree_hal_dim_t)k->nKV, (iree_hal_dim_t)rows, (iree_hal_dim_t)k->headDim};
  iree_status_t st = iree_hal_buffer_view_create(sub, 4, dims, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                                 IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(), out);
  iree_hal_buffer_release(sub);
  return st;
}
static iree_status_t read_i32(Kv* k, iree_hal_buffer_view_t* v, int32_t* out) {
  return iree_hal_device_transfer_d2h(k->dev, iree_hal_buffer_view_buffer(v), 0, out, 4,
                                      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

/* ---------- RoPE tables and chunk masks ---------- */
static void rope_rows(const Kv* k, int pos0, int rows, float base, float* cos_out, float* sin_out) {
  int hd = k->headDim, half = hd / 2;
  for (int r = 0; r < rows; ++r) {
    double p = (double)(pos0 + r);
    for (int i = 0; i < half; ++i) {
      double f = 1.0 / pow((double)base, (2.0 * i) / hd);
      double a = p * f;
      float c = (float)cos(a), s = (float)sin(a);
      cos_out[r * hd + i] = c; cos_out[r * hd + half + i] = c;
      sin_out[r * hd + i] = -s; sin_out[r * hd + half + i] = s;
    }
  }
}
/* Additive mask [1, nHeads, C, past+C]: key j < past is absolute (pos - past + j); key j >= past is chunk row j-past. */
static void chunk_mask(const Kv* k, int past, int nReal, int window, float* out) {
  int C = k->chunkC, K = past + C;
  for (int h = 0; h < k->nHeads; ++h) {
    for (int i = 0; i < C; ++i) {
      float* row = out + ((size_t)h * C + i) * K;
      int a = k->pos + i;
      for (int j = 0; j < K; ++j) {
        int absKey = (j < past) ? (k->pos - past + j) : (k->pos + (j - past));
        int ok = (i < nReal) ? (absKey <= a && (window <= 0 || absKey >= a - window + 1)) : (j == past + i);
        row[j] = ok ? 0.0f : MASK_NEG;
      }
    }
  }
}

/* ---------- calls ---------- */
static iree_status_t invoke(Graph* g, iree_hal_buffer_view_t** ins, int nIn, iree_hal_buffer_view_t** outs, int nOut) {
  iree_runtime_call_t c;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(g->sess, iree_make_cstring_view(g->fn), &c));
  iree_status_t st = iree_ok_status();
  for (int i = 0; i < nIn && iree_status_is_ok(st); ++i) st = iree_runtime_call_inputs_push_back_buffer_view(&c, ins[i]);
  if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
  for (int i = 0; i < nOut && iree_status_is_ok(st); ++i) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &outs[i]);
  iree_runtime_call_deinitialize(&c);
  return st;
}
static int is_global(const Kv* k, int l) { return (l % k->period) == k->period - 1; }

static void release_views(iree_hal_buffer_view_t** v, int n) { for (int i = 0; i < n; ++i) if (v[i]) { iree_hal_buffer_view_release(v[i]); v[i] = NULL; } }

/* Per-layer input views for the current cache: sliding layers see at most `window` rows (tail). Caller releases. */
static iree_status_t cache_inputs(Kv* k, iree_hal_buffer_view_t** outK, iree_hal_buffer_view_t** outV, int* pastS, int* pastG) {
  *pastS = 0; *pastG = 0;
  for (int l = 0; l < k->nLayers; ++l) {
    int len = k->len[l];
    int keep = is_global(k, l) ? len : (len > k->window ? k->window : len);
    if (is_global(k, l)) *pastG = keep; else *pastS = keep;
    if (keep == len) { outK[l] = k->kv[2 * l]; iree_hal_buffer_view_retain(outK[l]); outV[l] = k->kv[2 * l + 1]; iree_hal_buffer_view_retain(outV[l]); }
    else {
      IREE_RETURN_IF_ERROR(sub_rows(k, k->kv[2 * l], len - keep, keep, &outK[l]));
      IREE_RETURN_IF_ERROR(sub_rows(k, k->kv[2 * l + 1], len - keep, keep, &outV[l]));
    }
  }
  return iree_ok_status();
}
/* Adopt fresh K/V outputs (each [1,nKV,rows,headDim]); keep only the first `keepRows(l)` rows as a view. */
static iree_status_t adopt_outputs(Kv* k, iree_hal_buffer_view_t** outs, int addedRows, int realRows, int* pastPerLayer) {
  for (int l = 0; l < k->nLayers; ++l) {
    for (int kv = 0; kv < 2; ++kv) {
      iree_hal_buffer_view_t* full = outs[2 * l + kv];
      int rows = pastPerLayer[l] + realRows;
      iree_hal_buffer_view_t* keep = NULL;
      if (realRows == addedRows) { keep = full; iree_hal_buffer_view_retain(keep); }
      else { IREE_RETURN_IF_ERROR(sub_rows(k, full, 0, rows, &keep)); }
      if (k->kv[2 * l + kv]) iree_hal_buffer_view_release(k->kv[2 * l + kv]);
      k->kv[2 * l + kv] = keep;
    }
    k->len[l] = pastPerLayer[l] + realRows;
  }
  return iree_ok_status();
}

/* ---------- JNI: create / destroy ---------- */
static int jint_field(JNIEnv* env, jobject spec, const char* name) {
  jclass c = (*env)->GetObjectClass(env, spec);
  jfieldID f = (*env)->GetFieldID(env, c, name, "I");
  return f ? (*env)->GetIntField(env, spec, f) : 0;
}
static float jfloat_field(JNIEnv* env, jobject spec, const char* name) {
  jclass c = (*env)->GetObjectClass(env, spec);
  jfieldID f = (*env)->GetFieldID(env, c, name, "F");
  return f ? (*env)->GetFloatField(env, spec, f) : 0.f;
}

JNIEXPORT jlong JNICALL JNIFN(nativeCreate)(JNIEnv* env, jobject thiz, jstring jdev, jobject spec,
    jstring jvmfbWithPast, jstring jirpaWithPast, jstring jfnWithPast,
    jstring jvmfbChunk, jstring jirpaChunk, jstring jfnChunk,
    jstring jvmfbPrefill, jstring jirpaPrefill, jstring jfnPrefill) {
  Kv* k = calloc(1, sizeof(Kv));
  k->nLayers = jint_field(env, spec, "nLayers"); k->headDim = jint_field(env, spec, "headDim");
  k->nKV = jint_field(env, spec, "nKvHeads"); k->nHeads = jint_field(env, spec, "nHeads");
  k->hidden = jint_field(env, spec, "hiddenSize"); k->vocab = jint_field(env, spec, "vocabSize");
  k->window = jint_field(env, spec, "slidingWindow"); k->period = jint_field(env, spec, "globalLayerPeriod");
  k->chunkC = jint_field(env, spec, "chunk");
  k->baseS = jfloat_field(env, spec, "slidingRopeBase"); k->baseG = jfloat_field(env, spec, "globalRopeBase");
  if (k->nLayers <= 0 || k->headDim <= 0 || k->nKV != 1 || k->nHeads <= 0 || k->hidden <= 0 || k->vocab <= 0 || k->period <= 0 || k->chunkC <= 0) {
    throw_msg(env, "IreeKvSession: bad spec (nKvHeads must be 1 for zero-copy cache views; all sizes > 0)"); free(k); return 0;
  }
  k->kv = calloc((size_t)2 * k->nLayers, sizeof(void*)); k->len = calloc((size_t)k->nLayers, sizeof(int));

  const char* dev = (*env)->GetStringUTFChars(env, jdev, 0);
  const char* vWP = (*env)->GetStringUTFChars(env, jvmfbWithPast, 0); const char* iWP = (*env)->GetStringUTFChars(env, jirpaWithPast, 0); const char* fWP = (*env)->GetStringUTFChars(env, jfnWithPast, 0);
  const char* vCH = (*env)->GetStringUTFChars(env, jvmfbChunk, 0); const char* iCH = (*env)->GetStringUTFChars(env, jirpaChunk, 0); const char* fCH = (*env)->GetStringUTFChars(env, jfnChunk, 0);
  const char* vPF = jvmfbPrefill ? (*env)->GetStringUTFChars(env, jvmfbPrefill, 0) : NULL;
  const char* iPF = jirpaPrefill ? (*env)->GetStringUTFChars(env, jirpaPrefill, 0) : NULL;
  const char* fPF = jfnPrefill ? (*env)->GetStringUTFChars(env, jfnPrefill, 0) : NULL;

  iree_runtime_instance_options_t io; iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  iree_status_t st = iree_runtime_instance_create(&io, iree_allocator_system(), &k->inst);
  if (iree_status_is_ok(st)) st = iree_runtime_instance_try_create_default_device(k->inst, iree_make_cstring_view(dev), &k->dev);
  iree_io_parameter_index_t* wpIndex = NULL;
  if (iree_status_is_ok(st)) st = graph_open(k, &k->withPast, vWP, iWP, fWP, &wpIndex);
  if (iree_status_is_ok(st) && !locate_embedding(k, iWP, wpIndex)) st = iree_make_status(IREE_STATUS_NOT_FOUND, "no vocab x hidden bf16 embedding entry in the with-past archive");
  if (wpIndex) iree_io_parameter_index_release(wpIndex);
  if (iree_status_is_ok(st)) st = graph_open(k, &k->chunk, vCH, iCH, fCH, NULL);
  if (iree_status_is_ok(st) && vPF && iPF && fPF) { st = graph_open(k, &k->prefill, vPF, iPF, fPF, NULL); k->hasPrefill = iree_status_is_ok(st); }

  (*env)->ReleaseStringUTFChars(env, jdev, dev);
  (*env)->ReleaseStringUTFChars(env, jvmfbWithPast, vWP); (*env)->ReleaseStringUTFChars(env, jirpaWithPast, iWP); (*env)->ReleaseStringUTFChars(env, jfnWithPast, fWP);
  (*env)->ReleaseStringUTFChars(env, jvmfbChunk, vCH); (*env)->ReleaseStringUTFChars(env, jirpaChunk, iCH); (*env)->ReleaseStringUTFChars(env, jfnChunk, fCH);
  if (vPF) (*env)->ReleaseStringUTFChars(env, jvmfbPrefill, vPF); if (iPF) (*env)->ReleaseStringUTFChars(env, jirpaPrefill, iPF); if (fPF) (*env)->ReleaseStringUTFChars(env, jfnPrefill, fPF);

  if (!iree_status_is_ok(st)) {
    throw_status(env, "IreeKvSession.create", st);
    graph_close(&k->prefill); graph_close(&k->chunk); graph_close(&k->withPast);
    if (k->embMap) munmap(k->embMap, k->embMapLen);
    if (k->dev) iree_hal_device_release(k->dev); if (k->inst) iree_runtime_instance_release(k->inst);
    free(k->kv); free(k->len); free(k); return 0;
  }
  __android_log_print(ANDROID_LOG_INFO, TAG, "session open: device=%s layers=%d headDim=%d heads=%d window=%d chunk=%d prefill=%d", dev, k->nLayers, k->headDim, k->nHeads, k->window, k->chunkC, k->hasPrefill);
  return (jlong)(intptr_t)k;
}

JNIEXPORT void JNICALL JNIFN(nativeReleasePrefill)(JNIEnv* env, jobject thiz, jlong h) {
  Kv* k = (Kv*)(intptr_t)h; if (!k || !k->hasPrefill) return;
  graph_close(&k->prefill); k->hasPrefill = 0;
}
JNIEXPORT void JNICALL JNIFN(nativeDestroy)(JNIEnv* env, jobject thiz, jlong h) {
  Kv* k = (Kv*)(intptr_t)h; if (!k) return;
  release_views(k->kv, 2 * k->nLayers);
  graph_close(&k->prefill); graph_close(&k->chunk); graph_close(&k->withPast);
  if (k->embMap) munmap(k->embMap, k->embMapLen);
  if (k->dev) iree_hal_device_release(k->dev);
  if (k->inst) iree_runtime_instance_release(k->inst);
  free(k->kv); free(k->len); free(k);
}
JNIEXPORT jint JNICALL JNIFN(nativePosition)(JNIEnv* env, jobject thiz, jlong h) {
  Kv* k = (Kv*)(intptr_t)h; return k ? k->pos : -1;
}

/* ---------- prefill: the catalog prefix, once ---------- */
JNIEXPORT jint JNICALL JNIFN(nativePrefill)(JNIEnv* env, jobject thiz, jlong h, jintArray jtoks, jint n) {
  Kv* k = (Kv*)(intptr_t)h; if (!k) { throw_msg(env, "prefill: closed session"); return -1; }
  if (!k->hasPrefill) { throw_msg(env, "prefill: no prefill graph in this session (or already released)"); return -1; }
  jsize seq = (*env)->GetArrayLength(env, jtoks);
  if (n <= 0 || n > seq) { throw_msg(env, "prefill: n out of range"); return -1; }
  int32_t* toks = malloc((size_t)seq * 4); jint* src = (*env)->GetIntArrayElements(env, jtoks, 0);
  for (int i = 0; i < seq; ++i) toks[i] = (int32_t)src[i]; (*env)->ReleaseIntArrayElements(env, jtoks, src, JNI_ABORT);
  float* emb = malloc((size_t)seq * k->hidden * 4); embed_rows(k, toks, n, seq, emb);
  float* sel = calloc((size_t)seq, 4); sel[n - 1] = 1.0f;

  iree_hal_buffer_view_t* ins[3] = {0}; int nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_dim_t dT[1] = {(iree_hal_dim_t)seq}, dE[2] = {(iree_hal_dim_t)seq, (iree_hal_dim_t)k->hidden}, dS[2] = {1, (iree_hal_dim_t)seq};
  iree_status_t st = view_i32(k, toks, 1, dT, &ins[0]);
  if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[1]);
  if (iree_status_is_ok(st)) st = view_f32(k, sel, 2, dS, &ins[2]);
  if (iree_status_is_ok(st)) st = invoke(&k->prefill, ins, 3, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));   /* nothing cached before the prefill */
    release_views(k->kv, 2 * k->nLayers);
    st = adopt_outputs(k, outs, seq, n, past);
    free(past);
    k->pos = n;
  }
  release_views(ins, 3); release_views(outs, nOut); free(outs); free(toks); free(emb); free(sel);
  if (!iree_status_is_ok(st)) { throw_status(env, "prefill", st); return -1; }
  return tok;
}

/* ---------- chunk: the utterance, one call ---------- */
JNIEXPORT jint JNICALL JNIFN(nativeChunk)(JNIEnv* env, jobject thiz, jlong h, jintArray jtoks, jint n) {
  Kv* k = (Kv*)(intptr_t)h; if (!k) { throw_msg(env, "chunk: closed session"); return -1; }
  int C = k->chunkC;
  jsize got = (*env)->GetArrayLength(env, jtoks);
  if (n <= 0 || n > C || got < n) { throw_msg(env, "chunk: n must be in 1..chunk and <= tokens.length"); return -1; }
  if (k->len[0] == 0) { throw_msg(env, "chunk: empty cache — prefill first"); return -1; }
  int32_t* toks = calloc((size_t)C, 4); jint* src = (*env)->GetIntArrayElements(env, jtoks, 0);
  for (int i = 0; i < n; ++i) toks[i] = (int32_t)src[i]; (*env)->ReleaseIntArrayElements(env, jtoks, src, JNI_ABORT);
  int hd = k->headDim;
  float* emb = malloc((size_t)C * k->hidden * 4); embed_rows(k, toks, n, C, emb);
  float* cosS = malloc((size_t)C * hd * 4); float* sinS = malloc((size_t)C * hd * 4);
  float* cosG = malloc((size_t)C * hd * 4); float* sinG = malloc((size_t)C * hd * 4);
  rope_rows(k, k->pos, C, k->baseS, cosS, sinS); rope_rows(k, k->pos, C, k->baseG, cosG, sinG);
  float* sel = calloc((size_t)C, 4); sel[n - 1] = 1.0f;

  int nIn = 2 + 4 + 2 * k->nLayers + 2 + 1, nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** ins = calloc((size_t)nIn, sizeof(void*));
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_buffer_view_t** cK = calloc((size_t)k->nLayers, sizeof(void*)); iree_hal_buffer_view_t** cV = calloc((size_t)k->nLayers, sizeof(void*));
  int pastS = 0, pastG = 0;
  iree_status_t st = cache_inputs(k, cK, cV, &pastS, &pastG);
  float* maskS = NULL; float* maskG = NULL;
  iree_hal_dim_t dT[1] = {(iree_hal_dim_t)C}, dE[2] = {(iree_hal_dim_t)C, (iree_hal_dim_t)k->hidden}, dR[2] = {(iree_hal_dim_t)C, (iree_hal_dim_t)hd}, dSel[2] = {1, (iree_hal_dim_t)C};
  iree_hal_buffer_view_t *vCosS = NULL, *vSinS = NULL, *vCosG = NULL, *vSinG = NULL, *vMaskS = NULL, *vMaskG = NULL;
  if (iree_status_is_ok(st)) {
    maskS = malloc((size_t)k->nHeads * C * (pastS + C) * 4); chunk_mask(k, pastS, n, k->window, maskS);
    maskG = malloc((size_t)k->nHeads * C * (pastG + C) * 4); chunk_mask(k, pastG, n, 0, maskG);
    iree_hal_dim_t dMS[4] = {1, (iree_hal_dim_t)k->nHeads, (iree_hal_dim_t)C, (iree_hal_dim_t)(pastS + C)};
    iree_hal_dim_t dMG[4] = {1, (iree_hal_dim_t)k->nHeads, (iree_hal_dim_t)C, (iree_hal_dim_t)(pastG + C)};
    int i = 0;
    st = view_i32(k, toks, 1, dT, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, cosS, 2, dR, &vCosS);
    if (iree_status_is_ok(st)) st = view_f32(k, sinS, 2, dR, &vSinS);
    if (iree_status_is_ok(st)) st = view_f32(k, cosG, 2, dR, &vCosG);
    if (iree_status_is_ok(st)) st = view_f32(k, sinG, 2, dR, &vSinG);
    if (iree_status_is_ok(st)) st = view_f32(k, maskS, 4, dMS, &vMaskS);
    if (iree_status_is_ok(st)) st = view_f32(k, maskG, 4, dMG, &vMaskG);
    if (iree_status_is_ok(st)) {
      int introS = 0, introG = 0;
      for (int l = 0; l < k->nLayers; ++l) {
        int g = is_global(k, l);
        if (g && !introG) { ins[i++] = vCosG; ins[i++] = vSinG; }
        if (!g && !introS) { ins[i++] = vCosS; ins[i++] = vSinS; }
        ins[i++] = cK[l]; ins[i++] = cV[l];
        if (g && !introG) { ins[i++] = vMaskG; introG = 1; }
        if (!g && !introS) { ins[i++] = vMaskS; introS = 1; }
      }
      st = view_f32(k, sel, 2, dSel, &ins[i++]);
      if (iree_status_is_ok(st) && i != nIn) st = iree_make_status(IREE_STATUS_INTERNAL, "chunk: assembled %d inputs, expected %d", i, nIn);
    }
  }
  if (iree_status_is_ok(st)) st = invoke(&k->chunk, ins, nIn, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));
    for (int l = 0; l < k->nLayers; ++l) past[l] = is_global(k, l) ? pastG : pastS;
    st = adopt_outputs(k, outs, C, n, past);
    free(past);
    k->pos += n;
  }
  /* inputs: tokens, emb, select are owned here (ins[0], ins[1], last); cos/sin/mask views once each; cache views once each */
  if (ins[0]) iree_hal_buffer_view_release(ins[0]); if (ins[1]) iree_hal_buffer_view_release(ins[1]); if (ins[nIn - 1]) iree_hal_buffer_view_release(ins[nIn - 1]);
  if (vCosS) iree_hal_buffer_view_release(vCosS); if (vSinS) iree_hal_buffer_view_release(vSinS); if (vCosG) iree_hal_buffer_view_release(vCosG); if (vSinG) iree_hal_buffer_view_release(vSinG);
  if (vMaskS) iree_hal_buffer_view_release(vMaskS); if (vMaskG) iree_hal_buffer_view_release(vMaskG);
  release_views(cK, k->nLayers); release_views(cV, k->nLayers); release_views(outs, nOut);
  free(ins); free(outs); free(cK); free(cV); free(toks); free(emb); free(cosS); free(sinS); free(cosG); free(sinG); free(sel); free(maskS); free(maskG);
  if (!iree_status_is_ok(st)) { throw_status(env, "chunk", st); return -1; }
  return tok;
}

/* ---------- step: one generated token ---------- */
JNIEXPORT jint JNICALL JNIFN(nativeStep)(JNIEnv* env, jobject thiz, jlong h, jint token) {
  Kv* k = (Kv*)(intptr_t)h; if (!k) { throw_msg(env, "step: closed session"); return -1; }
  if (k->len[0] == 0) { throw_msg(env, "step: empty cache — prefill first"); return -1; }
  int hd = k->headDim; int32_t t = token;
  float* emb = malloc((size_t)k->hidden * 4); embed_rows(k, &t, 1, 1, emb);
  float cosS[1024], sinS[1024], cosG[1024], sinG[1024];
  if (hd > 1024) { throw_msg(env, "step: headDim > 1024 unsupported"); free(emb); return -1; }
  rope_rows(k, k->pos, 1, k->baseS, cosS, sinS); rope_rows(k, k->pos, 1, k->baseG, cosG, sinG);

  int nIn = 2 + 4 + 2 * k->nLayers, nOut = 2 * k->nLayers + 1;
  iree_hal_buffer_view_t** ins = calloc((size_t)nIn, sizeof(void*));
  iree_hal_buffer_view_t** outs = calloc((size_t)nOut, sizeof(void*));
  iree_hal_buffer_view_t** cK = calloc((size_t)k->nLayers, sizeof(void*)); iree_hal_buffer_view_t** cV = calloc((size_t)k->nLayers, sizeof(void*));
  int pastS = 0, pastG = 0;
  iree_status_t st = cache_inputs(k, cK, cV, &pastS, &pastG);
  iree_hal_dim_t dT[1] = {1}, dE[2] = {1, (iree_hal_dim_t)k->hidden}, dR[2] = {1, (iree_hal_dim_t)hd};
  iree_hal_buffer_view_t *vCosS = NULL, *vSinS = NULL, *vCosG = NULL, *vSinG = NULL;
  if (iree_status_is_ok(st)) {
    int i = 0;
    st = view_i32(k, &t, 1, dT, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, emb, 2, dE, &ins[i++]);
    if (iree_status_is_ok(st)) st = view_f32(k, cosS, 2, dR, &vCosS);
    if (iree_status_is_ok(st)) st = view_f32(k, sinS, 2, dR, &vSinS);
    if (iree_status_is_ok(st)) st = view_f32(k, cosG, 2, dR, &vCosG);
    if (iree_status_is_ok(st)) st = view_f32(k, sinG, 2, dR, &vSinG);
    if (iree_status_is_ok(st)) {
      int introS = 0, introG = 0;
      for (int l = 0; l < k->nLayers; ++l) {
        int g = is_global(k, l);
        if (g && !introG) { ins[i++] = vCosG; ins[i++] = vSinG; introG = 1; }
        if (!g && !introS) { ins[i++] = vCosS; ins[i++] = vSinS; introS = 1; }
        ins[i++] = cK[l]; ins[i++] = cV[l];
      }
      if (i != nIn) st = iree_make_status(IREE_STATUS_INTERNAL, "step: assembled %d inputs, expected %d", i, nIn);
    }
  }
  if (iree_status_is_ok(st)) st = invoke(&k->withPast, ins, nIn, outs, nOut);
  int32_t tok = -1;
  if (iree_status_is_ok(st)) st = read_i32(k, outs[nOut - 1], &tok);
  if (iree_status_is_ok(st)) {
    int* past = calloc((size_t)k->nLayers, sizeof(int));
    for (int l = 0; l < k->nLayers; ++l) past[l] = is_global(k, l) ? pastG : pastS;
    st = adopt_outputs(k, outs, 1, 1, past);
    free(past);
    k->pos += 1;
  }
  if (ins[0]) iree_hal_buffer_view_release(ins[0]); if (ins[1]) iree_hal_buffer_view_release(ins[1]);
  if (vCosS) iree_hal_buffer_view_release(vCosS); if (vSinS) iree_hal_buffer_view_release(vSinS); if (vCosG) iree_hal_buffer_view_release(vCosG); if (vSinG) iree_hal_buffer_view_release(vSinG);
  release_views(cK, k->nLayers); release_views(cV, k->nLayers); release_views(outs, nOut);
  free(ins); free(outs); free(cK); free(cV); free(emb);
  if (!iree_status_is_ok(st)) { throw_status(env, "step", st); return -1; }
  return tok;
}

/* ---------- snapshots: retain the current views, zero-copy ---------- */
JNIEXPORT jlong JNICALL JNIFN(nativeSnapshot)(JNIEnv* env, jobject thiz, jlong h) {
  Kv* k = (Kv*)(intptr_t)h; if (!k) { throw_msg(env, "snapshot: closed session"); return 0; }
  Snap* s = calloc(1, sizeof(Snap)); s->n = 2 * k->nLayers; s->pos = k->pos;
  s->kv = calloc((size_t)s->n, sizeof(void*)); s->len = calloc((size_t)k->nLayers, sizeof(int));
  for (int i = 0; i < s->n; ++i) { s->kv[i] = k->kv[i]; if (s->kv[i]) iree_hal_buffer_view_retain(s->kv[i]); }
  memcpy(s->len, k->len, sizeof(int) * (size_t)k->nLayers);
  return (jlong)(intptr_t)s;
}
JNIEXPORT void JNICALL JNIFN(nativeRestore)(JNIEnv* env, jobject thiz, jlong h, jlong hs) {
  Kv* k = (Kv*)(intptr_t)h; Snap* s = (Snap*)(intptr_t)hs;
  if (!k || !s) { throw_msg(env, "restore: closed session or null snapshot"); return; }
  if (s->n != 2 * k->nLayers) { throw_msg(env, "restore: snapshot from another session"); return; }
  for (int i = 0; i < s->n; ++i) {
    if (s->kv[i]) iree_hal_buffer_view_retain(s->kv[i]);
    if (k->kv[i]) iree_hal_buffer_view_release(k->kv[i]);
    k->kv[i] = s->kv[i];
  }
  memcpy(k->len, s->len, sizeof(int) * (size_t)k->nLayers);
  k->pos = s->pos;
}
JNIEXPORT void JNICALL JNIFN(nativeReleaseSnapshot)(JNIEnv* env, jobject thiz, jlong hs) {
  Snap* s = (Snap*)(intptr_t)hs; if (!s) return;
  release_views(s->kv, s->n); free(s->kv); free(s->len); free(s);
}
