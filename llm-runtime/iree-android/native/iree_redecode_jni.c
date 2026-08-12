/* iree_redecode_jni.c — generic fixed-seq redecode cartridge over the IREE runtime.
 *
 * Drives ANY DSL-compiled vmfb that follows the redecode graph contract established by
 * :llm-inference:smollm2's SmolLm2ExportHarness (and mirrored by any future per-model
 * export harness): ONE fixed-seq function, `tensor<1xSEQxi32> -> tensor<SEQxi32>`, with
 * the DSL in-graph argMax tail already applied (no host-side argmax over a [SEQ, vocab]
 * logits tensor). Weights are EXTERNAL (scope "model") — bound at session-create time
 * from a `.irpa` parameter archive via the io_parameters VM module, not baked into the
 * vmfb. The vmfb path, `.irpa` path, and exported function name are all caller-supplied —
 * this file knows nothing about any specific model. The host drives the re-decode loop
 * (GemmaDecoder pattern, see :llm-runtime:gemma-iree): pad tokens to SEQ, invoke, read
 * back all SEQ predicted ids, pick the one at the caller's requested position, append it,
 * and call again — causal masking makes padding-then-growing safe.
 *
 * Exposes to sk.ainet.transformers.iree.android.IreeRedecodeSession:
 *   long  nativeCreate(String device, String vmfbPath, String irpaPath, String functionName)
 *   int[] nativeStep(long h, int[] tokenIds)   // tokenIds.length == the vmfb's fixed SEQ
 *   void  nativeDestroy(long h)
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "iree/runtime/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/modules/io/parameters/module.h"

#define PARAM_SCOPE "model"
#define MAX_CONCURRENT_PARAM_OPS 16

typedef struct {
  iree_runtime_instance_t* inst;
  iree_hal_device_t* dev;
  iree_runtime_session_t* sess;
  iree_io_parameter_provider_t* param_provider;
  char* fn_name;
} Session;

#define JNIFN(name) Java_sk_ainet_transformers_iree_android_IreeRedecodeSession_##name

/* Loads the `.irpa` at |irpa| into a parameter-index provider scoped "model" and appends
 * the io_parameters module to |s->sess| BEFORE the compiled bytecode module — the compiled
 * module's util.global initializers resolve against it at link time. On success, |s->param_provider|
 * holds an extra retain so the provider outlives the session for teardown ordering. */
static iree_status_t append_parameters_module(Session* s, const char* irpa) {
  iree_allocator_t alloc = iree_allocator_system();
  iree_io_file_handle_t* file = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ, iree_make_cstring_view(irpa), alloc, &file));

  iree_io_parameter_index_t* index = NULL;
  iree_status_t st = iree_io_parameter_index_create(alloc, &index);
  if (iree_status_is_ok(st)) st = iree_io_parse_irpa_index(file, index, alloc);
  iree_io_file_handle_release(file);
  if (!iree_status_is_ok(st)) { if (index) iree_io_parameter_index_release(index); return st; }

  iree_io_parameter_provider_t* provider = NULL;
  st = iree_io_parameter_index_provider_create(
      iree_make_cstring_view(PARAM_SCOPE), index, MAX_CONCURRENT_PARAM_OPS, alloc, &provider);
  iree_io_parameter_index_release(index);
  if (!iree_status_is_ok(st)) return st;

  iree_vm_module_t* params_module = NULL;
  st = iree_io_parameters_module_create(
      iree_runtime_instance_vm_instance(s->inst), 1, &provider, alloc, &params_module);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_session_append_module(s->sess, params_module);
  }
  iree_vm_module_release(params_module);
  if (!iree_status_is_ok(st)) { iree_io_parameter_provider_release(provider); return st; }

  s->param_provider = provider;  /* transfers the caller's own reference — no extra retain */
  return st;
}

JNIEXPORT jlong JNICALL JNIFN(nativeCreate)(JNIEnv* env, jobject thiz,
    jstring jdev, jstring jvmfb, jstring jirpa, jstring jfn) {
  const char* dev = (*env)->GetStringUTFChars(env, jdev, 0);
  const char* vmfb = (*env)->GetStringUTFChars(env, jvmfb, 0);
  const char* irpa = (*env)->GetStringUTFChars(env, jirpa, 0);
  const char* fn = (*env)->GetStringUTFChars(env, jfn, 0);
  Session* s = calloc(1, sizeof(Session));
  jlong ret = 0;

  if (s) {
    size_t fnLen = strlen(fn) + 1;
    s->fn_name = malloc(fnLen);
    if (s->fn_name) memcpy(s->fn_name, fn, fnLen);
  }

  iree_runtime_instance_options_t io; iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  iree_status_t st = (s && s->fn_name)
      ? iree_runtime_instance_create(&io, iree_allocator_system(), &s->inst)
      : iree_status_from_code(IREE_STATUS_INTERNAL);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_instance_try_create_default_device(
        s->inst, iree_make_cstring_view(dev), &s->dev);
  }
  if (iree_status_is_ok(st)) {
    iree_runtime_session_options_t o; iree_runtime_session_options_initialize(&o);
    st = iree_runtime_session_create_with_device(
        s->inst, &o, s->dev, iree_runtime_instance_host_allocator(s->inst), &s->sess);
  }
  /* Parameters module BEFORE the compiled module — see append_parameters_module doc. */
  if (iree_status_is_ok(st)) st = append_parameters_module(s, irpa);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_session_append_bytecode_module_from_file(s->sess, vmfb);
  }
  if (iree_status_is_ok(st)) ret = (jlong)(intptr_t)s;

  (*env)->ReleaseStringUTFChars(env, jdev, dev);
  (*env)->ReleaseStringUTFChars(env, jvmfb, vmfb);
  (*env)->ReleaseStringUTFChars(env, jirpa, irpa);
  (*env)->ReleaseStringUTFChars(env, jfn, fn);
  if (!iree_status_is_ok(st)) {
    iree_status_ignore(st);
    if (s) {
      if (s->param_provider) iree_io_parameter_provider_release(s->param_provider);
      if (s->sess) iree_runtime_session_release(s->sess);
      if (s->dev) iree_hal_device_release(s->dev);
      if (s->inst) iree_runtime_instance_release(s->inst);
      free(s->fn_name);
      free(s);
    }
    return 0;
  }
  return ret;
}

JNIEXPORT jintArray JNICALL JNIFN(nativeStep)(JNIEnv* env, jobject thiz,
    jlong handle, jintArray jtoks) {
  Session* s = (Session*)(intptr_t)handle; if (!s) return NULL;
  jsize seq = (*env)->GetArrayLength(env, jtoks);
  if (seq <= 0) return NULL;

  int32_t* toks = malloc((size_t)seq * sizeof(int32_t));
  jint* src = (*env)->GetIntArrayElements(env, jtoks, 0);
  for (int i = 0; i < seq; ++i) toks[i] = (int32_t)src[i];
  (*env)->ReleaseIntArrayElements(env, jtoks, src, JNI_ABORT);

  iree_hal_allocator_t* alloc = iree_runtime_session_device_allocator(s->sess);
  iree_hal_buffer_view_t* in = NULL;
  iree_status_t st = iree_hal_buffer_view_allocate_buffer_copy(s->dev, alloc, 2,
      (iree_hal_dim_t[]){1, (iree_hal_dim_t)seq}, IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
      iree_make_const_byte_span(toks, (size_t)seq * 4), &in);
  free(toks);
  if (!iree_status_is_ok(st)) { iree_status_ignore(st); return NULL; }

  iree_runtime_call_t c;
  if (!iree_status_is_ok(iree_runtime_call_initialize_by_name(
          s->sess, iree_make_cstring_view(s->fn_name), &c))) {
    iree_hal_buffer_view_release(in); return NULL;
  }
  iree_runtime_call_inputs_push_back_buffer_view(&c, in);
  iree_hal_buffer_view_t* out = NULL;
  int ok = iree_status_is_ok(iree_runtime_call_invoke(&c, 0));
  if (ok) iree_runtime_call_outputs_pop_front_buffer_view(&c, &out);
  iree_runtime_call_deinitialize(&c);
  iree_hal_buffer_view_release(in);
  if (!ok || !out) { if (out) iree_hal_buffer_view_release(out); return NULL; }

  /* The redecode graph contract: output is rank-1, length SEQ (argMax already applied
   * in-graph — see the file doc). A caller pointing this at a vmfb that doesn't follow
   * the contract fails loudly here instead of silently misreading device memory. */
  iree_host_size_t outRank = iree_hal_buffer_view_shape_rank(out);
  iree_hal_dim_t outLen = outRank > 0 ? iree_hal_buffer_view_shape_dim(out, 0) : 0;
  if (outRank != 1 || outLen != (iree_hal_dim_t)seq) {
    iree_hal_buffer_view_release(out);
    return NULL;
  }

  int32_t* result = malloc((size_t)seq * sizeof(int32_t));
  ok = iree_status_is_ok(iree_hal_device_transfer_d2h(s->dev,
      iree_hal_buffer_view_buffer(out), 0, result, (size_t)seq * sizeof(int32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  iree_hal_buffer_view_release(out);
  if (!ok) { free(result); return NULL; }

  jintArray outArr = (*env)->NewIntArray(env, seq);
  if (outArr) (*env)->SetIntArrayRegion(env, outArr, 0, seq, (jint*)result);
  free(result);
  return outArr;
}

JNIEXPORT void JNICALL JNIFN(nativeDestroy)(JNIEnv* env, jobject thiz, jlong handle) {
  Session* s = (Session*)(intptr_t)handle; if (!s) return;
  if (s->param_provider) iree_io_parameter_provider_release(s->param_provider);
  if (s->sess) iree_runtime_session_release(s->sess);
  if (s->dev) iree_hal_device_release(s->dev);
  if (s->inst) iree_runtime_instance_release(s->inst);
  free(s->fn_name);
  free(s);
}
