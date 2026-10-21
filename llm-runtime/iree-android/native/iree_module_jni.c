/* iree_module_jni.c — generic single-shot runner for one compiled IREE module.
 *
 * Drives ANY vmfb whose entry function takes one float32 tensor and returns one float32 tensor
 * (`tensor<d0x…xf32> -> tensor<e0x…xf32>`, static shapes): a vision backbone with its head, a
 * classifier, an embedding graph. Weights are either baked into the vmfb or EXTERNAL (scope
 * "model") — bound at create time from a `.irpa` parameter archive via the io_parameters VM
 * module, exactly as in iree_redecode_jni.c. The vmfb path, the optional `.irpa` path and the
 * function name are caller-supplied; this file knows nothing about any specific model.
 *
 * Exposes to sk.ainet.transformers.iree.android.IreeModuleRunner:
 *   long    nativeCreate(String device, String vmfbPath, String irpaPath, String functionName,
 *                        int taskTopologyGroupCount)
 *             irpaPath "" = weights baked into the vmfb; group count <= 0 = IREE's own topology
 *   float[] nativeRun(long h, float[] input, int[] inputShape, int[] outputShape)
 *             outputShape (length >= 1 + MAX_RANK) receives [rank, d0, d1, …] of the result
 *   void    nativeDestroy(long h)
 *
 * Every failure is logged under the "skainet_iree" tag and returns 0 / NULL (see #404).
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "iree/base/tooling/flags.h"
#include "iree/runtime/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/modules/io/parameters/module.h"
#include <android/log.h>

#define TAG "skainet_iree"
#define PARAM_SCOPE "model"
#define MAX_CONCURRENT_PARAM_OPS 16
#define MAX_RANK 8

static void log_and_ignore(const char* what, iree_status_t st) {
  if (iree_status_is_ok(st)) return;
  char* buf = NULL; iree_host_size_t len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  if (iree_status_to_string(st, &alloc, &buf, &len) && buf) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s: %.*s", what, (int)len, buf);
    iree_allocator_free(alloc, buf);
  } else {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s: status code %d", what, (int)iree_status_code(st));
  }
  iree_status_ignore(st);
}

typedef struct {
  iree_runtime_instance_t* inst;
  iree_hal_device_t* dev;
  iree_runtime_session_t* sess;
  iree_io_parameter_provider_t* param_provider;
  char* fn_name;
} Runner;

#define JNIFN(name) Java_sk_ainet_transformers_iree_android_IreeModuleRunner_##name

/* The `.irpa` as a "model"-scoped parameter provider, appended BEFORE the bytecode module so its
 * util.global initializers resolve at link time. Same as iree_redecode_jni.c. */
static iree_status_t append_parameters_module(Runner* r, const char* irpa) {
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
      iree_runtime_instance_vm_instance(r->inst), 1, &provider, alloc, &params_module);
  if (iree_status_is_ok(st)) st = iree_runtime_session_append_module(r->sess, params_module);
  iree_vm_module_release(params_module);
  if (!iree_status_is_ok(st)) { iree_io_parameter_provider_release(provider); return st; }
  r->param_provider = provider;
  return st;
}

/* --task_topology_group_count for the next local-task device; <= 0 leaves IREE's detection. */
static iree_status_t apply_task_topology(int group_count) {
  if (group_count <= 0) return iree_ok_status();
  char flag[64];
  snprintf(flag, sizeof(flag), "--task_topology_group_count=%d", group_count);
  char* argv_storage[] = { "skainet_iree_module", flag };
  char** argv = argv_storage;
  int argc = 2;
  return iree_flags_parse(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
}

static void runner_free(Runner* r) {
  if (!r) return;
  if (r->param_provider) iree_io_parameter_provider_release(r->param_provider);
  if (r->sess) iree_runtime_session_release(r->sess);
  if (r->dev) iree_hal_device_release(r->dev);
  if (r->inst) iree_runtime_instance_release(r->inst);
  free(r->fn_name);
  free(r);
}

JNIEXPORT jlong JNICALL JNIFN(nativeCreate)(JNIEnv* env, jobject thiz,
    jstring jdev, jstring jvmfb, jstring jirpa, jstring jfn, jint group_count) {
  const char* dev = (*env)->GetStringUTFChars(env, jdev, 0);
  const char* vmfb = (*env)->GetStringUTFChars(env, jvmfb, 0);
  const char* irpa = (*env)->GetStringUTFChars(env, jirpa, 0);
  const char* fn = (*env)->GetStringUTFChars(env, jfn, 0);
  Runner* r = calloc(1, sizeof(Runner));
  if (r) r->fn_name = strdup(fn);

  iree_runtime_instance_options_t io; iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  iree_status_t st = (r && r->fn_name)
      ? iree_runtime_instance_create(&io, iree_allocator_system(), &r->inst)
      : iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  if (iree_status_is_ok(st)) st = apply_task_topology((int)group_count);
  if (iree_status_is_ok(st)) {
    st = iree_runtime_instance_try_create_default_device(r->inst, iree_make_cstring_view(dev), &r->dev);
  }
  if (iree_status_is_ok(st)) {
    iree_runtime_session_options_t o; iree_runtime_session_options_initialize(&o);
    st = iree_runtime_session_create_with_device(
        r->inst, &o, r->dev, iree_runtime_instance_host_allocator(r->inst), &r->sess);
  }
  if (iree_status_is_ok(st) && irpa[0] != '\0') st = append_parameters_module(r, irpa);
  if (iree_status_is_ok(st)) st = iree_runtime_session_append_bytecode_module_from_file(r->sess, vmfb);

  (*env)->ReleaseStringUTFChars(env, jdev, dev);
  (*env)->ReleaseStringUTFChars(env, jvmfb, vmfb);
  (*env)->ReleaseStringUTFChars(env, jirpa, irpa);
  (*env)->ReleaseStringUTFChars(env, jfn, fn);
  if (!iree_status_is_ok(st)) {
    log_and_ignore("IreeModuleRunner.nativeCreate (instance/device/session/parameters/module)", st);
    runner_free(r);
    return 0;
  }
  return (jlong)(intptr_t)r;
}

JNIEXPORT jfloatArray JNICALL JNIFN(nativeRun)(JNIEnv* env, jobject thiz,
    jlong handle, jfloatArray jin, jintArray jshape, jintArray joutShape) {
  Runner* r = (Runner*)(intptr_t)handle; if (!r) return NULL;
  jsize rank = (*env)->GetArrayLength(env, jshape);
  if (rank <= 0 || rank > MAX_RANK || (*env)->GetArrayLength(env, joutShape) < 1 + MAX_RANK) return NULL;

  iree_hal_dim_t dims[MAX_RANK];
  jint* sh = (*env)->GetIntArrayElements(env, jshape, 0);
  iree_host_size_t count = 1;
  for (int i = 0; i < rank; ++i) { dims[i] = (iree_hal_dim_t)sh[i]; count *= (iree_host_size_t)sh[i]; }
  (*env)->ReleaseIntArrayElements(env, jshape, sh, JNI_ABORT);
  if (count != (iree_host_size_t)(*env)->GetArrayLength(env, jin)) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "nativeRun: input has %d values, its shape needs %zu",
                        (int)(*env)->GetArrayLength(env, jin), (size_t)count);
    return NULL;
  }

  /* Copy straight from the pinned Java array into a device buffer. */
  jfloat* src = (*env)->GetPrimitiveArrayCritical(env, jin, 0);
  if (!src) return NULL;
  iree_hal_buffer_view_t* in = NULL;
  iree_status_t st = iree_hal_buffer_view_allocate_buffer_copy(
      r->dev, iree_runtime_session_device_allocator(r->sess), (iree_host_size_t)rank, dims,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
      iree_make_const_byte_span(src, count * sizeof(float)), &in);
  (*env)->ReleasePrimitiveArrayCritical(env, jin, src, JNI_ABORT);
  if (!iree_status_is_ok(st)) { log_and_ignore("nativeRun: input buffer_view", st); return NULL; }

  iree_runtime_call_t c;
  st = iree_runtime_call_initialize_by_name(r->sess, iree_make_cstring_view(r->fn_name), &c);
  if (!iree_status_is_ok(st)) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "nativeRun: function '%s' not found (names are module-qualified, e.g. module.main)", r->fn_name);
    log_and_ignore("nativeRun: call_initialize_by_name", st);
    iree_hal_buffer_view_release(in);
    return NULL;
  }
  st = iree_runtime_call_inputs_push_back_buffer_view(&c, in);
  iree_hal_buffer_view_release(in);
  if (iree_status_is_ok(st)) st = iree_runtime_call_invoke(&c, 0);
  iree_hal_buffer_view_t* out = NULL;
  if (iree_status_is_ok(st)) st = iree_runtime_call_outputs_pop_front_buffer_view(&c, &out);
  iree_runtime_call_deinitialize(&c);
  if (!iree_status_is_ok(st)) { log_and_ignore("nativeRun: invoke / output", st); if (out) iree_hal_buffer_view_release(out); return NULL; }

  iree_host_size_t outRank = iree_hal_buffer_view_shape_rank(out);
  if (iree_hal_buffer_view_element_type(out) != IREE_HAL_ELEMENT_TYPE_FLOAT_32 || outRank > MAX_RANK) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "nativeRun: '%s' must return one float32 tensor of rank <= %d (rank %u)",
                        r->fn_name, MAX_RANK, (unsigned)outRank);
    iree_hal_buffer_view_release(out);
    return NULL;
  }
  jint outShape[1 + MAX_RANK] = { (jint)outRank };
  iree_host_size_t outCount = 1;
  for (iree_host_size_t i = 0; i < outRank; ++i) {
    iree_hal_dim_t d = iree_hal_buffer_view_shape_dim(out, i);
    outShape[1 + i] = (jint)d; outCount *= (iree_host_size_t)d;
  }

  jfloatArray result = (*env)->NewFloatArray(env, (jsize)outCount);
  if (result) {
    jfloat* dst = (*env)->GetPrimitiveArrayCritical(env, result, 0);
    st = dst ? iree_hal_device_transfer_d2h(r->dev, iree_hal_buffer_view_buffer(out), 0, dst,
                                            outCount * sizeof(float), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                            iree_infinite_timeout())
             : iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    if (dst) (*env)->ReleasePrimitiveArrayCritical(env, result, dst, 0);
    if (!iree_status_is_ok(st)) { log_and_ignore("nativeRun: transfer_d2h", st); result = NULL; }
  }
  iree_hal_buffer_view_release(out);
  if (result) (*env)->SetIntArrayRegion(env, joutShape, 0, (jsize)(1 + outRank), outShape);
  return result;
}

JNIEXPORT void JNICALL JNIFN(nativeDestroy)(JNIEnv* env, jobject thiz, jlong handle) {
  runner_free((Runner*)(intptr_t)handle);
}
