/* whisper_session_check.c — drives libskainet_whisper_session.so on a device without an app.
 *
 * A minimal JNIEnv (only the functions the library calls) backed by plain C data: dlopen the library, then
 * nativeCreate → nativeDecode (repeated, for warm timings) → nativeStats → nativeDestroy, and print the ids.
 *
 *   whisper_session_check <lib.so> <encoderDevice[:decoderDevice]> <dir> <mel.bin> <geometry> <promptIds> <eot> [runs] [repeatNgram]
 *     dir       holds encoder.vmfb, prefill.vmfb, step.vmfb, params.irpa
 *     mel.bin   little-endian f32, nMels * melFrames values
 *     geometry  nLayers,dModel,audioCtx,nMels,melFrames,maxPositions,vocab   (from manifest.json)
 *     promptIds comma-separated, e.g. 50258,50261,50359,50363
 *
 * Build (same image as the library):
 *   docker run --rm -v "$PWD/native":/src -w /src --entrypoint sh skainet/iree-android:3.11.0 -c \
 *     '/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi29-clang \
 *        -O2 -o out/whisper_session_check whisper_session_check.c -ldl'
 */
#include <jni.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { jsize len; void* data; } Arr;
static int thrown;

static const char* GetStringUTFChars(JNIEnv* e, jstring s, jboolean* c) { if (c) *c = 0; return (const char*)s; }
static void ReleaseStringUTFChars(JNIEnv* e, jstring s, const char* p) {}
static jstring NewStringUTF(JNIEnv* e, const char* u) { return (jstring)strdup(u ? u : ""); }
static jsize GetArrayLength(JNIEnv* e, jarray a) { return ((Arr*)a)->len; }
static void GetIntArrayRegion(JNIEnv* e, jintArray a, jsize at, jsize n, jint* buf) { memcpy(buf, (jint*)((Arr*)a)->data + at, n * sizeof(jint)); }
static void SetIntArrayRegion(JNIEnv* e, jintArray a, jsize at, jsize n, const jint* buf) { memcpy((jint*)((Arr*)a)->data + at, buf, n * sizeof(jint)); }
static jintArray NewIntArray(JNIEnv* e, jsize n) { Arr* a = malloc(sizeof(Arr)); a->len = n; a->data = calloc(n ? n : 1, sizeof(jint)); return (jintArray)a; }
static jfloat* GetFloatArrayElements(JNIEnv* e, jfloatArray a, jboolean* c) { if (c) *c = 0; return ((Arr*)a)->data; }
static void ReleaseFloatArrayElements(JNIEnv* e, jfloatArray a, jfloat* p, jint m) {}
static jclass FindClass(JNIEnv* e, const char* n) { return (jclass)n; }
static jint ThrowNew(JNIEnv* e, jclass c, const char* m) { fprintf(stderr, "%s: %s\n", (const char*)c, m); thrown = 1; return 0; }

static Arr ints(const char* csv) {
  Arr a = { 0, calloc(512, sizeof(jint)) };
  for (char* p = (char*)csv; *p && a.len < 512; ) { ((jint*)a.data)[a.len++] = (jint)strtol(p, &p, 10); if (*p == ',') ++p; }
  return a;
}

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: %s lib.so device dir mel.bin geometry promptIds eot [runs] [repeatNgram]\n", argv[0]); return 1; }
  void* lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
#define SYM(n) dlsym(lib, "Java_sk_ainet_transformers_iree_android_IreeWhisperSession_" n)
  jlong (*create)(JNIEnv*, jobject, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jintArray) = SYM("nativeCreate");
  jintArray (*decode)(JNIEnv*, jobject, jlong, jfloatArray, jintArray, jint, jint, jint) = SYM("nativeDecode");
  jstring (*stats)(JNIEnv*, jobject, jlong) = SYM("nativeStats");
  void (*destroy)(JNIEnv*, jobject, jlong) = SYM("nativeDestroy");
  if (!create || !decode || !stats || !destroy) { fprintf(stderr, "dlsym failed\n"); return 3; }

  struct JNINativeInterface ni; memset(&ni, 0, sizeof ni);
  ni.GetStringUTFChars = GetStringUTFChars; ni.ReleaseStringUTFChars = ReleaseStringUTFChars; ni.NewStringUTF = NewStringUTF;
  ni.GetArrayLength = GetArrayLength; ni.GetIntArrayRegion = GetIntArrayRegion; ni.SetIntArrayRegion = SetIntArrayRegion;
  ni.NewIntArray = NewIntArray; ni.GetFloatArrayElements = GetFloatArrayElements;
  ni.ReleaseFloatArrayElements = ReleaseFloatArrayElements; ni.FindClass = FindClass; ni.ThrowNew = ThrowNew;
  const struct JNINativeInterface* nip = &ni;
  JNIEnv* env = (JNIEnv*)&nip;

  char enc[512], pre[512], step[512], par[512];
  snprintf(enc, sizeof enc, "%s/encoder.vmfb", argv[3]); snprintf(pre, sizeof pre, "%s/prefill.vmfb", argv[3]);
  snprintf(step, sizeof step, "%s/step.vmfb", argv[3]); snprintf(par, sizeof par, "%s/params.irpa", argv[3]);
  Arr geom = ints(argv[5]), prompt = ints(argv[6]);
  /* "vulkan:local-task" = encoder on the GPU, prefill and step on the CPU */
  char encdev[64]; snprintf(encdev, sizeof encdev, "%s", argv[2]);
  char* colon = strchr(encdev, ':');
  const char* decdev = colon ? colon + 1 : encdev;
  if (colon) *colon = 0;
  jlong h = create(env, NULL, (jstring)encdev, (jstring)decdev, (jstring)enc, (jstring)pre, (jstring)step, (jstring)par, (jstring)"model",
      (jstring)"whisper_encoder", (jstring)"whisper_prefill", (jstring)"whisper_step", (jintArray)&geom);
  if (!h || thrown) { fprintf(stderr, "nativeCreate failed\n"); return 4; }

  FILE* f = fopen(argv[4], "rb");
  if (!f) { fprintf(stderr, "no %s\n", argv[4]); return 5; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  Arr mel = { (jsize)(n / 4), malloc(n) };
  if (fread(mel.data, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 5; }
  fclose(f);

  int runs = argc > 8 ? atoi(argv[8]) : 1;
  int ngram = argc > 9 ? atoi(argv[9]) : 0;
  for (int r = 0; r < runs; ++r) {
    Arr* ids = (Arr*)decode(env, NULL, h, (jfloatArray)&mel, (jintArray)&prompt, atoi(argv[7]), 0, ngram);
    if (!ids || thrown) { fprintf(stderr, "nativeDecode failed\n"); return 6; }
    printf("run %d ids:", r);
    for (int i = 0; i < ids->len; ++i) printf("%s%d", i ? "," : " ", ((jint*)ids->data)[i]);
    printf("\nrun %d stats: %s\n", r, (const char*)stats(env, NULL, h));
  }
  destroy(env, NULL, h);
  return 0;
}
