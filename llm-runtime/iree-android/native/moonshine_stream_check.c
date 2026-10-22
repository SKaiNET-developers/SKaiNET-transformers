/* moonshine_stream_check.c — drives libskainet_moonshine_stream.so on a device without an app.
 *
 * A minimal JNIEnv (only the functions the library calls) backed by plain C data: dlopen the library,
 * nativeCreate once, then for every PCM file feed it in 40 ms chunks paced like a microphone, nativeFinish,
 * and print one line per file: name <TAB> transcript <TAB> stats. Stage timings also go to logcat
 * (tag moonshine-timing), as they do in an app.
 *
 *   moonshine_stream_check <lib.so> <device> <model dir> <pcm file or dir> [--fast]
 *     device     vulkan | local-task
 *     model dir  the eight files of a materialized model (frontend.vmfb ... dec_embed.bin), plus
 *                frontend_step.vmfb when the library should use the incremental frontend
 *     pcm        headerless little-endian s16 mono 16 kHz; a directory = every *.pcm in it, name order
 *     --fast     feed without pacing (timings then no longer resemble a live stream)
 *
 * Build (same image as the library):
 *   docker run --rm -v "$PWD/native":/src -w /src --entrypoint sh skainet/iree-android:3.11.0 -c \
 *     '/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi29-clang \
 *        -O2 -o out/moonshine_stream_check moonshine_stream_check.c -ldl'
 */
#include <jni.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { jsize len; void* data; } Arr;

static const char* GetStringUTFChars(JNIEnv* e, jstring s, jboolean* c) { if (c) *c = 0; return (const char*)s; }
static void ReleaseStringUTFChars(JNIEnv* e, jstring s, const char* p) {}
static jstring NewStringUTF(JNIEnv* e, const char* u) { return (jstring)strdup(u ? u : ""); }
static jsize GetArrayLength(JNIEnv* e, jarray a) { return ((Arr*)a)->len; }
static jfloat* GetFloatArrayElements(JNIEnv* e, jfloatArray a, jboolean* c) { if (c) *c = 0; return ((Arr*)a)->data; }
static void ReleaseFloatArrayElements(JNIEnv* e, jfloatArray a, jfloat* p, jint m) {}
static jclass FindClass(JNIEnv* e, const char* n) { return (jclass)n; }
static jint ThrowNew(JNIEnv* e, jclass c, const char* m) { fprintf(stderr, "%s: %s\n", (const char*)c, m); return 0; }

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6; }
static void sleep_until(double t) {
  double d = t - now_ms(); if (d <= 0) return;
  struct timespec ts = { (time_t)(d / 1000), (long)((d - (time_t)(d / 1000) * 1000) * 1e6) }; nanosleep(&ts, NULL);
}

static float* read_pcm(const char* path, int* n) {
  FILE* f = fopen(path, "rb"); if (!f) return NULL;
  fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
  short* s = malloc(bytes); if (fread(s, 1, bytes, f) != (size_t)bytes) { fclose(f); free(s); return NULL; }
  fclose(f);
  *n = (int)(bytes / 2);
  float* x = malloc(sizeof(float) * (*n));
  for (int i = 0; i < *n; ++i) x[i] = s[i] / 32768.0f;
  free(s); return x;
}

static int is_pcm(const struct dirent* d) { size_t l = strlen(d->d_name); return l > 4 && !strcmp(d->d_name + l - 4, ".pcm"); }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s <lib.so> <device> <model dir> <pcm file|dir> [--fast]\n", argv[0]); return 2; }
  int fast = argc > 5 && !strcmp(argv[5], "--fast");
  void* lib = dlopen(argv[1], RTLD_NOW); if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
#define SYM(n) dlsym(lib, "Java_sk_ainet_transformers_iree_android_IreeMoonshineStream_" n)
  jlong (*create)(JNIEnv*, jobject, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jstring, jstring) = SYM("nativeCreate");
  jstring (*feed)(JNIEnv*, jobject, jlong, jfloatArray) = SYM("nativeFeedPcm");
  jstring (*finish)(JNIEnv*, jobject, jlong) = SYM("nativeFinish");
  jstring (*stats)(JNIEnv*, jobject, jlong) = SYM("nativeStats");
  void (*destroy)(JNIEnv*, jobject, jlong) = SYM("nativeDestroy");
  if (!create || !feed || !finish || !stats || !destroy) { fprintf(stderr, "dlsym failed\n"); return 3; }

  struct JNINativeInterface ni; memset(&ni, 0, sizeof ni);
  ni.GetStringUTFChars = GetStringUTFChars; ni.ReleaseStringUTFChars = ReleaseStringUTFChars; ni.NewStringUTF = NewStringUTF;
  ni.GetArrayLength = GetArrayLength; ni.GetFloatArrayElements = GetFloatArrayElements;
  ni.ReleaseFloatArrayElements = ReleaseFloatArrayElements; ni.FindClass = FindClass; ni.ThrowNew = ThrowNew;
  const struct JNINativeInterface* nip = &ni;
  JNIEnv* env = (JNIEnv*)&nip;

  const char* names[] = {"frontend.vmfb", "encoder.vmfb", "adapter.vmfb", "prefill.vmfb", "step.vmfb", "params.irpa", "vocab.bin", "dec_embed.bin"};
  static char p[8][1024];
  for (int i = 0; i < 8; ++i) snprintf(p[i], sizeof p[i], "%s/%s", argv[3], names[i]);
  double t0 = now_ms();
  jlong h = create(env, NULL, (jstring)argv[2], (jstring)p[0], (jstring)p[1], (jstring)p[2], (jstring)p[3], (jstring)p[4], (jstring)p[5], (jstring)p[6], (jstring)p[7]);
  if (!h) { fprintf(stderr, "nativeCreate failed\n"); return 4; }
  fprintf(stderr, "create %.0fms\n", now_ms() - t0);

  struct dirent** list = NULL; int nfiles = 0;
  DIR* d = opendir(argv[4]);
  if (d) { closedir(d); nfiles = scandir(argv[4], &list, is_pcm, alphasort); }
  for (int k = 0; k < (d ? nfiles : 1); ++k) {
    char path[2048];
    if (d) snprintf(path, sizeof path, "%s/%s", argv[4], list[k]->d_name); else snprintf(path, sizeof path, "%s", argv[4]);
    int n = 0; float* x = read_pcm(path, &n);
    if (!x) { fprintf(stderr, "cannot read %s\n", path); continue; }
    const int CH = 640;   /* 40 ms */
    double start = now_ms();
    for (int off = 0; off < n; off += CH) {
      if (!fast) sleep_until(start + off / 16.0);
      Arr a = { n - off < CH ? n - off : CH, x + off };
      jstring part = feed(env, NULL, h, (jfloatArray)&a);
      if (part) free((void*)part);
    }
    if (!fast) sleep_until(start + n / 16.0);
    double tf = now_ms();
    jstring out = finish(env, NULL, h);
    double fin = now_ms() - tf;
    jstring st = stats(env, NULL, h);
    const char* base = strrchr(path, '/');
    printf("%s\t%s\tfinishWallMs=%.0f,%s\n", base ? base + 1 : path, out ? (const char*)out : "", fin, st ? (const char*)st : "");
    fflush(stdout);
    if (out) free((void*)out);
    if (st) free((void*)st);
    free(x);
  }
  destroy(env, NULL, h);
  return 0;
}
