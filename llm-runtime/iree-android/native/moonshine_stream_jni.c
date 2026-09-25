/* moonshine_stream_jni.c — streaming Moonshine v2 ASR over the IREE runtime (Vulkan or CPU).
 *
 * The streaming runtime for the five SKaiNET-DSL graphs of the Moonshine v2 export
 * (frontend / encoder / adapter / masked prefill / DYNAMIC with_past) plus the shared
 * decoder parameter archive, driven as a real streaming loop:
 *
 *   feed PCM ->[ring buffer]-> 64-frame windows, hop 44 -> frontend+encoder+adapter (device)
 *     -> finalized-memory append -> incremental greedy decode -> cumulative partial text
 *   finish() -> flush tail, EXACT full re-decode over the final memory -> final text
 *
 * Exposes to sk.ainet.transformers.iree.android.IreeMoonshineStream:
 *   long   nativeCreate(String device, String fe, String enc, String adp, String prefill,
 *                       String step, String params, String vocab, String embed)
 *   String nativeFeedPcm(long h, float[] pcm)   // 16 kHz mono [-1,1]; returns new partial or null
 *   String nativeFinish(long h)                 // end of utterance: exact final transcript; resets
 *   void   nativeReset(long h)                  // abort utterance, keep engine
 *   void   nativeDestroy(long h)
 *
 * Decode strategy (measured on a Mali GPU): the with_past step is ~120 ms and flat in
 * past length; a full re-decode per hop overruns the 0.88 s hop budget from ~the third hop.
 * Partials therefore reuse the self-KV across hops (prefix-locked, cross-K/V refreshed per
 * hop from the grown memory — an approximation, tokens never retracted), and finish() runs
 * one exact full re-decode so the transcript handed to NLU is bit-faithful.
 *
 * One engine = one utterance at a time; calls are single-threaded (host serializes).
 * Per-hop stage timings go to stderr/logcat as "moonshine-timing" lines.
 */
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef __ANDROID__
#include <android/log.h>
#define TLOG(...) __android_log_print(ANDROID_LOG_INFO, "moonshine-timing", __VA_ARGS__)
#else
#define TLOG(...) fprintf(stderr, __VA_ARGS__)
#endif
#include "iree/runtime/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/vm/api.h"

/* Decoder geometry — overridable at compile time (-DL=10 -DHD=64 -DDIM=512 -DENC_DIM=620
 * builds the `small` variant; defaults are the tiny/tiny-de geometry). DIM is the DECODER
 * stream width (embed rows, memory rows, step inputs); ENC_DIM is the frontend/encoder
 * feature width — identical for tiny (320/320), split for small (620 enc / 512 dec, the
 * adapter's proj bridges them inside the adapter vmfb). Vocab is NOT compiled in (read
 * from vocab.bin, see below). Making these runtime-dynamic is a filed follow-up. */
#ifndef L
#define L 6
#endif
#ifndef NH
#define NH 8
#endif
#ifndef HD
#define HD 40
#endif
#ifndef DIM
#define DIM 320
#endif
#ifndef ENC_DIM
#define ENC_DIM DIM
#endif
/* vocab size is NOT compiled in: it is read from vocab.bin's u32 count header
 * (EN tiny-streaming: 32768; the German v2 checkpoints: 12288) and drives the
 * dec_embed load and the logits argmax via e->vocab.n. */
#define BOS 1
#define EOS 2

#define SPF 320           /* audio samples per feature frame (4 x 80) */
#define CHUNK 64          /* encoder window, feature frames (1.28 s) */
#define WINDOW 16         /* encoder left context frames */
#define LOOKAHEAD 4       /* encoder lookahead frames */
#define HOP (CHUNK - WINDOW - LOOKAHEAD)   /* 44 frames = 0.88 s */
#define FE_LC 4           /* frontend causal left-context frames re-fed per window */
#define FE_IN ((CHUNK + FE_LC) * SPF)      /* 21760 samples */
#define FE_OUT (CHUNK + FE_LC)             /* 68 frames out */
#define MAXMEM 256        /* compiled cross-memory pad (5.12 s of finalized speech) */
#define MAXTOK 48
#define FINISH_TOKENS 24  /* hard ceiling for the final budget (see finish_budget) */
#define TOKENS_PER_SAMPLE (6.5f / 16000.0f)  /* the model card's cap: max_new_tokens = samples * 6.5/16000 + 2 */
#define HOP_TOKENS 6      /* max new tokens decoded per hop (budget guard) */
#define RESTART_HOPS 2    /* first hops re-decode exactly instead of incrementally */
#define PEEK_FRAMES 36    /* early-peek window: first partial at ~0.72 s instead of 1.28 s */
#define STEP_SLICE 2      /* tokens decoded per feed call — partials trickle out mid-hop */
#define MAXPCM ((MAXMEM + CHUNK) * SPF)    /* ring capacity, samples */
#define NEGMASK (-1.0e30f)

typedef struct { int n; char** t; } Vocab;

typedef struct {
  iree_runtime_instance_t* inst;
  iree_hal_device_t* dev;
  iree_hal_allocator_t* alloc;
  iree_runtime_session_t* fe, *enc, *adp, *pre, *step;
  iree_vm_module_t* params;
  Vocab vocab;
  float* embed;                    /* [vocab.n][DIM] */
  float rope_c[MAXTOK + 4][HD], rope_s[MAXTOK + 4][HD];

  /* utterance state */
  float* pcm; int npcm;            /* fed samples (capped) */
  int win;                         /* next window index */
  float mem[MAXMEM][DIM];          /* finalized memory rows */
  int nmem;
  /* incremental decode state */
  iree_hal_buffer_view_t* sk[L], *sv[L], *ck[L], *cv[L];
  int has_self, has_cross;
  int toks[MAXTOK]; int ntoks; int pos;
  int peeked;          /* early-peek window emitted (rows rewritten by the real window 0) */
  int ended_eos;       /* final decode ended on EOS (clean) vs budget/cycle (restart risk) */
  size_t shown_len;    /* longest partial surfaced so far — keeps the VIL text monotone */
  int pending;         /* tokens still budgeted for the current hop's decode */
  int halted;          /* decode hit EOS/cycle; stop continuing until the next hop */
  char text[4096];
} Engine;

static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- vocab (u32 count, then u16 len + utf8 piece per id; "\xE2\x96\x81" = word boundary) */
static Vocab vocab_load(const char* p) {
  Vocab v = {0, NULL}; FILE* f = fopen(p, "rb"); if (!f) return v;
  uint32_t n = 0; if (fread(&n, 4, 1, f) != 1 || n == 0 || n > 1000000) { fclose(f); return v; }
  v.n = (int)n; v.t = calloc(n, sizeof(char*));
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t l = 0; if (fread(&l, 2, 1, f) != 1) break;
    char* s = malloc((size_t)l + 1);
    if (l && fread(s, 1, l, f) != l) { free(s); break; }
    s[l] = 0;
    if (l >= 2 && s[0] == '<' && s[l - 1] == '>') s[0] = 0;   /* specials decode to nothing */
    v.t[i] = s;
  }
  fclose(f); return v;
}

static void detok_append(const Vocab* v, int id, char* out, size_t cap) {
  if (id < 0 || id >= v->n || !v->t[id] || !v->t[id][0]) return;
  const char* t = v->t[id];
  size_t ol = strlen(out);
  while (*t && ol + 4 < cap) {
    if ((unsigned char)t[0] == 0xE2 && (unsigned char)t[1] == 0x96 && (unsigned char)t[2] == 0x81) {
      out[ol++] = ' '; t += 3;
    } else {
      out[ol++] = *t++;
    }
  }
  out[ol] = 0;
}

/* ---- IREE plumbing (same patterns as whisper_iree_jni.c) ------------------------------- */
static iree_vm_module_t* load_params(Engine* e, const char* path) {
  iree_allocator_t host = iree_allocator_system();
  iree_io_parameter_index_t* index = NULL;
  if (!iree_status_is_ok(iree_io_parameter_index_create(host, &index))) return NULL;
  iree_io_file_handle_t* file = NULL;
  iree_vm_module_t* mod = NULL;
  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(iree_io_file_handle_open(
          IREE_IO_FILE_MODE_READ, iree_make_cstring_view(path), host, &file)) &&
      iree_status_is_ok(iree_io_parse_irpa_index(file, index, host)) &&
      iree_status_is_ok(iree_io_parameter_index_provider_create(
          iree_make_cstring_view("model"), index,
          IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS, host, &provider))) {
    iree_io_parameters_module_create(iree_runtime_instance_vm_instance(e->inst), 1, &provider, host, &mod);
  }
  if (provider) iree_io_parameter_provider_release(provider);
  if (index) iree_io_parameter_index_release(index);
  if (file) iree_io_file_handle_release(file);
  return mod;
}

static void log_status(const char* where, iree_status_t st) {
  char* buf = NULL; iree_host_size_t len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  if (iree_status_to_string(st, &alloc, &buf, &len) && buf) {
    TLOG("  %s: %.*s\n", where, (int)len, buf);
    iree_allocator_free(alloc, buf);
  } else {
    TLOG("  %s: (status code %d, no string)\n", where, (int)iree_status_code(st));
  }
}

static iree_runtime_session_t* load_session(Engine* e, const char* path, int with_params) {
  iree_runtime_session_options_t o; iree_runtime_session_options_initialize(&o);
  iree_runtime_session_t* s = NULL;
  iree_status_t st = iree_runtime_session_create_with_device(e->inst, &o, e->dev,
      iree_runtime_instance_host_allocator(e->inst), &s);
  if (!iree_status_is_ok(st)) { log_status("session_create", st); iree_status_ignore(st); return NULL; }
  if (with_params && e->params) {
    st = iree_runtime_session_append_module(s, e->params);
    if (!iree_status_is_ok(st)) { log_status("append_params", st); iree_status_ignore(st); return NULL; }
  }
  st = iree_runtime_session_append_bytecode_module_from_file(s, path);
  if (!iree_status_is_ok(st)) { log_status("append_bytecode", st); iree_status_ignore(st); return NULL; }
  return s;
}

static iree_hal_buffer_view_t* mkview(Engine* e, const iree_hal_dim_t* shape, iree_host_size_t rank,
    iree_hal_element_type_t et, const void* data, size_t bytes) {
  iree_hal_buffer_view_t* bv = NULL;
  if (!iree_status_is_ok(iree_hal_buffer_view_allocate_buffer_copy(e->dev, e->alloc, rank, shape, et,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){ .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT },
        iree_make_const_byte_span(data, bytes), &bv))) return NULL;
  return bv;
}

static int pull(Engine* e, iree_hal_buffer_view_t* bv, void* dst, size_t bytes) {
  return iree_status_is_ok(iree_hal_device_transfer_d2h(e->dev, iree_hal_buffer_view_buffer(bv),
      0, dst, bytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout())) ? 0 : -1;
}

static int argmax_logits(Engine* e, iree_hal_buffer_view_t* bv) {
  const int n = e->vocab.n;
  float* row = malloc((size_t)n * sizeof(float)); if (!row) return EOS;
  if (pull(e, bv, row, (size_t)n * sizeof(float))) { free(row); return EOS; }
  int best = 0; float b = row[0];
  for (int i = 1; i < n; ++i) if (row[i] > b) { b = row[i]; best = i; }
  free(row); return best;
}

static void rope_init(Engine* e) {
  const int HALF = HD / 2, ROT = 16;
  for (int p = 0; p < MAXTOK + 4; ++p) {
    for (int i = 0; i < HALF; ++i) {
      double c = 1.0, s = 0.0;
      if (i < ROT) { double a = p * pow(10000.0, -(double)i / ROT); c = cos(a); s = sin(a); }
      e->rope_c[p][2 * i] = (float)c; e->rope_c[p][2 * i + 1] = (float)c;
      e->rope_s[p][2 * i] = (float)-s; e->rope_s[p][2 * i + 1] = (float)s;
    }
  }
}

static void release_self(Engine* e) {
  if (!e->has_self) return;
  for (int l = 0; l < L; ++l) { iree_hal_buffer_view_release(e->sk[l]); iree_hal_buffer_view_release(e->sv[l]); }
  e->has_self = 0;
}
static void release_cross(Engine* e) {
  if (!e->has_cross) return;
  for (int l = 0; l < L; ++l) { iree_hal_buffer_view_release(e->ck[l]); iree_hal_buffer_view_release(e->cv[l]); }
  e->has_cross = 0;
}

/* ---- stages ----------------------------------------------------------------------------- */
static const char* FN_FE  = "module.moonshine_v2_frontend";
static const char* FN_ENC = "module.moonshine_v2_encoder";
static const char* FN_ADP = "module.moonshine_v2_adapter";
static const char* FN_PRE = "module.moonshine_v2_decoder_prefill";
static const char* FN_STEP = "module.moonshine_v2_decoder_with_past";

/* one-input/one-output helper (frontend, encoder) */
static iree_hal_buffer_view_t* run1(Engine* e, iree_runtime_session_t* s, const char* fn,
                                    iree_hal_buffer_view_t* in) {
  iree_runtime_call_t c;
  if (!iree_status_is_ok(iree_runtime_call_initialize_by_name(s, iree_make_cstring_view(fn), &c)))
    return NULL;
  iree_runtime_call_inputs_push_back_buffer_view(&c, in);
  iree_hal_buffer_view_t* out = NULL;
  if (iree_status_is_ok(iree_runtime_call_invoke(&c, 0)))
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &out);
  iree_runtime_call_deinitialize(&c);
  return out;
}

/* The model card's output-length cap, in tokens, for `npcm` samples of audio:
 *   max_new_tokens = samples * 6.5 / 16000 + 2
 * A fixed budget lets a short clip keep decoding long after the audio is spent, and the greedy
 * decode spends that budget restarting the utterance rather than stopping. Floored at 4 so a
 * one-word command still has room, ceilinged at the previous fixed budget so nothing decodes
 * longer than before — above ~3.4 s of audio the formula exceeds the ceiling and this is a no-op.
 * Measured on 183 German command-and-control recordings on a Mali device: clips below that
 * threshold finish 0.58 s sooner (paired median, faster in 135 of 172), clips above it are
 * unchanged (+0.09 s, 11 recordings), and the median word error rate does not move. */
static int finish_budget(int npcm) {
  int b = (int)(npcm * TOKENS_PER_SAMPLE) + 2;
  if (b < 4) b = 4;
  if (b > FINISH_TOKENS) b = FINISH_TOKENS;
  return b;
}

/* Process one encoder window starting at feature frame `start`; finalize rows into e->mem. */
static int process_window(Engine* e, int start, int produced, int flush, int peek) {
  double t0 = now_ms();
  /* frontend input: FE_LC frames of left context (zeros before utterance start) + the window */
  static float fein[FE_IN];
  int s0 = (start - FE_LC) * SPF;
  for (int i = 0; i < FE_IN; ++i) {
    int si = s0 + i;
    fein[i] = (si >= 0 && si < e->npcm) ? e->pcm[si] : 0.0f;
  }
  iree_hal_buffer_view_t* fev = mkview(e, (iree_hal_dim_t[]){1, FE_IN}, 2,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, fein, sizeof(fein));
  if (!fev) return -1;
  iree_hal_buffer_view_t* feo = run1(e, e->fe, FN_FE, fev);
  iree_hal_buffer_view_release(fev);
  if (!feo) return -1;
  static float feats[FE_OUT][ENC_DIM];
  if (pull(e, feo, feats, sizeof(feats))) { iree_hal_buffer_view_release(feo); return -1; }
  iree_hal_buffer_view_release(feo);
  double t1 = now_ms();

  /* encoder on the clean 64 frames (drop the FE_LC context rows) */
  iree_hal_buffer_view_t* encin = mkview(e, (iree_hal_dim_t[]){1, CHUNK, ENC_DIM}, 3,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, feats[FE_LC], CHUNK * ENC_DIM * sizeof(float));
  iree_hal_buffer_view_t* enco = encin ? run1(e, e->enc, FN_ENC, encin) : NULL;
  if (encin) iree_hal_buffer_view_release(encin);
  if (!enco) return -1;
  double t2 = now_ms();

  /* adapter: positions start..start+63 */
  int32_t posv[CHUNK];
  for (int i = 0; i < CHUNK; ++i) posv[i] = start + i;
  iree_hal_buffer_view_t* posbv = mkview(e, (iree_hal_dim_t[]){1, CHUNK}, 2,
      IREE_HAL_ELEMENT_TYPE_INT_32, posv, sizeof(posv));
  iree_runtime_call_t c;
  iree_hal_buffer_view_t* memo = NULL;
  if (posbv && iree_status_is_ok(iree_runtime_call_initialize_by_name(
          e->adp, iree_make_cstring_view(FN_ADP), &c))) {
    iree_runtime_call_inputs_push_back_buffer_view(&c, posbv);
    iree_runtime_call_inputs_push_back_buffer_view(&c, enco);
    if (iree_status_is_ok(iree_runtime_call_invoke(&c, 0)))
      iree_runtime_call_outputs_pop_front_buffer_view(&c, &memo);
    iree_runtime_call_deinitialize(&c);
  }
  if (posbv) iree_hal_buffer_view_release(posbv);
  iree_hal_buffer_view_release(enco);
  if (!memo) return -1;
  static float mem64[CHUNK][DIM];
  if (pull(e, memo, mem64, sizeof(mem64))) { iree_hal_buffer_view_release(memo); return -1; }
  iree_hal_buffer_view_release(memo);

  /* finalize: [nmem .. newFinal); at flush the lookahead holds nothing back */
  int limit = flush ? start + CHUNK : start + CHUNK - LOOKAHEAD;
  int new_final = limit < produced ? limit : produced;
  if (new_final > MAXMEM) new_final = MAXMEM;
  /* The real window 0 recomputes and OVERWRITES rows an early peek finalized with a
   * zero-padded tail; regular windows only append (their overlap rows were finalized with a
   * fuller attention band by the previous window and must not be downgraded). */
  int first = (!peek && start == 0 && e->peeked) ? 0 : e->nmem;
  for (int f = first; f < new_final; ++f)
    memcpy(e->mem[f], mem64[f - start], DIM * sizeof(float));
  if (new_final > e->nmem) e->nmem = new_final;
  if (!peek && start == 0) e->peeked = 0;
  double t3 = now_ms();
  TLOG("win %d fe %.0fms enc %.0fms adp+fin %.0fms nmem %d\n",
          e->win, t1 - t0, t2 - t1, t3 - t2, e->nmem);
  return 0;
}

/* Prefill over the current finalized memory; refresh cross-K/V (and start the token
 * sequence if none yet). Returns 0 on success. */
static int run_prefill(Engine* e, int start_tokens) {
  static float mem256[MAXMEM][DIM];
  static float mask[MAXMEM];
  memcpy(mem256, e->mem, sizeof(mem256));
  for (int f = e->nmem; f < MAXMEM; ++f) memset(mem256[f], 0, DIM * sizeof(float));
  for (int f = 0; f < MAXMEM; ++f) mask[f] = f < e->nmem ? 0.0f : NEGMASK;

  iree_hal_buffer_view_t* pe = mkview(e, (iree_hal_dim_t[]){1, 1, DIM}, 3,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->embed + (size_t)BOS * DIM, DIM * sizeof(float));
  iree_hal_buffer_view_t* memv = mkview(e, (iree_hal_dim_t[]){1, MAXMEM, DIM}, 3,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, mem256, sizeof(mem256));
  iree_hal_buffer_view_t* maskv = mkview(e, (iree_hal_dim_t[]){1, 1, 1, MAXMEM}, 4,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, mask, sizeof(mask));
  if (!pe || !memv || !maskv) return -1;

  iree_runtime_call_t c;
  if (!iree_status_is_ok(iree_runtime_call_initialize_by_name(
          e->pre, iree_make_cstring_view(FN_PRE), &c))) return -1;
  iree_runtime_call_inputs_push_back_buffer_view(&c, pe);
  iree_runtime_call_inputs_push_back_buffer_view(&c, memv);
  iree_runtime_call_inputs_push_back_buffer_view(&c, maskv);
  int rc = iree_status_is_ok(iree_runtime_call_invoke(&c, 0)) ? 0 : -1;
  iree_hal_buffer_view_release(pe); iree_hal_buffer_view_release(memv); iree_hal_buffer_view_release(maskv);
  if (rc) { iree_runtime_call_deinitialize(&c); return -1; }

  release_cross(e);
  iree_hal_buffer_view_t* psk[L], *psv[L];
  for (int l = 0; l < L; ++l) {
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &psk[l]);
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &psv[l]);
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &e->ck[l]);
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &e->cv[l]);
  }
  e->has_cross = 1;
  iree_hal_buffer_view_t* logits = NULL;
  iree_runtime_call_outputs_pop_front_buffer_view(&c, &logits);
  iree_runtime_call_deinitialize(&c);

  e->halted = 0; e->ended_eos = 0;
  if (start_tokens) {
    release_self(e);
    for (int l = 0; l < L; ++l) { e->sk[l] = psk[l]; e->sv[l] = psv[l]; }
    e->has_self = 1;
    int tok = argmax_logits(e, logits);
    e->ntoks = 0; e->pos = 1; e->text[0] = 0;
    if (tok != EOS) { e->toks[e->ntoks++] = tok; detok_append(&e->vocab, tok, e->text, sizeof(e->text)); }
  } else {
    for (int l = 0; l < L; ++l) { iree_hal_buffer_view_release(psk[l]); iree_hal_buffer_view_release(psv[l]); }
  }
  if (logits) iree_hal_buffer_view_release(logits);
  return 0;
}

static void rebuild_text(Engine* e);

/* On a silent tail the greedy decode restarts the utterance; a PARTIAL restart survives the
 * cycle guard ("to the next channel to the next" — 1.5 cycles). Any proper suffix equal to a
 * prefix of the token sequence is such a restart: drop it, repeat until stable. Applied only
 * to finals that did not end on EOS. */
static void drop_restart_suffix(Engine* e) {
  int again = 1;
  while (again && e->ntoks >= 2) {
    again = 0;
    for (int s2 = e->ntoks / 2; s2 >= 1; --s2) {
      if (memcmp(e->toks, e->toks + e->ntoks - s2, (size_t)s2 * sizeof(int)) == 0) {
        e->ntoks -= s2; again = 1; break;
      }
    }
  }
  rebuild_text(e);
}

/* Cycle guard (same rationale as the whisper cartridge): greedy decode on a truncated/silent
 * memory does not stop — it loops ("ever tried, ever tried, ...", "No matter No matter No
 * matter"). Periods >= 3 trip on the first repeat; periods 1-2 only on a TRIPLE, so a
 * legitimate doubled word or two-token phrase ("very very", one repeated "No matter") is never
 * trimmed while a runaway loop still is. */
static int repeat_period(const int* t, int n) {
  for (int p = 1; p <= n / 2; ++p) {
    int need = p >= 3 ? 2 : 3;               /* occurrences of the p-gram, incl. the original */
    if (n < p * need) continue;
    int same = 1;
    for (int i = 0; i < p * (need - 1) && same; ++i)
      if (t[n - 1 - i] != t[n - 1 - i - p]) same = 0;
    if (same) return p;
  }
  return 0;
}

static void rebuild_text(Engine* e) {
  e->text[0] = 0;
  for (int i = 0; i < e->ntoks; ++i) detok_append(&e->vocab, e->toks[i], e->text, sizeof(e->text));
}

/* Greedy steps from the current state; stops on EOS / cycle / budget / MAXTOK. `trim` also
 * drops the detected cycle's repeat from the emitted tokens (safe when the decode state is
 * rebuilt next hop or discarded at finish). Returns new-token count. */
static int run_steps(Engine* e, int budget, int trim) {
  static float mask[MAXMEM];
  for (int f = 0; f < MAXMEM; ++f) mask[f] = f < e->nmem ? 0.0f : NEGMASK;
  int added = 0;
  while (e->ntoks > 0 && e->ntoks < MAXTOK && added < budget && e->pos < MAXTOK) {
    int last = e->toks[e->ntoks - 1];
    iree_hal_buffer_view_t* ev = mkview(e, (iree_hal_dim_t[]){1, 1, DIM}, 3,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->embed + (size_t)last * DIM, DIM * sizeof(float));
    iree_hal_buffer_view_t* cq = mkview(e, (iree_hal_dim_t[]){1, HD}, 2,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->rope_c[e->pos], HD * sizeof(float));
    iree_hal_buffer_view_t* sq = mkview(e, (iree_hal_dim_t[]){1, HD}, 2,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, e->rope_s[e->pos], HD * sizeof(float));
    iree_hal_buffer_view_t* maskv = mkview(e, (iree_hal_dim_t[]){1, 1, 1, MAXMEM}, 4,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, mask, sizeof(mask));
    if (!ev || !cq || !sq || !maskv) return added;

    iree_runtime_call_t c;
    if (!iree_status_is_ok(iree_runtime_call_initialize_by_name(
            e->step, iree_make_cstring_view(FN_STEP), &c))) return added;
    iree_runtime_call_inputs_push_back_buffer_view(&c, ev);
    iree_runtime_call_inputs_push_back_buffer_view(&c, cq);
    iree_runtime_call_inputs_push_back_buffer_view(&c, sq);
    for (int l = 0; l < L; ++l) {
      iree_runtime_call_inputs_push_back_buffer_view(&c, e->sk[l]);
      iree_runtime_call_inputs_push_back_buffer_view(&c, e->sv[l]);
      iree_runtime_call_inputs_push_back_buffer_view(&c, e->ck[l]);
      iree_runtime_call_inputs_push_back_buffer_view(&c, e->cv[l]);
      if (l == 0) iree_runtime_call_inputs_push_back_buffer_view(&c, maskv);
    }
    int ok = iree_status_is_ok(iree_runtime_call_invoke(&c, 0));
    iree_hal_buffer_view_release(ev); iree_hal_buffer_view_release(cq);
    iree_hal_buffer_view_release(sq); iree_hal_buffer_view_release(maskv);
    if (!ok) { iree_runtime_call_deinitialize(&c); return added; }
    for (int l = 0; l < L; ++l) {
      iree_hal_buffer_view_release(e->sk[l]); iree_hal_buffer_view_release(e->sv[l]);
      iree_runtime_call_outputs_pop_front_buffer_view(&c, &e->sk[l]);
      iree_runtime_call_outputs_pop_front_buffer_view(&c, &e->sv[l]);
    }
    iree_hal_buffer_view_t* logits = NULL;
    iree_runtime_call_outputs_pop_front_buffer_view(&c, &logits);
    iree_runtime_call_deinitialize(&c);
    int nxt = argmax_logits(e, logits);
    if (logits) iree_hal_buffer_view_release(logits);
    e->pos++;
    if (nxt == EOS) { e->halted = 1; e->ended_eos = 1; break; }   /* tentative while streaming */
    e->toks[e->ntoks++] = nxt;
    detok_append(&e->vocab, nxt, e->text, sizeof(e->text));
    added++;
    int p = repeat_period(e->toks, e->ntoks);
    if (p > 0) {
      if (trim) {
        /* drop all repeats, keep one instance of the looping p-gram */
        int drop = p >= 3 ? p : 2 * p;
        e->ntoks -= drop; rebuild_text(e);
      }
      e->halted = 1;
      break;
    }
  }
  return added;
}

/* One throwaway zero-input pass through every graph: Vulkan compiles pipelines lazily at
 * each graph's FIRST dispatch, which would otherwise land inside the user's first
 * utterance (~+2 s on the first partial). Runs at create time, where init is async. */
static void warmup(Engine* e) {
  double t0 = now_ms();
  static float za[FE_IN];
  memset(za, 0, sizeof(za));
  iree_hal_buffer_view_t* v = mkview(e, (iree_hal_dim_t[]){1, FE_IN}, 2,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, za, sizeof(za));
  iree_hal_buffer_view_t* o = v ? run1(e, e->fe, FN_FE, v) : NULL;
  if (v) iree_hal_buffer_view_release(v);
  if (o) iree_hal_buffer_view_release(o);
  static float zf[CHUNK][DIM];
  memset(zf, 0, sizeof(zf));
  v = mkview(e, (iree_hal_dim_t[]){1, CHUNK, DIM}, 3,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, zf, sizeof(zf));
  iree_hal_buffer_view_t* enco = v ? run1(e, e->enc, FN_ENC, v) : NULL;
  if (v) iree_hal_buffer_view_release(v);
  if (enco) {
    int32_t posv[CHUNK];
    for (int i = 0; i < CHUNK; ++i) posv[i] = i;
    iree_hal_buffer_view_t* posbv = mkview(e, (iree_hal_dim_t[]){1, CHUNK}, 2,
        IREE_HAL_ELEMENT_TYPE_INT_32, posv, sizeof(posv));
    iree_runtime_call_t c;
    if (posbv && iree_status_is_ok(iree_runtime_call_initialize_by_name(
            e->adp, iree_make_cstring_view(FN_ADP), &c))) {
      iree_runtime_call_inputs_push_back_buffer_view(&c, posbv);
      iree_runtime_call_inputs_push_back_buffer_view(&c, enco);
      iree_hal_buffer_view_t* memo = NULL;
      if (iree_status_is_ok(iree_runtime_call_invoke(&c, 0)))
        iree_runtime_call_outputs_pop_front_buffer_view(&c, &memo);
      iree_runtime_call_deinitialize(&c);
      if (memo) iree_hal_buffer_view_release(memo);
    }
    if (posbv) iree_hal_buffer_view_release(posbv);
    iree_hal_buffer_view_release(enco);
  }
  /* decoder: fake 64 zero memory frames, prefill + one step, then reset */
  memset(e->mem, 0, sizeof(e->mem));
  e->nmem = CHUNK;
  if (run_prefill(e, 1) == 0 && e->ntoks > 0) run_steps(e, 1, 0);
  e->npcm = 0; e->win = 0; e->nmem = 0;
  e->peeked = 0; e->pending = 0; e->halted = 0; e->shown_len = 0;
  release_self(e); release_cross(e);
  e->ntoks = 0; e->pos = 0; e->text[0] = 0;
  TLOG("warmup %.0fms (all pipelines compiled)\n", now_ms() - t0);
}

static void reset_utterance(Engine* e) {
  e->npcm = 0; e->win = 0; e->nmem = 0;
  e->peeked = 0; e->pending = 0; e->halted = 0; e->shown_len = 0;
  release_self(e); release_cross(e);
  e->ntoks = 0; e->pos = 0; e->text[0] = 0;
}

/* ---- JNI -------------------------------------------------------------------------------- */
#define JNIFN(name) Java_sk_ainet_transformers_iree_android_IreeMoonshineStream_##name

JNIEXPORT jlong JNICALL JNIFN(nativeCreate)(JNIEnv* env, jobject thiz,
    jstring jdev, jstring jfe, jstring jenc, jstring jadp, jstring jpre, jstring jstep,
    jstring jparams, jstring jvocab, jstring jembed) {
  const char* dev = (*env)->GetStringUTFChars(env, jdev, 0);
  const char* fe = (*env)->GetStringUTFChars(env, jfe, 0);
  const char* enc = (*env)->GetStringUTFChars(env, jenc, 0);
  const char* adp = (*env)->GetStringUTFChars(env, jadp, 0);
  const char* pre = (*env)->GetStringUTFChars(env, jpre, 0);
  const char* stp = (*env)->GetStringUTFChars(env, jstep, 0);
  const char* par = (*env)->GetStringUTFChars(env, jparams, 0);
  const char* voc = (*env)->GetStringUTFChars(env, jvocab, 0);
  const char* emb = (*env)->GetStringUTFChars(env, jembed, 0);

  Engine* e = calloc(1, sizeof(Engine));
  jlong ret = 0;
  iree_runtime_instance_options_t io; iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_options_use_all_available_drivers(&io);
  if (e &&
      iree_status_is_ok(iree_runtime_instance_create(&io, iree_allocator_system(), &e->inst)) &&
      iree_status_is_ok(iree_runtime_instance_try_create_default_device(e->inst,
          iree_make_cstring_view(dev), &e->dev))) {
    e->params = load_params(e, par);
    TLOG("create: params(%s) -> %s\n", par, e->params ? "ok" : "FAIL");
    e->fe = load_session(e, fe, 0);
    TLOG("create: fe(%s) -> %s\n", fe, e->fe ? "ok" : "FAIL");
    if (e->fe && e->params) {
      e->alloc = iree_runtime_session_device_allocator(e->fe);
      e->enc = load_session(e, enc, 0);
      TLOG("create: enc(%s) -> %s\n", enc, e->enc ? "ok" : "FAIL");
      e->adp = load_session(e, adp, 0);
      TLOG("create: adp(%s) -> %s\n", adp, e->adp ? "ok" : "FAIL");
      e->pre = load_session(e, pre, 1);
      TLOG("create: pre(%s) -> %s\n", pre, e->pre ? "ok" : "FAIL");
      e->step = load_session(e, stp, 1);
      TLOG("create: step(%s) -> %s\n", stp, e->step ? "ok" : "FAIL");
      e->vocab = vocab_load(voc);
      TLOG("create: vocab(%s) -> n=%d\n", voc, e->vocab.n);
      FILE* f = e->vocab.n > 0 ? fopen(emb, "rb") : NULL;
      if (f) {
        const size_t want = (size_t)e->vocab.n * DIM;
        e->embed = malloc(want * sizeof(float));
        if (e->embed && fread(e->embed, sizeof(float), want, f) != want) {
          TLOG("dec_embed size mismatch: expected %zu floats (vocab %d x dim %d)\n",
               want, e->vocab.n, DIM);
          free(e->embed); e->embed = NULL;
        }
        fclose(f);
      }
      e->pcm = malloc(MAXPCM * sizeof(float));
      rope_init(e);
      if (e->enc && e->adp && e->pre && e->step && e->vocab.n > 0 && e->embed && e->pcm) {
        warmup(e);
        ret = (jlong)(intptr_t)e;
      }
    }
  }
  (*env)->ReleaseStringUTFChars(env, jdev, dev); (*env)->ReleaseStringUTFChars(env, jfe, fe);
  (*env)->ReleaseStringUTFChars(env, jenc, enc); (*env)->ReleaseStringUTFChars(env, jadp, adp);
  (*env)->ReleaseStringUTFChars(env, jpre, pre); (*env)->ReleaseStringUTFChars(env, jstep, stp);
  (*env)->ReleaseStringUTFChars(env, jparams, par); (*env)->ReleaseStringUTFChars(env, jvocab, voc);
  (*env)->ReleaseStringUTFChars(env, jembed, emb);
  if (!ret && e) { free(e->embed); free(e->pcm); free(e); }
  return ret;
}

/* Feed PCM; process any complete hops; return the partial transcript when it changed. */
JNIEXPORT jstring JNICALL JNIFN(nativeFeedPcm)(JNIEnv* env, jobject thiz,
    jlong handle, jfloatArray jpcm) {
  Engine* e = (Engine*)(intptr_t)handle; if (!e) return NULL;
  jsize n = (*env)->GetArrayLength(env, jpcm);
  jfloat* pcm = (*env)->GetFloatArrayElements(env, jpcm, 0);
  if (!pcm) return NULL;
  int take = n;
  if (e->npcm + take > MAXPCM) take = MAXPCM - e->npcm;
  if (take > 0) { memcpy(e->pcm + e->npcm, pcm, (size_t)take * sizeof(float)); e->npcm += take; }
  (*env)->ReleaseFloatArrayElements(env, jpcm, pcm, JNI_ABORT);

  int produced = e->npcm / SPF;
  int changed = 0;
  /* Early peek: flash the first words at ~0.72 s instead of waiting for the full 1.28 s
   * window. The peek rows are conservative (zero-padded tail, lookahead subtracted) and are
   * recomputed by the real window 0; the restart decode at hop 1 replaces the text anyway. */
  if (!e->peeked && e->win == 0 && produced >= PEEK_FRAMES && produced < CHUNK) {
    double t0 = now_ms();
    int usable = produced - LOOKAHEAD > 0 ? produced - LOOKAHEAD : produced;
    if (process_window(e, 0, usable, 0, 1) == 0) {
      e->peeked = 1;
      release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0;
      if (run_prefill(e, 1) == 0) run_steps(e, STEP_SLICE, 1);
      e->pending = 2 * HOP_TOKENS - STEP_SLICE;
      if (e->ntoks > 0) changed = 1;
      TLOG("peek total %.0fms nmem %d toks %d \"%s\"\n", now_ms() - t0, e->nmem, e->ntoks, e->text);
    }
  }
  int hopped = 0;
  while (e->win * HOP + CHUNK <= produced && e->nmem < MAXMEM) {
    double t0 = now_ms();
    if (process_window(e, e->win * HOP, produced, 0, 0)) break;
    e->win++;
    int prev = e->ntoks;
    /* Hybrid partials: the first hops re-decode from scratch each time (exact, affordable),
     * later hops go incremental with the prefix locked so the partial stays monotone.
     * Only STEP_SLICE tokens are decoded here — the rest of the hop budget is consumed
     * incrementally by subsequent feed calls (mid-hop emission), so text trickles out
     * instead of arriving in one lump per hop. finish() stays an exact full re-decode. */
    int restart = e->win <= RESTART_HOPS || e->ntoks == 0;
    if (restart) { release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0; }
    if (run_prefill(e, restart) == 0) run_steps(e, STEP_SLICE, restart);
    e->pending = (restart ? 2 * HOP_TOKENS : HOP_TOKENS) - STEP_SLICE;
    hopped = 1;
    if (e->ntoks != prev || e->win == 1) changed = 1;
    TLOG("hop %d total %.0fms toks %d \"%s\"\n",
            e->win, now_ms() - t0, e->ntoks, e->text);
  }
  if (!hopped && e->pending > 0 && !e->halted && e->ntoks > 0 && e->has_self && e->has_cross) {
    int budget = e->pending < STEP_SLICE ? e->pending : STEP_SLICE;
    int added = run_steps(e, budget, e->win <= RESTART_HOPS);
    e->pending -= added;
    if (added > 0) changed = 1;
  }
  /* Restart hops rebuild the text from scratch; mid-hop slices would expose that as the
   * partial briefly shrinking. Surface a partial only once it is at least as long as the
   * last one shown (the final always wins at finish()). */
  size_t len = strlen(e->text);
  if (changed && e->text[0] && len >= e->shown_len) {
    e->shown_len = len;
    return (*env)->NewStringUTF(env, e->text);
  }
  return NULL;
}

/* End of utterance: flush the tail, run one EXACT full re-decode, reset, return final text. */
JNIEXPORT jstring JNICALL JNIFN(nativeFinish)(JNIEnv* env, jobject thiz, jlong handle) {
  Engine* e = (Engine*)(intptr_t)handle; if (!e) return NULL;
  double t0 = now_ms();
  int produced = e->npcm / SPF;
  /* flush remaining windows (zero-padded), lookahead released */
  while (e->nmem < produced && e->nmem < MAXMEM) {
    if (process_window(e, e->win * HOP, produced, 1, 0)) break;
    e->win++;
  }
  jstring out = NULL;
  if (e->nmem > 0) {
    /* exact: fresh decode over the final memory */
    release_self(e); e->ntoks = 0; e->pos = 0; e->text[0] = 0;
    if (run_prefill(e, 1) == 0) run_steps(e, finish_budget(e->npcm), 1);
    if (!e->ended_eos) drop_restart_suffix(e);
    /* C&C utterances are single sentences; on a silent tail the greedy decode restarts the
     * utterance instead of emitting EOS ("Go to the next. Go to next") — same failure and
     * same cure as the whisper cartridge: the first sentence IS the utterance. */
    for (char* q = e->text; *q; ++q) {
      if (*q == '.' || *q == '!' || *q == '?') { q[1] = 0; break; }
    }
    const char* p = e->text; while (*p == ' ') ++p;
    out = (*env)->NewStringUTF(env, p);
    TLOG("finish %.0fms nmem %d toks %d/%d \"%s\"\n",
            now_ms() - t0, e->nmem, e->ntoks, finish_budget(e->npcm), p);
  }
  reset_utterance(e);
  return out;
}

JNIEXPORT void JNICALL JNIFN(nativeReset)(JNIEnv* env, jobject thiz, jlong handle) {
  Engine* e = (Engine*)(intptr_t)handle; if (e) reset_utterance(e);
}

JNIEXPORT void JNICALL JNIFN(nativeDestroy)(JNIEnv* env, jobject thiz, jlong handle) {
  Engine* e = (Engine*)(intptr_t)handle; if (!e) return;
  reset_utterance(e);
  if (e->fe) iree_runtime_session_release(e->fe);
  if (e->enc) iree_runtime_session_release(e->enc);
  if (e->adp) iree_runtime_session_release(e->adp);
  if (e->pre) iree_runtime_session_release(e->pre);
  if (e->step) iree_runtime_session_release(e->step);
  if (e->params) iree_vm_module_release(e->params);
  if (e->dev) iree_hal_device_release(e->dev);
  if (e->inst) iree_runtime_instance_release(e->inst);
  for (int i = 0; i < e->vocab.n; ++i) free(e->vocab.t[i]);
  free(e->vocab.t); free(e->embed); free(e->pcm); free(e);
}
