# FunctionGemma KV-cache 2-graph board loop — implementation + board bring-up

This completes **Phase 2** of the perf program (see the sl2610 demo's `PERF-LOGBOOK.md`). The DSL decoder
(`GemmaModel.forwardPrefill`/`forwardWithPast`) is CPU-verified token-for-token; the two board graphs export
and compile. The native board runtime loop:
- `IreeRuntime.invokeFiles(...)` — raw-bin file I/O (gemma-iree).
- `GemmaKvDecoder` — the prefill→with_past loop + host `splitHalfCosSin` + `Bin` raw-f32/i32 I/O (gemma-iree).
- Wired into the demo `Pipeline.runPipeline` behind `GEMMA_KV=1` (re-decode stays default).

## BOARD-VERIFIED (SL2610, 2026-08-11)

The full prefill→with_past loop reproduces the oracle `[262146,236769,3255,718,498,1373,262152,106]`
token-for-token (`"turn the light on"` → `set_lights`). Every open unknown resolved:

| unknown | resolution |
|---|---|
| per-block K-vs-V output order | **K then V** (`GemmaKvDecoder.kFirstInOutput=true`) — the draft's return-SSA "V,K" hint was wrong; the emitted MLIR's layer-0 return pair is `concat(K_in,·), concat(V_in,·)` (K = the RoPE'd+normed projection), and the K-first mapping matches the oracle |
| `--output=@file` format | extension-driven: `@file.bin` = RAW little-endian bytes (what `Bin` assumes); `@file.npy` would add a NumPy header |
| `gemma_with_past` input arg order | exactly as traced/documented below — confirmed against the compiled vmfb |
| dynamic `1x1x?x256` cache | the g165 Torq-fork `iree-compile` ACCEPTS the true-dynamic MLIR and one vmfb served every position (no `GEMMA_SENTINEL_PAST=1` fallback needed) |
| `--task_topology_group_count` | accepted by the board `iree-run-module` (g165) |
| **shared irpa — NEW finding** | the "one shared `gemma-gen.irpa`" contract is **invalid**: every trace numbers its `model` externals independently (`t0`, `t10`, …), so each graph needs the archive written from ITS trace. `exportPrefill`/`exportWithPast` now write `gemma-prefill.safetensors` / `gemma-with-past.safetensors` (converted to `gemma-prefill.irpa` / `gemma-with-past.irpa`); `GemmaKvDecoder` derives those paths from its `irpa` arg. Binding the wrong archive fails loudly (`NOT_FOUND … no parameter found in index with key 'tN'`). |

Measured (subprocess `iree-run-module`, `--task_topology_group_count=2`, 13-token prompt, 8 tokens):
re-decode **4419 ms/token**; KV loop **2139 ms/token** avg (steady-state **~1740 ms/token** once the irpa
mmap is warm) + one-off prefill 5956 ms.

Historical bring-up notes below (the constants/arg-order remain the reference):

## The two graphs (build with `GEMMA_KV=1 scripts/compile-gemma.sh board`)

- `gemma-prefill.vmfb` — `func @gemma_prefill`. Input: `{seq}xi32` token ids (fixed `seq`, prompt zero-padded).
  Outputs: `{seq}xi32` argMax tokens + **18 self-K + 18 self-V** `[1,1,seq,256]` (initial cache).
- `gemma-with-past.vmfb` — `func @gemma_with_past`. **Dynamic** `?` cache. See exact I/O below.
- Each binds its OWN parameter archive (`gemma-prefill.irpa` / `gemma-with-past.irpa`) — per-trace key
  numbering, see the board-verified table above.

## Model constants (gemma3-270M, probed from the GGUF)

```
nLayers = 18   headDim = 256   nKVHeads = 1   slidingWindow = 512
RoPE: SPLIT_HALF, FULL rotary (partialRotary forced 1.0), freqDenom = headDim, scaling factor = 1.0 (no-op)
  global base = 1_000_000.0   sliding base = 10_000.0
layer types = s s s s s G  s s s s s G  s s s s s G   (GLOBAL at layer i iff i % 6 == 5; else SLIDING)
```

## `gemma_with_past` exact I/O order (confirmed from the emitted MLIR)

**Inputs** — trace-first-use order. The board MUST build the input list by replicating this iteration:
```
arg0            = token           1xi32          (literal --input=1xi32=<id>, no file)
arg1, arg2      = cosSliding, sinSliding   1x256xf32   (first SLIDING layer = layer 0 introduces these)
arg3..arg12     = layers 0..4 self-K/V     1x1x{P}x256xf32   (5 layers x 2, order per-block — SEE CAVEAT)
arg13, arg14    = cosGlobal, sinGlobal     1x256xf32   (first GLOBAL layer = layer 5 introduces these)
arg15..arg40    = layers 5..17 self-K/V    1x1x{P}x256xf32   (13 layers x 2)
```
Reproduce with: `inputs=[token]; introduced={}; for i in 0..17 { t = (i%6==5)?global:sliding; if t not in
introduced { inputs += cos[t], sin[t]; introduced+=t }; inputs += K[i], V[i] }`.

**Outputs**: 36 `1x1x{P+1}x256xf32` (extended K/V) then `1xi32` token **last**.

### K-vs-V order — RESOLVED (board-verified 2026-08-11)
The emitted order is **K then V** per block (`kFirstInOutput=true`). The draft's return-SSA reading
("suggests V,K") was wrong: the return pairs list the LATER-defined SSA value first, and per layer the
K concat is emitted after the V concat's projection — read the defining ops, not the id order. Verified
in the MLIR (layer-0 pair = `concat(K_in,·), concat(V_in,·)`) and by board oracle parity.
(A tiny export-side `.manifest` mapping output slot → role is still a good follow-up.)

## Raw-bin tensor I/O (extend `IreeRuntime`, mirror `voicecc/asr/TorqRunModule`)

`iree-run-module` binds raw little-endian f32 bins: `--input=1x1x{P}x256xf32=@k12.bin` and writes outputs with
`--output=@out_k12.bin`. Add an `IreeRuntime.invokeFiles(module, function, inputs: List<Spec|Literal>,
outputs: List<String>): Boolean` (same `popen`+`--parameter_mode=file --parameters=model=…irpa` as today).
Cache tensors are **f32** (only the weight globals are bf16). `token`/`cos`/`sin` files are tiny; the 36 K/V
bins are `{P}*256*4` bytes each and grow by one row per step.

## Host cos/sin builder (native Kotlin — port of `RoPE.buildSplitHalfCosSin`, full rotary)

```kotlin
fun splitHalfCosSin(pos: Int, base: Float, headDim: Int = 256): Pair<FloatArray, FloatArray> {
    val half = headDim / 2
    val cos = FloatArray(headDim); val sin = FloatArray(headDim)   // seqLen = 1
    for (i in 0 until half) {
        val freq = 1.0 / Math.pow(base.toDouble(), 2.0 * i / headDim)   // freqDenom = headDim
        val a = pos * freq
        val c = kotlin.math.cos(a).toFloat(); val s = kotlin.math.sin(a).toFloat()
        cos[i] = c; cos[half + i] = c; sin[i] = -s; sin[half + i] = s   // sign baked into first half
    }
    return cos to sin
}
// Build ONCE per step: (cosSliding,sinSliding)=splitHalfCosSin(pos,10_000f); (cosGlobal,sinGlobal)=…(pos,1_000_000f)
```

## The loop (replaces the re-decode loop in `GemmaDecoder`, or a new `GemmaKvDecoder`)

```
1. Encode prompt (BOS + Octopus template). P = prompt length. Pad tokens to `seq`.
2. PREFILL: run gemma_prefill(paddedTokens) -> read the 36 K/V output bins + argMax bins.
   first token = argMax[P-1]. SLICE each K/V bin to [1,1,P,256] (drop padded positions P..seq-1).
3. pos = P. loop (until eot/eos or maxTokens):
   a. (cosS,sinS)=splitHalfCosSin(pos,1e4); (cosG,sinG)=splitHalfCosSin(pos,1e6). Write 4 cos/sin bins.
   b. Write the 36 current K/V bins.
   c. run gemma_with_past with inputs in the arg order above (token literal + cos/sin + K/V files),
      outputs = 36 new K/V bins + token bin.
   d. Read token bin -> next token. Read the 36 new K/V bins (each [1,1,pos+1,256]) -> current cache. pos++.
4. Detokenize; CompactCodec.parse.
```

## Verification ladder (on board)
1. `GEMMA_KV=1 … compile-gemma.sh board` compiles all three vmfbs (dynamic `?` in with_past must compile —
   the first real test of the sentinel-relax; if iree rejects the dynamic concat, fall back to fixed-pad+mask).
2. Prefill-then-1-step reproduces the re-decode logits at that position (isolates the cache seam + K/V order).
3. Full loop reproduces the oracle `[262146,236769,3255,718,498,1373,262152,106]`.
4. `VOICECC_PROFILE=1 voicecc gen "turn the light on"` -> append the ms/token row to `PERF-LOGBOOK.md`
   (expect the O(seq²)->O(seq) collapse vs the ~6 s/token baseline).
```
