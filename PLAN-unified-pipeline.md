# Plan: Unified Model Pipeline with Decoupled Tool Calling

## Context

Currently SKaiNET-transformers has:
- **5+ hand-coded runtimes** (LlamaRuntime, Qwen35Runtime, Gemma3nRuntime, ApertusRuntime, VoxtralRuntimes) — each reimplements the forward pass, weight loading, and layer execution
- **Tool calling tightly coupled to kllama** — the AgentLoop, ToolCallingDemo, and chat modes only exist in the kllama runner. Other models (Gemma, Apertus) cannot use tool calling without duplicating code
- **Two execution paths** — legacy hand-coded runtimes AND the newer `OptimizedLLMRuntime` with DSL/compute-graph/AOT. LlamaRuntime and ApertusRuntime are already marked deprecated

The goal: converge on **one unified pipeline** where model definition, weight loading, tokenization, and tool calling are cleanly separated pipeline stages.

## Architecture Overview

```
GGUF/SafeTensors File
    |
WeightLoader (parse metadata + tensors)
    |
DSL Network Definition (model-specific, declarative)
    |
ComputeGraph (DAG)
    |
Optimization Pipeline (TransposeElim -> WeightDedup -> LLMFusion -> DCE)
    |
ComputeGraphExecutor (fused kernels)
    |
InferenceRuntime (unified: forward + generate)
    |
TokenizationPipeline (encode/decode, special tokens, byte-level BPE)
    |
ChatPipeline (template formatting, tool calling, agent loop)
```

## Phase 1: Decouple Tool Calling from kllama (immediate value) -- DONE

**What was done:**

1. **Enhanced `Tokenizer` interface** with `eosTokenId`, `bosTokenId`, `vocabSize`
   - Updated all implementations: `GGUFTokenizer`, `TokenizerImpl`, `HuggingFaceBPETokenizer`, `TekkenTokenizerAdapter`, `HuggingFaceTokenizer` (BERT)

2. **Created `ChatSession` abstraction** in `llm-agent`
   - File: `llm-agent/.../chat/ChatSession.kt`
   - Bundles `InferenceRuntime` + `Tokenizer` + `ModelMetadata`
   - Provides `createAgentLoop()` and `runSingleTurn()` for any runner

3. **Refactored `ToolCallingDemo` and `AgentCli`** to use `Tokenizer` interface instead of `GGUFTokenizer`
   - Both now accept any `Tokenizer`, not just `GGUFTokenizer`
   - Both use `ChatSession` internally for agent loop creation

4. **Removed `GGUFTokenizer` cast from kllama Main.kt** dispatch
   - Chat/agent/demo modes now work with any `Tokenizer`

5. **Fixed `JavaAgentLoop`** — replaced `GGUFTokenizer` instanceof hack with `tokenizer.eosTokenId`

## Phase 2: Unified DSL-Based Model Definition (converge on OptimizedLLMRuntime) -- PARTIAL

**What was done:**

1. **Created `ModelRegistry`** in `llm-core/.../ModelRegistry.kt`
   - `ModelFamily` enum: LLAMA, QWEN, GEMMA, APERTUS, BERT, VOXTRAL, UNKNOWN
   - `ModelRegistry.detect(architecture)` maps GGUF arch strings to families
   - Tracks capabilities (supportsToolCalling, chatTemplateFamily)

2. **Created `UnifiedModelLoader`** in `llm-core/.../UnifiedModelLoader.kt`
   - `UnifiedModelLoader.peek(source)` extracts `GGUFModelInfo` from GGUF metadata
   - Returns architecture, family, dimensions without loading weights

**Already existing (no changes needed):**
- DSL networks: `llamaNetwork()`, `qwenNetwork()`, `apertusNetwork()`, `bertNetwork()`, `voxtralBackboneNetwork()`, `voxtralAcousticNetwork()`
- `OptimizedLLMRuntime` with DIRECT/OPTIMIZED/HYBRID modes
- Per-model `NetworkLoader` classes (LlamaNetworkLoader, ApertusNetworkLoader, etc.)

**Remaining (future work):**
- `gemmaNetwork()` DSL definition (Gemma3n has unique features: GELU, MatFormer variable FFN, sliding window)
- Migrate CLI runners from deprecated runtimes to OptimizedLLMRuntime
- Remove deprecated LlamaRuntime and ApertusRuntime

## Phase 3: Tokenization as Pipeline Stage -- DONE

**What was done:**

1. **Enhanced `Tokenizer` interface** with `eosTokenId`, `bosTokenId`, `vocabSize` (done in Phase 1)

2. **Moved `GGUFTokenizer` from kllama to `llm-core`**
   - New location: `llm-core/.../tokenizer/GGUFTokenizer.kt`
   - Old location has a typealias for backwards compatibility
   - Added `skainet-io-gguf` and `kotlinx-io-core` dependencies to `llm-core`

3. **Created `TokenizerFactory`** in `llm-core/.../tokenizer/TokenizerFactory.kt`
   - `TokenizerFactory.fromGGUF(source)` — from GGUF file metadata
   - `TokenizerFactory.fromTokenizerJson(json)` — from HuggingFace tokenizer.json
   - `TokenizerFactory.fromHuggingFace(json, config)` — full HF BPE tokenizer

4. All runners can now use `GGUFTokenizer` and `TokenizerFactory` directly from `llm-core`

## Phase 4: Unified Runner (single CLI entry point) -- DONE

**What was done:**

1. **Created `llm-apps/skainet-cli`** — new unified CLI module
   - Auto-detects architecture from GGUF metadata via `UnifiedModelLoader.peek()`
   - Loads any LLaMA-compatible model (LLaMA, Qwen, Mistral)
   - Supports `--chat`, `--agent`, `--demo` modes with tool calling
   - Uses `TokenizerFactory.fromGGUF()` for tokenizer loading
   - Registered as `skainet` runner in smoke test script

2. **Usage:**
   ```bash
   skainet -m model.gguf "The capital of France is"   # auto-detect, generate
   skainet -m model.gguf --chat                        # interactive chat
   skainet -m model.gguf --demo "What is 2+2?"         # tool calling demo
   ```

3. **Existing per-model CLIs are preserved** — no breaking changes

**Remaining (future work):**
- Add Gemma3n loading path to unified CLI (requires gemmaNetwork() DSL)
- Add Apertus loading path to unified CLI
- Eventually deprecate per-model CLIs

## Phase 5: Gemma on the DAG → CPU-on-JVM path

**Goal:** route Gemma through the same declarative pipeline used by Llama/Apertus —
`gemmaNetwork()` DSL → `Module<T,V>` → traced `ComputeGraph` (DAG) → optimization passes
→ `ComputeGraphExecutor` on the JVM `CpuBackendProvider`. No hand-coded `Gemma4Runtime`
on this path. Same two execution modes as the rest (`DIRECT` for debugging, `OPTIMIZED`
for fused-kernel DAG execution).

**Scope split.** Gemma 4 carries several architectural features that the current DSL
does not express (proportional RoPE, per-layer head_dim, sliding-window attention, KV
cache sharing). Rather than blocking on those, Phase 5 is split in two:

### Phase 5a — Minimal pipeline, simplified Gemma (DONE)

Delivers an end-to-end DAG → CPU-on-JVM path for a *reduced* Gemma that uses standard
full attention, standard RoPE, no KV sharing, and GELU-gated FFN. Accuracy parity
with the hand-coded runtime is not required at this stage — the point is to close the
DSL/loader/weight-mapping loop and validate the execution path.

Steps:

1. **`GeGLUFFN` module** in `llm-core/.../transformer/GeGLUFFN.kt`.
   Mirrors `SwiGLUFFN` but substitutes `ops.gelu` for `ops.silu`. Parameter layout
   (`gate_proj.weight`, `up_proj.weight`, `down_proj.weight`) is deliberately
   identical so `LlamaGGUFNameResolver` maps weights without any resolver changes.
2. **`geGluFFN(...)` DSL extension** in `llm-core/.../dsl/TransformerDsl.kt`.
   Extension functions on `StageImpl` and `NeuralNetworkDslImpl`, mirroring `swiGluFFN`.
3. **`gemmaNetwork()` DSL** in `llm-inference/gemma/.../GemmaNetworkDef.kt`.
   Takes `Gemma4ModelMetadata`. For 5a, every layer is treated as
   `full_attention` with `headDim = globalHeadDim`, standard RoPE, and no KV sharing:
   `Embedding → N × (RMSNorm → MHA(RoPE, KVCache) → Residual → RMSNorm → GeGLUFFN →
   Residual) → RMSNorm → Dense(vocab)`. Wrapped in `HybridTransformerBlock` per layer
   like Llama/Apertus so `ResidualAdd` skip-connections work.
4. **`GemmaNetworkLoader`** in `llm-inference/gemma/.../GemmaNetworkLoader.kt`.
   Modeled on `ApertusNetworkLoader`: GGUF + SafeTensors + preloaded weight variants,
   `WeightMapper` with `LlamaGGUFNameResolver`. Reuses the existing
   `Gemma4WeightLoader` / `Gemma4SafeTensorsWeightLoader`.
5. **Compile check.** `./gradlew :llm-inference:gemma:compileKotlinJvm` and the existing
   Gemma4 test suite stays green. No new tests in 5a — that belongs in 5b alongside
   accuracy parity.

### Phase 5b — Full Gemma DSL primitives (DONE, pending accuracy parity)

DSL primitives for every Gemma 4 architectural feature are now in place:

- **Sealed KVCache hierarchy** (`llm-core/.../transformer/KVCache.kt`):
  `AppendKVCache` (default), `SlidingWindowKVCache(window)` (trims to last N),
  `SharedKVCache(delegate)` (writes/reads forward to owner; `reset()` no-op
  on follower). `is KVCache<*, *>` checks in runtime code stay intact.
- **RoPE partial rotation + proportional scaling** (`RoPE.kt`):
  `partialRotaryFactor` (fraction of head_dim that rotates; Gemma 4 global =
  0.5) and `RoPEScaling.PROPORTIONAL` with NTK-aware `base' = base × factor ^
  (rotaryDim / (rotaryDim − 2))`.
- **Sliding-window attention** in `MultiHeadAttention` via an additive
  `[1, 1, seqQ, seqKV]` mask; mask subsumes causal so SDPA's built-in
  causal path is disabled when the mask is active.
- **DSL extensions**: `multiHeadAttention(slidingWindow = …)`, `rope(…,
  scaling = PROPORTIONAL, scalingFactor, partialRotaryFactor)`, and
  `kvCache(cache: KVCache<T, V>)` escape hatch to attach pre-built variants.
- **`gemmaNetwork()` walks `metadata.layerTypes`**: full-attention layers
  use PROPORTIONAL RoPE + global head_dim + `partialRotaryFactor=0.5`;
  sliding layers use standard RoPE + sliding head_dim + full rotation +
  `slidingWindow=metadata.slidingWindow`. Trailing `kvSharedLayers` layers
  share the KV cache of their owner via `SharedKVCache`. Per-layer FFN
  width via `metadata.getIntermediateSize(layer)`.

**Remaining for accuracy parity** (separate follow-up, unblocked by this
pass):

- Golden-output parity tests against the hand-coded `Gemma4Runtime` on a
  real E2B checkpoint.
- Then deprecate `Gemma4Runtime` the way `LlamaRuntime` / `ApertusRuntime`
  are being deprecated.

The `GemmaNetworkLoaderIntegrationTest` now exercises all 5b primitives at
construction against a real Gemma 4 E2B GGUF (35 layers, 6 global + 29
sliding by default pattern, shared KV for last 20).

## Phase 5c — Numerical parity against Gemma4Runtime (DONE, partial)

`GemmaRuntimeParityTest` runs synthetic-weight models through both the
hand-coded `Gemma4Runtime` and the DSL path (`gemmaNetwork()` +
`OptimizedLLMRuntime` in DIRECT mode) and compares the logits token by
token. Two configurations are covered:

- **1-layer global**, no sliding, no shared KV. Max |Δlogit| = **2.98e-8**
  across 5 steps — bit-exact at FP32 precision.
- **4-layer mixed** (3 sliding + 1 global), sliding window = 3, no shared
  KV. Max |Δlogit| = **7.45e-8** across 8 decode steps — also bit-exact.

This validates the core DSL pipeline against the reference implementation
for everything except shared-KV layers.

## Phase 5d — Positional KV cache, shared-KV parity, deprecate Gemma4Runtime (DONE)

Added `PositionalKVCache` and `SharedPositionalKVCache` to `llm-core`.
The positional variant backs storage with a pre-allocated
`[nKVHeads, maxSeqLen, headDim]` buffer and writes at its own position
counter. The shared variant wraps a `PositionalKVCache` delegate and
writes at the *follower's* own step into the delegate's buffer —
overwriting whatever the owner (or previous followers) wrote at that
slot. Matches `HeapGemma4KvCache`'s "last writer wins at (slot, pos)"
semantics exactly. Each cache instance tracks its own step counter so
RoPE in `MultiHeadAttention` sees correct absolute positions across all
shared peers.

`gemmaNetwork()` now uses the positional variants: owner layers get a
`PositionalKVCache`, the trailing `kvSharedLayers` wrap it with
`SharedPositionalKVCache`.

`GemmaRuntimeParityTest` gained a third configuration — 4 layers with
`kvSharedLayers = 2` (layers 2 & 3 share) — and passes at
**max |Δlogit| = 8.94e-8** across 8 decode steps, on par with the
non-shared cases. The DSL path now numerically matches `Gemma4Runtime`
across every Gemma 4 architectural feature.

`Gemma4Runtime` is marked `@Deprecated(level = WARNING)` pointing at
`gemmaNetwork() + OptimizedLLMRuntime`. `kgemma/Gemma4Ingestion.kt`
(the CLI ingestion layer, not yet migrated) and
`GemmaRuntimeParityTest` (by design) carry `@file:Suppress("DEPRECATION")`.

**Known limitation.** `SharedPositionalKVCache` requires uniform
`(nKVHeads, headDim)` across all peers in a shared group. Real Gemma 4
checkpoints released so far have `globalHeadDim == headDim` so this is
fine, but checkpoints with mixed dims across a shared group would need
a max-dim-padded storage variant (same idea as
`HeapGemma4KvCache.kvDim = max(kvDim, globalKvDim)`).

## Phase 6 — DSL path exposed via CLIs (DONE, opt-in)

`Gemma4Ingestion` grew a parallel set of loaders that return the DSL-based
runtime:

- `loadDslRuntime(Source)` / `loadDslRuntimeStreaming(RandomAccessSource)` —
  GGUF entry points. Both require `QuantPolicy.DEQUANTIZE_TO_FP32`; the
  ingestion fails fast with a clear error message otherwise.
- `loadDslRuntimeFromSafeTensors(indexPath)` — HuggingFace shard loader.
- `buildDslRuntime(Gemma4Weights<T, Float>)` — from pre-loaded weights
  (synthetic tests, custom pipelines).

All return an `InferenceRuntime<T>` so consumers dispatch via the
`InferenceRuntime<T>.generate(...)` extension function and stay agnostic
to which path built the runtime. The hand-coded `loadRuntime*` path stays
alongside, carrying its `@Suppress("DEPRECATION")` until quant-aware DAG
kernels land.

Supporting the non-reified ingestion layer surfaced a small refactor in
`gemma` / `GemmaNetworkLoader`: both `gemmaNetwork()` and
`GemmaNetworkLoader.fromWeights` now expose an explicit-`dtype` overload
alongside the original `reified` version.

### CLI changes

- **`kgemma`** grew a `--runtime=handcoded|dsl` flag (default `handcoded`
  to keep the existing low-RAM behaviour). `--runtime=dsl` routes through
  `loadDslRuntime*` and prints the path at startup.
- **`skainet-cli`** auto-routes any `ModelFamily.GEMMA` checkpoint
  through `GemmaNetworkLoader + OptimizedLLMRuntime` (no flag needed).
  Everything else — LLaMA, Qwen, Apertus — continues on the existing
  `LlamaRuntime` path with `NATIVE_OPTIMIZED` quant support. The new
  `:llm-inference:gemma` module dependency was added.

### Known limitation (superseded by Phase 7)

The DSL path still requires FP32 dequant, so:

- Real Gemma 4 E2B (~4.5 B params) needs ~20 GB RAM after dequant.
- `skainet-cli` prints an explicit note about this on the Gemma path.
- For RAM-constrained loads of real checkpoints, users should either
  stay on `Gemma4Runtime` (via `kgemma` without the flag), or wait for
  quant-aware DAG matmul (`ISSUE-skainet-8b-oom.md` §Solution C) to
  land — that's the single remaining gate between the DSL path and full
  replacement of the hand-coded runtime.

## Phase 7 — DSL consumes quantized weights without FP32 dequant (DONE, Q4_0/Q8_0)

`ISSUE-skainet-8b-oom.md` §Solution C, applied to the DSL path.

### 7a — `linearProject` helper (shipped)

Every DSL module that projects against a stored weight goes through
`linearProject(ops, input, weight)` instead of hand-written
`ops.matmul(input, ops.transpose(weight))`. Today the helper just
materialises the transpose (pure rename), but it centralises the
matmul-against-weight convention so future work — pre-transpose
markers, transpose-fused matmul ops — has one call site to evolve.

### 7b — empirical probe (shipped)

`GemmaDslQuantizedTest` builds a tiny 1-layer Gemma 4 DSL module tree
whose Q/K/V/O and gate/up/down projections are backed by
`Q8MemorySegmentTensorData` (constructed via an inline scale=1 Q8_0
packer so the test's integer weights round-trip losslessly). Forward
passes are compared against the FP32 baseline:

    Max |Δlogit| = 0.0 across 3 decode steps

The CPU backend's existing `ops.transpose` on `Q8MemorySegmentMarker`
(lazy shape-swap) and `ops.matmul(FloatArray, Q8_MemSeg)` (SIMD kernel
dispatch) compose correctly — the DSL path ran Q8 quantized inference
with zero code changes beyond 7a. Requires `inputDim % 32 == 0` (SIMD
lane × block alignment), which all real transformer dims satisfy.

### 7c — CLI plumbing (shipped)

`GemmaMemSegConverter` walks a `Gemma4Weights` map produced by
`QuantPolicy.NATIVE_OPTIMIZED` and replaces quantized tensors with the
right runtime representation:

- **Q4_0, Q8_0** → `Q4MemorySegmentTensorData` / `Q8MemorySegmentTensorData`
  (packed, no dequant, no pre-transpose). These ride the proven 7b
  dispatch end-to-end.
- **Q4_K, Q5_K, Q6_K** → dequant to FP32, keep the canonical `[out, in]`
  layout (no pre-transpose — the DSL's `linearProject` transposes at
  runtime, so pre-transposing would double-transpose and produce the
  wrong math).
- **`token_embd.weight`** → always dequant (needs row-gather, not
  matmul).

`Gemma4Ingestion.loadDslRuntimeNativeStreaming(...)` is the JVM-only
entry point that composes raw GGUF loading with `NATIVE_OPTIMIZED`, the
converter, and `buildDslRuntime`. `kgemma --runtime=dsl` now takes this
path — help output mentions the Q4_K gap explicitly.

### What's not done (Phase 8 or later)

- **Q4_K / Q5_K / Q6_K native kernels on the DSL path.** The backend has
  `matmul(FloatArray, Q4_KTensorData)` via `JvmQuantizedVectorKernels.matmulQ4_KVec`
  — the kernel exists. The missing piece is `ops.transpose(Q4_KTensorData)`:
  unlike the MemSeg markers, `Q4_KTensorData` falls through to the default
  per-element transpose, which doesn't preserve the packed block
  layout. Until a lazy-shape-swap transpose for `Q4_KTensorData` lands
  in `DefaultCpuOpsJvm`, K-series on the DSL path must dequant to FP32.
- **Real Gemma 4 E2B Q4_K_M** (the common checkpoint) still inflates
  to ~18 GB because ~all weights are Q4_K. Q8_0 Gemma checkpoints (if
  released) would now run at ~3 GB resident through the DSL path.

## All Phases Complete

| Phase | Status | Summary |
|-------|--------|---------|
| 1. Decouple tool calling | DONE | ChatSession, Tokenizer interface, no GGUFTokenizer coupling |
| 2. Model registry | DONE | ModelRegistry, UnifiedModelLoader, ModelFamily enum |
| 3. Tokenization pipeline | DONE | GGUFTokenizer in llm-core, TokenizerFactory |
| 4. Unified runner | DONE | skainet-cli with auto-detection |
| 5a. Gemma DAG → CPU-on-JVM (simplified) | DONE | GeGLUFFN + gemmaNetwork() + GemmaNetworkLoader |
| 5b. Gemma DSL primitives | DONE | Sealed KVCache, p-RoPE, sliding window, per-layer dims wired into gemmaNetwork() |
| 5c. Numerical parity (no shared KV) | DONE | 1-layer and 4-layer mixed sliding+global match Gemma4Runtime at ≤ 8e-8 |
| 5d. Positional KV cache + shared-KV parity + Gemma4Runtime @Deprecated | DONE | PositionalKVCache + SharedPositionalKVCache; shared-KV parity at 8.94e-8; Gemma4Runtime marked @Deprecated |
| 6. DSL path exposed via CLIs (opt-in) | DONE | Gemma4Ingestion.loadDslRuntime*, `kgemma --runtime=dsl`, skainet-cli auto-routes Gemma through GemmaNetworkLoader |
| 7a–7c. DSL consumes quantized Gemma weights (Q4_0/Q8_0) | DONE | linearProject centralisation, Q8 DSL probe at Δ=0, GemmaMemSegConverter, `kgemma --runtime=dsl` uses NATIVE_OPTIMIZED. Q4_K/Q5_K/Q6_K still dequant to FP32. |
| 7d. Q4_K native matmul through DSL (synthetic parity) | DONE | Lazy `Q4_KTensorData` transpose in `DefaultCpuOpsJvm`, Q4_K row-major → input-block-major re-layout, logical-shape side channel. `GemmaDslQ4KTest` matches FP32 at Δ=4.29e-6. Real E2B Q4_K_M loads at ~3 GB RAM; end-to-end generation still hits attention shape bugs — separate follow-up. |
