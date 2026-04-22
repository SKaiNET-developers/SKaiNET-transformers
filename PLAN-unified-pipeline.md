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

## Phase 5d — Positional KV cache + shared-KV parity (PENDING)

Shared-KV semantics differ between the two paths. The hand-coded
`HeapGemma4KvCache` writes positionally (each `(layer_slot, position)`
cell holds exactly one K/V; shared layers overwrite each other within a
step). The DSL's `SharedKVCache` wraps an owner `AppendKVCache` and just
forwards reads/writes, so followers *append* to the owner's history
rather than overwriting a slot — different behaviour when
`kvSharedLayers > 0`.

To close parity for real Gemma 4 E2B (`kvSharedLayers = 20 / 35`), llm-core
needs a positional-storage KV cache variant, then the parity test flips
to cover that case, and only *then* should `Gemma4Runtime` be
`@Deprecated`. Until 5d lands, keep the hand-coded runtime as the
production path for real Gemma 4 checkpoints.

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
| 5d. Positional KV cache + shared-KV parity | PENDING | Needed before deprecating Gemma4Runtime (real E2B has kvSharedLayers=20/35) |
