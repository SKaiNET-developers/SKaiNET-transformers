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

## Phase 2: Unified DSL-Based Model Definition (converge on OptimizedLLMRuntime)

**Problem:** Each model has a hand-coded runtime. `OptimizedLLMRuntime` already supports DSL -> graph -> optimized execution, but only some models use it.

**Changes:**

1. **Define DSL networks for all model families:**
   - `llamaNetwork(config)` — LLaMA/Mistral/Qwen2/3 (standard transformer)
   - `qwen35Network(config)` — Qwen3.5 (hybrid DeltaNet + full attention)
   - `gemmaNetwork(config)` — Gemma (GELU, MatFormer FFN, sliding window)
   - `apertusNetwork(config)` — Apertus (xIELU, ungated MLP, QK-norm)
   - Each is a pure function returning a `Network<T>` from the DSL

2. **Unified model loading flow:**
   ```
   detectArchitecture(ggufMetadata) -> ModelFamily
   ModelFamily.createNetwork(config) -> Network<T>
   WeightLoader.loadAndMap(file, network) -> weights
   OptimizedLLMRuntime(network, weights, mode=HYBRID) -> InferenceRuntime
   ```

3. **Remove deprecated hand-coded runtimes** once DSL equivalents are validated:
   - `LlamaRuntime` -> `llamaNetwork()` + `OptimizedLLMRuntime`
   - `ApertusRuntime` -> `apertusNetwork()` + `OptimizedLLMRuntime`

**Critical files:**
- `llm-core/.../OptimizedLLMRuntime.kt` — already exists, extend
- `llm-core/.../dsl/TransformerDsl.kt` — already has embedding, MHA, SwiGLU, RMSNorm
- `llm-core/.../weights/LLMWeightNameResolvers.kt` — already maps DSL paths -> GGUF names
- New: per-model DSL network definitions

## Phase 3: Tokenization as Pipeline Stage

**Problem:** Tokenization is split between `GGUFTokenizer` (kllama module), `QwenByteLevelBPETokenizer` (llm-core), and model-specific code. The byte-level BPE fix we just made shows the fragility.

**Changes:**

1. **Enhance `Tokenizer` interface** (`llm-core`):
   ```kotlin
   interface Tokenizer {
       fun encode(text: String): IntArray
       fun decode(token: Int): String
       fun decode(tokens: IntArray): String
       val eosTokenId: Int
       val bosTokenId: Int
       val vocabSize: Int
       val specialTokens: Set<String>
   }
   ```

2. **Unified tokenizer factory:**
   - `TokenizerFactory.fromGGUF(source)` — auto-detects BPE/SentencePiece/WordPiece
   - `TokenizerFactory.fromTokenizerJson(json)` — HuggingFace format
   - Returns the correct implementation (byte-level BPE for GPT-2/Qwen, SentencePiece for LLaMA, etc.)

3. **Move `GGUFTokenizer` to `llm-core`** so all runners can use it without depending on kllama

## Phase 4: Unified Runner (single CLI entry point)

**Problem:** 6 separate CLI apps with duplicated argument parsing, model loading, and dispatch logic.

**Changes:**

1. **Single `skainet` CLI** that auto-detects model architecture from GGUF metadata:
   ```bash
   skainet -m model.gguf "prompt"                    # auto-detect, generate
   skainet -m model.gguf --chat                      # auto-detect, chat mode
   skainet -m model.gguf --demo "What is 2+2?"       # auto-detect, tool calling
   ```

2. **Architecture registry:**
   ```kotlin
   ModelRegistry.register("llama", ::llamaNetwork)
   ModelRegistry.register("qwen3", ::qwenNetwork)
   ModelRegistry.register("gemma", ::gemmaNetwork)
   ```

3. **Auto-detection from GGUF metadata** (already exists in `peekGgufMetadata()`)

## Verification

- All existing unit tests pass (`llm-agent`, `llm-runtime:kllama`, `llm-core`)
- Smoke test suite passes (generation + tool calling)
- Basic generation produces identical output for all model families
- Tool calling works for any model that supports ChatML/Qwen/Llama3 templates
- `OptimizedLLMRuntime` in HYBRID mode matches hand-coded runtime output

## Suggested Implementation Order

1. **Phase 1** first — immediately unblocks tool calling for all models
2. **Phase 3** next — reduces fragility (the GGUFTokenizer byte-level BPE issue)
3. **Phase 2** then — biggest refactor, needs per-model validation
4. **Phase 4** last — depends on all other phases
