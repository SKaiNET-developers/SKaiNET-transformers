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
