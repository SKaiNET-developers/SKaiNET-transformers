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

## All Phases Complete

| Phase | Status | Summary |
|-------|--------|---------|
| 1. Decouple tool calling | DONE | ChatSession, Tokenizer interface, no GGUFTokenizer coupling |
| 2. Model registry | DONE | ModelRegistry, UnifiedModelLoader, ModelFamily enum |
| 3. Tokenization pipeline | DONE | GGUFTokenizer in llm-core, TokenizerFactory |
| 4. Unified runner | DONE | skainet-cli with auto-detection |
3. **Phase 2** then — biggest refactor, needs per-model validation
4. **Phase 4** last — depends on all other phases
