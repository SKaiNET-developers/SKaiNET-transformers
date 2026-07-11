# SKaiNET-transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.transformers/skainet-transformers-agent.svg)](https://central.sonatype.com/artifact/sk.ainet.transformers/skainet-transformers-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/SKaiNET-developers/SKaiNET-transformers)

Tranformers based LLM application layer on top of the [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET) engine. Provides model-specific inference, agentic chat with tool calling, and a unified CLI for transformer-based models, all in Kotlin Multiplatform.

> [!WARNING]
> **Project status — early / experimental.**
> This repository is an **initial version**. Nothing here is stable, and there is
> **no support or status guarantee for any feature, model, or API**. Model
> coverage, tool calling, and the runtime APIs are all work in progress and may
> not work for a given model or model version — for example, tool calling can
> fail to trigger or parse even on a model that generates plain text correctly.
> The capabilities described below are **goals, not promises**. Treat everything
> as a preview and expect things to break.

## Start in 5 minutes

SKaiNET Transformers is Kotlin Multiplatform. The fastest way to verify it on
your machine is the unified `skainet-cli`:

1. Get a local **GGUF** model file (e.g. a small quantized TinyLlama or Qwen).
2. Run the CLI, pointing it at the model.
3. Confirm the prompt returns a generated answer.

```bash
./gradlew :llm-apps:skainet-cli:run \
  --args="-m /absolute/path/to/model.gguf 'The capital of France is'"
```

Expected result: the CLI auto-detects the model architecture, loads the model,
and streams a generated answer. See the
[getting-started tutorial](docs/modules/ROOT/pages/tutorials/getting-started.adoc)
for model setup notes.

Working in Java? SKaiNET Transformers ships first-class Java support — see the
[`kllama-java-sample`](llm-apps/kllama-java-sample/README.md) starter and the
[Java getting-started guide](docs/modules/ROOT/pages/tutorials/getting-started-java.adoc).

Use the version shown in this README as the source of truth for first-run snippets.

## Key features

> The list below describes the project's **intended** scope. Maturity varies
> widely per item and many paths are unverified — see the project-status note above.

- **Multi-model support (in progress).** Architecture code exists for Llama / Mistral, Qwen 2 / 3, Gemma 2 / 3 / 3n, Apertus (Swiss AI) and BERT. Llama is the most exercised path; the other families are at varying, often early, stages and are not all verified end-to-end.
- **Native CPU performance.** Auto-discovers SKaiNET's priority-100 FFM (Foreign Function & Memory) native kernel provider when present (4–6× faster Q4_K matmul, 1.5–1.8× faster FP32 SGEMM vs the priority-50 Panama Vector path; Linux x86_64 / macOS ARM64 / Windows x86_64 in the published JAR — no manual setup).
- **Tool calling (experimental).** Family-specific chat templates and tool-call parsers (Llama 3, Qwen, Gemma, Apertus, ChatML/Hermes) and a Java surface (`KLlamaJava`, `JavaTools.definition`, `JavaAgentLoop`) exist, but tool calling is **not reliable yet** — it may fail to trigger or parse even when plain generation works.
- **GGUF + SafeTensors loading.** Streaming reader for any model size; `NATIVE_OPTIMIZED` quant policy keeps weights in their packed SIMD-friendly form.
- **Kotlin Multiplatform.** JVM, Android, Kotlin/Native (Linux x64/ARM64, macOS ARM64, iOS arm64/sim arm64), JS, Wasm targets where applicable.

## Roadmap

### Architecture goal

SKaiNET Transformers follows the SKaiNET engine's core path: **a transformer model
is defined once in the Kotlin DSL, captured as a tape or DAG, and then either
compiled to native code or executed eagerly — without rewriting it.**

1. **Define** the model with the decoder DSL (`llamaNetwork()`, `apertusNetwork()`, …).
2. **Capture** it as a *tape* (traced execution) or a *DAG* (explicit graph).
3. **Run** it one of two ways:
   - **Compile** — lower the graph to MLIR / StableHLO and compile to **native** code.
   - **Eager** — execute directly on a backend. On the **JVM this is the primary, go-to path.**

```mermaid
flowchart LR
    DSL["Transformer model — Kotlin DSL"] --> Graph["Tape / DAG"]
    Graph --> HLO["MLIR / StableHLO"]
    Graph --> Eager["Eager backend (JVM, …)"]
    HLO --> Native["Native code"]
```

The **eager JVM path** is the primary way every model family runs today. The
StableHLO / native path is shared with the engine and wired for the first
families: FunctionGemma exports a compiled edge build (0.35.0), and the BERT
encoder traces to an optimized ComputeGraph and lowers to StableHLO (0.36.0);
full generative-model coverage is still in progress.

### Where each architecture fits

Honest status — see the project-status note at the top of this README.

| Architecture | State |
|---|---|
| **Llama / Mistral** | Most exercised path — basic text generation works on the eager JVM path. |
| **Qwen 2 / 3** | DSL + loaders present; runs through the shared decoder path. Early; Qwen3 RoPE / QK-norm fixes landed in 0.23.2. |
| **Gemma 2 / 3 / 3n** | DSL + loaders present (Gemma 4 via the SafeTensors path); has the most test coverage, but not verified end-to-end. |
| **Apertus** | DSL + loaders present; declared end-to-end in 0.23.1, still early. |
| **BERT** | Sentence embeddings on the DSL path (`bertNetwork()` + `BertEncoderRuntime`, eager or traced/fused) — verified against sentence-transformers on MongoDB/mdbr-leaf. One-call `BertEmbeddingModel.fromHuggingFace(...)` with built-in Hub download. No text generation, no tool calling. |
| **T5 / GTR** | Encoder-decoder runtime (hand-coded, batch 1, no KV cache) + `GtrEmbedder`, powering the **vec2text** embedding-inversion pipeline — verified with a real-weights gtr-base round-trip test. |
| **Voxtral** | TTS / voice; architecture code only — no runtime facade or CLI yet. |

### Near term

- Make the **eager JVM path reliable per family** — including tool calling —
  before extending scope.
- Verify each generative architecture end-to-end with smoke tests.
- Wire the **StableHLO / native compilation path** for full transformer models.
  As of 0.28.1 a full gemma3 graph exports to StableHLO and `iree-compile`s to a
  `vmfb` (`GemmaMlirDumpTest`); next is running the compiled module and extending
  the same path to the other families.

## Current release

The current release is **0.36.0** (against **SKaiNET 0.35.0**) — **BERT is now completely
defined in the SKaiNET NN DSL**, and the deprecated hand-coded eager BERT stack is **removed
(BREAKING)** in the same release:

- `bertNetwork()` is a numerically complete `tokens → hidden-states` encoder: the new
  `BertEmbeddings` module adds the absolute-position and token-type embeddings the DSL definition
  previously omitted, and each encoder layer is wired as two post-norm blocks so every residual
  lands on the right value.
- `BertEncoderRuntime` runs the same definition **eagerly** (`DIRECT`, default) or as a traced,
  LLM-pipeline-**optimized ComputeGraph** (`OPTIMIZED`, bit-exact vs eager), adds masked mean
  pooling, the optional sentence-transformers `2_Dense` projection, and L2 normalization — and
  `exportTape(...)` lowers the encoder to StableHLO.
- One-call consumption: `BertEmbeddingModel.fromHuggingFace("MongoDB/mdbr-leaf-mt")` /
  `fromSafeTensors(dir)` behind the neutral `EmbeddingModel` SPI, with built-in Hub download
  (`HF_TOKEN`-aware, cached, offline-safe after the first run).
- Downstream effect: indexing the leaf-cli reference corpus dropped **676.9 s → 44.5 s (~15×)**
  with identical embeddings. Migration notes for the removed `BertRuntime` stack are in the
  [CHANGELOG](CHANGELOG.md) and the
  [BERT-as-DSL explanation](docs/modules/ROOT/pages/explanation/bert-dsl.adoc).

0.36.0 also adds a **T5 encoder-decoder** runtime (`llm-inference/t5`) with `GtrEmbedder`, and a
**vec2text embedding-inversion** pipeline (`llm-inference/vec2text`) that iteratively reconstructs
text from a GTR embedding — verified end-to-end against real
`sentence-transformers/gtr-t5-base` weights.

It builds on **0.35.0**, which added **FunctionGemma** self-compiled from the SKaiNET NN DSL: a
one-dependency function-calling sLLM (`skainet-transformers-runtime-kgemma`) with an eager one-line
API (`FunctionGemma.fromGguf(gguf).call("turn the light on")` → `ToolCall(set_lights, {state="on"})`,
runs anywhere on CPU, no iree) **and** a no-Python compiled edge export
(`FunctionGemma.exportCompiled` / `compile-gemma.sh`) verified token-for-token against llama.cpp on
the SL2610 board, using the engine's new `argMax` op to fold the `logits → token-ids` argmax tail
into the DSL trace; and on **0.34.1** — a patch that layer-qualifies the
Moonshine encoder's attention/LayerNorm parameter names so by-name weight loading can tell the
layers apart (no public API change) — and on **0.34.0**, which adds the first **Moonshine**
speech-to-text encoder authored entirely in the SKaiNET NN DSL (`skainet-transformers-inference-moonshine`,
bf16-native) — it emits portable StableHLO and transcribes correctly on both CPU and the Synaptics Torq
NPU. Supporting this, `transformer-core` RoPE now computes its rotation and `cos`/`sin` tables in **f32**
and uses the **full-head (ONNX) interleaved form** for bit-exact accuracy on NPU targets, and gains
**partial-rotary** support (`partialRotaryFactor` / `freqDenomRotaryDim`); `VoidDense` gains an optional
`addBias` for faithful FFNs. On top of the **0.33.0** engine adoption, the **0.32.1**
streaming-detokenization fix and the **0.32.0** real-GGUF **Llama** eager + StableHLO/IREE export work:

- The eager **`NATIVE_OPTIMIZED` path now works for Llama** (`Q4_K`/`Q6_K`): weights stay
  packed and `LlamaNetworkLoader.fromGguf(NATIVE_OPTIMIZED) + OptimizedLLMRuntime` decodes
  coherently, matching llama.cpp — fixing the packed token-embedding
  `gather: unsupported input rank 1`.
- **Fused decode-attention** (`seqQ == 1`) skips the `repeatKVHeads` concat + SDPA plumbing
  for a faster decode loop (~1.5×), bit-identical output.
- **Interleaved RoPE is now traceable**, so Llama/Mistral/GGUF graphs export to StableHLO
  (and `iree-compile` to a `vmfb`) instead of baking a disconnected constant.

The earlier `transformer-core` extraction (0.31.1) and the Gemma `NATIVE_OPTIMIZED`
footprint work (0.31.0) still apply.

The recommended way to consume is via the BOM. It pins every published `skainet-transformers-*` artifact and re-exports the upstream `sk.ainet:skainet-bom`, so the engine-side `sk.ainet.core:skainet-*` artifacts get the matching version too — you only need to declare the BOM version in one place.

```kotlin
dependencies {
    implementation(platform("sk.ainet.transformers:skainet-transformers-bom:0.36.0"))

    // Versions resolved from the BOM:
    implementation("sk.ainet.transformers:skainet-transformers-core")
    implementation("sk.ainet.transformers:skainet-transformers-runtime-kllama") // or runtime-kgemma, inference-qwen, inference-apertus
    implementation("sk.ainet.transformers:skainet-transformers-agent")          // chat templates + tool calling
}
```

To opt in to the native FFM CPU provider (recommended for JVM consumers):

```kotlin
dependencies {
    implementation("sk.ainet.core:skainet-backend-cpu")        // priority-50 Panama Vector
    implementation("sk.ainet.core:skainet-backend-native-cpu") // priority-100 FFM (auto-discovered)
}
```

`KernelRegistry` picks the highest-priority available provider; on hosts where the native lib doesn't load (sandboxed JDKs, unsupported arches), it cleanly falls back to Panama with no functional regression.

## Project structure

| Module               | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `llm-api`            | Framework-neutral interfaces (`ChatModel`, `EmbeddingModel`, `ToolDefinition`) — Spring AI-shaped. |
| `transformer-core`   | Framework NN primitives (attention, KV-cache family, embedding, norms, RoPE, FFNs, linear projection). `lang-core`-only → **all targets incl. `androidNative`**; re-exported by `llm-core`. |
| `llm-core`           | `OptimizedLLMRuntime`, `ModelRegistry`, `UnifiedModelLoader`, shared abstractions. |
| `llm-inference/<arch>` | Per-architecture network DSLs and weight loaders (`llama`, `gemma`, `qwen`, `apertus`, `bert`, `t5`, `vec2text`). |
| `llm-runtime/<arch>` | Per-architecture runtime facades (`kllama`, `kgemma`, `kqwen`, `kapertus`). |
| `llm-agent`          | Chat templates, tool-call parsers, agent loops; Java surface.           |
| `llm-apps`           | CLIs: `skainet-cli` (unified), `kllama-cli`, `kbert-cli`, plus `kllama-java-sample`. |
| `llm-test/llm-test-java` | JUnit 5 end-to-end tests for the Java surface (gated on `TINYLLAMA_MODEL_PATH`). |

## Getting started

### Prerequisites

- JDK 21 or higher
- Gradle 8.10+

### CLI: unified `skainet-cli`

```bash
# Plain generation
./gradlew :llm-apps:skainet-cli:shadowJar
java -jar llm-apps/skainet-cli/build/libs/skainet-all.jar \
  -m /path/to/model.gguf "The capital of France is"

# Tool-calling demo (calculator + file-listing tools auto-registered)
java -jar skainet-all.jar -m model.gguf --demo --template=llama3 "What is 17 * 23?"

# Interactive agent
java -jar skainet-all.jar -m model.gguf --agent --template=apertus
```

`--template` accepts `llama3`, `chatml`, `qwen`, `gemma`, `apertus` (auto-detected from GGUF metadata if omitted).

### Embeddings: LEAF in one call

Sentence embeddings with MongoDB's compact LEAF retrieval models need a single factory call —
the model downloads from the Hugging Face Hub and is cached on first use:

```kotlin
import sk.ainet.llm.providers.BertEmbeddingModel

BertEmbeddingModel.fromHuggingFace("MongoDB/mdbr-leaf-ir").use { model ->
    val vector = model.embed("The quick brown fox")   // L2-normalized FloatArray
}
```

See the [Getting Started with LEAF tutorial](docs/modules/ROOT/pages/tutorials/getting-started-leaf.adoc)
and the [BERT-as-DSL explanation](docs/modules/ROOT/pages/explanation/bert-dsl.adoc).

### Java consumers

```java
try (KLlamaSession session = KLlamaJava.loadGGUF(modelPath, /* systemPrompt */ null)) {
    JavaTool calc = new JavaTool() {
        @Override public ToolDefinition getDefinition() {
            return JavaTools.definition(
                "calculator", "Evaluate an arithmetic expression.",
                "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}"
            );
        }
        @Override public String execute(Map<String, ?> args) { /* ... */ }
    };
    JavaAgentLoop agent = JavaAgentLoop.builder()
        .session(session).tool(calc).template("llama3").build();
    String response = agent.chat("What is 17 * 23?");
}
```

See `llm-test/llm-test-java/src/test/java/.../KLlamaJavaToolCallingTest.java` for a runnable reference.

## What's new in 0.32.1

- **Streaming detokenization keeps word spaces.** A generation loop decoding one token at a time
  (`tokenizer.decode(tokenId)`) no longer runs words together. `SentencePieceSpecialTokens` and
  `UpstreamTokenizerAdapter` route `decode(Int)` through engine 0.32.4's `Tokenizer.decodeToken`,
  which preserves each SentencePiece piece's leading space (llama.cpp `token_to_piece` semantics).
  Engine pin `0.32.2 → 0.32.4`.

## What's new in 0.32.0

- **Eager `NATIVE_OPTIMIZED` for real-GGUF Llama.** `LlamaNetworkLoader.fromGguf(NATIVE_OPTIMIZED)`
  now keeps `Q4_K`/`Q6_K` weights packed and runs them through `OptimizedLLMRuntime`, mirroring the
  Gemma path (new `LlamaQuantLayout` + `LlamaPackedWeights.convertLlamaWeightsPacked`). Output is
  coherent and matches llama.cpp; fixes the packed token-embedding `gather: unsupported input rank 1`.
  This is the low-footprint path real-GGUF Llama inference on constrained ARM was missing. (ccbd87e)
- **Fused decode-attention fast path.** For the decode step (`seqQ == 1`), `MultiHeadAttention` runs
  scores → softmax → GQA-weighted-V straight from the cached K/V, bypassing the `repeatKVHeads` concat
  and the `unsqueeze → SDPA → squeeze → permute` chain. ~1.5× decode throughput on the JVM eager path;
  bit-for-bit-equivalent output. Prefill keeps the general SDPA path. (3791f88)
- **Traceable interleaved RoPE (graph export).** `RoPE` in `INTERLEAVED` mode (Llama / Mistral / most
  GGUF) used a raw-array path (`copyToFloatArray` / `fromFloatArray`) that, under graph tracing, recorded
  the rotated Q/K as a *disconnected constant* — severing them from the projection weights and crashing
  `iree-compile` downstream. It now records the rotation as tensor ops when tracing (gated on the tracing
  wrapper; eager keeps the fast raw-array path byte-identical). Unblocks TinyLlama → StableHLO → IREE. (019b049)
- **Engine pin `skainet 0.31.0 → 0.32.2`.**

## What's new in 0.31.1

- **`transformer-core` module — NN primitives reusable on all targets incl. `androidNative`.** The
  attention / KV-cache / embedding / norm / RoPE / FFN / linear-projection primitives were trapped in
  `llm-core` (whose io/compile/backend deps lack `androidNative`); they only need `skainet-lang-core`
  (which has it), so they're extracted into `transformer-core` and `llm-core` re-exports them. Existing
  consumers are unaffected; ARM-native downstreams (on-device whisper, future models) reuse them instead of
  reimplementing. Ships against engine **0.31.0** (additive, no engine change). (#183)

## What's new in 0.31.0

- **Tied Q8_0 lm_head stays packed (eager `NATIVE_OPTIMIZED`).** FunctionGemma's
  `token_embd` is Q8_0 and tied, so `convertGemmaWeightsPacked` was dequantizing
  *both* `token_embd` and `output` to FP32 (2×~0.67 GB) — OOM on the 1.9 GB
  SL2610. `output`/lm_head now packs as Q8_0 (runs on the NEON Q8_0 kernel);
  `token_embd` stays FP32 (it's gathered) but is wrapped no-copy. Footprint
  ~1.34 GB → ~0.76 GB; byte-identical decode (`GemmaQ5KPackedParityTest`),
  stable ~1.06 GB load on the SL2610.
- **`GemmaNetworkLoader.load(maxInferenceLen = …)`** — cap the context so the KV
  cache + RoPE tables stay tiny on constrained devices (default
  `min(contextLength, 4096)`).
- **Engine pin `skainet 0.30.0 → 0.31.0`** — picks up `ops.transpose`'s
  lazy-rewrap fix for all packed matmul dtypes (Q8_0/Q4_0), required so the
  packed lm_head transposes through `linearProject` instead of `ClassCastException`.

## What's new in 0.30.0

- **Q5_K stays packed in the eager Gemma runtime.** `GemmaMemSegConverter` used to
  dequantize Q5_K weights to FP32 on load; SKaiNET 0.30.0 provides a first-class
  Q5_K packed matmul (`Q5_KBlockTensorData` + `Q5KMatmulKernel`), so the converter
  now relayouts the GGUF bytes to block-major and keeps them packed (176 B/block).
  FunctionGemma-270M (`Q5_K_M`) decodes byte-identically to the FP32 baseline
  (`GemmaQ5KPackedParityTest`).
- **Gemma `NATIVE_OPTIMIZED` path is Kotlin/Native–ready.** The reusable layout +
  packing helpers (`GemmaQuantLayout.kt`, `GemmaPackedWeights.kt`) moved to
  `commonMain`, and `GemmaNetworkLoader.load()` now runs `convertGemmaWeightsPacked`
  under `NATIVE_OPTIMIZED` — so the board binary keeps K-quant weights packed with
  no `java.lang.foreign` MemSeg dependency. Verified on JVM and `linuxX64`.
- **Engine pin `skainet 0.28.1 → 0.30.0`** — released Q5_K packed matmul, NEON
  native kernels, and Kotlin/Native cinterop. The `mavenLocal()`-first dev shim is
  reverted; the release resolves the engine from Maven Central.
- **Fixes.** Kernel-less quant types under `NATIVE_OPTIMIZED` now dequant to FP32
  `[out, in]` instead of crashing on a rank-1 transpose; `DecoderGgufMemSegConverter`
  dequantizes Q4_1 and every other non-packed quant type instead of passing raw
  bytes through to a matmul crash ([#654](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/654)).

## What's new in 0.28.1

- **Engine pin `skainet 0.27.0 → 0.28.1`.** Picks up the completed Kotlin DSL →
  StableHLO → IREE export path. Every shape-changing op now declares its inferred
  output type (`reshape`/`matmul`/`concatenate`, [#673](https://github.com/SKaiNET-developers/SKaiNET/issues/673);
  `conv1d`/`gather`/pooling/`flatten`, [#675](https://github.com/SKaiNET-developers/SKaiNET/issues/675)),
  and `reduce_window` is emitted in IREE's generic region form — so a full gemma3
  graph traced via `GemmaMlirDumpTest` lowers to StableHLO that `iree-compile`s to
  a `vmfb`. No transformers-side API changes; existing callers compile unchanged.
- Verified end-to-end: `:llm-inference:gemma:jvmTest` green against the published
  0.28.1 (`GemmaMlirDumpTest`, `GemmaTraceTest` pass).

## What's new in 0.25.0

- **`DTypePolicy` on every `*NetworkLoader.fromGguf` / `.fromSafeTensors`
  entry.** A sealed `DTypePolicy` type (`Any | Require | Prefer | OneOf`,
  upstream of SKaiNET 0.25.0) is now accepted on every loader companion in
  `LlamaNetworkLoader`, `QwenNetworkLoader`, `GemmaNetworkLoader`,
  `ApertusNetworkLoader`, and `VoxtralNetworkLoader`. The policy is
  validated eagerly via `sk.ainet.apps.llm.DTypePolicyValidation` —
  `Require(BF16)` rejects on GGUF paths (no KEEP_NATIVE GGUF yet),
  accepts on SafeTensors paths. Default `DTypePolicy.Any` keeps the
  existing adaptive behaviour; every existing caller compiles
  unchanged.
- **SafeTensors BF16 KEEP_NATIVE** in `DecoderSafeTensorsLoader`. With
  `Require(BF16)` (or `Prefer(BF16)` / `OneOf` containing BF16) the
  loader stops dequanting BF16 SafeTensors weights and instead wraps
  the packed 2-bytes-per-element buffer in `Bf16DenseTensorData`. The
  matmul dispatch in `DefaultCpuOpsJvm` detects `Bf16TensorData` at
  runtime and routes to the SIMD BF16 kernel — a BF16 checkpoint now
  stays near its on-disk footprint in RAM instead of ~2× FP32 inflation.
- **Catalog goes BOM-only.** Every `skainet-*` alias in
  `gradle/libs.versions.toml` is now coordinate-only (no `version.ref`).
  Versions come from the `sk.ainet:skainet-bom` platform constraint
  re-exported by `:llm-bom`, and every consumer module pulls in
  `implementation(project.dependencies.platform(project(":llm-bom")))`
  in each affected source set. Engine bumps are still a one-line edit
  at the top of the catalog, but every internal build now exercises
  the BOM end-to-end — a missing-from-BOM regression fails locally
  instead of leaking into a published artifact.
- **Three reference smoke tests with `@Tag("smoke-reference")`** —
  the smoke tier that pins the architectures we always want to run end-
  to-end: `Qwen3ReferenceSmokeTest` (Qwen3-1.7B Q8 GGUF; exercises the
  new 0.25.0 `Q8_0MatmulKernel` + Qwen's `RoPEMode.SPLIT_HALF` +
  QK-Norm), `Gemma4ReferenceSmokeTest` (Gemma-4 E2B SafeTensors;
  sliding-window attention + per-layer KV sharing), and
  `BertLeafReferenceSmokeTest` (MongoDB `mdbr-leaf-ir` SafeTensors via
  the Java `KBertJava` surface). Run with
  `./gradlew test -PsmokeReference -PincludeIntegration`. Each test
  self-skips via JUnit `Assumptions` when the model file isn't
  reachable through the standard `~/.lmstudio/models/` /
  `~/.cache/huggingface/hub/` / env-var fallback chain.

### Earlier in the 0.23.x line

**0.23.5** — `skainet-cli` reliability on JDKs without the
`jdk.incubator.vector` module: `--enable-preview --add-modules
jdk.incubator.vector` flags reach the generated launchers (previously
only `gradle :run`); detection of scalar-fallback CPU ops with auto
weight dequant to FP32; backend label printed after the real ops
probe so it can't disagree with the warning beside it.

**0.23.4** — BOM is now correct and self-maintaining: `:llm-inference:apertus`
and `:llm-inference:voxtral` were missing from the BOM's constraints and are now
covered, so consumers pulling them through the BOM get proper version alignment;
the constraint list is auto-discovered by a `buildSrc/` convention plugin. The
README and tutorial dependency snippets were also fixed to use the published
artifact IDs (`skainet-transformers-core` etc.) via the BOM pattern.

**0.23.3** — Prefill progress callback: `generateUntilStop` and
`AgentLoop` expose `(done, total)` progress during the autoregressive
prefill loop via a default-no-op `AgentListener.onPrefillProgress`
method, so UIs on CPU-only runtimes can show that work is happening
between round start and the first generated token.

**0.23.2** — `kllama-cli`, `kllama-native`, `kllama-wasm`, and
`KLlamaJava` swapped to the DSL path (`OptimizedLLMRuntime` +
`llamaNetwork()`); GPU stubs deleted; SentencePiece + GGUF tokenizers
unified through upstream `sk.ainet.io.tokenizer`; markdown-fenced Llama 3
JSON tool calls now parse correctly; Qwen3 NEOX RoPE pairing fix; QK-norm
RMSNorm-eps wiring fix.

**0.23.1** — Apertus end-to-end (routing through `OptimizedLLMRuntime` +
`apertusNetwork()`, chat template + tool calling, real-GGUF Q4_K
loading); Gemma 4 chat-model JVM facade with mmap-arena cleanup; multi-id
EOS / stop-token support in the chat layer; SentencePiece auto-detect in
`fromTokenizerJson`; LEAF + Llama 3 single-JVM smoke test;
`ServiceLoader` shadow-jar fix-up so the priority-100 native-cpu provider
is picked up post-merge.

See [`CHANGELOG.md`](CHANGELOG.md) for the full set of changes.

## Engine

This project uses [**SKaiNET**](https://github.com/SKaiNET-developers/SKaiNET) as its underlying execution engine — tensor ops, neural-network DSL, kernel SPI, GGUF / SafeTensors I/O.

## License

MIT — see [LICENCE](LICENCE).
