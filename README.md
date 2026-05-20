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

Today every model family runs through the **eager JVM path**. The StableHLO /
native path is shared with the engine and not yet wired for full transformer
models.

### Where each architecture fits

Honest status — see the project-status note at the top of this README.

| Architecture | State |
|---|---|
| **Llama / Mistral** | Most exercised path — basic text generation works on the eager JVM path. |
| **Qwen 2 / 3** | DSL + loaders present; runs through the shared decoder path. Early; Qwen3 RoPE / QK-norm fixes landed in 0.23.2. |
| **Gemma 2 / 3 / 3n** | DSL + loaders present (Gemma 4 via the SafeTensors path); has the most test coverage, but not verified end-to-end. |
| **Apertus** | DSL + loaders present; declared end-to-end in 0.23.1, still early. |
| **BERT** | Encoder for embeddings only — no text generation, no tool calling. |
| **Voxtral** | TTS / voice; architecture code only — no runtime facade or CLI yet. |

### Near term

- Make the **eager JVM path reliable per family** — including tool calling —
  before extending scope.
- Verify each generative architecture end-to-end with smoke tests.
- Wire the **StableHLO / native compilation path** for full transformer models.

## Current release

The current release is **0.23.5** — a transformers-only release on the **0.23.x** line (no SKaiNET engine bump), focused on `skainet-cli` reliability on JDKs where the `jdk.incubator.vector` module is unavailable.

The recommended way to consume is via the BOM. It pins every published `skainet-transformers-*` artifact and re-exports the upstream `sk.ainet:skainet-bom`, so the engine-side `sk.ainet.core:skainet-*` artifacts get the matching version too — you only need to declare the BOM version in one place.

```kotlin
dependencies {
    implementation(platform("sk.ainet.transformers:skainet-transformers-bom:0.23.5"))

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
| `llm-core`           | `OptimizedLLMRuntime`, `ModelRegistry`, `UnifiedModelLoader`, shared abstractions. |
| `llm-inference/<arch>` | Per-architecture network DSLs and weight loaders (`llama`, `gemma`, `qwen`, `apertus`, `bert`). |
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

## What's new in 0.23.5

- **Vector API flags now reach the generated launchers.** `--enable-preview
  --add-modules jdk.incubator.vector` was only applied to `gradle :run`; the
  generated `bin/skainet-cli` and shadow launcher shipped without them, so a
  direct `java -jar` invocation hit the scalar fallback and `ClassCastException`-ed
  on the first Q8 attention projection. The flags moved into
  `application { applicationDefaultJvmArgs }` so both launchers inherit them.
- **No more hard crash on runtimes without the Vector API.** When the CPU ops
  factory falls back to the scalar `DefaultCpuOpsBase` (older JDK, missing
  `--add-modules`, or unsupported platforms), `skainet-cli` now detects this at
  startup, warns about the ~4× memory hit, and loads weights with
  `QuantPolicy.DEQUANTIZE_TO_FP32` so every op route works regardless of backend.
- **Backend label now matches the real code path.** The "Backend: …" startup line
  is printed after the actual ops probe and reports either "Vector API SIMD" or
  "scalar fallback", so it can no longer disagree with the warning beside it.

### Earlier in the 0.23.x line

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
