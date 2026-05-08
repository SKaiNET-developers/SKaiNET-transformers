# SKaiNET-transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.transformers/skainet-transformers-agent.svg)](https://central.sonatype.com/artifact/sk.ainet.transformers/skainet-transformers-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/SKaiNET-developers/SKaiNET-transformers)

High-performance LLM application layer on top of the [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET) engine. Provides model-specific inference, agentic chat with tool calling, and a unified CLI for transformer-based models, all in Kotlin Multiplatform.

## Key features

- **Multi-model support.** Llama, BERT.
- **Native CPU performance in JVM.** Auto-discovers SKaiNET's priority-100 FFM (Foreign Function & Memory) native kernel provider when present (4–6× faster Q4_K matmul, 1.5–1.8× faster FP32 SGEMM vs the priority-50 Panama Vector path.
- **Native tool calling.** Family-specific chat templates and tool-call parsers for Llama 3, Gemma 4, Qwen, Apertus, and ChatML/Hermes. Includes a Java surface (`KLlamaJava`, `JavaTools.definition`, `JavaAgentLoop`) for plain-Java consumers.
- **Kotlin Multiplatform.** JVM, Android, Kotlin/Native (Linux x64/ARM64, macOS ARM64, iOS arm64/sim arm64), JS, Wasm targets where applicable.

## Current release

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
java  --add-modules jdk.incubator.vector  -jar llm-apps/skainet-cli/build/libs/skainet-all.jar \
  -m /path/to/model.gguf "The capital of France is"
```  

```bash
# Tool-calling demo (calculator + file-listing tools auto-registered)
java -jar skainet-all.jar -m model.gguf --demo --template=llama3 "What is 17 * 23?"
```  

```bash
# Interactive agent
java  --add-modules jdk.incubator.vector  -jar skainet-all.jar -m model.gguf --agent --template=llama3 
```

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

Transformers-only release; no SKaiNET engine bump. Three `skainet-cli`
fixes that together make Q4/Q8 GGUF inference work reliably on JDKs
where the Vector API incubator module is unavailable or not loaded.

- **Vector API flags now bake into the launcher scripts.**
  `--enable-preview --add-modules jdk.incubator.vector` moved from
  `tasks.withType<JavaExec>` (which only affected `gradle :run`) into
  `application { applicationDefaultJvmArgs }`, so the generated
  `bin/skainet-cli` and shadow launcher both inherit them. Previously
  these were missing on direct `java -jar` invocations and the runtime
  silently fell back to `DefaultCpuOpsBase`, where Q8-weighted matmul
  ClassCasted Byte→Float at the first attention projection.
- **Graceful dequantize-at-load fallback.** When `PlatformCpuOpsFactory`
  resolves to `DefaultCpuOpsBase` (older JDK, missing
  `--add-modules jdk.incubator.vector`, or platforms where the Vector
  API intrinsic isn't backed — notably some macOS JDK builds),
  `skainet-cli` now detects this at startup and passes
  `QuantPolicy.DEQUANTIZE_TO_FP32` to the loader. Weights land as
  plain FP32 and every op route works regardless of which CPU ops
  backend is active. A startup warning calls out the ~4× memory hit.
- **Backend label reflects the actual Vector API path.** The
  "Backend: …" startup line was printed from the kernel registry's
  static `displayName`, so users running without the incubator module
  still saw "CPU (SIMD)" while the scalar fallback was doing the work.
  The label is now printed after the `DefaultCpuOpsJvm` probe and
  appends either "Vector API SIMD" or "scalar fallback (Vector API not
  loaded)" — it can no longer disagree with the warning that follows.

### Earlier in the 0.23.x line

**0.23.4** — BOM coverage fix (`:llm-inference:apertus` and
`:llm-inference:voxtral` added; constraint list now auto-discovered by
a `buildSrc/` convention plugin); README and tutorial dependency
snippets use real published artifact IDs (`skainet-transformers-core`
etc.) and the BOM pattern.


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

## License

MIT — see [LICENCE](LICENCE).
