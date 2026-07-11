# Spring AI adapter for SKaiNET-transformers (separate companion repo)

**Repository (target):** `SKaiNET-developers/SKaiNET-spring-ai` (new, to be created)
**Depends on:** `SKaiNET-developers/SKaiNET-transformers` — modules `llm-api`, `llm-providers`
**Labels:** enhancement, integration, spring
**Milestone:** —

---

## Summary

Build a thin Spring AI adapter + Spring Boot starter that exposes the existing
`sk.ainet.llm.api.ChatModel` / `EmbeddingModel` SPI through Spring AI's
`org.springframework.ai.chat.model.ChatModel` and
`org.springframework.ai.embedding.EmbeddingModel`. The adapter lives in a
**separate repository** so the SKaiNET core stays Kotlin Multiplatform and free
of Spring transitives.

This is the planned follow-up to the neutral SPI work that landed on
`feature/llm-api-neutral-spi` (modules `llm-api` and `llm-providers`).

## Motivation

- Spring AI's `ChatModel`/`EmbeddingModel` is the most familiar provider-SPI on the
  JVM today. Many users will reach for `spring-ai-starter-model-*` first.
- We **do not** want Spring as a dependency in the SKaiNET core. The neutral SPI
  in `llm-api` already mirrors Spring AI's shape; the adapter is a translation layer
  of a few hundred lines.
- A separate repo keeps Spring's release cadence (1.1.x ↔ 2.0.x) decoupled from
  SKaiNET-transformers' release cadence.

## What's already done (in this repo)

On branch `feature/llm-api-neutral-spi`:

- `llm-api/` — KMP module with `ChatModel`, `StreamingChatModel`, `EmbeddingModel`,
  `ChatRequest`/`Response`/`Chunk`, `ChatOptions`, `EmbeddingRequest`/`Response`,
  `Message`/`Role`, `ToolDefinition`/`ToolCall`, `Usage`, `FinishReason`. Kotlin
  `Flow` for streaming. Deps: `kotlin-stdlib` + `kotlinx-coroutines` only.
- `llm-providers/` — JVM module with `SkaiNetChatModel<T>` (wraps any
  `InferenceRuntime<T>` + `Tokenizer` + `ChatTemplate`) and `SkaiNetEmbeddingModel<T>`
  (wraps `BertEncoderRuntime<T>`).
- BOM updated; binary-compat baseline (`api/`) generated; basic unit tests for
  mappers and stop-sequence helper.

## Scope (this issue)

A new repository `SKaiNET-spring-ai` with two artifacts.

### 1. `spring-ai-skainet` — adapter library

```
sk.ainet.llm.spring/
  SpringSkaiNetChatModel    implements org.springframework.ai.chat.model.ChatModel
                                       , org.springframework.ai.chat.model.StreamingChatModel
  SpringSkaiNetEmbeddingModel implements org.springframework.ai.embedding.EmbeddingModel
  PromptMapper              Spring AI Prompt/Message  ↔ sk.ainet.llm.api.ChatRequest/Message
  OptionsMapper             Spring AI ChatOptions     ↔ sk.ainet.llm.api.ChatOptions
  StreamingBridge           kotlinx.coroutines.flow.Flow → reactor.core.publisher.Flux
                            (via kotlinx-coroutines-reactor `asFlux`)
```

Translation rules:
- Spring `Prompt.getInstructions()` → neutral `List<Message>`. Map roles 1:1.
- Spring `ChatOptions` → neutral `ChatOptions`. `temperature`, `topK`, `topP`,
  `maxTokens`, `stopSequences`, `seed`, `model` map directly. Spring-only knobs
  (`frequencyPenalty`, `presencePenalty`) are dropped with a debug log
  (the underlying SKaiNET runtime does not honor them today).
- `ChatResponse` ← neutral `ChatResponse`. Wrap each neutral `Generation` in a
  Spring `org.springframework.ai.chat.model.Generation`. Carry `Usage` into
  `ChatResponseMetadata`.
- Streaming: `stream(Prompt)` returns `Flux<ChatResponse>`. Each upstream
  `ChatResponseChunk` becomes a `ChatResponse` whose single `Generation` carries the
  delta as content. Final chunk includes finishReason + final usage.

### 2. `spring-ai-starter-model-skainet` — Spring Boot starter

```
sk.ainet.llm.spring.boot/
  SkaiNetAutoConfiguration
    @AutoConfiguration
    @ConditionalOnClass(SpringSkaiNetChatModel.class)
    @EnableConfigurationProperties(SkaiNetProperties.class)
    @ConditionalOnProperty(prefix = "spring.ai.skainet",
                           name = "enabled", havingValue = "true",
                           matchIfMissing = true)
  SkaiNetProperties  prefix = "spring.ai.skainet"
  resources/META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
```

Properties:

```yaml
spring.ai.skainet:
  enabled: true
  chat:
    model-path: /models/qwen3-0.6b.gguf       # required if chat enabled
    model-format: gguf                        # gguf | safetensors
    chat-template: auto                       # auto | qwen | gemma | llama3 | chatml
    options:
      temperature: 0.7
      max-tokens: 512
      top-k: 40                               # accepted, ignored at runtime today
      top-p: 0.95                             # accepted, ignored at runtime today
      stop-sequences: ["</s>"]
  embedding:
    model-path: /models/bge-small-en          # required if embedding enabled
    options:
      model: bge-small-en
```

Conditional bean wiring:
- `@Bean @ConditionalOnMissingBean ChatModel skaiNetChatModel(...)` —
  builds an `OptimizedLLMRuntime` from the configured GGUF path, picks the
  `ChatTemplate` via `ModelRegistry.detect(...)` (or the explicit override),
  wraps in `SkaiNetChatModel`, then in `SpringSkaiNetChatModel`.
- `@Bean @ConditionalOnMissingBean EmbeddingModel skaiNetEmbeddingModel(...)` —
  same pattern with `BertEncoderRuntime` (the DSL-path encoder) +
  `SkaiNetEmbeddingModel`, or simply `BertEmbeddingModel.fromHuggingFace(...)`.

## Open dependencies / blockers

The SPI is callable today, but to make the **starter** usable we need a
"one-call model loader" inside SKaiNET. Two options:

1. Add a `ModelLoader.fromGguf(path): InferenceRuntime<*>` (or similar) inside
   `llm-providers` and call it from the starter. Cleanest.
2. Have the starter replicate the wiring from
   `llm-apps/skainet-cli/Main.kt` (Arena, ExecutionContext, weight loader,
   tokenizer). Works, but bigger surface to maintain.

Recommend option (1) as a small follow-up PR in this repo before standing up
the Spring repo. Track separately if needed.

## Acceptance Criteria

- [ ] `SKaiNET-spring-ai` repo exists with `spring-ai-skainet` and
      `spring-ai-starter-model-skainet` modules and CI green on JDK 17 + 21.
- [ ] Sample Spring Boot app: 30 lines of YAML + a `@RestController` injecting
      `org.springframework.ai.chat.client.ChatClient` returns a non-empty
      response for a Qwen3-0.6B GGUF.
- [ ] `ChatClient` streaming endpoint (`text/event-stream`) emits tokens
      progressively, not in one chunk.
- [ ] `EmbeddingModel.embed("hello").size == dimensions` for a small
      sentence-transformers model.
- [ ] Spring AI 1.1.x compatibility documented; build sets `spring-ai-bom` as
      the dependency-version pin.
- [ ] No Spring or Reactor classes appear anywhere in SKaiNET-transformers' core
      modules.

## Reproduction / Test Plan

Once the companion repo is up:

1. Publish SKaiNET-transformers `llm-api` + `llm-providers` artifacts (snapshot
   to mavenLocal is fine for first iteration).
2. In the new repo, depend on those + `org.springframework.ai:spring-ai-bom:1.1.x`.
3. Run the sample Boot app against `Qwen3-0.6B-Q8_0.gguf`:
   - `POST /chat` with body `"Hello, who are you?"` → 200, non-empty body
   - `GET /chat/stream?q=hello` → `text/event-stream`, multiple chunks
4. Run the embedding sample against a `bge-small-en` checkpoint and check that
   two semantically similar sentences yield cosine similarity > 0.7.

## Reference

- Spring AI Ollama starter
  (`org.springframework.ai:spring-ai-starter-model-ollama`) is the closest
  structural analog for the autoconfig + properties layout.
- Streaming bridge: `kotlinx.coroutines.reactor.asFlux` (artifact
  `org.jetbrains.kotlinx:kotlinx-coroutines-reactor`).
- Memory note in this repo:
  `feedback_neutral_spi_over_framework_coupling.md` — keep Spring out of core.

## Related

- Plan file (this repo, this branch): `.claude/plans/spring-ai-s-own-docs-partitioned-toucan.md`
- Modules: `llm-api/`, `llm-providers/`
- BOM: `llm-bom/build.gradle.kts` already exposes both new modules.
