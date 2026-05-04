# Changelog

All notable changes to **SKaiNET-transformers** are documented here. The
version line is kept in lock-step with the underlying SKaiNET engine
(`sk.ainet.core:*`) — a transformers `X.Y.Z` ships against engine `X.Y.Z`.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.23.1] — 2026-05-04

Version-aligned with **SKaiNET 0.23.1**.

### Added

- **Apertus end-to-end.** Real-GGUF loading now works on top of skainet 0.23.x's
  block-major Q4_K `TensorData` wiring. Routing fix to go through
  `OptimizedLLMRuntime` + `apertusNetwork()`, plus chat template, tool calling,
  and integration tests against `Apertus-8B-Q4_K_S`. See
  [`APERTUS_ROLLOUT.md`](APERTUS_ROLLOUT.md).
- **Gemma 4 chat-model JVM facade** (`Gemma4ChatModel`) for embedded text-only
  deployments. `close()` now propagates to the mmap arena. The PLE mmap path
  consumes upstream `loadTensorStorageMapped` rather than maintaining a fork.
- **Multi-id EOS / stop-token support** in the chat layer — needed for templates
  that emit several end-of-sequence markers (e.g. ChatML / Apertus).
- **End-to-end smoke test** in `llm-test/llm-test-java`
  (`Llama3LeafSmokeTest`) that wires LEAF (`mdbr-leaf-mt`, via `KBertJava`) and
  Llama 3.2-1B (`KLlamaJava`) in one JVM, gated on env vars / cache fallbacks
  so CI without the checkpoints cleanly skips.
- **Apertus tool calling** as a first-class family alongside Llama 3, Gemma 4,
  Qwen, and ChatML/Hermes.

### Changed

- `gradle/libs.versions.toml` `skainet` pin: 0.22.1 → **0.23.1**.
- `VERSION_NAME`: 0.21.1 → **0.23.1** (no 0.22.x transformers release was tagged;
  the version line jumps to keep the engine and consumer artifacts in sync).
- `kllama-cli` and `skainet-cli` shadow-jar builds now apply the
  `ServiceLoader` `META-INF/services` merge fix-up so the priority-100
  `skainet-backend-native-cpu` provider is picked up at runtime.
- `llm-test/llm-test-java` `maxHeapSize` 8g → 16g — the previous cap OOM'd
  while loading both Llama 3.2-1B + LEAF in a single JVM.

### Fixed

- `fix(apertus): force-dequant token_embd under NATIVE_OPTIMIZED` — Apertus
  was producing garbage on quantized embeddings; we now dequant the token
  embedding tensor regardless of policy, matching upstream behaviour.
- `fix(tokenizer): auto-detect SentencePiece marker in fromTokenizerJson` —
  models that ship a `tokenizer.json` without the explicit
  `pre_tokenizer.type = SentencePiece` marker now decode correctly.
- `fix(gemma4): produce coherent text on real SafeTensors checkpoint` — the
  loader path for full HF-format Gemma 4 checkpoints (not just the GGUF
  variant) now produces coherent generations end-to-end.
- `fix(apertus): route through OptimizedLLMRuntime + apertusNetwork()` —
  the legacy direct-runtime path was bypassed; Apertus now flows through the
  optimized DAG runtime like every other family.

### Tests / CI

- `test(apertus): real-GGUF loader integration test against Apertus-8B-Q4_K_S`.
- `test(apertus): pin weight-loader fixes with regression tests`.
- `test(kgemma): fast tokenizer parity guard against HF reference`.
- `test(kgemma): tighten tool-call probe budget + add env override`.
- Native-cpu provider now wired into the `qwen` and `llama` JVM test runs so
  the priority-100 FFM kernels are exercised during CI.

### Docs

- `docs(apertus): document chat-template format` plus the staged-rollout plan
  at the repo root (`APERTUS_ROLLOUT.md`).
- README refreshed: lead with native FFM CPU performance numbers, current
  release coordinates at 0.23.1, "What's new" section in place of the previous
  "In develop, not in X yet" callout.

### Removed

- `chore(apertus): close out rollout — remove deprecated runtimes`. The
  pre-rollout direct-runtime entry points for Apertus are gone.

## [0.21.1] — 2026-04-30

Hotfix release: add missing `POM_NAME` for the `apertus`, `voxtral`, and
`llm-performance` modules so Maven Central publishing succeeds.

## [0.21.0] — 2026-04-29

Version-aligned with **SKaiNET 0.21.0**.

- `chore(release): bump SKaiNET to 0.21.0, prepare transformers 0.21.0` —
  mirror the engine version in the transformers line so the coupling is
  explicit for Maven Central consumers. Engine highlights (delivered via
  the bump): Panama Vector FP32 matmul kernel auto-discovered via
  `ServiceLoader`, `ScratchPool` SPI, Q4_K SIMD-fused matmul kernel,
  Q6_K dequant via `ByteVector ql` + `qh` extraction, canonical ggml
  layout for Q4_K + Q5_K, FP32 `MemSeg` arena leak fix.
- `VERSION_NAME` jumps 0.18.0 → 0.21.0 to align tags with the engine; no
  0.17.0 / 0.19.x / 0.20.0 transformers releases were ever tagged.

## [0.18.0] — earlier

Last published transformers release before the engine-aligned version line.
See `git log v0.16.0..0.18.0` for details.

[0.23.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.23.1
[0.21.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.1
[0.21.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.0
[0.18.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.18.0
