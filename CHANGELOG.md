# Changelog

All notable changes to **SKaiNET-transformers** are documented here. The
version line is kept in lock-step with the underlying SKaiNET engine
(`sk.ainet.core:*`) — a transformers `X.Y.Z` ships against engine `X.Y.Z`.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.31.0] — 2026-06-15

Version-aligned with **SKaiNET 0.31.0**. Completes the eager board-decode path
for FunctionGemma: the tied **Q8_0 lm_head now stays packed** (paired with the
engine's `ops.transpose` fix for all packed dtypes), and `load()` can cap the
context to fit constrained devices.

### Added

- **`maxInferenceLen` on `GemmaNetworkLoader.load()`** — an optional cap on the
  context length the eager network sizes its KV cache + RoPE tables for (default
  `min(contextLength, 4096)`, threaded through `applyWeightsToNetwork` →
  `gemmaNetwork`). A constrained-device consumer (e.g. the 1.9 GB SL2610 board)
  can pass a small value (e.g. `32` for a short tool-call prompt) to shrink the
  KV cache ~100×, which otherwise allocates ~0.4 GB at the first forward and OOMs
  the board after the weights load. Default `null` preserves existing behaviour. (#180)

### Changed

- **`gradle/libs.versions.toml` `skainet` pin: 0.30.0 → 0.31.0.** Picks up the
  engine's `ops.transpose` lazy-rewrap fix for **all** packed matmul dtypes
  (Q8_0/Q4_0 added) — required so the packed Q8_0 lm_head below transposes
  through `linearProject` instead of throwing `ClassCastException`. Downstream
  consumers get the upstream SKaiNET BOM transparently via `:llm-bom`.
- **`gradle.properties` `VERSION_NAME=0.31.0`.** Lock-step with the engine.
- **`com.networknt:json-schema-validator` → 3.0.4.** (#175)

### Fixed

- **Tied Q8_0 lm_head stays packed in the eager `NATIVE_OPTIMIZED` Gemma path.**
  FunctionGemma's `token_embd` is Q8_0 and tied, so `convertGemmaWeightsPacked`
  was dequantizing **both** `token_embd` and `output` to FP32 (2×~0.67 GB) —
  OOM on the 1.9 GB SL2610. `output`/lm_head now packs as Q8_0
  (`packGemmaKQuant` gained a Q8_0 case; the row-major→block-major relayout is
  generalized with a `blockSize` param) and runs on the (NEON) Q8_0 kernel;
  `token_embd` stays FP32 (it is gathered, not matmul'd) but is wrapped no-copy
  via `DenseFloatArrayTensorData` instead of `ctx.fromFloatArray` (which
  allocated a second ~0.67 GB buffer). Tied embed/lm_head footprint
  ~1.34 GB → ~0.76 GB. Verified byte-identical decode parity
  (`GemmaQ5KPackedParityTest`) and a stable ~1.06 GB load on the SL2610. (#179)

## [0.30.0] — 2026-06-14

Version-aligned with **SKaiNET 0.30.0**. Skips 0.29.x — SKaiNET-transformers
tracked the engine internally across that window (the in-progress Q5_K kernel
shipped as a local `0.29.1`) without a tagged release. The headline is
**Q5_K stays packed in the eager Gemma runtime** and the **Gemma
`NATIVE_OPTIMIZED` packed-weight path is now Kotlin/Native–ready** — the board
binary can keep K-quant weights packed without the JVM's `java.lang.foreign`
MemSeg path.

### Added

- **Q5_K packed in-kernel dequant in the eager Gemma runtime.** FunctionGemma-270M
  ships as `Q5_K_M`, but `GemmaMemSegConverter` previously dequantized Q5_K
  weights to FP32 on load ("no native matmul kernel yet for Q5_K"), giving up
  both the memory saving and the in-kernel dequant. SKaiNET 0.30.0 provides a
  first-class Q5_K packed matmul (`Q5_KBlockTensorData` + `Q5KMatmulKernel`:
  scalar / Panama / native), so the converter now relayouts the GGUF bytes to
  block-major and wraps them as `Q5_KBlockTensorData` (176 B/block). Dispatch and
  the lazy transpose reach the kernel through `DefaultCpuOps`. Verified by
  `GemmaQ5KPackedParityTest` (`-PincludeIntegration`): the Q5_K packed path
  decodes FunctionGemma byte-identically to the FP32 baseline —
  `[262146, 236769, 3255, 718, 498, 1373, 262152, 106]` →
  `<tool_0>(state="on")<end>` for *"Turn the light on."*
- **Kotlin/Native–ready Gemma packed-weight path.** The `NATIVE_OPTIMIZED`
  packed conversion was `jvmMain`-only (it built `MemSeg`/`Arena`-backed tensors
  via `java.lang.foreign`), so the Kotlin/Native board binary couldn't keep
  K-quant weights packed. The platform-neutral pieces now live in `commonMain`:
  - **`GemmaQuantLayout.kt`** (`commonMain`) — `logicalShapeFor`,
    `relayoutKSeriesRowMajorToBlockMajor` (KMP-safe `copyInto`), and
    `packGemmaKQuant<T>()`, which builds heap-packed Q4_K/Q5_K/Q6_K
    `BlockTensorData` directly with no `MemSeg`/`Arena`.
  - **`GemmaPackedWeights.kt`** (`commonMain`) — `convertGemmaWeightsPacked`
    packs Q4/Q5/Q6_K matmul weights to heap `Q*_KBlockTensorData`, dequants
    `token_embd`/`output` to FP32 (gathered, no transpose) and any other quant
    type to FP32 `[out, in]`. `extractRawBytes` reads the loader's bytes back
    across both backings (JVM `IntArrayTensorData` / native `Byte`-typed).
  - **`GemmaNetworkLoader.load()`** now runs `convertGemmaWeightsPacked` before
    `applyWeightsToNetwork` under `NATIVE_OPTIMIZED`, so `load(NATIVE_OPTIMIZED)`
    yields a runnable network on the board *and* the JVM (previously it could not
    be built from raw-byte weights at all). `GemmaMemSegConverter` (`jvmMain`)
    now shares the `commonMain` helpers; only the `MemSeg`/FFM conversion and the
    FP32 fallbacks stay JVM-only.
  Verified on JVM and `linuxX64` (`GemmaQuantLayoutTest`): relayout, packing, and
  the native byte-extraction round-trip run on every target, and
  `GemmaQ5KPackedParityTest` confirms all three paths (FP32 baseline, `jvmMain`
  MemSeg-packed, `load()` packed) produce the identical token sequence.

### Changed

- **`gradle/libs.versions.toml` `skainet` pin: 0.28.1 → 0.30.0.** Picks up the
  released Q5_K packed matmul, the NEON native kernels, and the Kotlin/Native
  cinterop. Downstream consumers get the upstream SKaiNET BOM transparently via
  `:llm-bom`, so no per-consumer migration is needed.
- **`gradle.properties` `VERSION_NAME=0.30.0`.** Lock-step with the engine.
- **`settings.gradle.kts` reverts the `mavenLocal()`-first dev shim.** The
  ordering added while consuming the in-progress local SKaiNET `0.29.1` is no
  longer needed now that 0.30.0 is on Maven Central; the release resolves the
  engine purely from Central. The opt-in `-PuseLocalSkainet` composite build is
  unchanged for local engine work.

### Fixed

- **`fix(gemma): dequant kernel-less quant types in `NATIVE_OPTIMIZED` instead of
  leaving raw bytes`.** Loading a Gemma GGUF whose attention/FFN weights used a
  quant type with no packed SIMD kernel (e.g. Q5_1) under
  `QuantPolicy.NATIVE_OPTIMIZED` crashed at the first decode step
  (`Transpose requires at least 2 dimensions` in `MultiHeadAttention` →
  `linearProject`): `GemmaMemSegConverter.convertOne` left every unhandled quant
  type as raw 1-D bytes. Kernel-less types now dequantize to a correct FP32
  `[out, in]` weight via a new `dequantPackedToFp32` helper (mirroring the proven
  `Gemma4WeightLoader.createTensor` column-major → row-major transpose). The
  supported packed types (Q4_0/Q8_0/Q4_K/Q6_K) keep their fast SIMD form; only
  kernel-less types pay the FP32 dequant.
- **`fix(llama): dequantize Q4_1 (and all non-packed quant types) in
  `DecoderGgufMemSegConverter``.** The converter handled only Q4_0/Q8_0 (packed)
  and Q4_K/Q5_K/Q6_K (dequant); every other quant type fell through an `else`
  branch that logged a warning and passed the raw quant bytes through unchanged,
  crashing deep inside matmul (e.g. `unsupported quant type Q4_1 for
  blk.0.ffn_down.weight` on Q4_1 Qwen3 models). The `else` branch now routes
  through `DequantOps.dequantFromBytes` to FP32, covering Q4_1, Q5_0, Q5_1, Q8_1,
  IQ4_NL/XS, TQ1/2_0, etc.; genuinely unknown types now fail explicitly at load
  time instead of crashing later inside matmul. Closes
  [#654](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/654).

### Tests / CI

- **`GemmaQ5KPackedParityTest`** — byte-identical decode parity across the FP32
  baseline, the `jvmMain` MemSeg-packed path, and the `load(NATIVE_OPTIMIZED)`
  `commonMain` packed path.
- **`GemmaQuantLayoutTest`** (`commonTest`) — block-transpose relayout, packing,
  and the byte-extraction round-trip; runs on JVM and `linuxX64`.
- **`DecoderGgufMemSegConverterTest`** — regression that a Q4_1 weight is
  dequantized to its logical 2-D FP32 shape rather than passed through as 1-D
  bytes.
- **`fix(gemma): macosArm64 target for `gemma-iree``** and CI parity fixes:
  MLIR-dump tests write to a portable build dir instead of a hardcoded local
  path; browser Mocha gets a 60 s timeout (parity with the engine repo).
- **`test(gemma): repoint stale FunctionGemma GGUF path`** — six real-model
  integration tests now point at the in-repo
  `sl2610-function-calling/models/` location, matching
  `GemmaQ5KPackedParityTest`; all pass against the published SKaiNET 0.30.0
  (`-PincludeIntegration`).

## [0.28.1] — 2026-06-06

Version-aligned with **SKaiNET 0.28.1**. Skips 0.26.x / 0.27.x —
SKaiNET-transformers tracked the engine internally across that window without a
tagged release.

### Changed

- **`gradle/libs.versions.toml` `skainet` pin: 0.27.0 → 0.28.1.** Picks up the
  completed Kotlin DSL → StableHLO → IREE export path. SKaiNET 0.28.0/0.28.1
  closed the remaining DAG-DSL export bugs: shape-changing ops now declare their
  inferred output type instead of echoing operand-0 — `reshape`/`matmul`/`concatenate`
  ([SKaiNET #673](https://github.com/SKaiNET-developers/SKaiNET/issues/673)) and
  `conv1d`/`gather`/`maxpool2d`/`avgpool2d`/`flatten`
  ([SKaiNET #675](https://github.com/SKaiNET-developers/SKaiNET/issues/675)) — and
  `reduce_window` is emitted in IREE's generic region form. A full gemma3 graph
  traced through `GemmaMlirDumpTest` / `GemmaTraceTest` now lowers to StableHLO
  that `iree-compile`s to a `vmfb`. No transformers-side API changes; existing
  callers compile unchanged.

### Verified

- `:llm-inference:gemma:jvmTest` green against the published SKaiNET 0.28.1
  (`GemmaMlirDumpTest` 1/1, `GemmaTraceTest` 1/1).

## [0.25.0] — 2026-05-25

Version-aligned with **SKaiNET 0.25.0**. Skips 0.24.x — SKaiNET-transformers has
been on 0.23.4 since 2026-05-08; the engine bumped 0.23.1 → 0.25.0 in the same
window without a tagged 0.24.x release on either side.

### Added

- **`DTypePolicy` accepted on every `*NetworkLoader.fromGguf` / `.fromSafeTensors`
  entrypoint.** SKaiNET 0.25.0 introduced the
  [hybrid adaptive DSL with optional dtype constraints RFC](https://github.com/SKaiNET-developers/SKaiNET/pull/616)
  — a sealed `DTypePolicy` type (`Any | Require | Prefer | OneOf`) carrying
  execution-side dtype intent through the loader / DAG / resolution pipeline.
  `LlamaNetworkLoader`, `QwenNetworkLoader`, `GemmaNetworkLoader`,
  `ApertusNetworkLoader`, and `VoxtralNetworkLoader` now each accept
  `dtypePolicy: DTypePolicy = DTypePolicy.Any` on every public companion
  factory. The policy is eagerly validated against the loader's actual
  output dtypes at construction time (via the new
  `sk.ainet.apps.llm.DTypePolicyValidation` helper), matching the SKaiNET
  0.25.0 `StreamingGgufParametersLoader.validatePolicy()` /
  `SafeTensorsParametersLoader.mapPolicyToBf16()` semantics:
  - GGUF entrypoints accept `Any` / `Prefer` / `OneOf` / `Require(FP32)` and
    reject `Require(BF16)` / `Require(FP16)` / `Require(other)` with the same
    error messages as SKaiNET's own GGUF loader.
  - SafeTensors entrypoints additionally accept `Require(BF16)` (matching the
    `KEEP_NATIVE` precedent that `Bf16LoadPolicy.toDTypePolicy()` is built on
    upstream).
  - All entrypoints fall through with no behavioural change on the default
    `Any` value, so the bump is fully back-compat.
- **`decoderTransformerNetwork(dtypePolicy = …)`** parameter on the shared
  decoder-only builder in `llm-core` — declarative slot for the top-level
  block policy. Forward-compat surface; not yet propagated into the underlying
  `DagBuilder.op(..., dtypePolicy = …)` slot SKaiNET 0.25.0 introduced
  (`HybridTransformerBlock.compile()` will read this in a follow-up). Setting
  a non-`Any` value compiles today and starts taking effect when the
  compile-step plumbing lands — no API change at consumers.
- **SafeTensors BF16 KEEP_NATIVE** in `DecoderSafeTensorsLoader`. When the
  consumer attaches a `DTypePolicy` that admits BF16 (`Require(BF16)`,
  `Prefer(BF16)`, or `OneOf` containing BF16), the loader stops dequanting
  BF16 tensors and instead wraps the packed 2-bytes-per-element buffer in
  `Bf16DenseTensorData`. The matmul dispatch in `DefaultCpuOpsJvm` (SKaiNET
  0.25.0) detects `Bf16TensorData` at runtime and routes to the SIMD BF16
  kernel — so a BF16 SafeTensors checkpoint now stays near its on-disk
  footprint in RAM instead of inflating ~2× to FP32. Threaded through
  `LlamaNetworkLoader` / `QwenNetworkLoader` / `VoxtralNetworkLoader`
  (each forwards `loader.dtypePolicy` into the
  `DecoderSafeTensorsLoader<T>(ctx, T::class, metadata, tied, dtypePolicy)`
  constructor). The default value remains `DTypePolicy.Any` — adaptive
  FP32 dequant, no behavioural change for existing callers. Validation
  errors still fire at the `LlamaNetworkLoader.withDtypePolicy(...)`
  boundary: `LlamaNetworkLoaderDTypePolicyTest` pins each policy arm.
- **Three reference smoke tests with `@Tag("smoke-reference")`.** The new
  smoke tier exists alongside the existing `@Tag("integration")` filter and
  pins the three architectures we always want to run end-to-end:
  - `llm-runtime/kllama` — `Qwen3ReferenceSmokeTest` (Qwen3-1.7B Q8_0 GGUF;
    exercises the new SKaiNET 0.25.0 `Q8_0MatmulKernel` end-to-end +
    Qwen's `RoPEMode.SPLIT_HALF` + QK-Norm).
  - `llm-runtime/kgemma` — `Gemma4ReferenceSmokeTest` (Gemma-4 E2B SafeTensors;
    sliding-window attention + per-layer KV sharing).
  - `llm-test/llm-test-java` — `BertLeafReferenceSmokeTest` (MongoDB
    `mdbr-leaf-ir` SafeTensors via the Java `KBertJava` consumer surface,
    with a cosine-similarity sanity check on paraphrase embeddings).
  Run with `./gradlew test -PsmokeReference -PincludeIntegration`. Each test
  self-skips via JUnit `Assumptions.assumeTrue` when the model artifact isn't
  resolvable through the standard `~/.lmstudio/models/` /
  `~/.cache/huggingface/hub/` / env-var fallback chain, so CI without model
  files stays green.

### Changed

- **`gradle/libs.versions.toml` `skainet → 0.25.0`.** Downstream consumers
  already get the upstream SKaiNET BOM transparently via `:llm-bom`
  (`api(platform("sk.ainet:skainet-bom:${libs.versions.skainet.get()}"))`,
  unchanged since 0.23.4 when the BOM auto-discovery convention plugin
  landed) — no per-consumer migration needed.
- **`gradle.properties` `VERSION_NAME=0.25.0`.** Lock-step with the engine.
- **`tasks.withType<Test>().configureEach { ... }`** at the root build now
  honors a `-PsmokeReference` project property — symmetric to the existing
  `-PincludeIntegration`. When set, JUnit Platform is filtered to
  `@Tag("smoke-reference")` so the smoke tier runs in isolation
  (`./gradlew test -PsmokeReference -PincludeIntegration`).
- **`tests/smoke/smoke-models.json`** gains a `"reference": true` flag on
  the three reference entries (`Qwen3-1.7B-Q8`, `Gemma4-E4B-GGUF`,
  `MongoDB-mdbr-leaf-ir`) so the shell smoke harness and the JVM smoke
  tier point at the same artifacts. The `smoke-test.sh` script does not
  yet consume the flag — follow-up.
- **`smoke-reference` GitHub Actions workflow.** New
  `.github/workflows/smoke-reference.yml` triggers the three
  `@Tag("smoke-reference")` tests via `./gradlew test -PsmokeReference
  -PincludeIntegration`. `workflow_dispatch`-only (manual) with three
  optional URL inputs — supply each artifact URL via the dispatch form
  and the staging steps download it into `RUNNER_TEMP`, set the env var
  the test reads (`QWEN3_1B7_MODEL_PATH` / `GEMMA4_E2B_SAFETENSORS_PATH`
  / `LEAF_MODEL_DIR`), and the smoke tier actually exercises the models.
  Run with empty inputs and every test self-skips via JUnit
  `Assumptions` — the workflow is green either way, so it's safe to
  promote to `push: branches: [develop]` later once a self-hosted
  runner with pre-cached checkpoints is available.
- **Catalog goes BOM-only.** Every `skainet-*` alias in
  `gradle/libs.versions.toml` is now coordinate-only (no `version.ref`);
  versions are supplied by the `sk.ainet:skainet-bom` platform
  constraint re-exported by `:llm-bom`. Every consumer module gains
  `implementation(project.dependencies.platform(project(":llm-bom")))`
  in each source set that pulls a `skainet-*` artifact. Bumping the
  engine is still a one-line change at the top of the catalog (the
  `[versions] skainet = "X.Y.Z"` line drives the BOM platform
  reference in `llm-bom/build.gradle.kts`), but every internal build
  now exercises the BOM — so a BOM-coverage regression fails locally
  instead of leaking into a published artifact. Mirrors the
  `llm-test/llm-test-java` reference pattern that landed in 0.23.4.

### Deferred

These pieces of the dtype-policy RFC integration are intentionally not in
this release. The threading surface accepts the API so consumers can
compile against the eventual implementation; the actual behavioural
changes land in follow-up PRs.

- **Per-DSL-layer dtype-policy parameters** on `TransformerDsl.kt` factories
  (`embedding` / `rmsNorm` / `multiHeadAttention` / `swiGluFFN` / `geGluFFN`
  / `xielu`). The DSL is module-based and would need a `Module`-level
  metadata side-map to carry the policy down to compile time; landing
  that without a consumer that reads it would add maintenance surface
  for no behavioural value today.
- **`HybridTransformerBlock.compile()` honoring the policy on
  `DagBuilder.op(..., dtypePolicy = …)` per the W6 SKaiNET PR.** Blocked
  on the side-map above.
- **`DecoderGgufWeightLoader` per-tensor policy enforcement.** The GGUF
  loader still dequants BF16 → FP32 unconditionally — SKaiNET 0.25.0's
  `StreamingGgufParametersLoader.validatePolicy()` itself rejects
  `Require(BF16)` for GGUF today (no KEEP_NATIVE GGUF backing yet), so
  this is parked until the engine grows that path. *(SafeTensors BF16
  KEEP_NATIVE shipped in this release — see Added.)*

## [0.23.4] — 2026-05-08

Transformers-only release; no SKaiNET engine bump in this version. The
focus is the BOM and the consumer-facing docs.

### Fixed

- **BOM coverage gap.** `:llm-inference:apertus` and `:llm-inference:voxtral`
  ship to Maven Central but were missing from `skainet-transformers-bom`'s
  constraints. Consumers who imported the BOM and pulled either of these
  artifacts got no version alignment for them.
- **Wrong artifact IDs in the README and tutorials.** The "Current release"
  snippet in `README.md` and the two tutorial pages
  (`getting-started-java.adoc`, `llama3-tool-calling.adoc`) showed
  `sk.ainet.transformers:llm-core` / `llm-runtime-kllama` / `llm-agent` —
  those are project paths, not published artifact IDs. The real
  coordinates are `skainet-transformers-core`,
  `skainet-transformers-runtime-kllama`, `skainet-transformers-agent`;
  anyone copy-pasting hit a "module not found" error. Fixed and switched
  the snippets to the BOM pattern so future version bumps only need to
  touch one line.

### Changed

- **BOM internals: auto-discovery.** The constraint list in
  `llm-bom/build.gradle.kts` is no longer hand-maintained. A new
  convention plugin in `buildSrc/` (`sk.ainet.transformers.bom-coverage`)
  auto-discovers every sibling subproject that applies
  `com.vanniktech.maven.publish` and adds it as an `api` constraint on
  the BOM. The only manual input left is the exclusion list (currently
  just `:llm-performance`); the BOM is coherent by construction —
  missing or drifting modules can no longer happen.
- **`llm-test-java` consumes SKaiNET through the BOM** so the BOM is
  exercised during the build itself; a regression in BOM constraints
  fails locally instead of leaking into a published artifact.
- **Removed dead `group = "sk.ainet.llm"` override** from the root build.
  The published group has always been `sk.ainet.transformers` (sourced
  from `gradle.properties`); the override was being overridden in turn
  by vanniktech at publish time. The in-memory project group now matches
  the published group, which removes a footgun for anyone trying to
  resolve internal modules by GAV.

## [0.23.3] — 2026-05-06

Version-aligned with **SKaiNET 0.23.3**.

### Added

- **Prefill progress callback.** `generateUntilStop` gains an optional
  `onPrefill: ((Int, Int) -> Unit)?` parameter that fires once per prompt
  token during the autoregressive prefill loop, with `(done, total)` —
  `done` is 1-based, `total` is `prompt.size`. Plumbed through both
  `AgentLoop.run` and `AgentLoop.runWithEncoder` as a new
  default-no-op `AgentListener.onPrefillProgress(done, total)` method.

  Why this matters: prefill is autoregressive in 0.23.x (the comment on
  `generateUntilStop` documents the `forwardBatched` correctness
  regression we reverted), so on a CPU-only runtime with a 300-token
  prompt the first `onToken` lands tens of seconds to minutes after the
  agent loop starts — UIs previously had no way to show the loop was
  alive. The new callback closes that gap (e.g. `prefill: 32/282 (11%)`).

  Backwards compatible — the new parameter and interface method default
  to null/no-op, so existing `AgentListener` implementations and callers
  compile and behave unchanged.

### Tests

- New tests for the prefill callback in `GenerateExtensionsTest`:
  - `generateUntilStopReportsPrefillProgressForEachPromptToken` —
    one `(done, total)` pair per prompt token, in order, with `done`
    1-based and `total = prompt.size`.
  - `generateUntilStopWithEmptyPromptDoesNotInvokePrefillCallback` —
    callback never fires for an empty prompt.

## [0.23.2] — 2026-05-05

Version-aligned with **SKaiNET 0.23.2**.

### Added

- **Llama 3 tool-calling walkthrough** — end-to-end docs for app integrators,
  covering chat template, JSON tool-call format, and `JavaAgentLoop` wiring.
- **Llama-3.2-1B-Instruct smoke test** with a tool-calling assertion.
- **MongoDB / mdbr-leaf-ir embedding entry** in the smoke runner catalogue.
- **`kllama-cli`**: prompts, raw responses, and tools list now logged by
  `ToolCallingDemo`.

### Changed

- **`kllama-cli`, `kllama-native`, and `kllama-wasm` swapped to the DSL
  path** (`OptimizedLLMRuntime` + `llamaNetwork()`); placeholder GPU
  attention/tensor stubs deleted; native benchmark scenario renamed to
  `native-cpu-throughput`.
- **`KLlamaJava` facade swapped to the DSL path.**
- **`llm-core`**: SentencePiece decorator + GGUF tokenizer now route
  through upstream `sk.ainet.io.tokenizer` instead of a local fork; fixes
  Qwen / GPT-2 BPE GGUF tokenization.

### Fixed

- `fix(tool-calling): tolerate markdown code fences around Llama 3 JSON
  tool calls` — the parser previously skipped fenced JSON, causing the
  agent loop to keep generating until `maxTokensPerRound` instead of
  executing the call.
- `fix(qwen): NEOX (SPLIT_HALF) RoPE pairing for Qwen3 GGUFs.`
- `fix(transformer): thread metadata RMSNorm eps through QK-norm.`
- `fix(llama): inject logical 2D shape and dequant token_embd in DSL
  converter.`
- `fix(kllama-cli): route Llama GGUF/SafeTensors back to eager
  `LlamaRuntime`` — the DSL Q4/Q8 path is functionally correct but needs
  first-class Q4/Q8 DTypes to match the SIMD perf of the legacy path.
  Tracked as a followup.
- `fix(kllama-cli): apply application plugin so :run task is wired.`
- `fix(smoke): tolerate runners that don't emit tok/s (embedding models).`

### Removed

- `:llm-runtime:kqwen` module and `LlamaIngestionBlocking.kt` deleted.

### Docs

- API dumps refreshed for 0.23.2 (`api/` directory).

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

[0.31.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.31.0
[0.30.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.30.0
[0.28.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.28.1
[0.23.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.23.1
[0.21.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.1
[0.21.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.0
[0.18.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.18.0
