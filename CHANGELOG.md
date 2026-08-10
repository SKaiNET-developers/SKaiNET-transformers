# Changelog

All notable changes to **SKaiNET-transformers** are documented here. The
version line is kept in lock-step with the underlying SKaiNET engine
(`sk.ainet.core:*`) — a transformers `X.Y.Z` ships against engine `X.Y.Z`.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **iOS artifacts for the runtime facades.** `llm-runtime/kllama` and `llm-runtime/kgemma` now
  declare `iosArm64` + `iosSimulatorArm64` and publish the corresponding klibs. kllama's
  `src/iosMain` (the `registerPlatformBackends` actual) predated the targets and was silently dead —
  these modules set `kotlin.mpp.applyDefaultHierarchyTemplate=false`, so the `iosMain` source set is
  now wired by hand (`iosMain → nativeMain`, mirroring `llm-core`). All commonMain dependencies of
  both modules already published iOS. No CLI executables are declared for the Apple targets —
  consumers link the klib into their app. Closes
  [#271](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/271).
- **Supported-targets matrix in the README.** A module-vs-target table (derived from each module's
  `build.gradle.kts`) replaces the "where applicable" hand-wave, so which artifact runs on iOS /
  Android / Wasm is now documented rather than discoverable only by browsing Maven Central
  ([#271](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/271)).

## [0.38.0] — 2026-07-31

Ships against **SKaiNET engine 0.38.0**, which adds first-class dynamic tensor shapes (`Dim`) plus
the narrow-float codec (`Fp16DenseTensorData`, FP16 matmul kernels, codec-driven dispatch — engine
PR #886). Two headlines: **Moonshine v2 streaming ASR authored end-to-end in the SKaiNET NN DSL**
(the last vendor-ONNX graph is gone) and **narrow-float `KEEP_NATIVE` weights** across the LLM loaders.

### Added

- **Moonshine v2 — the complete streaming pipeline in the NN DSL**, self-compiled DSL → StableHLO →
  IREE with no vendor neural binaries (`skainet-transformers-inference-moonshine`):
  - **Audio frontend** (`MoonshineV2Frontend`): CMVN → `asinh` compression → filterbank matmul →
    SiLU → two causal `Conv1d(k5,s2)` — the last vendor-ONNX graph, now DSL-authored (bit-exact vs
    `frontend.onnx`, cos > 0.999).
  - **Encoder** (position-free sliding-window local attention) and **adapter** (learned absolute
    positional embedding, pos-embed add only) bridging the position-free memory to the decoder.
  - **Decoder** authored in the DSL, reusing the shared KV-cache decoder.
- **True-dynamic KV-cache decode graphs.** `MOONSHINE_V2_TRUE_DYNAMIC` / `GEMMA_TRUE_DYNAMIC` trace
  the cache seq dim as a real dynamic extent (`Dim.DYNAMIC`), so one compiled vmfb serves every
  autoregressive position instead of a fixed-shape re-decode. Requires engine 0.38.0's `Dim`.
- **Fixed-max-pad cross-attention mask** for streaming decode: pad the encoder memory to a fixed MAX
  and mask the padding, so one prefill + one `with_past` pair serve any encoder length ≤ MAX while the
  self-cache stays dynamic (growing). `transformer-core`'s `MultiHeadAttention` gains an optional
  trailing `crossMask` (default `null` → byte-identical for existing callers).
- **Gemma row-dequant of the packed `token_embd`** in the shared `Embedding`, cutting host memory at
  load.
- **FP16 KEEP_NATIVE on the SafeTensors path.** `DecoderSafeTensorsLoader` gains the F16 arm
  that BF16 has had since 0.25.0: with a `DTypePolicy` admitting FP16 (`Require(FP16)`,
  `Prefer(FP16)`, or `OneOf` containing FP16) it stops widening F16 tensors and wraps the
  on-disk 2-bytes-per-element buffer in `Fp16DenseTensorData`. The arm was missing only
  because no such storage type existed. `DefaultCpuOpsJvm` matches `NarrowFloatTensorData`
  and picks the kernel by codec, so an F16 checkpoint now stays near its on-disk footprint
  instead of inflating ~2× as FP32. Covers LLaMA, Qwen, and Voxtral, which share this loader.
- **Narrow-float KEEP_NATIVE on the GGUF path — `DTypePolicy` is honored there at all now.**
  `DecoderGgufWeightLoader` accepts a `dtypePolicy` and keeps F16 / BF16 source tensors packed
  instead of widening every one to FP32. `LlamaNetworkLoader`, `QwenNetworkLoader`, and
  `VoxtralNetworkLoader` plumb the policy attached via `withDtypePolicy` down into it; before
  this the GGUF branches constructed the loader without the policy and silently ignored it.
  This is the KEEP_NATIVE GGUF path the 0.25.0 notes parked, and it is what makes
  `Require(BF16)` real on GGUF.

  The packed path mirrors the FP32 path's layout handling exactly: for rank 2 it swaps the
  shape to `[cols, rows]` and **moves no bytes**. GGUF header dims are reversed relative to the
  logical row-major shape, so the "column-major → row-major" step is a reinterpretation, not a
  permutation (`DequantOps.transposeColumnMajorToRowMajor` returns its input unchanged). An
  actual element transpose here would have handed the matmul kernel a silently transposed
  weight matrix. The result is genuinely zero-copy — the on-disk buffer becomes the storage.

### Changed

- **`DTypePolicyValidation` capability model is per-format.** `validate(policy, loaderName,
  keepNative: Set<DType>)` replaces the BF16-only `allowBf16Require: Boolean` (kept as a
  `@Deprecated` overload). A caller declares which narrow-float formats its chain actually
  hands through packed, and a `Require` naming one is accepted only by a chain that can honor
  it. The boolean could express neither "keeps FP16 but not BF16" nor the empty case.

  The two formats are tracked separately and never interchangeably: `Require(BF16)` still
  widens F16 sources, and vice versa. Both are 2 bytes per element, so mis-tagging F16 bytes as
  BF16 decodes to plausible-looking garbage rather than throwing. `DTypePolicyValidation
  .keepsNative(policy, native)` is the single decision point both loader chains share, mirroring
  the engine's `mapPolicyToNarrow` / `keepsNative`.
- **`Require(FP16)` is now accepted** by `LlamaNetworkLoader`, `QwenNetworkLoader`, and
  `VoxtralNetworkLoader` (both GGUF and SafeTensors), and **`Require(BF16)` is now accepted on
  their GGUF paths**. Both previously threw.
- **Binary-breaking (source-compatible): `DecoderGgufWeightLoader` constructors** gain a
  trailing `dtypePolicy: DTypePolicy = DTypePolicy.Any`, which changes their JVM descriptors.
  Kotlin and Java callers compile unchanged; already-compiled callers must be rebuilt.
  Behaviour with the default is identical to before.

### Fixed

- **`GemmaNetworkLoader` and `ApertusNetworkLoader` no longer accept a `Require(BF16)` they
  ignore.** Both have their own weight chains (`Gemma4WeightLoader` /
  `Gemma4SafeTensorsWeightLoader`, `ApertusWeightLoader` / `ApertusSingleSafeTensorsLoader`)
  which widen every narrow float to FP32 and have no KEEP_NATIVE path. Their SafeTensors
  entrypoints nevertheless passed `allowBf16Require = true`, so `Require(BF16)` validated and
  was then silently disregarded at load — the exact failure the eager validator exists to
  prevent. They now declare `keepNative = emptySet()` and reject it. **Callers relying on the
  old acceptance must switch to `Prefer(BF16)`** (a soft constraint, which still passes) until
  those chains grow a KEEP_NATIVE path.
- Moonshine v2 encoder sliding-window off-by-one (was cos 0.991 vs ONNX); the v2 config is set to the
  real tiny-streaming dims; the adapter is pos-embed add only (no LayerNorm).
- kgemma heavy-trace test heap raised to 12 g, with honest skips.

## [0.36.1] — 2026-07-17

Patch on **0.36.0** (same SKaiNET engine 0.36.0). Two additions: **BGE embedding models** on the
BERT DSL path (CLS pooling + retrieval prefixes), and **beam search** for the T5 decoder and the
vec2text inversion loop. Both are additive — existing consumers are untouched, and the vec2text
greedy path is unchanged when both beam widths are 1.

### Added

- **BGE embedding models** (`BAAI/bge-small-en-v1.5` and siblings) run on the BERT DSL path:
  - **CLS pooling.** `BertPooling { MEAN, CLS }` on `BertEncoderRuntime` /
    `createBertEncoderRuntime`; auto-detected from the sentence-transformers
    `1_Pooling/config.json` (absent file → `MEAN`, unsupported max/sqrt-len modes rejected
    loudly). Pooling stays outside the traced graph — OPTIMIZED mode and StableHLO export
    are unaffected.
  - **Query/document asymmetry.** `EmbeddingModel` gains `embedQuery` / `embedDocument` /
    `embedDocuments` (defaults delegate to `embed` — additive, existing consumers untouched).
    `PrefixedEmbeddingModel` + `EmbeddingModelProfiles` apply retrieval instruction prefixes
    per repo id (E5 `query: `/`passage: `, BGE query instruction); `fromHuggingFace` wires
    them automatically, `fromSafeTensors` accepts explicit `prefixes`.
  - **Integer checkpoint buffers no longer break loads.** BGE-style snapshots persist an
    I64 `embeddings.position_ids` buffer; the interim `FloatSafeTensorsLoader` skips
    non-float buffers (the index-free encoder never needs them). Drop when the engine's
    loader gains a tensor filter ([SKaiNET#822](https://github.com/SKaiNET-developers/SKaiNET/issues/822)).
  - Design + traceable plan: `docs/specs/embedding-model-coverage.md` (E5 multilingual
    follows in Phase 2 — Unigram tokenizer).
- **Token-level beam search on the T5 decoder.** `T5Runtime.generateBeam(memory, numBeams,
  maxLength, lengthPenalty)` returns up to `numBeams` sequences, best-first by length-normalized
  log-probability. It shares a new `decoderLastLogits()` step with greedy `generate`, and adds
  `logSoftmax` plus linear top-k helpers. There is still no KV cache, so decode cost scales
  roughly linearly with `numBeams`.
- **Sequence-level beam search across correction rounds.** `Vec2TextInverter.invert(...,
  sequenceBeamWidth, tokenBeams)` and `invertEmbedding()` keep `beamWidth` hypotheses between
  correction steps, ranked by cosine similarity to the target embedding — the oracle the beam
  exploits. `InversionModel.invertBeam` / `CorrectorModel.correctBeam` expose the top-N candidates
  from each stage.
- Verified end-to-end on real gtr-base weights: at one correction step, beam (sequence width 3,
  token beams 3) improves cosine **0.765 → 0.818** over greedy on the round-trip test's example
  sentence, with a visibly closer reconstruction. Covered by `Vec2TextRoundTripTest`
  (`invert_beamBeatsGreedy`), which skips unless `VEC2TEXT_MODELS_DIR` is set.

## [0.36.0] — 2026-07-12

Ships against **SKaiNET engine 0.36.0**. Headline: **BERT is now completely defined on the DSL
path** — the legacy hand-coded eager stack is removed (**BREAKING**, see *Removed*), and sentence
embeddings get a one-call factory with built-in Hugging Face Hub download. Also new: a **T5
encoder-decoder** runtime and a **vec2text embedding-inversion** pipeline (invert GTR embeddings
back to text). Downstream impact:
indexing the leaf-cli reference corpus (56 chunks) drops from 676.9 s to 44.5 s (~15×) with
identical embeddings.

### Added

- **BERT sentence embeddings completed on the DSL path.** `bertNetwork()` is now a numerically
  complete `tokens → hidden-states` encoder: the new `BertEmbeddings` module adds absolute-position
  and token-type embeddings (index-free `narrow`-based lookups, single-segment) that the DSL
  definition previously omitted. New `BertEncoderRuntime` executes it eagerly (**DIRECT**, default)
  or as a traced, optimized **ComputeGraph** (**OPTIMIZED**, shape-specialized per sequence length
  with an LRU cache) and adds masked mean pooling, the optional sentence-transformers `2_Dense`
  projection, and L2 normalization on top of the pure encoder graph. The encoder trace lowers to
  StableHLO (gather / dot_general / SDPA preserved) — export is gate-tested; IREE *execution* of the
  exported module stays out of scope for now. Verified against the PyTorch-validated legacy runtime
  on real MongoDB/mdbr-leaf-mt (hidden-state parity ≤ 2.2e-6) and DIRECT-vs-OPTIMIZED bit-exact.
- **One-call embedding factory with built-in Hugging Face download.**
  `BertEmbeddingModel.fromHuggingFace("MongoDB/mdbr-leaf-mt")` (llm-providers) downloads the
  snapshot via the engine's `skainet-data-source` (`hf://` URIs, `HF_TOKEN`-aware) into
  `~/.cache/skainet/models/`, streamed with `.part` + atomic rename, offline-safe after the first
  run; `fromSafeTensors(dir)` loads a local snapshot, auto-detecting weights, config, tokenizer
  (`vocab.txt` → `tokenizer.json`), and the `2_Dense/` head. `kbert-cli` accepts an HF repo id
  directly: `kbert MongoDB/mdbr-leaf-mt "query" "doc"`.
- `BertConfigParser` — shared `config.json` (+ `2_Dense/config.json` → `projectionDim`) parser,
  consolidating the copies previously living in `KBertJava` and downstream apps.
- **T5 encoder-decoder runtime** (`llm-inference/t5`, `sk.ainet.models.t5`). Hand-coded in the
  direct tensor-ops style (per-head attention via narrow/matmul/softmax, batch 1, no KV cache —
  the greedy decoder recomputes the stack per step), handling T5's specifics: no 1/√d attention
  scaling, learned relative-position bias (`T5RelativeBias`, block-0 table shared per stack,
  none in cross-attention), RMSNorm-style T5LayerNorm, un-gated ReLU FFN, tied embeddings with
  `d_model^-0.5` logit scaling. Includes `GtrEmbedder` — GTR sentence embeddings exactly as
  vec2text consumes them (raw T5 encoder + mean pooling; deliberately no Dense projection and no
  L2 normalization) — with a parity test against real `sentence-transformers/gtr-t5-base` weights.
- **vec2text embedding inversion** (`llm-inference/vec2text`, `sk.ainet.models.vec2text`).
  Port of vec2text's greedy corrector loop (`sequence_beam_width = 1`): `InversionModel`
  produces an initial hypothesis from a target GTR embedding, then `CorrectorModel` iteratively
  re-embeds and corrects it, early-stopping when the cosine score plateaus — `Vec2TextInverter`
  returns the best reconstruction plus the full step trace. Verified with an end-to-end
  round-trip test on real gtr-base weights.

### Fixed

- **BERT post-norm residual wiring.** The single-block-per-layer `bertNetwork()` definition wired the
  FFN residual to the pre-LayerNorm value — the transformer blocks' residual rule fits pre-norm
  decoder stacks, but BERT is post-norm. Each encoder layer is now two blocks (`attn` / `ffn`) so
  every residual segment starts at the correct value.
- **Bias-free `2_Dense` projection heads were silently dropped.** The legacy eager runtime required
  projection weight *and* bias; LEAF models ship `bias=false`, so it skipped the projection entirely
  (returning 384-dim vectors while advertising 1024). `BertEncoderRuntime` applies bias-free
  projections; `KBertJava` now picks up `2_Dense/` heads it previously ignored.
- **Graph replay dropped `permute` axes.** The ComputeGraph executor's builtin dispatch replayed
  `permute` as a plain last-two-dims transpose, breaking every multi-token attention trace —
  single-token decode never hit it. Fixed upstream in engine 0.36.0
  ([SKaiNET#803](https://github.com/SKaiNET-developers/SKaiNET/pull/803)), which this release
  consumes; the interim axes-aware `permute` handler in `LLMFusedOpHandlers` (never in a published
  release) is removed again.

### Removed

- **BREAKING: the deprecated hand-coded BERT stack is gone** — `BertRuntime`, `BertRuntimeWeights`,
  `BertLayerWeights`, `loadBertWeights`, `BertWeightMapper`, `BertTensorNames`, `BertIngestion`, and
  `BertNetworkLoader.fromRuntimeWeights`. Migrate to `createBertEncoderRuntime(config, tensors, ctx)`
  (tensors from `BertNetworkLoader.loadWeightTensors`) or, one level up, to
  `BertEmbeddingModel.fromSafeTensors(...)` / `fromHuggingFace(...)`. `SkaiNetEmbeddingModel`'s
  constructor now takes `BertEncoderRuntime`; `KBertJava` / `KBertSession` keep their method surface
  (`loadSafeTensors` / `encode` / `similarity`) with the constructor type changing. `BertModelConfig`
  and `MDBR_LEAF_IR_CONFIG` moved to `BertConfig.kt` (same package — imports unaffected). The
  `docs/optimizable-LLM-NNs-DAG.md` reference in the old deprecation pointed at a document that never
  existed; the real migration guide is `explanation/dsl-vs-handcoded.adoc`.

## [0.35.0] — 2026-07-09

Ships against **SKaiNET engine 0.35.0**, whose new `argMax` op this release uses to fold the LLM
`logits → token-ids` tail into the DSL trace.

### Added

- **FunctionGemma self-compile from the SKaiNET DSL** (`sk.ainet.transformers:…-kgemma`). One reusable
  dependency for the FunctionGemma-270M function-calling sLLM, in **both** SKaiNET execution modes:
  - `FunctionGemma.fromGguf(gguf).call("turn the light on")` → `ToolCall(set_lights, {state="on"})` —
    **eager** (DirectCpu + `OptimizedLLMRuntime(DIRECT)` + Octopus-v2 template + `CompactCodec`), runs
    anywhere on CPU, no iree. (The `partialRotary = 1.0` gemma3 rotary fix is applied.)
  - `FunctionGemma.exportCompiled(outDir)` / `FunctionGemmaExport.export(…)` — **compiled** edge path:
    traces `gemmaNetwork()` ending in `ops.argMax(logits, -1)` (the engine op), emits StableHLO with
    **bf16 external params** (bf16 globals + convert-on-load + bf16 safetensors). Promotes the former
    `RealGemmaBakeIrpaTest` and retires the Python argmax/f16 MLIR rewrites. Verified token-for-token
    against llama.cpp on the SL2610 board.
  - `exportFunctionGemma` Gradle task (for `scripts/compile-gemma.sh`); `kgemma` jvm deps gain
    `skainet-compile-hlo`/`-dag` + `gemma-iree` (`CompactCodec`).

### Changed

- **Engine → 0.35.0.** Adopts the new engine line; the compiled FunctionGemma export depends on the
  engine's new `argMax` op. (engine 0.35.0)

## [0.34.1] — 2026-07-05

Patch on **0.34.0** (same SKaiNET engine 0.34.0). Fixes Moonshine encoder parameter naming.

### Fixed

- **Layer-qualified Moonshine encoder parameter names.** The encoder's attention and LayerNorm
  parameters were not prefixed with the layer (`attn.q_proj.weight`, `attn_norm.weight` repeated
  identically every layer), while the FFN parameters were (`enc.$layer.ffn_*`). By-name weight
  loading could therefore not distinguish the layers. All parameter names are now unique and
  layer-qualified (`enc.$layer.attn.*`, `enc.$layer.attn_norm.*`, `enc.$layer.ffn_norm.*`),
  matching the FFN convention. No public API change — `moonshineEncoder()` is unchanged.

## [0.34.0] — 2026-07-05

Ships against **SKaiNET engine 0.34.0**. Headline: the first **Moonshine** speech-to-text encoder
authored entirely in the SKaiNET NN DSL, plus the RoPE work that makes transformer exports
bit-exact on a real NPU.

### Added

- **`skainet-transformers-inference-moonshine`** (new, first published module) — the Moonshine-tiny
  audio **encoder** built in the NN DSL, bf16-native, emitting portable (hardware-agnostic) StableHLO.
  It compiles through the SKaiNET pipeline and transcribes correctly on both CPU and the Synaptics
  Torq NPU. The exported IR carries no target-specific ops — backend optimizations plug in from
  outside core (see the vendor-plugin pattern).
- **Partial rotary embeddings** in `transformer-core`: `RoPE` gains `partialRotaryFactor` (rotate only
  the leading fraction of each head, the rest passes through) and `freqDenomRotaryDim` (compute
  `inv_freq` over the rotary dim rather than the full head dim). `TransformerDsl.rope()` threads both.
  Matches models like Moonshine (rotate 32 of 36 head dims), verified against the reference ONNX.
- **`VoidDense(addBias = true)`** — a projection can now add its `$name.bias` term, keeping traced
  FFNs faithful to reference checkpoints that carry `fc1.bias` / `fc2.bias`.

### Changed

- **RoPE precision & form.** The interleaved rotation and its `cos`/`sin` tables are computed in
  **f32** (upcast, then back to model dtype), and the interleaved path uses the **full-head (ONNX)
  form** — numerically identical to the split-recombine form but bit-exact once accelerator layout
  passes sit between the split and merge. Fixes low-precision RoPE drift on NPU targets.
- **Engine → 0.34.0.** Transformer models inherit the engine's 0.34.0 work (f32 LayerNorm
  decomposition, the pluggable target-optimizer / op-granularity seam that keeps exported StableHLO
  portable).

## [0.33.0] — 2026-06-29

Ships against **SKaiNET engine 0.33.0**. No transformers API changes — this release adopts the new
engine line and routine dependency updates.

### Changed

- **Engine → 0.33.0.** Transformer models authored with this layer inherit the engine's 0.33.0 work;
  most relevant here, `layerNorm` / `rmsNorm` now lower to real `stablehlo.reduce`, so transformer
  exports compile and run on stock IREE (engine #769). The engine also fixes a silent autodiff
  gradient-drop (`elu`/`leakyRelu`/`permute`) and adds new differentiable ops (`cos`/`sin`/`gather`/…),
  available to model authors. (engine 0.33.0)
- **Dependencies:** Ktor client `3.5.1` (#198), Logback `1.5.36` (#199).

## [0.32.1] — 2026-06-26

Fixes streaming detokenization — generated text no longer runs words together
(`"the process"` → `"theprocess"`). Ships against engine **0.32.4**.

### Fixed

- **Per-token streaming decode preserves word-boundary spaces.** `SentencePieceSpecialTokens.decode(Int)`
  and `UpstreamTokenizerAdapter.decode(Int)` now route through the engine's new `Tokenizer.decodeToken(id)`
  (engine 0.32.4), which keeps each SentencePiece piece's leading space instead of stripping it per token
  (the sequence-level `addSpacePrefix` strip is only correct once per sequence). Fixes correct-but-spaceless
  output in streaming generation (kllama, agent loops). Adds `SentencePieceSpecialTokensStreamingTest`.

### Changed

- **Engine pin `skainet 0.32.2 → 0.32.4`** (adds `Tokenizer.decodeToken`).

## [0.32.0] — 2026-06-25

Brings the real-GGUF **Llama** eager path up to the Gemma standard (packed
`NATIVE_OPTIMIZED`) and **unblocks StableHLO/IREE export for Llama-family models**
(traceable interleaved RoPE). Ships against engine **0.32.2**.

### Added

- **Eager `NATIVE_OPTIMIZED` packed path for Llama.** `LlamaNetworkLoader.fromGguf(NATIVE_OPTIMIZED)`
  keeps `Q4_K`/`Q6_K` weights packed and runs them through `OptimizedLLMRuntime` — new `LlamaQuantLayout`
  + `LlamaPackedWeights.convertLlamaWeightsPacked`, mirroring `convertGemmaWeightsPacked`. Coherent
  output matching llama.cpp; the low-footprint path real-GGUF Llama inference on constrained ARM was
  missing. (ccbd87e)

### Changed

- **Fused decode-attention fast path.** `MultiHeadAttention`'s decode step (`seqQ == 1`) now computes
  scores → softmax → GQA-weighted-V directly from the cached K/V, bypassing the `repeatKVHeads` concat
  and the `unsqueeze → SDPA → squeeze → permute` chain — ~1.5× decode throughput, bit-identical output.
  Prefill (`seqLen > 1`) keeps the general SDPA path. (3791f88)
- **Engine pin `skainet 0.31.0 → 0.32.2`** (0.32.2 is the first engine release exposing
  `ExecutionContext.isRecording`, required by the trace-faithful KV-cache path).

### Fixed

- **Packed token-embedding gather for Llama** — `fromGguf(NATIVE_OPTIMIZED)` no longer fails with
  `gather: unsupported input rank 1`; the packed embedding is wired through the canonical loader. (ccbd87e)
- **Interleaved RoPE is now traceable.** In `INTERLEAVED` mode (Llama / Mistral / most GGUF) the rotation
  used a raw float-array path (`copyToFloatArray` / `fromFloatArray`) that, under graph tracing, baked the
  rotated Q/K as a *disconnected constant* — severing them from the projection weights and crashing
  `iree-compile` (null-deref in constant folding) on the exported graph. `RoPE` now records the rotation
  as tensor ops when running under the tracing wrapper; eager execution keeps the byte-identical raw-array
  fast path. Unblocks Llama/Mistral/GGUF StableHLO/IREE export. (019b049)

## [0.31.1] — 2026-06-17

Adds **`transformer-core`** — the framework NN primitives (attention, the KV-cache family, embedding,
norms, RoPE, SwiGLU/GeGLU FFN, residual, linear projection) extracted from `llm-core` so they build on the
**full Kotlin target matrix including `androidNative`** (32-bit + 64-bit ARM). `llm-core` re-exports it, so
existing consumers are unaffected; ARM-native downstreams (e.g. on-device whisper) can now reuse the
primitives instead of reimplementing them.

### Added

- **`transformer-core` module** (`sk.ainet.transformers:skainet-transformers-transformer-core`) — the
  lang-core-only NN primitives, reusable on every target incl. `androidNativeArm32`/`androidNativeArm64`.
  Depends only on `skainet-lang-core`. Added to the BOM. (#183)

### Changed

- **`llm-core` now `api`-depends on `transformer-core` and re-exports it** (no behaviour change). The NN
  primitive sources moved out of `llm-core` into `transformer-core`; `dsl/decoder/*` stayed (it needs the
  compile-opt-coupled `HybridTransformerBlock`). `MultiHeadAttention`'s diagnostic `dumpStats` is decoupled
  via a settable `mhaStatSink` that `HybridTransformerBlock` wires to llm-core's platform `dumpStats`.

### Notes

- **Engine pin unchanged (`skainet = 0.31.0`).** `transformer-core` needs nothing new from the engine (only
  `skainet-lang-core`, already in 0.31.0), so this patch ships against engine **0.31.0** — the one case the
  transformers-`X.Y.Z` ↔ engine-`X.Y.Z` alignment is intentionally relaxed (additive + engine-independent).

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

[0.38.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.38.0
[0.36.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.36.1
[0.36.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.36.0
[0.31.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.31.0
[0.30.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.30.0
[0.28.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.28.1
[0.23.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.23.1
[0.21.1]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.1
[0.21.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.21.0
[0.18.0]: https://github.com/SKaiNET-developers/SKaiNET-transformers/releases/tag/0.18.0
