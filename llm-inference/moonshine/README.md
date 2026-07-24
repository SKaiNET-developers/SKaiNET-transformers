# skainet-transformers-inference-moonshine

Reusable **Moonshine** speech-to-text (Whisper-family encoder–decoder) authored in the SKaiNET NN DSL.
This module is the **model only** — a portable graph producer. It carries **no runtime, board, or Torq
code**, so you can drop it into any project and compile it for whatever target you need.

- **Coordinate:** `sk.ainet.transformers:skainet-transformers-inference-moonshine:0.35.0`
- **Targets:** `jvm`, `linuxX64`, `linuxArm64`
- **What's inside** (`src/commonMain/.../moonshine/`): `moonshineEncoder()`, `MoonshineDecoderModel` /
  `MoonshineDecoderLayer` (causal self-attn + cross-attn + gated-SiLU MLP + tied LM head, with
  `forwardPrefill` / `forwardWithPast` KV-cache trace entry points), `MoonshinePreprocessor` (conv frontend),
  `MoonshineConfig`.

## Use it standalone

```kotlin
dependencies {
    implementation("sk.ainet.transformers:skainet-transformers-inference-moonshine:0.35.0")
}
```

```kotlin
import sk.ainet.models.moonshine.moonshineEncoder
import sk.ainet.lang.types.FP32          // or BF16 — see "dtype is a target choice"

val cfg = MoonshineConfig(/* tiny defaults */)
val encoder = moonshineEncoder<FP32, Float>(cfg, FP32::class)   // a DSL Module you can trace → StableHLO
```

The module builds the graph with **placeholder parameters**; bake real weights from a checkpoint before
compiling a runnable vmfb (the demo's `voicecc/export/MoonshineWeights.kt` + `convert_moonshine_weights.py`
show the by-name mapping — `enc.$layer.*` / `dec.$layer.*`).

## dtype is a **target choice**, not a model property

`moonshineEncoder(cfg, dtype)` and the decoder are parameterized on the element type, so the **same graph**
lowers to any IREE backend. Pick the dtype to match the target:

| Target | dtype | How |
|---|---|---|
| Host CPU (llvm-cpu / AVX) or GPU (CUDA/Vulkan) | **`FP32`** | portable default — no target-specific passes needed |
| Synaptics Torq NPU | **`BF16`** | weights must stay bf16 at the matmul (fp32 crashes the torq compiler's `getWeightMemoryFormat`) |

## Compile path (model → runnable)

`DSL Module → trace → StableHLO → iree-compile (target of choice) → .vmfb`. The Torq/NPU passes are
**not** here — they live in the quarantined `sk.ainet.vendors:synaptics-torq` plugin and are only applied
when you target `"torq"`. Worked examples of both paths (host CPU vs Torq NPU) are in the SL2610 demo:
`scripts/iree-compile-cpu.sh` (host llvm-cpu, AVX or aarch64 NEON) and `scripts/iree-compile-torq-docker.sh`
(Torq NPU).

**Multi-target, verified (2026-07-23):** the same encoder StableHLO that targets the Torq NPU also compiles
to a **host x64 llvm-cpu** vmfb — the portability proof for standalone reuse:
```bash
iree-compile moonshine-encoder.mlir --iree-input-type=stablehlo \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host -o moonshine-encoder-hostcpu.vmfb   # exit 0, ~115 KB vmfb
```
(A full *run* needs baked weights; compilation proves the graph lowers for a non-Torq target.)

## Roadmap
- **Streaming (Moonshine v2):** a position-free sliding-window *causal* encoder + adapter for low-latency
  chunked ASR (the current v1 encoder is bidirectional/non-causal). The causal KV-cache decoder here is
  already streaming-ready.
- **GPU targets:** the lowering is target-agnostic (the `TargetOptimizers` registry in
  `skainet-compile-opt` lets a backend register its own passes); a CUDA/Vulkan compile path is a
  documented next step, not yet shipped.
