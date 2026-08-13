# skainet-transformers-inference-gemma

Reusable **Gemma** model (incl. the FunctionGemma tool-calling fine-tune) authored in the SKaiNET NN DSL —
a portable graph producer with **no runtime/board/Torq code**. Pair it with the runtime module below to
decode on-device.

- **Coordinate:** `sk.ainet.transformers:skainet-transformers-inference-gemma:0.40.2`
- **Targets:** `android`, `iosArm64`, `iosSimulatorArm64`, `macosArm64`, `linuxX64`, `linuxArm64` (broadly
  portable — mobile through server).
- **Entry point:** `gemmaNetwork()` / `GemmaNetworkLoader` (loads a GGUF, builds the DSL graph, incl. the
  `argMax` tail for FunctionGemma's functional-token decode).

## The reuse pair

| Module | Coordinate | Role |
|---|---|---|
| model | `…:skainet-transformers-inference-gemma` | the DSL graph (`gemmaNetwork()`) — trace → StableHLO → compile |
| runtime | `…:skainet-transformers-runtime-gemma-iree` | on-device decode: `GemmaDecoder`, `GemmaKvDecoder`, `IreeRuntime`, `CompactCodec` (drives the compiled vmfb) |

FunctionGemma has a one-liner facade in `…:skainet-transformers-runtime-kgemma`
(`FunctionGemma.fromGguf(...).call("turn the light on")` eager, or `.exportCompiled(dir)` for the edge path).

## Use it standalone

```kotlin
dependencies {
    implementation("sk.ainet.transformers:skainet-transformers-inference-gemma:0.40.2")
    implementation("sk.ainet.transformers:skainet-transformers-runtime-gemma-iree:0.40.2") // on-device decode
}
```

## dtype is a **target choice**

Like Moonshine, the same DSL graph lowers to any IREE target — `FP32` for portable host/GPU builds,
`BF16` for the Torq NPU (board A/B proved bf16 is a bit-exact drop-in for f16 on FunctionGemma). Weights are
emitted as external params (`.irpa`); the Torq passes live in `sk.ainet.vendors:synaptics-torq`, applied only
when targeting `"torq"`.

## Compile path
`gemmaNetwork() → trace → StableHLO → iree-compile (llvm-cpu host / Torq board) → .vmfb` + external `.irpa`.
The SL2610 demo's `scripts/compile-gemma.sh` is a Python-free, one-command worked example (GGUF → mlir →
irpa → vmfb), including the optional KV-cache 2-graph decode (`GEMMA_KV=1`) and per-row int8 (`GEMMA_QUANT=int8`).
