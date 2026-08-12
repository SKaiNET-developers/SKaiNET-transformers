# SmolLM2 compiled vmfb — reference

Generated from real `SKaiNET-iree-toolchain` (IREE `3.11.0`) tool output, not
hand-written — regenerate by re-running the commands below against a fresh
export from `SmolLm2ExportHarness.export()`. This is the artifact
`llm-runtime/iree-android`'s `IreeRedecodeDecoder` drives on Android; see
that module's README for the runtime side.

## Redecode graph contract

One fixed-`seq` function, `tensor<1x24xi32> -> tensor<24xi32>` — token ids in
(batch=1, seq=24, causally padded), predicted-next-token ids out **per
position** (the DSL's in-graph `argMax` already ran — there is no
`[24, 49152]` logits tensor to transfer or argmax host-side). The caller
reads the position it needs (`prompt.size - 1 + step`) and grows the buffer;
see `IreeRedecodeDecoder.generate()`.

## Function signature (tool-derived)

Captured by pointing `iree-run-module` at the compiled module with its
parameters bound but no `--function=` — it reports the mismatch against the
real exported signature:

```
$ iree-run compiler run-module --device=local-task \
    --module=smollm2-gen-host.vmfb --parameters=model=smollm2.irpa

EXEC @smollm2
INVALID_ARGUMENT; input list and function mismatch; expected 1 arguments but passed 0;
invoking function 'smollm2';
`sync func @smollm2(%input0: tensor<1x24xi32>) -> (%output0: tensor<24xi32>)`
```

## External parameter table (tool-derived)

Full table via `iree-dump-parameters` (393 entries, one per externalized
weight tensor — `t0`, `t10`, `t12`, … `t2712`), captured with:

```
$ iree-run compiler shell -c "iree-dump-parameters --parameters=model=smollm2.irpa"
```

First and last few rows (see `iree-dump-parameters` output in full for all 393):

```
//===--------------------------------------------------------------------------------------------------------------===//
// Parameter scope `model` (393 entries, 326021760 total bytes)
//===------------+------------------+------------------+-----------------------------------------------------------===//
//         Start |              End |           Length | Key
//---------------+------------------+------------------+--------------------------------------------------------------//
           33344 |         56656448 |         56623104 | `t0`
        56656448 |         56657600 |             1152 | `t10`
        56657600 |         57321152 |           663552 | `t12`
        57321152 |         57542336 |           221184 | `t15`
        57542336 |         57763520 |           221184 | `t18`
        ⋮
       267661376 |        269430848 |          1769472 | `t2696`
       269430848 |        269432000 |             1152 | `t2710`
       269432000 |        326055104 |         56623104 | `t2712`
```

`t0` (56,623,104 bytes at bf16 = 49152 × 576 × 2) is the token-embedding
table — `49152` vocab, `576` embedding dim, matching SmolLM2-135M's real
architecture. `t2712`, same size, is the (tied) output projection.

## Artifact sizes

| Artifact | Target | Size | Notes |
|---|---|---|---|
| `smollm2-gen-host.vmfb` | `llvm-cpu`, host x86_64 | 205,064 B | numeric verification only (see below) — not shipped |
| `smollm2-gen-arm64.vmfb` | `llvm-cpu`, `aarch64-linux-android29` cortex-a76+dotprod | 221,574 B | bundled for `arm64-v8a` |
| `smollm2-gen-arm32.vmfb` | `llvm-cpu`, `armv7a-linux-androideabi29` cortex-a55+neon | 206,790 B | bundled for `armeabi-v7a` |
| `smollm2.irpa` | n/a (portable weight data, bf16) | 326,057,984 B (~311 MiB) | shared across all vmfb targets/ABIs |

The vmfb is machine code — architecture-specific, one per ABI. The `.irpa` is
portable tensor data — one file serves every ABI.

## Vulkan (GPU) — currently unsupported for this export

`iree-compile --iree-hal-target-backends=vulkan-spirv` (targeting `valhall4`,
a representative Mali GPU) **fails** on this graph:

```
error: failed to legalize operation 'vector.step'
  %v393 = "stablehlo.gather"(%v0, %arg0) ... : (tensor<49152x576xf32>, tensor<1x24xi32>) -> tensor<1x24x576xf32>
```

This is a real IREE `3.11.0` SPIR-V codegen gap for the token-embedding
`stablehlo.gather` pattern this export emits — not something fixable from
the JNI/build side. `llm-runtime/iree-android`'s runtime and `.so` already
support `device = "vulkan"` generically (built with `--vulkan`, symbols
verified) for whenever a compatible vmfb exists; it's specifically this
SmolLM2 export's gather lowering that the current toolchain can't compile
for Vulkan yet. `local-task` (CPU) is the only currently-working device for
this graph.

## Numeric verification (host only, not on-device)

`smollm2-gen-host.vmfb` (the `--target host` build, real x86_64 machine
code, distinct from the `arm64`/`arm32` builds above) was driven by hand
through the `GemmaDecoder`-style re-decode loop via `iree-run-module` for
the prompt "The capital of France is" (tokens `1,504,3575,282,4649,314`):
8 greedy steps decode to **"the city of Paris, a city of"** — correct and
coherent. The `arm64`/`arm32`-target vmfbs are real ARM machine code and
cannot execute on this x86_64 host (no emulation attempted); their
correctness rests on being compiled from the identical `smollm2-gen.mlir`
that produced the verified host build, not on independent execution.
On-device (physical or emulated) verification is still outstanding.
