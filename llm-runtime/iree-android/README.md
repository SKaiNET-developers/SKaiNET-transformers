# `llm-runtime:iree-android`

Generic Android JNI runtime for the DSL → StableHLO → IREE **compiled path**
(transformers#305) — the compiled-path counterpart to `skainet-backend-jni-cpu`
(engine repo) for the **eager** path: both serve any model without knowing
which one is calling them.

This module drives **any** DSL-compiled vmfb that follows the redecode graph
contract established by `:llm-inference:smollm2`'s `SmolLm2ExportHarness`
(the first producer of a compatible triple — see its
[`docs/smollm2-vmfb.md`](../../llm-inference/smollm2/docs/smollm2-vmfb.md)):

- one fixed-`seq` exported function, `tensor<1xSEQxi32> -> tensor<SEQxi32>`
- the DSL's in-graph `argMax` already applied (small per-step output, no
  host-side argmax over a `[SEQ, vocab]` logits tensor)
- weights **external** — bound at session-create time from a `.irpa`
  parameter archive under scope `"model"`, not baked into the vmfb

Unlike `:llm-runtime:gemma-iree` (Kotlin/Native, drives `iree-run-module` as
a subprocess on a Linux board — structurally unusable on Android, which runs
on ART/JVM, not K/N), this is the first real JNI-against-the-IREE-C-API code
in this repo.

## Quick start

```kotlin
val decoder = IreeRedecodeDecoder.fromAssets(
    context,
    vmfbAsset = "smollm2/smollm2-gen-arm64.vmfb",   // pick per Build.SUPPORTED_ABIS
    irpaAsset = "smollm2/smollm2.irpa",
    functionName = "module.smollm2",
    seq = 24,
    cacheDirName = "skainet_smollm2",
)
val generated = decoder.generate(promptTokenIds, eosTokenId = tokenizer.eosTokenId)
decoder.close()
```

No tokenizer dependency — pair this with whatever BPE tokenizer your app
already has.

### API surface (package `sk.ainet.transformers.iree.android`)

| Class | Role |
|---|---|
| `IreeRedecodeDecoder` | Facade: `fromAssets(context, vmfbAsset, irpaAsset, functionName, seq, cacheDirName, device)`, `generate(promptIds, eosTokenId, maxNewTokens)` |
| `IreeRedecodeSession` | Raw JNI wrapper: padded `[seq]` token buffer in → `[seq]` predicted-next-token ids out. Package/class name is the JNI symbol contract with the `.so` — do not move/rename. |

## Devices: CPU and GPU

`device` is just an IREE HAL driver string:

- `IreeRedecodeSession.DEFAULT_DEVICE` (`"local-task"`, CPU) — the primary,
  numerically-verified path (see the smollm2 docs linked above).
- `IreeRedecodeSession.VULKAN_DEVICE` (`"vulkan"`, GPU — Mali, Adreno, etc.,
  portable SPIR-V) — the `.so` in this module is built with `--vulkan` and
  the Vulkan HAL driver compiled in, so it's ready whenever a caller has a
  `vulkan-spirv`-compiled vmfb. SmolLM2's own export currently does **not**
  compile for Vulkan (a real IREE 3.11.0 SPIR-V codegen gap on its
  token-embedding gather — see the smollm2 docs) — that's a limitation of
  that specific export, not of this runtime.

## How the `.so` is built

`native/iree_redecode_jni.c` — see its header doc for the full IREE C API
call sequence (session create → parse `.irpa` → `io_parameters` VM module →
append *before* the compiled bytecode module, since its `util.global`
initializers resolve against it at link time → invoke by name → transfer
result → rank/length-guard against the redecode contract).

Cross-built via [`SKaiNET-iree-toolchain`](https://github.com/SKaiNET-developers/SKaiNET-iree-toolchain)'s
`skainet/iree-android:3.11.0` image:

```bash
native/build-iree-redecode.sh arm64-v8a --vulkan
native/build-iree-redecode.sh armeabi-v7a --vulkan
# copy native/out/libskainet_iree_redecode.so to src/main/jniLibs/<abi>/
```

Because weights are **external** (unlike `skainet-embedder-android`'s
baked-in-weights precedent), the `.so` links three IREE targets beyond the
default `iree_runtime_unified`: `iree_modules_io_parameters_parameters`,
`iree_io_parameter_index`, `iree_io_parameter_index_provider`,
`iree_io_formats_irpa_irpa` — any external-weights vmfb needs these, so
they're unconditional in the build script, not per-model.

The `.so` is a **checked-in artifact**, not built by CI — regenerate it by
re-running the script when `iree_redecode_jni.c` changes. Migrating to
AGP's own `externalNativeBuild`/CMake (matching `skainet-backend-jni-cpu`'s
in-repo build, against IREE's `export-android-sdk` output) is real future
work: it means vendoring/consuming the whole IREE runtime build, not a
handful of hand-written kernel `.c` files, and deserves its own
investigation.

## Status

- [x] Generic, model-agnostic native shim + Kotlin API
- [x] Both ABIs (`arm64-v8a`, `armeabi-v7a`) cross-built, JNI symbols verified
- [x] Both CPU (`local-task`) and GPU (`vulkan`) HAL drivers compiled into the `.so`
- [x] Standalone module build verified (`./gradlew :llm-runtime:iree-android:assembleRelease`)
- [ ] Verified running on a physical Android device or emulator (none
      available in the environment this was built in)
- [ ] Wired into `kllama`'s `registerPlatformBackends` facade for
      zero-config app consumption (matches how NEON kernels are
      auto-discovered on the eager path) — natural follow-up, not done here
