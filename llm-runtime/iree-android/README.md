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

## KV session (`IreeKvSession` / `IreeKvDecoder`, `libskainet_iree_kv.so`)

The stateful counterpart of the redecode session, for the FunctionGemma contract addendum
(`gemma_prefill_at`, `gemma_prefill_with_past`, `gemma_with_past`; host-gather variants with an
`emb` input right after the tokens — the embedding rows are read from the with-past archive by the
native side, so callers pass token ids only):

```kotlin
val spec = IreeKvSpec.functionGemma270m(chunk = 32)           // or IreeKvSpec.fromManifest(json)
val session = IreeKvSession(spec, IreeKvSession.VULKAN_DEVICE,
    "$dir/gemma-with-past-hostgather-valhall4.vmfb", "$dir/gemma-with-past.irpa",
    "$dir/gemma-prefill-with-past-hostgather-valhall4.vmfb", "$dir/gemma-prefill-with-past.irpa",
    "$dir/gemma-prefill-at-hostgather-valhall4.vmfb", "$dir/gemma-prefill-at.irpa")
val decoder = IreeKvDecoder(session, prefillSeq = 1024)
val catalog = decoder.prefillPrefix(catalogPromptIds)          // once per process (25 s on an arm32 Mali device, 843 tokens)
session.releasePrefill()                                       // drops the prefill archive mapping
val ids = decoder.generate(catalog, utteranceIds, eosTokenId = 106, maxNewTokens = 32)   // per turn
```

What the native side does per call: sliding layers (`l % globalLayerPeriod != period-1`) only ever
see their last `slidingWindow` cache positions through zero-copy tail views; RoPE cos/sin tables
(split-half, sign folded into the first half) and the chunk graph's per-head additive masks
(causal band, padding, window) are built for the absolute positions; K/V outputs replace the
retained views; `snapshot()`/`restore()` retain/release views without copying. Every failure is
thrown as `IllegalStateException` with the formatted IREE status (also logged under `skainet_iree_kv`).

Measured (arm32 Android device, Mali via Vulkan, bf16 archives, chunk 32, 843-token catalog prefix,
16 decode tokens): open 12.4 s, prefix 25.2 s once, then **p50 5.9 s per utterance** (one chunk call
≈ 2.0 s + 16 × 0.245 s), restore 0 ms, RSS ≈ 1.5 GB in the 32-bit process. Rebuild the library
with `native/build-iree-kv.sh <abi> --vulkan` (same image and links as the redecode `.so`).

## Moonshine streaming (`IreeMoonshineStream`, `libskainet_moonshine_stream.so`)

Streaming Moonshine v2 speech-to-text over the five graphs of `MoonshineV2ExportCli`
(`:llm-inference:moonshine:exportMoonshineV2`) — frontend, encoder, adapter, masked prefill, dynamic
with-past step — plus the shared decoder parameter archive (`iree-compile --iree-opt-export-parameters`
on the decoder graphs) and the two host-side tables the exporter writes. The native side windows the audio
(64 frames, hop 0.88 s), runs the encode stages on the device, decodes incrementally and hands out
cumulative partials; `finish()` runs one exact full re-decode. Geometry (tiny: 320 / 6 / 8) is compiled in;
attention bands and vocabulary size come from the files.

```kotlin
val stream = IreeMoonshineStream(IreeMoonshineStream.VULKAN_DEVICE, IreeMoonshineStream.filesIn(modelDir))
stream.feedPcm(pcm16k)?.let { partial -> show(partial) }       // as audio arrives; null = unchanged
val text = stream.finish()                                      // end of utterance; the stream resets
stream.close()
```

`native/build-moonshine-stream.sh <abi> --vulkan` builds it like the other two (`GEOMETRY="-DL=10 -DHD=64
-DDIM=512 -DENC_DIM=620"` for the split-width `small` checkpoints).


## Whisper (`IreeWhisperSession`, `libskainet_whisper_session.so`)

Speech-to-text for one utterance over the three graphs of `WhisperExportCli`
(`:llm-inference:whisper:exportWhisper`, whisper contract version 2): encoder over a log-mel window, masked
prefill of the decode prompt, then greedy KV steps until end-of-text. Prefill and step share one parameter
archive. Nothing about the checkpoint is compiled in — geometry, function names, parameter scope and the
prompt come from the export's `manifest.json` — so one library serves every Whisper size and the
English-only variants.

```kotlin
val contract = IreeWhisperSession.Contract.fromManifest(File(dir, "manifest.json").readText())
val session = IreeWhisperSession(IreeWhisperSession.VULKAN_DEVICE, IreeWhisperSession.filesIn(dir), contract,
    decoderDevice = IreeWhisperSession.CPU_DEVICE)        // encoder on the GPU, prefill and step on the CPU
val mel = MelSpectrogram(MelFilterbanks.slaney(80, 400, 16_000), MelConfig.whisper(maxFrames = contract.melFrames))
    .logMel(pcm16k).data                                         // SKaiNET-audio's Whisper frontend
val ids = session.decode(mel)                                   // generated ids, end-of-text excluded
val text = tokenizer.decode(ids)                                // the checkpoint's tokenizer.json
```

The step graph's self K/V are `[1, maxPositions, dModel]`, rewritten by the graph at a one-hot write vector;
the attention mask is additive. Both are built natively per step, the K/V stay on the device and only the
logits row comes back.

**Repetition guard.** Greedy decoding over a short window can transcribe the utterance and then start it
again instead of ending: on the German 4 s reference clip end-of-text is the runner-up after the full stop
(18.05 vs 19.07), and the PyTorch reference loops the same way. `decode(..., repeatNgram = 3)` (the
default) stops as soon as the last three tokens already occurred and drops them; `0` decodes until
end-of-text, `maxNewTokens` or a full cache.

**Two devices.** The encoder can run on another device than prefill and step:
`IreeWhisperSession("vulkan", files, contract, decoderDevice = "local-task")`. The encoder is one large batch
over the whole window, where a GPU pays; the step is a chain of small matrix-vector products whose cost is close
on both. The encoder features cross once through host memory; each vmfb is compiled for the device it runs on.

Measured on an arm32 Android device with a Mali GPU, whisper-tiny German, 4 s window, the reference clip
(three runs each, guard on):

| Encoder / decoder | Encoder | Prefill | Steps (10) | Total |
|---|---|---|---|---|
| CPU / CPU (`llvm-cpu` arm32) | ≈ 1.98 s | ≈ 0.45 s | ≈ 2.0 s | **4.4 s** (11.3 s without the guard) |
| Vulkan / CPU | ≈ 0.47 s | ≈ 0.45 s | ≈ 2.1 s | **3.05 s** |

The ids are identical in both setups, across runs, and to the eager JVM model and PyTorch over all 44 tokens
those references decode. The **step graph does not compile for Vulkan** with IREE 3.11 yet (SPIR-V codegen of
the M = 1 matrix-vector products); encoder and prefill do — hence the CPU decoder above.

`native/build-whisper-session.sh <abi> --vulkan` builds the library like the others.
`native/whisper_session_check.c` drives it on a device without an app (a minimal `JNIEnv` over `dlopen`) —
how the numbers above were taken.

## Single-shot modules (`IreeModuleRunner`, `libskainet_iree_module.so`)

For batch graphs that are a plain tensor function — one float32 tensor in, one float32 tensor out, static
shapes: a detector with its decoded head, a classifier, an embedding graph. Nothing model-specific: the vmfb,
the optional `.irpa` (scope `model`; `null` when the vmfb carries its weights) and the function name are the
caller's.

```kotlin
IreeModuleRunner("$dir/yolov8n.vmfb", "$dir/params.irpa", "yolov8_detect", IreeModuleRunner.VULKAN_DEVICE).use { runner ->
    val head: FloatTensor = runner.run(FloatTensor(pixels, intArrayOf(1, 3, 640, 640)))   // head.shape = [1, 84, 8400]
}
```

Failures throw, with the IREE status in logcat under `skainet_iree`. One `run` at a time per instance.
`native/build-iree-module.sh <abi> --vulkan` builds it like the redecode `.so` (same image and links).
