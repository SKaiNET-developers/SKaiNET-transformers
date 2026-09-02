# skainet-decode-android

The Android activity leg of `skainet-decode` (SKaiNET#1244): load a GGUF, decode, and report the
same `GenerationMetrics` block as the JVM CLI — on the platform the 2 GB memory arc actually
targets. The decode flow lives in `:llm-apps:skainet-decode-core` (`DecodeSession`), so the JVM
and Android legs cannot diverge; this module adds only the rows a device can answer:

- **major-fault rate for mapped weights** — the loader maps the GGUF (`MappedRandomAccessSource`),
  so cold weight pages fault in from storage; `MemoryProbe` is sampled inside every decode span
  and `GenerationMetrics` derives faults/s from the counters.
- **RSS on a constrained device** — sampled per decode step, with a first/last footer.

## Running it

There is no model picker and no download path — the audience is measurement, and the app's own
external-files dir needs no permissions:

```sh
./gradlew :llm-apps:skainet-decode-android:installDebug
adb push smollm2-135m-q8_0.gguf \
  /sdcard/Android/data/sk.ainet.apps.decode/files/model.gguf
```

Launch **skainet-decode**, adjust prompt/steps, press **Run**. The pre-flight renders the
device-fit verdict (`AndroidGguf.fits`, header-only — the refusal happens before a byte of
payload is read) and the run ends with `GenerationMetrics.render()` plus the device footer.

The full report mirrors to logcat (`adb logcat -s SkDecode`) and lands beside the model:

```sh
adb pull /sdcard/Android/data/sk.ainet.apps.decode/files/decode-report.md
```

## The numbers this lane owns (SKEEP-002)

The engine repo's off-heap storage docs cite two device measurements that belong to this sample:

- **≤ 40 MB managed heap for SmolLM2-135M Q8_0** — measured as shipped (`largeHeap="true"`;
  the heap *cap* is larger, the claim is about heap *use* with mapped, keep-packed weights).
- **a ~600 MB Q4_K model loads on a 256 MB heap** — measured with `largeHeap` **off**: flip the
  manifest attribute to `false` for this run. Llama-family 135M models need `largeHeap` on only
  because their dense FP32 `token_embd` (~113 MiB) lands on the managed heap (transformers#272).

## Caveats

- **Kernel fallback skews tok/s, not correctness.** Kernels arrive via the self-healing
  ServiceLoader SPI in the `skainet-backend-jni-cpu` AAR (0.52.0). If a quant/view combination
  has no pack, dispatch falls back to the decoding reference kernel — the run is correct but
  slow, and the *kernel/adapter share* rows in the report are how you notice. Packaging merges
  `META-INF/services/**`; if a minified build ever strips them, everything silently slows.
- x86_64 emulator runs keep the lane executable but prove nothing about performance.
- `RecordingTraceSink` is single-threaded by design: the whole session runs on one worker
  thread; the UI only receives strings.
