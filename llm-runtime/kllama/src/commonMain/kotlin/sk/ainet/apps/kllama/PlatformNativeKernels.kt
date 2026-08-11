package sk.ainet.apps.kllama

/**
 * Installs the platform's native-cinterop packed-quant kernel provider
 * (priority 100 — Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/Q5_0/Q5_1) into the process-wide
 * `KernelRegistry`, on targets where one exists and no `ServiceLoader` can
 * register it automatically.
 *
 * On JVM and Android the engine's own ops-factory `ServiceLoader` lookup
 * finds the FFM / JNI providers without any call here — this is a no-op
 * there. Kotlin/Native has no `ServiceLoader`, so `DirectCpuExecutionContext`
 * registers only the scalar (and, on Apple, Accelerate dense-FP32) provider
 * by default; without this call every K/N target runs packed-quant matmul
 * scalar. JS/Wasm have no native backend. Idempotent — safe to call from
 * every context-creation site (#300).
 */
internal expect fun installPlatformNativeKernels()
