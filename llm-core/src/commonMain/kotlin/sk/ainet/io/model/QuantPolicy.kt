package sk.ainet.io.model

/**
 * Controls how the *transformers-side* weight loaders (LLaMA, Gemma, Apertus) handle quantized
 * tensors.
 *
 * History: this enum lived in the SKaiNET engine (`skainet-io-core`) through 0.39 and was removed
 * there in 0.40 (`SKaiNET@9ada6f88` — "remove the legacy loader axes"), replaced by the engine's
 * `WeightForm`/`EncodingRequest` surface. It was never actually consumed by any engine API this
 * repo calls — our loaders parse GGUF themselves and branch on it internally — so the type moves
 * here, same package, and becomes what it always effectively was: transformers-owned plumbing.
 *
 * When migrating a call path to the engine's loader, the mapping is:
 * `NATIVE_OPTIMIZED` → `EncodingRequest.KeepAsStored`, `DEQUANTIZE_TO_FP32` →
 * `EncodingRequest.DequantizeTo(FP32)`; `RAW_BYTES` has no engine counterpart.
 */
public enum class QuantPolicy {
    /** Keep quantized payloads as raw bytes (Int8 tensor) with quantized shape. */
    RAW_BYTES,

    /** Dequantize to FP32 on load. */
    DEQUANTIZE_TO_FP32,

    /**
     * Mixed mode: dequantize F32/F16/BF16 tensors to FP32, but keep quantized
     * weight tensors (Q4_0, Q8_0, etc.) as raw bytes for native kernel consumption.
     *
     * This allows loading with dtype=FP32 while preserving quantized weights
     * for platform-specific optimized kernels (e.g. MemorySegment-backed SIMD).
     */
    NATIVE_OPTIMIZED,
}
