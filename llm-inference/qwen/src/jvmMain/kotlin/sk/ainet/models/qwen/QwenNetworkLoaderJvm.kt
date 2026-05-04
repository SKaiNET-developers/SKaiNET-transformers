package sk.ainet.models.qwen

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.loadDecoderGgufWeightsNative
import java.lang.foreign.Arena

/**
 * Streaming GGUF load for the DSL Qwen3 path with native-quantized weights.
 *
 * Produces a [Module] whose Q4_0 / Q8_0 weights remain `MemorySegment`-
 * packed end-to-end, so the upstream `DefaultCpuOpsJvm.matmul` quant
 * dispatch fires at forward time without dequantising to FP32. K-quants
 * (Q4_K / Q5_K / Q6_K) are dequantised to FP32 — packed K-quant kernels
 * are not on the DSL hot path yet.
 *
 * Caller manages the [arena] lifecycle. Tying it to the inference
 * `ExecutionContext` lifecycle is the typical pattern.
 *
 * Companion to [QwenNetworkLoader.fromGguf], which always dequantises to
 * FP32 and is the multiplatform-friendly default. This entry point is
 * JVM-only — `MemorySegment` and `Arena` are JDK-21+ APIs.
 */
public suspend fun QwenNetworkLoader.Companion.fromGgufNative(
    randomAccessProvider: () -> RandomAccessSource,
    ctx: ExecutionContext,
    arena: Arena,
): Module<FP32, Float> {
    val weights = loadDecoderGgufWeightsNative(
        randomAccessProvider = randomAccessProvider,
        acceptedArchitectures = QWEN_ARCHITECTURES,
        ctx = ctx,
        arena = arena,
    )
    return QwenNetworkLoader.fromWeights(weights)
}
