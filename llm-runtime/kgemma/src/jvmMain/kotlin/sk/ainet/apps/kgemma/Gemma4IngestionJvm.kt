package sk.ainet.apps.kgemma

import java.lang.foreign.Arena
import kotlinx.io.Source
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.Gemma4Weights
import sk.ainet.models.gemma.convertGemmaWeightsToMemSeg
import kotlin.reflect.KClass

/**
 * JVM-only extensions to [Gemma4Ingestion] that compose raw quantized GGUF
 * loading, the MemSeg-aware weight converter, and the DSL runtime builder
 * into a single call. The converter uses `java.lang.foreign.Arena` and
 * `MemorySegment` so it can't live in commonMain.
 *
 * The CLI uses these for `--runtime=dsl`: Q4_0 and Q8_0 weights stay
 * packed end-to-end, K-series dequant to FP32 (no pre-transpose — the DSL
 * always transposes). See `GemmaMemSegConverter` for details and known
 * limitations (Q4_K native dispatch isn't wired yet).
 */

/** Load a Gemma 4 GGUF with `NATIVE_OPTIMIZED`, convert quantized tensors to
 *  MemorySegment-backed Q4/Q8 form, and build a DSL-based [InferenceRuntime]. */
public suspend fun <T : DType> Gemma4Ingestion<T>.loadDslRuntimeNativeStreaming(
    randomAccessProvider: () -> RandomAccessSource,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    arena: Arena
): InferenceRuntime<T> {
    val rawWeights = Gemma4WeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
    ).loadToMapStreaming<T, Float>(ctx, dtype)

    @Suppress("UNCHECKED_CAST")
    val convertedAny = convertGemmaWeightsToMemSeg(rawWeights, ctx, arena) as Gemma4Weights<T, Float>
    // DIAGNOSTIC env flag: GEMMA4_NO_KV_SHARING=1 forces every layer to use
    // its own KV cache (kvSharedLayers=0). Used to isolate whether the
    // length-dependent confidence collapse is caused by the shared-KV-cache
    // architecture or something elsewhere in the stack.
    val finalWeights = if (System.getenv("GEMMA4_NO_KV_SHARING") == "1") {
        println("[diag] GEMMA4_NO_KV_SHARING=1 → forcing kvSharedLayers=0")
        convertedAny.copy(metadata = convertedAny.metadata.copy(kvSharedLayers = 0))
    } else convertedAny
    return buildDslRuntime(finalWeights)
}

/** Sequential-Source variant of [loadDslRuntimeNativeStreaming] for GGUFs under 2 GB. */
public suspend fun <T : DType> Gemma4Ingestion<T>.loadDslRuntimeNative(
    sourceProvider: () -> Source,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    arena: Arena
): InferenceRuntime<T> {
    val rawWeights = Gemma4WeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
    ).loadToMap<T, Float>(ctx, dtype)

    @Suppress("UNCHECKED_CAST")
    val converted = convertGemmaWeightsToMemSeg(rawWeights, ctx, arena) as Gemma4Weights<T, Float>
    return buildDslRuntime(converted)
}
