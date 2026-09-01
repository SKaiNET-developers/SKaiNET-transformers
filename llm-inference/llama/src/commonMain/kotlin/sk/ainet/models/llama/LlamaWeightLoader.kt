package sk.ainet.models.llama

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy

/** GGUF `general.architecture` values the Llama family accepts. */
public val LLAMA_ARCHITECTURES: Set<String> = setOf("llama", "mistral")

/**
 * The Llama family's weight loader (#346's `<F>WeightLoader` row, transformers#372) — the
 * family-prefixed thin wrapper over the shared [DecoderGgufWeightLoader] in `llm-core`, pinned
 * to [LLAMA_ARCHITECTURES]. Same shape as `BitNetWeightLoader`: the family states *which*
 * architectures and *what form*; the engine materializes.
 */
public object LlamaWeightLoader {

    /** Sequential [Source] load — models under 2GB, always dequantizes to dense FP32. */
    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext,
        noinline sourceProvider: () -> Source,
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ): DecoderGgufWeights<T, V> = DecoderGgufWeightLoader(
        sourceProvider = sourceProvider,
        acceptedArchitectures = LLAMA_ARCHITECTURES,
        dtypePolicy = dtypePolicy,
    ).loadToMap(ctx)

    /**
     * Streaming random-access load — any size; `weightForm` `null` keeps the decoder loader's
     * keep-packed MAPPED default.
     */
    @OptIn(ExperimentalMemoryApi::class)
    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        noinline randomAccessProvider: () -> RandomAccessSource,
        weightForm: WeightForm? = null,
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ): DecoderGgufWeights<T, V> = DecoderGgufWeightLoader(
        randomAccessProvider = randomAccessProvider,
        acceptedArchitectures = LLAMA_ARCHITECTURES,
        dtypePolicy = dtypePolicy,
        weightForm = weightForm,
    ).loadToMapStreaming(ctx)
}
