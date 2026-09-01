package sk.ainet.models.qwen

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy

/**
 * GGUF `general.architecture` values the Qwen family accepts. `qwen2` covers
 * Qwen2/Qwen2.5 (attention projection biases, no QK-norm), `qwen3`/`qwen35`
 * the QK-norm generations without biases.
 */
public val QWEN_ARCHITECTURES: Set<String> = setOf("qwen2", "qwen3", "qwen35")

/**
 * The Qwen family's weight loader (#346's `<F>WeightLoader` row) — the family-prefixed thin
 * wrapper over the shared [DecoderGgufWeightLoader] in `llm-core`, pinned to
 * [QWEN_ARCHITECTURES]. Same shape as `LlamaWeightLoader` / `BitNetWeightLoader`: the family
 * states *which* architectures and *what form*; the engine materializes. The decoder loader
 * carries Qwen2/2.5's `blk.N.attn_{q,k,v}.bias` tensors as optionals (#352).
 */
public object QwenWeightLoader {

    /** Sequential [Source] load — models under 2GB, always dequantizes to dense FP32. */
    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext,
        noinline sourceProvider: () -> Source,
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ): DecoderGgufWeights<T, V> = DecoderGgufWeightLoader(
        sourceProvider = sourceProvider,
        acceptedArchitectures = QWEN_ARCHITECTURES,
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
        acceptedArchitectures = QWEN_ARCHITECTURES,
        dtypePolicy = dtypePolicy,
        weightForm = weightForm,
    ).loadToMapStreaming(ctx)
}
