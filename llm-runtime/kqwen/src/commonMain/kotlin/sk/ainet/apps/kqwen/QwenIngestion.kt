package sk.ainet.apps.kqwen

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaWeightMapper
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

public data class QwenLoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

/**
 * Facade for loading Qwen2/Qwen3 models from GGUF files.
 *
 * Qwen uses the same tensor layout as LLaMA, so this delegates to [DecoderGgufWeightLoader]
 * with `acceptedArchitectures = setOf("qwen2", "qwen3")`.
 */
public class QwenIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: QwenLoadConfig = QwenLoadConfig()
) {
    private companion object {
        val QWEN_ARCHITECTURES = setOf("qwen2", "qwen3")
    }

    /**
     * Load Qwen runtime weights using streaming API.
     * Parses metadata only (~1MB memory), loads tensors on-demand.
     */
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): LlamaRuntimeWeights<T> {
        val loader = DecoderGgufWeightLoader(
            randomAccessProvider = randomAccessProvider,
            quantPolicy = config.quantPolicy,
            acceptedArchitectures = QWEN_ARCHITECTURES
        )
        val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
        if (!config.allowQuantized && loaded.quantTypes.isNotEmpty()) {
            error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors.")
        }
        return LlamaWeightMapper.map(loaded)
    }
}
