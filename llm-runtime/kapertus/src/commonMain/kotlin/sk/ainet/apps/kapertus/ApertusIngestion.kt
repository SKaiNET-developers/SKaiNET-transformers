package sk.ainet.apps.kapertus

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.apertus.ApertusModelMetadata
import sk.ainet.models.apertus.ApertusQuantizedRuntimeWeights
import sk.ainet.models.apertus.ApertusRuntimeWeights
import sk.ainet.models.apertus.ApertusSafeTensorsLoader
import sk.ainet.models.apertus.loadApertusQuantizedWeights
import sk.ainet.models.apertus.loadApertusQuantizedWeightsStreaming
import sk.ainet.models.apertus.loadApertusRuntimeWeights
import sk.ainet.models.apertus.loadApertusRuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Configuration for loading Apertus weights.
 */
public data class ApertusLoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val preTransposed: Boolean = false
)

/**
 * Thin facade around the GGUF and SafeTensors loaders for the Apertus model.
 */
public class ApertusIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: ApertusLoadConfig = ApertusLoadConfig()
) {

    /**
     * Load Apertus runtime weights from GGUF source (sequential).
     * Suitable for models under 2GB.
     */
    public suspend fun load(sourceProvider: () -> Source): ApertusRuntimeWeights<T> {
        return loadApertusRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            preTransposed = config.preTransposed
        )
    }

    /**
     * Load Apertus runtime weights from GGUF source (streaming).
     * Suitable for models of any size.
     */
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): ApertusRuntimeWeights<T> {
        return loadApertusRuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            preTransposed = config.preTransposed
        )
    }

    /**
     * Load Apertus runtime weights from HuggingFace SafeTensors.
     */
    public suspend fun loadSafeTensors(indexPath: String): ApertusRuntimeWeights<T> {
        val loader = ApertusSafeTensorsLoader(indexPath)
        val weights = loader.loadToMap<T>(ctx, dtype)
        return sk.ainet.models.apertus.ApertusWeightMapper.map(weights)
    }

    /**
     * Load Apertus weights in quantized form for lazy dequantization (sequential).
     * Large weight matrices stay quantized; small tensors (norms) are FP32.
     */
    public suspend fun loadQuantized(sourceProvider: () -> Source): ApertusQuantizedRuntimeWeights {
        return loadApertusQuantizedWeights(ctx = ctx, sourceProvider = sourceProvider)
    }

    /**
     * Load Apertus weights in quantized form for lazy dequantization (streaming).
     * Uses ~4-8x less memory than [loadStreaming] with DEQUANTIZE_TO_FP32.
     */
    public suspend fun loadQuantizedStreaming(
        randomAccessProvider: () -> RandomAccessSource
    ): ApertusQuantizedRuntimeWeights {
        return loadApertusQuantizedWeightsStreaming(ctx = ctx, randomAccessProvider = randomAccessProvider)
    }
}
