package sk.ainet.apps.kapertus

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.apertus.ApertusModelMetadata
import sk.ainet.models.apertus.ApertusRuntimeWeights
import sk.ainet.models.apertus.ApertusSafeTensorsLoader
import sk.ainet.models.apertus.loadApertusRuntimeWeights
import sk.ainet.models.apertus.loadApertusRuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Configuration for loading Apertus weights.
 */
public data class ApertusLoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
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
            quantPolicy = config.quantPolicy
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
            quantPolicy = config.quantPolicy
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
}
