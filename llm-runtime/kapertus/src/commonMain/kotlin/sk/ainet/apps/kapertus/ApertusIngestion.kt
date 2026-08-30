package sk.ainet.apps.kapertus

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.apertus.ApertusRuntimeWeights
import sk.ainet.models.apertus.ApertusSafeTensorsLoader
import sk.ainet.models.apertus.ApertusWeightMapper
import sk.ainet.models.apertus.loadApertusRuntimeWeights
import sk.ainet.models.apertus.loadApertusRuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Thin facade around the GGUF and SafeTensors loaders for the Apertus model.
 */
public class ApertusIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>
) {

    /**
     * Load Apertus runtime weights from GGUF source (sequential).
     * Suitable for models under 2GB.
     */
    public suspend fun load(sourceProvider: () -> Source): ApertusRuntimeWeights<T> {
        return loadApertusRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype
        )
    }

    /**
     * Load Apertus runtime weights from GGUF source (streaming).
     * Suitable for models of any size; quantized projection matrices keep
     * their stored block encoding as packed tensor data.
     */
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): ApertusRuntimeWeights<T> {
        return loadApertusRuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype
        )
    }

    /**
     * Load Apertus runtime weights from HuggingFace SafeTensors.
     */
    public suspend fun loadSafeTensors(indexPath: String): ApertusRuntimeWeights<T> {
        val loader = ApertusSafeTensorsLoader(indexPath)
        val weights = loader.loadToMap<T>(ctx, dtype)
        return ApertusWeightMapper.map(weights)
    }
}
