package sk.ainet.models.llama

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader
import sk.ainet.lang.types.DType

/**
 * The llama-typed half of the SafeTensors load (#372): the family-neutral
 * [DecoderSafeTensorsLoader.loadToMap] lives in `llm-core`; mapping the flat tensors into
 * structured [LlamaRuntimeWeights] is llama's own concern and stays here as an extension.
 */
public fun <T : DType> DecoderSafeTensorsLoader<T>.load(
    randomAccessProvider: () -> RandomAccessSource,
): LlamaRuntimeWeights<T> = LlamaWeightMapper.map(loadToMap(randomAccessProvider))
