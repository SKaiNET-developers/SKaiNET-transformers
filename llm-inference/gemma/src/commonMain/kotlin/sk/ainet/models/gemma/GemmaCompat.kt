@file:Suppress("DEPRECATION")

package sk.ainet.models.gemma

/**
 * Pre-#374 versioned names, kept one release for source compatibility. The unversioned names
 * are the #346 convention — and the factual ones: the loader/config/metadata stack serves
 * gemma3 (FunctionGemma) and gemma4 alike.
 */
@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaWeightLoader"))
public typealias Gemma4WeightLoader = GemmaWeightLoader

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaSafeTensorsLoader"))
public typealias Gemma4SafeTensorsWeightLoader = GemmaSafeTensorsLoader

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaConfigParser"))
public typealias Gemma4ConfigParser = GemmaConfigParser

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaModelMetadata"))
public typealias Gemma4ModelMetadata = GemmaModelMetadata

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaRopeConfig"))
public typealias Gemma4RopeConfig = GemmaRopeConfig

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaLayerWeights"))
public typealias Gemma4LayerWeights<T> = GemmaLayerWeights<T>

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaRuntimeWeights"))
public typealias Gemma4RuntimeWeights<T> = GemmaRuntimeWeights<T>

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaWeights"))
public typealias Gemma4Weights<T, V> = GemmaWeights<T, V>

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaTensorNames"))
public typealias Gemma4TensorNames = GemmaTensorNames

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaWeightMapper"))
public typealias Gemma4WeightMapper = GemmaWeightMapper
