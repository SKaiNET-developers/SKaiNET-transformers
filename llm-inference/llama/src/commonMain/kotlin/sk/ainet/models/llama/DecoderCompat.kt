@file:Suppress("DEPRECATION")

package sk.ainet.models.llama

/**
 * Pre-#372 names of the shared decoder machinery, kept one release for source compatibility.
 * The canonical home is `sk.ainet.lang.nn.dsl.decoder` in `llm-core` — the loader half joining
 * the architecture half (`DecoderModelMetadata`, `decoderTransformerNetwork`) that already
 * lived there.
 */
@Deprecated(
    "Moved to llm-core as GgufDecoderMetadata (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata"),
)
public typealias LlamaModelMetadata = sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

@Deprecated(
    "Moved to llm-core (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader"),
)
public typealias DecoderGgufWeightLoader = sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader

@Deprecated(
    "Moved to llm-core (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights"),
)
public typealias DecoderGgufWeights<T, V> = sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights<T, V>

@Deprecated(
    "Moved to llm-core (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader"),
)
public typealias DecoderSafeTensorsLoader<T> = sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader<T>

@Deprecated(
    "Moved to llm-core (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.HfTensorNameMapper"),
)
public typealias HfTensorNameMapper = sk.ainet.lang.nn.dsl.decoder.HfTensorNameMapper

@Deprecated(
    "Moved to llm-core as DecoderTensorNames (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames"),
)
public typealias LlamaTensorNames = sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames

@Deprecated(
    "Moved to llm-core as DecoderGgufTensorNames (transformers#372).",
    ReplaceWith("sk.ainet.lang.nn.dsl.decoder.DecoderGgufTensorNames"),
)
public typealias LlamaGgufTensorNames = sk.ainet.lang.nn.dsl.decoder.DecoderGgufTensorNames
