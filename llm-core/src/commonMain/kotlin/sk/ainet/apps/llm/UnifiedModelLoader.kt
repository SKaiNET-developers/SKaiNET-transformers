package sk.ainet.apps.llm

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.types.DType

/**
 * Metadata extracted from a GGUF file for model detection and loading.
 */
public data class GGUFModelInfo(
    val architecture: String,
    val family: ModelFamily,
    val contextLength: Int,
    val vocabSize: Int,
    val blockCount: Int,
    val embeddingLength: Int,
    val fields: Map<String, Any?>
)

/**
 * Unified model loader that auto-detects model architecture from GGUF metadata
 * and delegates to the appropriate network loader.
 *
 * Usage:
 * ```kotlin
 * // Peek at model info without loading weights
 * val info = UnifiedModelLoader.peek(source)
 * println("Architecture: ${info.architecture}, Family: ${info.family}")
 *
 * // Register a loader for a model family
 * UnifiedModelLoader.register(ModelFamily.LLAMA) { info, source, ctx ->
 *     LlamaNetworkLoader.fromGguf(source).load(ctx)
 * }
 * ```
 *
 * Network loaders register themselves at startup. The loader detects the
 * architecture from GGUF metadata and delegates to the registered handler.
 */
public object UnifiedModelLoader {

    /**
     * Peek at a GGUF file to extract model info without loading weights.
     *
     * @param sourceProvider Provides a [RandomAccessSource] to the GGUF file.
     * @return Model information including architecture, family, and dimensions.
     */
    public fun peek(sourceProvider: () -> RandomAccessSource): GGUFModelInfo {
        return sourceProvider().use { source ->
            StreamingGGUFReader.open(source).use { reader ->
                val fields = reader.fields
                val arch = (fields["general.architecture"] as? String) ?: "unknown"
                val family = ModelRegistry.detect(arch)

                GGUFModelInfo(
                    architecture = arch,
                    family = family,
                    contextLength = (fields["${arch}.context_length"] as? Number)?.toInt() ?: 4096,
                    vocabSize = (fields["${arch}.vocab_size"] as? Number)?.toInt()
                        ?: ((fields["tokenizer.ggml.tokens"] as? List<*>)?.size ?: 0),
                    blockCount = (fields["${arch}.block_count"] as? Number)?.toInt() ?: 0,
                    embeddingLength = (fields["${arch}.embedding_length"] as? Number)?.toInt() ?: 0,
                    fields = fields
                )
            }
        }
    }
}
