package sk.ainet.models.gemma

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Adapter that loads Gemma 3n weights from GGUF files.
 *
 * Key differences from DecoderGgufWeightLoader:
 * - Architecture validation: accepts "gemma3n", "gemma3", "gemma" prefixes
 * - Variable intermediate (FFN) sizes per layer
 * - Per-layer embedding support
 * - Hybrid attention metadata extraction
 */
public class Gemma3nWeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val loadTensorData: Boolean = true,
    private val quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
) {
    /**
     * Primary constructor for sequential Source-based loading.
     * Loads entire file into memory - suitable for models under 2GB.
     */
    public constructor(
        sourceProvider: () -> Source,
        loadTensorData: Boolean = true,
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        loadTensorData = loadTensorData,
        quantPolicy = quantPolicy
    )

    /**
     * Secondary constructor for streaming RandomAccessSource-based loading.
     * Parses metadata only (~1MB memory) and loads tensors on-demand.
     * Suitable for models of any size (100+ GB).
     */
    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = true,
        quantPolicy = quantPolicy
    )


    /**
     * Load weights and invoke [onTensorLoaded] for each required tensor. Returns parsed metadata.
     */
    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma3nModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma3nModelMetadata = load(ctx, T::class, onTensorLoaded)

    /** Convenience helper that collects tensors into a map alongside metadata. */
    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma3nWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return Gemma3nWeights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): Gemma3nWeights<T, V> = loadToMap(ctx, T::class)

    // ============== Streaming API (for large files >2GB) ==============

    /**
     * Load weights using streaming API - parses metadata only, loads tensors on-demand.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma3nModelMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma3nModelMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    /**
     * Load weights to map using streaming API.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma3nWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromStreamingGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return Gemma3nWeights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): Gemma3nWeights<T, V> = loadToMapStreaming(ctx, T::class)

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): Gemma3nModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Gemma 3n GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor. Use loadFromStreamingGguf for RandomAccessSource."
        }

        val reader = sourceProvider.invoke().buffered().use { src ->
            GGUFReader(src, loadTensorData = loadTensorData)
        }

        val metadata = metadataFromGguf(reader.fields, reader.tensors)
        validateMetadata(metadata)

        val required = requiredTensorNames(metadata)
        val tensorByName = reader.tensors.associateBy { it.name }

        required.forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
            onTensorLoaded(name, tensor)
            if (quantPolicy == QuantPolicy.RAW_BYTES && rt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(name, rt.tensorType)
            }
        }

        // Output weight: use dedicated tensor or fall back to weight tying (reuse token embeddings)
        val outputRt = tensorByName[Gemma3nTensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt, metadata)
            onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
            if (quantPolicy == QuantPolicy.RAW_BYTES && outputRt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(Gemma3nTensorNames.OUTPUT_WEIGHT, outputRt.tensorType)
            }
        } else {
            // Weight tying: reuse token_embd.weight as output.weight (common in Gemma models)
            val embedRt = tensorByName[Gemma3nTensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight — cannot resolve LM head")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, embedRt, metadata)
            onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
        }

        // Optional tensors
        loadOptionalTensors(ctx, dtype, reader, tensorByName, onTensorLoaded, metadata)

        return metadata
    }

    private fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): Gemma3nModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Gemma 3n GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor. Use loadFromGguf for Source."
        }

        val source = randomAccessProvider.invoke()
        return StreamingGGUFReader.open(source).use { reader ->
            val metadata = metadataFromStreamingGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }

            required.forEach { name ->
                val st = tensorByName[name]
                    ?: error("Missing required tensor in GGUF payload: $name")
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
                if (quantPolicy == QuantPolicy.RAW_BYTES && st.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(name, st.tensorType)
                }
            }

            // Output weight: use dedicated tensor or fall back to weight tying (reuse token embeddings)
            val outputSt = tensorByName[Gemma3nTensorNames.OUTPUT_WEIGHT]
            if (outputSt != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, outputSt, metadata)
                onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
                if (quantPolicy == QuantPolicy.RAW_BYTES && outputSt.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(Gemma3nTensorNames.OUTPUT_WEIGHT, outputSt.tensorType)
                }
            } else {
                // Weight tying: reuse token_embd.weight as output.weight (common in Gemma models)
                val embedSt = tensorByName[Gemma3nTensorNames.TOKEN_EMBEDDINGS]
                    ?: error("Missing both output.weight and token_embd.weight — cannot resolve LM head")
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, embedSt, metadata)
                onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
            }

            // Optional tensors
            loadOptionalStreamingTensors(ctx, dtype, reader, tensorByName, onTensorLoaded, metadata)

            metadata
        }
    }

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
        tensors: List<ReaderTensor>
    ): Gemma3nModelMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "unknown"

        // Try multiple prefixes for Gemma architecture
        val prefix = findArchPrefix(fields, listOf("gemma3n", "gemma3", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val perLayerEmbedding = fields["$prefix.per_layer_embedding_length"]?.scalarInt()
            ?: fields["$prefix.embedding_length_per_layer_input"]?.scalarInt()
            ?: 256
        val contextLength = fields["$prefix.context_length"]?.scalarInt() ?: 8192
        val blockCount = fields["$prefix.block_count"]?.scalarInt() ?: 35
        val headCount = fields["$prefix.attention.head_count"]?.scalarInt() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.scalarInt() ?: 2
        val headDim = fields["$prefix.attention.head_dim"]?.scalarInt() ?: 256
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)

        // Gemma 3n specific
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.scalarInt()
            ?: Gemma3nModelMetadata.DEFAULT_SLIDING_WINDOW
        val ropeBaseLocal = fields["$prefix.rope.freq_base_local"]?.scalarFloat()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_LOCAL
        val ropeBaseGlobal = fields["$prefix.rope.freq_base_global"]?.scalarFloat()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_GLOBAL
        val kvSharedLayers = fields["$prefix.kv_shared_layers"]?.scalarInt()
            ?: fields["$prefix.attention.shared_kv_layers"]?.scalarInt()
            ?: Gemma3nModelMetadata.DEFAULT_KV_SHARED_LAYERS

        // Extract FFN sizes per layer (MatFormer)
        val ffnLengths = extractFeedForwardLengths(fields, prefix, blockCount, embeddingLength)

        // Layer pattern (default: 4 sliding + 1 global)
        val layerPattern = extractLayerPattern(fields, prefix)

        // AltUp fields (E4B)
        val numAltupInputs = fields["$prefix.altup.num_inputs"]?.scalarInt() ?: 1
        val altupActiveIdx = fields["$prefix.altup.active_idx"]?.scalarInt() ?: 0

        // Activation sparsity
        val activationSparsityPattern = extractActivationSparsityPattern(fields, prefix)
        val activationSparsityScale = fields["$prefix.activation_sparsity_scale"]?.scalarFloat() ?: 0f

        return Gemma3nModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            perLayerEmbeddingLength = perLayerEmbedding,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLengths = ffnLengths,
            headDim = headDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            ropeBaseLocal = ropeBaseLocal,
            ropeBaseGlobal = ropeBaseGlobal,
            kvSharedLayers = kvSharedLayers,
            layerPattern = layerPattern,
            numAltupInputs = numAltupInputs,
            altupActiveIdx = altupActiveIdx,
            activationSparsityPattern = activationSparsityPattern,
            activationSparsityScale = activationSparsityScale
        )
    }

    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): Gemma3nModelMetadata {
        val arch = (fields["general.architecture"] as? String) ?: "unknown"

        // Try multiple prefixes for Gemma architecture
        val prefix = findStreamingArchPrefix(fields, listOf("gemma3n", "gemma3", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.toIntValue()
            ?: inferEmbeddingFromStreamingTensor(tensors)
        val perLayerEmbedding = fields["$prefix.per_layer_embedding_length"]?.toIntValue()
            ?: fields["$prefix.embedding_length_per_layer_input"]?.toIntValue()
            ?: 256
        val contextLength = fields["$prefix.context_length"]?.toIntValue() ?: 8192
        val blockCount = fields["$prefix.block_count"]?.toIntValue() ?: 35
        val headCount = fields["$prefix.attention.head_count"]?.toIntValue() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.toIntValue() ?: 2
        val headDim = fields["$prefix.attention.head_dim"]?.toIntValue() ?: 256
        val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)

        // Gemma 3n specific
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.toIntValue()
            ?: Gemma3nModelMetadata.DEFAULT_SLIDING_WINDOW
        val ropeBaseLocal = fields["$prefix.rope.freq_base_local"]?.toFloatValue()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_LOCAL
        val ropeBaseGlobal = fields["$prefix.rope.freq_base_global"]?.toFloatValue()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_GLOBAL
        val kvSharedLayers = fields["$prefix.kv_shared_layers"]?.toIntValue()
            ?: fields["$prefix.attention.shared_kv_layers"]?.toIntValue()
            ?: (fields["$prefix.attention.shared_kv_layers"] as? Number)?.toInt()
            ?: Gemma3nModelMetadata.DEFAULT_KV_SHARED_LAYERS

        // Extract FFN sizes per layer (MatFormer)
        val ffnLengths = extractStreamingFeedForwardLengths(fields, prefix, blockCount, embeddingLength)

        // Layer pattern (default: 4 sliding + 1 global)
        val layerPattern = extractStreamingLayerPattern(fields, prefix)

        // AltUp fields (E4B)
        val numAltupInputs = fields["$prefix.altup.num_inputs"]?.toIntValue() ?: 1
        val altupActiveIdx = fields["$prefix.altup.active_idx"]?.toIntValue() ?: 0

        // Activation sparsity
        val activationSparsityPattern = extractStreamingActivationSparsityPattern(fields, prefix)
        val activationSparsityScale = fields["$prefix.activation_sparsity_scale"]?.toFloatValue() ?: 0f

        return Gemma3nModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            perLayerEmbeddingLength = perLayerEmbedding,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLengths = ffnLengths,
            headDim = headDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            ropeBaseLocal = ropeBaseLocal,
            ropeBaseGlobal = ropeBaseGlobal,
            kvSharedLayers = kvSharedLayers,
            layerPattern = layerPattern,
            numAltupInputs = numAltupInputs,
            altupActiveIdx = altupActiveIdx,
            activationSparsityPattern = activationSparsityPattern,
            activationSparsityScale = activationSparsityScale
        )
    }

    private fun findArchPrefix(fields: Map<String, ReaderField>, prefixes: List<String>): String {
        for (prefix in prefixes) {
            if (fields["$prefix.embedding_length"] != null || fields["$prefix.block_count"] != null) {
                return prefix
            }
        }
        return prefixes.first()
    }

    private fun findStreamingArchPrefix(fields: Map<String, Any?>, prefixes: List<String>): String {
        for (prefix in prefixes) {
            if (fields["$prefix.embedding_length"] != null || fields["$prefix.block_count"] != null) {
                return prefix
            }
        }
        return prefixes.first()
    }

    private fun extractFeedForwardLengths(
        fields: Map<String, ReaderField>,
        prefix: String,
        blockCount: Int,
        embeddingLength: Int
    ): List<Int> {
        // Try to get per-layer FFN sizes
        val perLayerKey = "$prefix.feed_forward_lengths"
        val perLayerField = fields[perLayerKey]
        if (perLayerField != null) {
            return try {
                perLayerField.intListValue()
            } catch (e: Exception) {
                List(blockCount) { embeddingLength * 4 }
            }
        }

        // Fall back to single FFN length
        val ffnLength = fields["$prefix.feed_forward_length"]?.scalarInt() ?: (embeddingLength * 4)
        return List(blockCount) { ffnLength }
    }

    private fun extractStreamingFeedForwardLengths(
        fields: Map<String, Any?>,
        prefix: String,
        blockCount: Int,
        embeddingLength: Int
    ): List<Int> {
        val perLayerKey = "$prefix.feed_forward_lengths"
        val perLayerValue = fields[perLayerKey]
        if (perLayerValue != null && perLayerValue is List<*>) {
            return perLayerValue.mapNotNull { (it as? Number)?.toInt() }
        }

        // Fall back to single FFN length
        val ffnLength = fields["$prefix.feed_forward_length"]?.toIntValue() ?: (embeddingLength * 4)
        return List(blockCount) { ffnLength }
    }

    private fun extractLayerPattern(fields: Map<String, ReaderField>, prefix: String): List<String> {
        val patternKey = "$prefix.attention.layer_pattern"
        val patternField = fields[patternKey]
        if (patternField != null) {
            return try {
                patternField.stringListValue()
            } catch (e: Exception) {
                Gemma3nModelMetadata.DEFAULT_LAYER_PATTERN
            }
        }
        return Gemma3nModelMetadata.DEFAULT_LAYER_PATTERN
    }

    private fun extractStreamingLayerPattern(fields: Map<String, Any?>, prefix: String): List<String> {
        val patternKey = "$prefix.attention.layer_pattern"
        val patternValue = fields[patternKey]
        if (patternValue != null && patternValue is List<*>) {
            return patternValue.mapNotNull { it as? String }
        }
        return Gemma3nModelMetadata.DEFAULT_LAYER_PATTERN
    }

    private fun extractActivationSparsityPattern(
        fields: Map<String, ReaderField>,
        prefix: String
    ): List<Float> {
        val key = "$prefix.activation_sparsity_pattern"
        val field = fields[key]
        if (field != null) {
            return try {
                field.floatListValue()
            } catch (e: Exception) {
                emptyList()
            }
        }
        return emptyList()
    }

    private fun extractStreamingActivationSparsityPattern(
        fields: Map<String, Any?>,
        prefix: String
    ): List<Float> {
        val key = "$prefix.activation_sparsity_pattern"
        val value = fields[key]
        if (value is List<*>) {
            return value.mapNotNull { (it as? Number)?.toFloat() }
        }
        return emptyList()
    }

    private fun validateMetadata(metadata: Gemma3nModelMetadata) {
        val validArchs = setOf("gemma3n", "gemma3", "gemma", "llama", "unknown")
        require(metadata.architecture in validArchs || metadata.architecture.startsWith("gemma")) {
            "Unsupported architecture: ${metadata.architecture}. Expected gemma3n, gemma3, gemma, or compatible."
        }
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: Gemma3nModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += Gemma3nTensorNames.TOKEN_EMBEDDINGS
        names += Gemma3nTensorNames.OUTPUT_NORM
        // OUTPUT_WEIGHT is handled separately — many Gemma models use weight tying
        // (no output.weight tensor; the token embedding is reused as the LM head).

        repeat(metadata.blockCount) { layer ->
            names += Gemma3nTensorNames.inputLayernorm(layer)
            names += Gemma3nTensorNames.attnQ(layer)
            names += Gemma3nTensorNames.attnK(layer)
            names += Gemma3nTensorNames.attnV(layer)
            names += Gemma3nTensorNames.attnOut(layer)
            names += Gemma3nTensorNames.postAttentionLayernorm(layer)
            names += Gemma3nTensorNames.ffnGate(layer)
            names += Gemma3nTensorNames.ffnDown(layer)
            names += Gemma3nTensorNames.ffnUp(layer)
        }
        return names
    }

    private fun <T : DType, V> loadOptionalTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        tensorByName: Map<String, ReaderTensor>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        metadata: Gemma3nModelMetadata
    ) {
        // RoPE tables
        listOf(
            Gemma3nTensorNames.ROPE_FREQS_REAL,
            Gemma3nTensorNames.ROPE_FREQS_IMAG
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null && rt.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // Per-layer embeddings (optional)
        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma3nTensorNames.perLayerInput(layer),
                Gemma3nTensorNames.perLayerOutput(layer)
            ).forEach { name ->
                val rt = tensorByName[name]
                if (rt != null) {
                    val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }

        // E4B per-layer AltUp + additional tensors
        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma3nTensorNames.altupPredictCoef(layer),
                Gemma3nTensorNames.altupCorrectCoef(layer),
                Gemma3nTensorNames.altupCorrectScale(layer),
                Gemma3nTensorNames.altupRouter(layer),
                Gemma3nTensorNames.altupRouterNorm(layer),
                Gemma3nTensorNames.attnQNorm(layer),
                Gemma3nTensorNames.attnKNorm(layer),
                Gemma3nTensorNames.postAttentionNorm(layer),
                Gemma3nTensorNames.postFfwNorm(layer),
                Gemma3nTensorNames.postNorm(layer),
                Gemma3nTensorNames.inputGate(layer),
                Gemma3nTensorNames.proj(layer),
                Gemma3nTensorNames.laurelL(layer),
                Gemma3nTensorNames.laurelR(layer),
                Gemma3nTensorNames.laurelPostNorm(layer)
            ).forEach { name ->
                val rt = tensorByName[name]
                if (rt != null) {
                    val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }

        // E4B global AltUp tensors
        listOf(
            Gemma3nTensorNames.ALTUP_PROJ,
            Gemma3nTensorNames.ALTUP_UNEMBD_PROJ
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // E4B global per-layer embedding tensors
        listOf(
            Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma3nTensorNames.PER_LAYER_MODEL_PROJ,
            Gemma3nTensorNames.PER_LAYER_PROJ_NORM
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                onTensorLoaded(name, tensor)
            }
        }
    }

    private fun <T : DType, V> loadOptionalStreamingTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        tensorByName: Map<String, StreamingTensorInfo>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        metadata: Gemma3nModelMetadata
    ) {
        // RoPE tables
        listOf(
            Gemma3nTensorNames.ROPE_FREQS_REAL,
            Gemma3nTensorNames.ROPE_FREQS_IMAG
        ).forEach { name ->
            val st = tensorByName[name]
            if (st != null && st.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // Per-layer embeddings (optional)
        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma3nTensorNames.perLayerInput(layer),
                Gemma3nTensorNames.perLayerOutput(layer)
            ).forEach { name ->
                val st = tensorByName[name]
                if (st != null) {
                    val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }

        // E4B per-layer AltUp + additional tensors
        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma3nTensorNames.altupPredictCoef(layer),
                Gemma3nTensorNames.altupCorrectCoef(layer),
                Gemma3nTensorNames.altupCorrectScale(layer),
                Gemma3nTensorNames.altupRouter(layer),
                Gemma3nTensorNames.altupRouterNorm(layer),
                Gemma3nTensorNames.attnQNorm(layer),
                Gemma3nTensorNames.attnKNorm(layer),
                Gemma3nTensorNames.postAttentionNorm(layer),
                Gemma3nTensorNames.postFfwNorm(layer),
                Gemma3nTensorNames.postNorm(layer),
                Gemma3nTensorNames.inputGate(layer),
                Gemma3nTensorNames.proj(layer),
                Gemma3nTensorNames.laurelL(layer),
                Gemma3nTensorNames.laurelR(layer),
                Gemma3nTensorNames.laurelPostNorm(layer)
            ).forEach { name ->
                val st = tensorByName[name]
                if (st != null) {
                    val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }

        // E4B global AltUp tensors
        listOf(
            Gemma3nTensorNames.ALTUP_PROJ,
            Gemma3nTensorNames.ALTUP_UNEMBD_PROJ
        ).forEach { name ->
            val st = tensorByName[name]
            if (st != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // E4B global per-layer embedding tensors
        listOf(
            Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma3nTensorNames.PER_LAYER_MODEL_PROJ,
            Gemma3nTensorNames.PER_LAYER_PROJ_NORM
        ).forEach { name ->
            val st = tensorByName[name]
            if (st != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }
    }

    // ============== Tensor conversion using DecoderGgufWeightLoader.Dequant ==============

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor,
        metadata: Gemma3nModelMetadata
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                val floats = (if (rt.data.isEmpty()) reader.materialize(rt) else rt.data) as List<Float>
                createTensor(ctx, dtype, shape, floats.toFloatArray())
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes = DequantOps.toByteArray(raw, rt.name)
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val floats = when (rt.tensorType) {
                            GGMLQuantizationType.F16 -> DequantOps.dequantF16(raw)
                            GGMLQuantizationType.BF16 -> DequantOps.dequantBF16(raw)
                            else -> error("Unreachable")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL,
            GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0,
            GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes = DequantOps.toByteArray(raw, rt.name)
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val floats = dequantize(raw, rt.tensorType, rt.nElements)
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            else -> {
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes = DequantOps.toByteArray(raw, rt.name)
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> streamingTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        st: StreamingTensorInfo,
        metadata: Gemma3nModelMetadata
    ): Tensor<T, V> {
        val shape = Shape(*st.shape.map { it.toInt() }.toIntArray())
        val bytes = reader.loadTensorData(st)

        return when (st.tensorType) {
            GGMLQuantizationType.F32 -> {
                val floats = bytesToFloatArray(bytes)
                createTensor(ctx, dtype, shape, floats)
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        val floats = when (st.tensorType) {
                            GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
                            GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
                            else -> error("Unreachable")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL,
            GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0,
            GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        val floats = dequantFromBytes(bytes, st.tensorType, st.nElements.toInt())
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            else -> {
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    private fun dequantize(raw: List<Any>, tensorType: GGMLQuantizationType, nElems: Int): FloatArray =
        DequantOps.dequantFromList(raw, tensorType, nElems)

    private fun dequantFromBytes(bytes: ByteArray, tensorType: GGMLQuantizationType, nElems: Int): FloatArray =
        DequantOps.dequantFromBytes(bytes, tensorType, nElems)

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        data: FloatArray
    ): Tensor<T, V> {
        return if (originalShape.rank == 2) {
            val rows = originalShape[0]
            val cols = originalShape[1]
            val transposed = DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, V>
        } else {
            ctx.fromFloatArray<T, Float>(originalShape, dtype, data) as Tensor<T, V>
        }
    }

    // ============== Helper methods (delegating to DequantOps) ==============

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray = DequantOps.bytesToFloatArray(bytes)
    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantF16FromBytes(bytes)
    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantBF16FromBytes(bytes)

    private fun Any?.toIntValue(): Int? = when (this) {
        is Int -> this
        is UInt -> this.toInt()
        is Long -> this.toInt()
        is ULong -> this.toInt()
        is Short -> this.toInt()
        is UShort -> this.toInt()
        is Byte -> this.toInt()
        is UByte -> this.toInt()
        else -> null
    }

    private fun Any?.toFloatValue(): Float? = when (this) {
        is Float -> this
        is Double -> this.toFloat()
        is Number -> this.toFloat()
        else -> null
    }

    private fun ReaderField.scalarInt(): Int {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val value = (part as List<*>).firstOrNull()
            ?: error("Empty data part for field $name")
        return when (value) {
            is Int -> value
            is UInt -> value.toInt()
            is Long -> value.toInt()
            is ULong -> value.toInt()
            is Short -> value.toInt()
            is UShort -> value.toInt()
            is Byte -> value.toInt()
            is UByte -> value.toInt()
            else -> error("Unsupported scalar type ${value::class} for field $name")
        }
    }

    private fun ReaderField.scalarFloat(): Float {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val value = (part as List<*>).firstOrNull()
            ?: error("Empty data part for field $name")
        return when (value) {
            is Float -> value
            is Double -> value.toFloat()
            is Number -> value.toFloat()
            else -> error("Unsupported scalar type ${value::class} for field $name")
        }
    }

    private fun ReaderField.stringValue(): String {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        val bytes = (part as List<Any>).mapNotNull {
            when (it) {
                is UByte -> it.toByte()
                is Byte -> it
                else -> null
            }
        }
        return bytes.toByteArray().decodeToString()
    }

    private fun ReaderField.intListValue(): List<Int> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { (it as? Number)?.toInt() }
    }

    private fun ReaderField.floatListValue(): List<Float> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { (it as? Number)?.toFloat() }
    }

    private fun ReaderField.stringListValue(): List<String> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { it as? String }
    }

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma3nTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma3nTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma3nTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma3nTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }
}
