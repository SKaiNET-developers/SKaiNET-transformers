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
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
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
 *
 * The random-access path delegates tensor materialization to the engine's
 * [StreamingGgufParametersLoader] — quantized tensors keep their stored
 * block encoding with logical `[out, in]` shapes by default; pass
 * [GEMMA_DEQUANTIZE_ALL] as [weightForm] for a dense FP32 load. The token
 * embedding is always dequantized and the PLE table always stays packed
 * (see [GemmaWeightLoader]'s kdoc — same overrides). The sequential
 * [Source] path dequantizes everything to dense floats.
 */
@OptIn(ExperimentalMemoryApi::class)
public class Gemma3nWeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val loadTensorData: Boolean = true,
    private val weightForm: WeightForm? = null,
) {
    /**
     * Primary constructor for sequential Source-based loading.
     * Loads entire file into memory - suitable for models under 2GB.
     */
    public constructor(
        sourceProvider: () -> Source,
        loadTensorData: Boolean = true,
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        loadTensorData = loadTensorData,
    )

    /**
     * Secondary constructor for streaming RandomAccessSource-based loading.
     * Parses metadata only (~1MB memory) and loads tensors on-demand.
     * Suitable for models of any size (100+ GB).
     */
    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        weightForm: WeightForm? = null,
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = true,
        weightForm = weightForm,
    )


    /**
     * Load weights and invoke [onTensorLoaded] for each required tensor. Returns parsed metadata.
     */
    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma3nModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded)
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
        val meta = loadFromGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return Gemma3nWeights(meta, byName)
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
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded)
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
        val meta = loadFromStreamingGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return Gemma3nWeights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): Gemma3nWeights<T, V> = loadToMapStreaming(ctx, T::class)

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
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

        // Retained so the tied-output fallback below can alias the loaded
        // embedding instead of re-reading it.
        var embedTensor: Tensor<T, V>? = null

        required.forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
            if (name == Gemma3nTensorNames.TOKEN_EMBEDDINGS) embedTensor = tensor
            onTensorLoaded(name, tensor)
        }

        // Output weight: use dedicated tensor or fall back to weight tying (reuse token embeddings)
        val outputRt = tensorByName[Gemma3nTensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt)
            onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
        } else {
            // Weight tying: reuse token_embd.weight as output.weight (common in Gemma models)
            val embedRt = tensorByName[Gemma3nTensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight — cannot resolve LM head")
            val tensor: Tensor<T, V> = embedTensor
                ?: readerTensorToTensor(ctx, dtype, reader, embedRt)
            onTensorLoaded(Gemma3nTensorNames.OUTPUT_WEIGHT, tensor)
        }

        // Optional tensors
        optionalTensorNames(metadata).forEach { name ->
            val rt = tensorByName[name] ?: return@forEach
            onTensorLoaded(name, readerTensorToTensor(ctx, dtype, reader, rt))
        }

        return metadata
    }

    private suspend fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
    ): Gemma3nModelMetadata {
        require(dtype == FP32::class) {
            "Gemma 3n engine-backed GGUF loading delivers FP32-typed tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor. Use loadFromGguf for Source."
        }

        // Header pass: metadata, required-tensor check, tied-embedding
        // detection, wanted-name set (the engine loader delivers every
        // tensor in the file).
        val metadata: Gemma3nModelMetadata
        val wanted: Set<String>
        val tiedEmbeddings: Boolean
        val ggufTypeByName: Map<String, GGMLQuantizationType>
        StreamingGGUFReader.open(randomAccessProvider.invoke()).use { reader ->
            metadata = metadataFromStreamingGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }
            ggufTypeByName = reader.tensors.associate { it.name to it.tensorType }

            val missing = required.filter { it !in tensorByName }
            require(missing.isEmpty()) {
                "Missing required tensor(s) in GGUF payload: ${missing.joinToString()}"
            }

            tiedEmbeddings = tensorByName[Gemma3nTensorNames.OUTPUT_WEIGHT] == null
            wanted = buildSet {
                addAll(required)
                add(Gemma3nTensorNames.OUTPUT_WEIGHT)
                optionalTensorNames(metadata).forEach { if (it in tensorByName) add(it) }
            }
        }

        // Payload pass through the engine loader — same form policy as
        // [GemmaWeightLoader]: caller's [weightForm] (default keep-packed
        // [out, in]) with token-embedding and PLE overrides.
        val defaultForm = weightForm ?: WeightForm(
            shape = WeightShapeOrientation.OUT_IN,
            // MAPPED residency default (#342 arc, P5): servable encodings are
            // served zero-copy from file-backed pages; the engine heap-stages
            // the rest. The PLE override below stays HEAP — its row-dequant
            // wrapper needs a heap byte view of the packed table.
            residency = WeightResidency.MAPPED,
        )
        val engineLoader = StreamingGgufParametersLoader(
            sourceProvider = randomAccessProvider,
            weightForm = defaultForm,
            weightFormFor = { name ->
                when (name) {
                    Gemma3nTensorNames.TOKEN_EMBEDDINGS -> GEMMA_DEQUANTIZE_ALL
                    Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD ->
                        WeightForm(shape = WeightShapeOrientation.OUT_IN)
                    else -> null
                }
            }
        )
        var tokenEmbedding: Tensor<T, V>? = null
        engineLoader.load(ctx, dtype) { name: String, tensor: Tensor<T, V> ->
            if (name !in wanted) return@load
            val delivered =
                if (name == Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD) {
                    wrapGemmaPleIfPacked(ctx, dtype, tensor, ggufTypeByName[name])
                } else {
                    tensor
                }
            if (name == Gemma3nTensorNames.TOKEN_EMBEDDINGS) tokenEmbedding = delivered
            onTensorLoaded(name, delivered)
        }
        if (tiedEmbeddings) {
            // Weight tying: reuse token_embd.weight as output.weight.
            onTensorLoaded(
                Gemma3nTensorNames.OUTPUT_WEIGHT,
                tokenEmbedding
                    ?: error("Tied embeddings detected but ${Gemma3nTensorNames.TOKEN_EMBEDDINGS} was not delivered")
            )
        }

        return metadata
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

    private fun optionalTensorNames(metadata: Gemma3nModelMetadata): List<String> {
        val names = mutableListOf(
            Gemma3nTensorNames.ROPE_FREQS_REAL,
            Gemma3nTensorNames.ROPE_FREQS_IMAG,
            Gemma3nTensorNames.ALTUP_PROJ,
            Gemma3nTensorNames.ALTUP_UNEMBD_PROJ,
            Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma3nTensorNames.PER_LAYER_MODEL_PROJ,
            Gemma3nTensorNames.PER_LAYER_PROJ_NORM,
        )
        repeat(metadata.blockCount) { layer ->
            names += Gemma3nTensorNames.perLayerInput(layer)
            names += Gemma3nTensorNames.perLayerOutput(layer)
            names += Gemma3nTensorNames.altupPredictCoef(layer)
            names += Gemma3nTensorNames.altupCorrectCoef(layer)
            names += Gemma3nTensorNames.altupCorrectScale(layer)
            names += Gemma3nTensorNames.altupRouter(layer)
            names += Gemma3nTensorNames.altupRouterNorm(layer)
            names += Gemma3nTensorNames.attnQNorm(layer)
            names += Gemma3nTensorNames.attnKNorm(layer)
            names += Gemma3nTensorNames.postAttentionNorm(layer)
            names += Gemma3nTensorNames.postFfwNorm(layer)
            names += Gemma3nTensorNames.postNorm(layer)
            names += Gemma3nTensorNames.inputGate(layer)
            names += Gemma3nTensorNames.proj(layer)
            names += Gemma3nTensorNames.laurelL(layer)
            names += Gemma3nTensorNames.laurelR(layer)
            names += Gemma3nTensorNames.laurelPostNorm(layer)
        }
        return names
    }

    // ============== Tensor conversion using DecoderGgufWeightLoader.Dequant ==============

    /** Dequantize any GGUF tensor to a dense tensor of the requested dtype. */
    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor,
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
        val floats = when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                (raw as List<Float>).toFloatArray()
            }
            GGMLQuantizationType.F16 -> DequantOps.dequantF16(raw)
            GGMLQuantizationType.BF16 -> DequantOps.dequantBF16(raw)
            else -> {
                val bytes = DequantOps.toByteArray(raw, rt.name)
                DequantOps.dequantFromBytes(bytes, rt.tensorType, rt.nElements)
            }
        }
        return createTensor(ctx, dtype, shape, floats)
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
