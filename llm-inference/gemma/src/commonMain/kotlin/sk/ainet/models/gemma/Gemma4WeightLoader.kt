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
import kotlin.reflect.KClass

/**
 * Adapter that loads Gemma 4 weights from GGUF files.
 *
 * Recognizes architecture prefixes: "gemma4", "gemma", "llama".
 * Extracts Gemma 4 specific metadata: global_head_dim, proportional RoPE, layer types.
 */
public class Gemma4WeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val loadTensorData: Boolean = true,
    private val quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
) {
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

    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = true,
        quantPolicy = quantPolicy
    )

    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata = load(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma4Weights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return Gemma4Weights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): Gemma4Weights<T, V> = loadToMap(ctx, T::class)

    // Streaming API

    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded, null)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma4Weights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val quantTypes = linkedMapOf<String, GGMLQuantizationType>()
        val meta = loadFromStreamingGguf(ctx, dtype, { name, tensor -> byName[name] = tensor }) { name, qt ->
            quantTypes[name] = qt
        }
        return Gemma4Weights(meta, byName, quantTypes)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): Gemma4Weights<T, V> = loadToMapStreaming(ctx, T::class)

    // ============== Internal loading ==============

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?
    ): Gemma4ModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Gemma 4 GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor."
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

        // Output weight with weight tying fallback
        val outputRt = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt, metadata)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
            if (quantPolicy == QuantPolicy.RAW_BYTES && outputRt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, outputRt.tensorType)
            }
        } else {
            val embedRt = tensorByName[Gemma4TensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, embedRt, metadata)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
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
    ): Gemma4ModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Gemma 4 GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor."
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

            // Output weight with weight tying fallback
            val outputSt = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT]
            if (outputSt != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, outputSt, metadata)
                onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
                if (quantPolicy == QuantPolicy.RAW_BYTES && outputSt.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, outputSt.tensorType)
                }
            } else {
                val embedSt = tensorByName[Gemma4TensorNames.TOKEN_EMBEDDINGS]
                    ?: error("Missing both output.weight and token_embd.weight")
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, embedSt, metadata)
                onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
            }

            loadOptionalStreamingTensors(ctx, dtype, reader, tensorByName, onTensorLoaded, metadata)

            metadata
        }
    }

    // ============== Metadata extraction ==============

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
        tensors: List<ReaderTensor>
    ): Gemma4ModelMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "unknown"
        val prefix = findArchPrefix(fields, listOf("gemma4", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.scalarInt() ?: 131072
        val blockCount = fields["$prefix.block_count"]?.scalarInt() ?: 34
        val headCount = fields["$prefix.attention.head_count"]?.scalarInt() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.scalarInt() ?: 4
        val headDim = fields["$prefix.attention.head_dim"]?.scalarInt() ?: 256
        val globalHeadDim = fields["$prefix.attention.global_head_dim"]?.scalarInt() ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)
        val intermediateSize = fields["$prefix.feed_forward_length"]?.scalarInt()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.scalarInt()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        val kvSharedLayers = fields["$prefix.kv_shared_layers"]?.scalarInt()
            ?: fields["$prefix.attention.shared_kv_layers"]?.scalarInt()
            ?: Gemma4ModelMetadata.DEFAULT_KV_SHARED_LAYERS

        val layerTypes = extractLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.scalarFloat() ?: 1000000f
        val ropeBaseLocal = fields["$prefix.rope.freq_base_local"]?.scalarFloat() ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.scalarFloat() ?: 1.0f
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.scalarFloat() ?: 1.0f

        return Gemma4ModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            intermediateSize = intermediateSize,
            headDim = headDim,
            globalHeadDim = globalHeadDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            kvSharedLayers = kvSharedLayers,
            layerTypes = layerTypes,
            ropeParametersFull = Gemma4RopeConfig(
                base = ropeBase,
                ropeType = "proportional",
                factor = ropeFactor,
                partialRotaryFactor = partialRotaryFactor
            ),
            ropeParametersSliding = Gemma4RopeConfig(
                base = ropeBaseLocal,
                ropeType = "default"
            ),
            maxPositionEmbeddings = contextLength
        )
    }

    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): Gemma4ModelMetadata {
        val arch = (fields["general.architecture"] as? String) ?: "unknown"
        val prefix = findStreamingArchPrefix(fields, listOf("gemma4", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.toIntValue()
            ?: inferEmbeddingFromStreamingTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.toIntValue() ?: 131072
        val blockCount = fields["$prefix.block_count"]?.toIntValue() ?: 34
        val headCount = fields["$prefix.attention.head_count"]?.toIntValue() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.toIntValue() ?: 4
        val headDim = fields["$prefix.attention.head_dim"]?.toIntValue() ?: 256
        val globalHeadDim = fields["$prefix.attention.global_head_dim"]?.toIntValue() ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)
        val intermediateSize = fields["$prefix.feed_forward_length"]?.toIntValue()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.toIntValue()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        val kvSharedLayers = fields["$prefix.kv_shared_layers"]?.toIntValue()
            ?: fields["$prefix.attention.shared_kv_layers"]?.toIntValue()
            ?: Gemma4ModelMetadata.DEFAULT_KV_SHARED_LAYERS

        val layerTypes = extractStreamingLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.toFloatValue() ?: 1000000f
        val ropeBaseLocal = fields["$prefix.rope.freq_base_local"]?.toFloatValue() ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.toFloatValue() ?: 1.0f
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.toFloatValue() ?: 1.0f

        return Gemma4ModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            intermediateSize = intermediateSize,
            headDim = headDim,
            globalHeadDim = globalHeadDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            kvSharedLayers = kvSharedLayers,
            layerTypes = layerTypes,
            ropeParametersFull = Gemma4RopeConfig(
                base = ropeBase,
                ropeType = "proportional",
                factor = ropeFactor,
                partialRotaryFactor = partialRotaryFactor
            ),
            ropeParametersSliding = Gemma4RopeConfig(
                base = ropeBaseLocal,
                ropeType = "default"
            ),
            maxPositionEmbeddings = contextLength
        )
    }

    // ============== Helpers ==============

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

    private fun extractLayerTypes(
        fields: Map<String, ReaderField>,
        prefix: String,
        blockCount: Int
    ): List<String> {
        val patternField = fields["$prefix.attention.layer_types"]
            ?: fields["$prefix.attention.layer_pattern"]
        if (patternField != null) {
            return try {
                patternField.stringListValue()
            } catch (_: Exception) {
                buildDefaultLayerTypes(blockCount)
            }
        }
        return buildDefaultLayerTypes(blockCount)
    }

    private fun extractStreamingLayerTypes(
        fields: Map<String, Any?>,
        prefix: String,
        blockCount: Int
    ): List<String> {
        val value = fields["$prefix.attention.layer_types"]
            ?: fields["$prefix.attention.layer_pattern"]
        if (value is List<*>) {
            return value.mapNotNull { it as? String }
        }
        return buildDefaultLayerTypes(blockCount)
    }

    private fun buildDefaultLayerTypes(blockCount: Int): List<String> {
        return List(blockCount) { idx ->
            if (idx == blockCount - 1) "full_attention"
            else if ((idx + 1) % 6 == 0) "full_attention"
            else "sliding_attention"
        }
    }

    private fun validateMetadata(metadata: Gemma4ModelMetadata) {
        val validArchs = setOf("gemma4", "gemma", "llama", "unknown")
        require(metadata.architecture in validArchs || metadata.architecture.startsWith("gemma")) {
            "Unsupported architecture: ${metadata.architecture}. Expected gemma4 or compatible."
        }
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: Gemma4ModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += Gemma4TensorNames.TOKEN_EMBEDDINGS
        names += Gemma4TensorNames.OUTPUT_NORM

        repeat(metadata.blockCount) { layer ->
            names += Gemma4TensorNames.inputLayernorm(layer)
            names += Gemma4TensorNames.attnQ(layer)
            names += Gemma4TensorNames.attnK(layer)
            names += Gemma4TensorNames.attnV(layer)
            names += Gemma4TensorNames.attnOut(layer)
            names += Gemma4TensorNames.postAttentionLayernorm(layer)
            names += Gemma4TensorNames.ffnGate(layer)
            names += Gemma4TensorNames.ffnDown(layer)
            names += Gemma4TensorNames.ffnUp(layer)
        }
        return names
    }

    private fun <T : DType, V> loadOptionalTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        tensorByName: Map<String, ReaderTensor>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        metadata: Gemma4ModelMetadata
    ) {
        // RoPE tables
        listOf(Gemma4TensorNames.ROPE_FREQS_REAL, Gemma4TensorNames.ROPE_FREQS_IMAG).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null && rt.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // PLE and optional per-layer tensors
        listOf(
            Gemma4TensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma4TensorNames.PER_LAYER_MODEL_PROJ,
            Gemma4TensorNames.PER_LAYER_PROJ_NORM
        ).forEach { name ->
            val rt = tensorByName[name]
            if (rt != null) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma4TensorNames.perLayerInput(layer),
                Gemma4TensorNames.perLayerOutput(layer),
                Gemma4TensorNames.attnQNorm(layer),
                Gemma4TensorNames.attnKNorm(layer)
            ).forEach { name ->
                val rt = tensorByName[name]
                if (rt != null) {
                    val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }
    }

    private fun <T : DType, V> loadOptionalStreamingTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        tensorByName: Map<String, StreamingTensorInfo>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        metadata: Gemma4ModelMetadata
    ) {
        listOf(Gemma4TensorNames.ROPE_FREQS_REAL, Gemma4TensorNames.ROPE_FREQS_IMAG).forEach { name ->
            val st = tensorByName[name]
            if (st != null && st.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        listOf(
            Gemma4TensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma4TensorNames.PER_LAYER_MODEL_PROJ,
            Gemma4TensorNames.PER_LAYER_PROJ_NORM
        ).forEach { name ->
            val st = tensorByName[name]
            if (st != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma4TensorNames.perLayerInput(layer),
                Gemma4TensorNames.perLayerOutput(layer),
                Gemma4TensorNames.attnQNorm(layer),
                Gemma4TensorNames.attnKNorm(layer)
            ).forEach { name ->
                val st = tensorByName[name]
                if (st != null) {
                    val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                    onTensorLoaded(name, tensor)
                }
            }
        }
    }

    // ============== Tensor conversion ==============

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor,
        metadata: Gemma4ModelMetadata
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())

        return when {
            quantPolicy == QuantPolicy.DEQUANTIZE_TO_FP32 || rt.tensorType == GGMLQuantizationType.F32 -> {
                val data = if (rt.tensorType == GGMLQuantizationType.F32) {
                    DequantOps.bytesToFloatArray(rt.data)
                } else {
                    DequantOps.dequantize(rt.data, rt.tensorType, shape.totalElements)
                }
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            rt.tensorType == GGMLQuantizationType.F16 -> {
                val data = DequantOps.dequantF16FromBytes(rt.data)
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            rt.tensorType == GGMLQuantizationType.BF16 -> {
                val data = DequantOps.dequantBF16FromBytes(rt.data)
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            else -> {
                // Raw quantized bytes as Int8 tensor
                val byteData = rt.data
                val int8Shape = Shape(byteData.size)
                ctx.fromByteArray<T>(int8Shape, dtype, byteData) as Tensor<T, V>
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> streamingTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        st: StreamingTensorInfo,
        metadata: Gemma4ModelMetadata
    ): Tensor<T, V> {
        val shape = Shape(*st.shape.map { it.toInt() }.toIntArray())
        val bytes = reader.loadTensorData(st)

        return when {
            quantPolicy == QuantPolicy.DEQUANTIZE_TO_FP32 || st.tensorType == GGMLQuantizationType.F32 -> {
                val data = if (st.tensorType == GGMLQuantizationType.F32) {
                    DequantOps.bytesToFloatArray(bytes)
                } else {
                    DequantOps.dequantize(bytes, st.tensorType, shape.totalElements)
                }
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            st.tensorType == GGMLQuantizationType.F16 -> {
                val data = DequantOps.dequantF16FromBytes(bytes)
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            st.tensorType == GGMLQuantizationType.BF16 -> {
                val data = DequantOps.dequantBF16FromBytes(bytes)
                ctx.fromFloatArray<T, Float>(shape, dtype, data) as Tensor<T, V>
            }
            else -> {
                val int8Shape = Shape(bytes.size)
                ctx.fromByteArray<T>(int8Shape, dtype, bytes) as Tensor<T, V>
            }
        }
    }

    // ============== Inference helpers ==============

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val embd = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
        return embd?.shape?.getOrNull(1)?.toInt() ?: 2304
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val embd = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
        return embd?.shape?.getOrNull(0)?.toInt() ?: 262144
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val embd = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
        return embd?.shape?.getOrNull(1)?.toInt() ?: 2304
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val embd = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
        return embd?.shape?.getOrNull(0)?.toInt() ?: 262144
    }

    private fun ReaderField.scalarInt(): Int? = when (val v = this.value) {
        is Number -> v.toInt()
        is List<*> -> (v.firstOrNull() as? Number)?.toInt()
        else -> null
    }

    private fun ReaderField.scalarFloat(): Float? = when (val v = this.value) {
        is Number -> v.toFloat()
        is List<*> -> (v.firstOrNull() as? Number)?.toFloat()
        else -> null
    }

    private fun ReaderField.stringValue(): String? = when (val v = this.value) {
        is String -> v
        is ByteArray -> v.decodeToString()
        is List<*> -> (v.firstOrNull() as? String)
        else -> v?.toString()
    }

    private fun ReaderField.stringListValue(): List<String> = when (val v = this.value) {
        is List<*> -> v.mapNotNull { it as? String }
        else -> emptyList()
    }

    private fun ReaderField.intListValue(): List<Int> = when (val v = this.value) {
        is List<*> -> v.mapNotNull { (it as? Number)?.toInt() }
        else -> emptyList()
    }

    private fun Any?.toIntValue(): Int? = when (this) {
        is Number -> this.toInt()
        is List<*> -> (this.firstOrNull() as? Number)?.toInt()
        else -> null
    }

    private fun Any?.toFloatValue(): Float? = when (this) {
        is Number -> this.toFloat()
        is List<*> -> (this.firstOrNull() as? Number)?.toFloat()
        else -> null
    }
}
