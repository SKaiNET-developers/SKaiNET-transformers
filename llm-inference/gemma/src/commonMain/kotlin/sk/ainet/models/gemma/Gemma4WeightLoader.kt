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
        quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
        loadTensorData: Boolean = true
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = loadTensorData,
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
        val logicalShapes = linkedMapOf<String, Shape>()
        val meta = loadFromGguf(
            ctx,
            dtype,
            onTensorLoaded = { name, tensor -> byName[name] = tensor },
            quantCallback = { name, qt -> quantTypes[name] = qt },
            logicalShapeCallback = { name, shape -> logicalShapes[name] = shape }
        )
        return Gemma4Weights(meta, byName, quantTypes, logicalShapes)
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
        val logicalShapes = linkedMapOf<String, Shape>()
        val meta = loadFromStreamingGguf(
            ctx,
            dtype,
            onTensorLoaded = { name, tensor -> byName[name] = tensor },
            quantCallback = { name, qt -> quantTypes[name] = qt },
            logicalShapeCallback = { name, shape -> logicalShapes[name] = shape }
        )
        return Gemma4Weights(meta, byName, quantTypes, logicalShapes)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): Gemma4Weights<T, V> = loadToMapStreaming(ctx, T::class)

    // ============== Internal loading ==============

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?,
        logicalShapeCallback: ((String, Shape) -> Unit)? = null
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
            if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && rt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(name, rt.tensorType)
            }
            logicalShapeCallback?.invoke(
                name,
                Shape(*rt.shape.map { it.toInt() }.reversed().toIntArray())
            )
        }

        // Output weight with weight tying fallback
        val outputRt = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt, metadata)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
            if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && outputRt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, outputRt.tensorType)
            }
        } else {
            val embedRt = tensorByName[Gemma4TensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, embedRt, metadata)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
            // See loadFromStreamingGguf for why the tied-output path must
            // also fire quantCallback + logicalShapeCallback.
            if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && embedRt.tensorType != GGMLQuantizationType.F32) {
                quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, embedRt.tensorType)
            }
            logicalShapeCallback?.invoke(
                Gemma4TensorNames.OUTPUT_WEIGHT,
                Shape(*embedRt.shape.map { it.toInt() }.reversed().toIntArray())
            )
        }

        // Optional tensors
        loadOptionalTensors(ctx, dtype, reader, tensorByName, onTensorLoaded, metadata)

        return metadata
    }

    private fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)?,
        logicalShapeCallback: ((String, Shape) -> Unit)? = null
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

            if (!loadTensorData) return@use metadata

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }

            required.forEach { name ->
                val st = tensorByName[name]
                    ?: error("Missing required tensor in GGUF payload: $name")
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
                if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && st.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(name, st.tensorType)
                }
                logicalShapeCallback?.invoke(name, reversedShape(st.shape))
            }

            // Output weight with weight tying fallback
            val outputSt = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT]
            if (outputSt != null) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, outputSt, metadata)
                onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
                if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && outputSt.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, outputSt.tensorType)
                }
                logicalShapeCallback?.invoke(
                    Gemma4TensorNames.OUTPUT_WEIGHT,
                    reversedShape(outputSt.shape)
                )
            } else {
                val embedSt = tensorByName[Gemma4TensorNames.TOKEN_EMBEDDINGS]
                    ?: error("Missing both output.weight and token_embd.weight")
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, embedSt, metadata)
                onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
                // Weight-tied output shares the embedding's quant type — the
                // MemSeg converter needs this callback to convert the shared
                // tensor into a matmul-ready layout. Without it, the output
                // stays as a 1-D byte blob and linearProject fails.
                if ((quantPolicy == QuantPolicy.RAW_BYTES || quantPolicy == QuantPolicy.NATIVE_OPTIMIZED) && embedSt.tensorType != GGMLQuantizationType.F32) {
                    quantCallback?.invoke(Gemma4TensorNames.OUTPUT_WEIGHT, embedSt.tensorType)
                }
                logicalShapeCallback?.invoke(
                    Gemma4TensorNames.OUTPUT_WEIGHT,
                    reversedShape(embedSt.shape)
                )
            }

            loadOptionalStreamingTensors(
                ctx, dtype, reader, tensorByName, onTensorLoaded, metadata,
                quantCallback, logicalShapeCallback
            )

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
        // GGUF uses attention.key_length_swa for sliding head dim, attention.key_length for global
        val headDim = fields["$prefix.attention.key_length_swa"]?.scalarInt()
            ?: fields["$prefix.attention.head_dim"]?.scalarInt()
            ?: 256
        val globalHeadDim = fields["$prefix.attention.key_length"]?.scalarInt()
            ?: fields["$prefix.attention.global_head_dim"]?.scalarInt()
            ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)
        val ffnField = fields["$prefix.feed_forward_length"]
        val perLayerIntermediateSize: List<Int> = ffnField
            ?.let { runCatching { it.intListValue() }.getOrNull() }
            ?: emptyList()
        val intermediateSize = ffnField?.scalarInt()
            ?: perLayerIntermediateSize.firstOrNull()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.scalarInt()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        val kvSharedLayers = fields["$prefix.attention.shared_kv_layers"]?.scalarInt()
            ?: fields["$prefix.kv_shared_layers"]?.scalarInt()
            ?: Gemma4ModelMetadata.DEFAULT_KV_SHARED_LAYERS
        val perLayerEmbeddingLength = fields["$prefix.embedding_length_per_layer_input"]?.scalarInt() ?: 0

        val layerTypes = extractLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.scalarFloat() ?: 1000000f
        // GGUF uses rope.freq_base_swa for sliding window RoPE base
        val ropeBaseLocal = fields["$prefix.rope.freq_base_swa"]?.scalarFloat()
            ?: fields["$prefix.rope.freq_base_local"]?.scalarFloat()
            ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.scalarFloat() ?: 1.0f
        // GGUF does not carry partial_rotary_factor for Gemma 4. HF config for
        // google/gemma-4-* says 0.25 for full_attention; use that default for
        // any gemma* architecture. Non-gemma archs keep the historical 1.0.
        val partialRotaryDefault = if (arch.startsWith("gemma")) 0.25f else 1.0f
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.scalarFloat()
            ?: partialRotaryDefault
        val finalLogitSoftcapping = fields["$prefix.final_logit_softcapping"]?.scalarFloat() ?: 0f

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
            maxPositionEmbeddings = contextLength,
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            perLayerIntermediateSize = perLayerIntermediateSize,
            finalLogitSoftcapping = finalLogitSoftcapping
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
        val headDim = fields["$prefix.attention.key_length_swa"]?.toIntValue()
            ?: fields["$prefix.attention.head_dim"]?.toIntValue()
            ?: 256
        val globalHeadDim = fields["$prefix.attention.key_length"]?.toIntValue()
            ?: fields["$prefix.attention.global_head_dim"]?.toIntValue()
            ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)
        val ffnField = fields["$prefix.feed_forward_length"]
        val perLayerIntermediateSize: List<Int> = (ffnField as? List<*>)
            ?.mapNotNull { (it as? Number)?.toInt() }
            ?: emptyList()
        val intermediateSize = ffnField?.toIntValue()
            ?: perLayerIntermediateSize.firstOrNull()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.toIntValue()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        val kvSharedLayers = fields["$prefix.attention.shared_kv_layers"]?.toIntValue()
            ?: fields["$prefix.kv_shared_layers"]?.toIntValue()
            ?: Gemma4ModelMetadata.DEFAULT_KV_SHARED_LAYERS
        val perLayerEmbeddingLength = fields["$prefix.embedding_length_per_layer_input"]?.toIntValue() ?: 0

        val layerTypes = extractStreamingLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.toFloatValue() ?: 1000000f
        val ropeBaseLocal = fields["$prefix.rope.freq_base_swa"]?.toFloatValue()
            ?: fields["$prefix.rope.freq_base_local"]?.toFloatValue()
            ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.toFloatValue() ?: 1.0f
        // See the non-streaming metadata parse above — Gemma 4 defaults
        // partial_rotary_factor to 0.25 (GGUF doesn't store it).
        val partialRotaryDefault = if (arch.startsWith("gemma")) 0.25f else 1.0f
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.toFloatValue()
            ?: partialRotaryDefault
        val finalLogitSoftcapping = fields["$prefix.final_logit_softcapping"]?.toFloatValue() ?: 0f

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
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            perLayerIntermediateSize = perLayerIntermediateSize,
            finalLogitSoftcapping = finalLogitSoftcapping,
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
        fields["$prefix.attention.sliding_window_pattern"]?.let { field ->
            runCatching { field.boolListValue() }.getOrNull()?.let { bools ->
                if (bools.size == blockCount) {
                    return bools.map { if (it) "sliding_attention" else "full_attention" }
                }
            }
        }
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
        // sliding_window_pattern is a boolean list: True = sliding_attention, False = full_attention.
        val slidingPattern = fields["$prefix.attention.sliding_window_pattern"]
        if (slidingPattern is List<*>) {
            val bools = slidingPattern.mapNotNull { it as? Boolean }
            if (bools.size == blockCount) {
                return bools.map { if (it) "sliding_attention" else "full_attention" }
            }
        }
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

    /**
     * Convert GGUF's native-order tensor dims (ne[0] fastest-changing/innermost)
     * into the PyTorch-style `[outer, inner]` layout used by the runtime. For a
     * standard 2-D weight tensor this flips `[ne[0]=in, ne[1]=out]` into
     * `[out, in]`, matching `Gemma4WeightMapper.require2D(outDim, inDim)` and
     * the `createTensor` transpose that the `DEQUANTIZE_TO_FP32` path does
     * inline.
     */
    private fun reversedShape(ggufDims: List<UInt>): Shape =
        Shape(*ggufDims.map { it.toInt() }.reversed().toIntArray())

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
                Gemma4TensorNames.attnKNorm(layer),
                Gemma4TensorNames.postAttentionNorm(layer),
                Gemma4TensorNames.postFfwNorm(layer),
                Gemma4TensorNames.layerOutputScale(layer),
                Gemma4TensorNames.pleInpGate(layer),
                Gemma4TensorNames.plePostNorm(layer),
                Gemma4TensorNames.pleProj(layer)
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
        metadata: Gemma4ModelMetadata,
        quantCallback: ((String, GGMLQuantizationType) -> Unit)? = null,
        logicalShapeCallback: ((String, Shape) -> Unit)? = null
    ) {
        listOf(Gemma4TensorNames.ROPE_FREQS_REAL, Gemma4TensorNames.ROPE_FREQS_IMAG).forEach { name ->
            val st = tensorByName[name]
            if (st != null && st.tensorType == GGMLQuantizationType.F32) {
                val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                onTensorLoaded(name, tensor)
            }
        }

        // PLE top-level tensors. per_layer_token_embd is Q6_K on Gemma 4 E2B
        // and has > Int.MAX_VALUE elements (vocab × num_layers × ple_dim), so
        // the dequant-to-FP32 path in `tryLoadOptionalStreamingTensor` can't
        // materialize it. Route through `streamingTensorToTensor` which
        // respects the quant policy (NATIVE_OPTIMIZED keeps bytes as
        // Int8/ByteArray storage ≈ 1.8 GB for E2B, fits in a single JVM
        // ByteArray). PerLayerEmbedding.compute dequants rows on demand.
        // per_layer_token_embd gets a dedicated path because it's the one
        // optional tensor large enough that avoiding a single extra byte
        // copy matters (Q6_K on E2B = 1.8 GB). Bytes go straight into a
        // GemmaPerLayerTokenEmbedTensorData row-dequant wrapper so the
        // ByteTensorDataImpl.data.copyOf() roundtrip is skipped.
        tensorByName[Gemma4TensorNames.PER_LAYER_TOKEN_EMBD]?.let { st ->
            val isPackedQuant = st.tensorType in setOf(
                GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K,
                GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K
            )
            if (quantPolicy == QuantPolicy.NATIVE_OPTIMIZED && isPackedQuant) {
                val bytes = reader.loadTensorData(st)
                val logicalShape = reversedShape(st.shape)
                @Suppress("UNCHECKED_CAST")
                val data = GemmaPerLayerTokenEmbedTensorData(logicalShape, st.tensorType, bytes)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
                val tensor: Tensor<T, V> = ctx.fromData(data, dtype)
                onTensorLoaded(Gemma4TensorNames.PER_LAYER_TOKEN_EMBD, tensor)
                quantCallback?.invoke(Gemma4TensorNames.PER_LAYER_TOKEN_EMBD, st.tensorType)
                logicalShapeCallback?.invoke(Gemma4TensorNames.PER_LAYER_TOKEN_EMBD, logicalShape)
            } else {
                tryLoadOptionalStreamingTensor(ctx, dtype, reader, st, Gemma4TensorNames.PER_LAYER_TOKEN_EMBD, onTensorLoaded)
            }
        }

        listOf(
            Gemma4TensorNames.PER_LAYER_MODEL_PROJ,
            Gemma4TensorNames.PER_LAYER_PROJ_NORM
        ).forEach { name ->
            val st = tensorByName[name] ?: return@forEach
            tryLoadOptionalStreamingTensor(ctx, dtype, reader, st, name, onTensorLoaded)
        }

        repeat(metadata.blockCount) { layer ->
            listOf(
                Gemma4TensorNames.perLayerInput(layer),
                Gemma4TensorNames.perLayerOutput(layer),
                Gemma4TensorNames.attnQNorm(layer),
                Gemma4TensorNames.attnKNorm(layer),
                Gemma4TensorNames.postAttentionNorm(layer),
                Gemma4TensorNames.postFfwNorm(layer),
                Gemma4TensorNames.layerOutputScale(layer),
                Gemma4TensorNames.pleInpGate(layer),
                Gemma4TensorNames.plePostNorm(layer),
                Gemma4TensorNames.pleProj(layer)
            ).forEach { name ->
                val st = tensorByName[name] ?: return@forEach
                // Route Q-series tensors through streamingTensorToTensor so
                // NATIVE_OPTIMIZED keeps packed bytes (same path the required
                // tensors use). tryLoadOptionalStreamingTensor otherwise drops
                // Q-types silently when the policy isn't DEQUANTIZE_TO_FP32.
                val isPackedQuant = when (st.tensorType) {
                    GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
                    GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1,
                    GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1,
                    GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K,
                    GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                    GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K -> true
                    else -> false
                }
                if (quantPolicy == QuantPolicy.NATIVE_OPTIMIZED && isPackedQuant) {
                    val tensor: Tensor<T, V> = streamingTensorToTensor(ctx, dtype, reader, st, metadata)
                    onTensorLoaded(name, tensor)
                    quantCallback?.invoke(name, st.tensorType)
                    logicalShapeCallback?.invoke(name, reversedShape(st.shape))
                } else {
                    tryLoadOptionalStreamingTensor(ctx, dtype, reader, st, name, onTensorLoaded)
                }
            }
        }
    }

    /** Load an optional streaming tensor, skipping silently if it exceeds size limits. */
    private fun <T : DType, V> tryLoadOptionalStreamingTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        st: StreamingTensorInfo,
        name: String,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        // Dequantized FloatArray index is Int; tensors with > Int.MAX_VALUE elements
        // (e.g. PLE tables in quantized E2B/E4B) cannot be materialized as FP32 here.
        if (st.nElements > Int.MAX_VALUE.toLong()) return
        try {
            val shape = Shape(*st.shape.map { it.toInt() }.toIntArray())
            val bytes = reader.loadTensorData(st)
            val floats = when (st.tensorType) {
                GGMLQuantizationType.F32 -> bytesToFloatArray(bytes)
                GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
                GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
                else -> when (quantPolicy) {
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> dequantFromBytes(bytes, st.tensorType, st.nElements.toInt())
                    else -> null
                }
            }
            if (floats != null) {
                val tensor = createTensor<T, V>(ctx, dtype, shape, floats)
                onTensorLoaded(name, tensor)
            }
        } catch (_: IllegalArgumentException) {
            // Streaming reader size limits.
        } catch (_: IllegalStateException) {
            // kotlinx-io throws IllegalStateException ("Can't create an array of size N")
            // when a dequant output would exceed the JVM FloatArray size cap.
            // Typed as IllegalStateException so this stays in commonMain (the old
            // NegativeArraySizeException is JVM-only and breaks native/JS/WASM compile).
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
                    QuantPolicy.RAW_BYTES -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes = DequantOps.toByteArray(raw, rt.name)
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // See streaming counterpart: 1-D byte shape so the factory
                        // accepts packed quantized bytes.
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes = DequantOps.toByteArray(raw, rt.name)
                        val byteShape = Shape(bytes.size)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(byteShape, Int8::class, bytes) as Tensor<T, V>
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
        metadata: Gemma4ModelMetadata
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
                    QuantPolicy.RAW_BYTES -> {
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        // Store raw quantized bytes with a 1-D byte shape (matches
                        // LlamaWeightLoader). The factory would otherwise reject the
                        // 2-D logical shape because `byte count != elements`. Downstream
                        // converters (MemSegWeightConverter for Llama, GemmaMemSegConverter
                        // for Gemma) recover the logical shape from the metadata.
                        val byteShape = Shape(bytes.size)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromByteArray<Int8, Byte>(byteShape, Int8::class, bytes) as Tensor<T, V>
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

    // ============== Helper methods ==============

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray = DequantOps.bytesToFloatArray(bytes)
    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantF16FromBytes(bytes)
    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantBF16FromBytes(bytes)

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: return 2304
        // GGUF stores shapes in column-major order; embedding is the smaller dimension
        return token.shape.map { it.toInt() }.minOrNull() ?: 2304
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: return 262144
        // GGUF stores shapes in column-major order; vocab is the larger dimension
        return token.shape.map { it.toInt() }.maxOrNull() ?: 262144
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

    private fun ReaderField.stringListValue(): List<String> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { it as? String }
    }

    private fun ReaderField.intListValue(): List<Int> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { (it as? Number)?.toInt() }
    }

    private fun ReaderField.boolListValue(): List<Boolean> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        return (part as List<*>).mapNotNull { it as? Boolean }
    }

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
}
