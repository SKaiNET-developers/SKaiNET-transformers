package sk.ainet.models.apertus

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
 * Loads Apertus weights from GGUF files.
 *
 * Compared to the LLaMA loader, Apertus has:
 * - QK-norm tensors: `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight`
 * - xIELU scalar params: `blk.N.mlp.act_fn.{alpha_p,alpha_n,beta,eps}` (scalar BF16/F32)
 * - No `ffn_gate` (ungated MLP)
 * - Metadata prefix `apertus.*` instead of `llama.*`
 */
public class ApertusWeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    private val preTransposed: Boolean = false
) {

    public companion object {
        public fun fromSource(
            sourceProvider: () -> Source,
            quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
            preTransposed: Boolean = false
        ): ApertusWeightLoader = ApertusWeightLoader(
            sourceProvider = sourceProvider,
            randomAccessProvider = null,
            quantPolicy = quantPolicy,
            preTransposed = preTransposed
        )

        public fun fromRandomAccess(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
            preTransposed: Boolean = false
        ): ApertusWeightLoader = ApertusWeightLoader(
            sourceProvider = null,
            randomAccessProvider = randomAccessProvider,
            quantPolicy = quantPolicy,
            preTransposed = preTransposed
        )
    }

    /**
     * Load all weights into a map with metadata.
     */
    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): ApertusWeights<T, V> {
        return if (randomAccessProvider != null) {
            loadFromStreamingGguf(ctx, dtype)
        } else {
            loadFromGguf(ctx, dtype)
        }
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): ApertusWeights<T, V> = loadToMap(ctx, T::class)

    // ============== Sequential loading ==============

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): ApertusWeights<T, V> {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Apertus GGUF loader supports FP32 and FP16 (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor."
        }

        val reader = sourceProvider.invoke().buffered().use { src ->
            GGUFReader(src, loadTensorData = true)
        }

        val metadata = metadataFromGguf(reader.fields, reader.tensors)
        validateMetadata(metadata)

        val tensorByName = reader.tensors.associateBy { it.name }
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val xieluParams = mutableMapOf<Int, ApertusXIELUParams>()

        // Load required tensors
        requiredTensorNames(metadata).forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            byName[name] = readerTensorToTensor(ctx, dtype, reader, rt)
        }

        // Load optional rope_freqs tensor
        tensorByName[ApertusTensorNames.ROPE_FREQS]?.let { rt ->
            byName[ApertusTensorNames.ROPE_FREQS] = readerTensorToTensor(ctx, dtype, reader, rt)
        }

        // Extract xIELU params: try metadata fields first, then per-layer tensors
        extractXIELUParams(reader.fields, metadata.blockCount, xieluParams)
        if (xieluParams.isEmpty()) {
            extractXIELUParamsFromReader(reader, tensorByName, metadata.blockCount, xieluParams)
        }

        return ApertusWeights(metadata, byName, xieluParams, preTransposed)
    }

    // ============== Streaming loading ==============

    private fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): ApertusWeights<T, V> {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Apertus GGUF loader supports FP32 and FP16 (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor."
        }

        val source = randomAccessProvider.invoke()
        return StreamingGGUFReader.open(source).use { reader ->
            val metadata = metadataFromStreamingGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)

            val tensorByName = reader.tensors.associateBy { it.name }
            val byName = linkedMapOf<String, Tensor<T, V>>()
            val xieluParams = mutableMapOf<Int, ApertusXIELUParams>()

            requiredTensorNames(metadata).forEach { name ->
                val st = tensorByName[name]
                    ?: error("Missing required tensor in GGUF payload: $name")
                byName[name] = streamingTensorToTensor(ctx, dtype, reader, st)
            }

            // Load optional rope_freqs tensor
            tensorByName[ApertusTensorNames.ROPE_FREQS]?.let { st ->
                byName[ApertusTensorNames.ROPE_FREQS] = streamingTensorToTensor(ctx, dtype, reader, st)
            }

            // Extract xIELU params: try metadata fields first, then per-layer tensors
            extractXIELUParamsFromStreamingMeta(reader.fields, metadata.blockCount, xieluParams)
            if (xieluParams.isEmpty()) {
                extractXIELUParamsFromStreaming(reader, tensorByName, metadata.blockCount, xieluParams)
            }

            ApertusWeights(metadata, byName, xieluParams, preTransposed)
        }
    }

    // ============== xIELU parameter extraction ==============

    /**
     * Extract xIELU params from GGUF metadata fields.
     *
     * Fields are arrays of FLOAT32 with one value per layer:
     * xielu.alpha_p, xielu.alpha_n, xielu.beta, xielu.eps
     */
    private fun extractXIELUParams(
        fields: Map<String, ReaderField>,
        blockCount: Int,
        out: MutableMap<Int, ApertusXIELUParams>
    ) {
        val alphaPField = fields["xielu.alpha_p"] ?: return
        val alphaNField = fields["xielu.alpha_n"] ?: return
        val betaField = fields["xielu.beta"] ?: return
        val epsField = fields["xielu.eps"] ?: return

        val alphaPArr = alphaPField.floatArray()
        val alphaNArr = alphaNField.floatArray()
        val betaArr = betaField.floatArray()
        val epsArr = epsField.floatArray()

        for (layer in 0 until blockCount) {
            out[layer] = ApertusXIELUParams(
                alphaP = alphaPArr.getOrElse(layer) { alphaPArr.first() },
                alphaN = alphaNArr.getOrElse(layer) { alphaNArr.first() },
                beta = betaArr.getOrElse(layer) { betaArr.first() },
                eps = epsArr.getOrElse(layer) { epsArr.first() }
            )
        }
    }

    /**
     * Extract xIELU params from streaming GGUF metadata.
     *
     * Values are per-layer arrays (one FLOAT32 per layer).
     */
    private fun extractXIELUParamsFromStreamingMeta(
        fields: Map<String, Any?>,
        blockCount: Int,
        out: MutableMap<Int, ApertusXIELUParams>
    ) {
        val alphaPArr = fields["xielu.alpha_p"]?.asFloatArray() ?: return
        val alphaNArr = fields["xielu.alpha_n"]?.asFloatArray() ?: return
        val betaArr = fields["xielu.beta"]?.asFloatArray() ?: return
        val epsArr = fields["xielu.eps"]?.asFloatArray() ?: return

        for (layer in 0 until blockCount) {
            out[layer] = ApertusXIELUParams(
                alphaP = alphaPArr.getOrElse(layer) { alphaPArr.first() },
                alphaN = alphaNArr.getOrElse(layer) { alphaNArr.first() },
                beta = betaArr.getOrElse(layer) { betaArr.first() },
                eps = epsArr.getOrElse(layer) { epsArr.first() }
            )
        }
    }

    /** Fallback: extract xIELU from per-layer tensors (if stored as tensors). */
    private fun extractXIELUParamsFromReader(
        reader: GGUFReader,
        tensorByName: Map<String, ReaderTensor>,
        blockCount: Int,
        out: MutableMap<Int, ApertusXIELUParams>
    ) {
        for (layer in 0 until blockCount) {
            val alphaP = readScalarFromReader(reader, tensorByName, xieluTensorName(layer, "alpha_p"))
            val alphaN = readScalarFromReader(reader, tensorByName, xieluTensorName(layer, "alpha_n"))
            val beta = readScalarFromReader(reader, tensorByName, xieluTensorName(layer, "beta"))
            val eps = readScalarFromReader(reader, tensorByName, xieluTensorName(layer, "eps"))
            out[layer] = ApertusXIELUParams(alphaP, alphaN, beta, eps)
        }
    }

    private fun readScalarFromReader(
        reader: GGUFReader,
        tensorByName: Map<String, ReaderTensor>,
        name: String
    ): Float {
        val rt = tensorByName[name] ?: error("Missing xIELU tensor: $name")
        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                (raw as List<Float>).first()
            }
            GGMLQuantizationType.BF16 -> {
                DequantOps.dequantBF16(raw).first()
            }
            GGMLQuantizationType.F16 -> {
                DequantOps.dequantF16(raw).first()
            }
            else -> error("Unexpected type ${rt.tensorType} for scalar xIELU tensor $name")
        }
    }

    private fun extractXIELUParamsFromStreaming(
        reader: StreamingGGUFReader,
        tensorByName: Map<String, StreamingTensorInfo>,
        blockCount: Int,
        out: MutableMap<Int, ApertusXIELUParams>
    ) {
        for (layer in 0 until blockCount) {
            val alphaP = readScalarFromStreaming(reader, tensorByName, xieluTensorName(layer, "alpha_p"))
            val alphaN = readScalarFromStreaming(reader, tensorByName, xieluTensorName(layer, "alpha_n"))
            val beta = readScalarFromStreaming(reader, tensorByName, xieluTensorName(layer, "beta"))
            val eps = readScalarFromStreaming(reader, tensorByName, xieluTensorName(layer, "eps"))
            out[layer] = ApertusXIELUParams(alphaP, alphaN, beta, eps)
        }
    }

    private fun readScalarFromStreaming(
        reader: StreamingGGUFReader,
        tensorByName: Map<String, StreamingTensorInfo>,
        name: String
    ): Float {
        val st = tensorByName[name] ?: error("Missing xIELU tensor: $name")
        val bytes = reader.loadTensorData(st)
        return when (st.tensorType) {
            GGMLQuantizationType.F32 -> DequantOps.bytesToFloatArray(bytes).first()
            GGMLQuantizationType.BF16 -> DequantOps.dequantBF16FromBytes(bytes).first()
            GGMLQuantizationType.F16 -> DequantOps.dequantF16FromBytes(bytes).first()
            else -> error("Unexpected type ${st.tensorType} for scalar xIELU tensor $name")
        }
    }

    private fun xieluTensorName(layer: Int, param: String): String =
        "blk.$layer.mlp.act_fn.$param"

    // ============== Metadata extraction ==============

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
        tensors: List<ReaderTensor>
    ): ApertusModelMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "apertus"
        val prefix = arch // "apertus" or "llama" depending on how the GGUF was exported

        val embeddingLength = fields["$prefix.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.scalarInt() ?: 2048
        val blockCount = fields["$prefix.block_count"]?.scalarInt() ?: 0
        val headCount = fields["$prefix.attention.head_count"]?.scalarInt() ?: 0
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.scalarInt() ?: headCount
        val feedForwardLength = fields["$prefix.feed_forward_length"]?.scalarInt() ?: 0
        val ropeDim = fields["$prefix.rope.dimension_count"]?.scalarInt()
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)
        val ropeTheta = fields["$prefix.rope.freq_base"]?.scalarFloat() ?: 12000000f
        val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.scalarFloat() ?: 1e-5f
        val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.scalarInt() ?: 1
        val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.scalarInt() ?: 2

        return ApertusModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLength = feedForwardLength,
            ropeDimensionCount = ropeDim,
            vocabSize = vocabSize,
            ropeTheta = ropeTheta,
            rmsNormEps = rmsNormEps,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId
        )
    }

    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): ApertusModelMetadata {
        val arch = (fields["general.architecture"] as? String) ?: "apertus"
        val prefix = arch

        val embeddingLength = fields["$prefix.embedding_length"]?.toIntValue()
            ?: inferEmbeddingFromStreamingTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.toIntValue() ?: 2048
        val blockCount = fields["$prefix.block_count"]?.toIntValue() ?: 0
        val headCount = fields["$prefix.attention.head_count"]?.toIntValue() ?: 0
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.toIntValue() ?: headCount
        val feedForwardLength = fields["$prefix.feed_forward_length"]?.toIntValue() ?: 0
        val ropeDim = fields["$prefix.rope.dimension_count"]?.toIntValue()
        val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)
        val ropeTheta = fields["$prefix.rope.freq_base"]?.toFloatValue() ?: 12000000f
        val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.toFloatValue() ?: 1e-5f
        val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.toIntValue() ?: 1
        val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.toIntValue() ?: 2

        return ApertusModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLength = feedForwardLength,
            ropeDimensionCount = ropeDim,
            vocabSize = vocabSize,
            ropeTheta = ropeTheta,
            rmsNormEps = rmsNormEps,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId
        )
    }

    private fun validateMetadata(metadata: ApertusModelMetadata) {
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: ApertusModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += ApertusTensorNames.TOKEN_EMBEDDINGS
        names += ApertusTensorNames.OUTPUT_NORM
        names += ApertusTensorNames.OUTPUT_WEIGHT

        repeat(metadata.blockCount) { layer ->
            names += ApertusTensorNames.attnNorm(layer)
            names += ApertusTensorNames.attnQ(layer)
            names += ApertusTensorNames.attnK(layer)
            names += ApertusTensorNames.attnV(layer)
            names += ApertusTensorNames.attnOut(layer)
            names += ApertusTensorNames.attnQNorm(layer)
            names += ApertusTensorNames.attnKNorm(layer)
            names += ApertusTensorNames.ffnNorm(layer)
            names += ApertusTensorNames.ffnDown(layer)
            names += ApertusTensorNames.ffnUp(layer)
        }
        return names
    }

    // ============== Tensor conversion ==============

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                val floats = (if (rt.data.isEmpty()) reader.materialize(rt) else rt.data) as List<Float>
                createTensor(ctx, dtype, shape, floats.toFloatArray())
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "F16/BF16 tensor ${rt.name} requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
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

            GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES, QuantPolicy.NATIVE_OPTIMIZED -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                        val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                        val floats = DequantOps.dequantFromBytes(bytes, rt.tensorType, rt.nElements)
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            else -> {
                println("WARNING: Unhandled tensor type ${rt.tensorType} for '${rt.name}'. Storing as raw bytes.")
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes: ByteArray = DequantOps.toByteArray(raw, rt.name)
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> streamingTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingGGUFReader,
        st: StreamingTensorInfo
    ): Tensor<T, V> {
        val shape = Shape(*st.shape.map { it.toInt() }.toIntArray())
        val bytes = reader.loadTensorData(st)

        return when (st.tensorType) {
            GGMLQuantizationType.F32 -> {
                val floats = DequantOps.bytesToFloatArray(bytes)
                createTensor(ctx, dtype, shape, floats)
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES -> {
                        require(dtype == Int8::class) {
                            "F16/BF16 tensor ${st.name} requires dtype Int8 with quantPolicy=RAW_BYTES"
                        }
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32,
                    QuantPolicy.NATIVE_OPTIMIZED -> {
                        val floats = when (st.tensorType) {
                            GGMLQuantizationType.F16 -> DequantOps.dequantF16FromBytes(bytes)
                            GGMLQuantizationType.BF16 -> DequantOps.dequantBF16FromBytes(bytes)
                            else -> error("Unreachable")
                        }
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q8_1,
            GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K,
            GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0 -> {
                when (quantPolicy) {
                    QuantPolicy.RAW_BYTES, QuantPolicy.NATIVE_OPTIMIZED -> {
                        ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
                    }
                    QuantPolicy.DEQUANTIZE_TO_FP32 -> {
                        val floats = DequantOps.dequantFromBytes(bytes, st.tensorType, st.nElements.toInt())
                        createTensor(ctx, dtype, shape, floats)
                    }
                }
            }

            else -> {
                println("WARNING: Unhandled tensor type ${st.tensorType} for '${st.name}'. Storing as raw bytes.")
                ctx.fromByteArray<Int8, Byte>(shape, Int8::class, bytes) as Tensor<T, V>
            }
        }
    }

    /**
     * Create a tensor from dequantized float data.
     *
     * For 2D tensors from GGUF (stored column-major with shape [out, in]):
     * - Normal mode: transposes to row-major [in, out] (requires `.t()` in runtime)
     * - Pre-transposed mode: interprets column-major as row-major [in, out] directly,
     *   skipping the data transpose. The weights can then be used directly in matmul
     *   without `.t()`, saving ~50% memory.
     */
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
            if (preTransposed) {
                // Column-major [out, in] is equivalent to row-major [in, out]
                // Skip data transpose — weights are already in matmul-ready layout
                val newShape = Shape(cols, rows)
                ctx.fromFloatArray<T, Float>(newShape, dtype, data) as Tensor<T, V>
            } else {
                val transposed = DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)
                val newShape = Shape(cols, rows)
                ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, V>
            }
        } else {
            ctx.fromFloatArray<T, Float>(originalShape, dtype, data) as Tensor<T, V>
        }
    }

    // ============== Field helpers ==============

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
            is Int -> value.toFloat()
            is UInt -> value.toFloat()
            is Long -> value.toFloat()
            else -> error("Unsupported float type ${value::class} for field $name")
        }
    }

    /**
     * Extract a float array from a ReaderField (GGUF ARRAY of FLOAT32).
     * Each element is stored as a separate part; data indices point to them.
     */
    private fun ReaderField.floatArray(): FloatArray {
        return FloatArray(data.size) { idx ->
            val partIdx = data[idx]
            val part = parts.getOrNull(partIdx) ?: error("Missing part $partIdx for field $name")
            val value = (part as List<*>).firstOrNull()
                ?: error("Empty part for field $name at index $idx")
            when (value) {
                is Float -> value
                is Double -> value.toFloat()
                is Number -> value.toFloat()
                else -> error("Unsupported array element type ${value::class} for field $name")
            }
        }
    }

    private fun ReaderField.stringValue(): String {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val bytes = (part as List<*>).mapNotNull {
            when (it) {
                is UByte -> it.toByte()
                is Byte -> it
                else -> null
            }
        }
        return bytes.toByteArray().decodeToString()
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
        is Int -> this.toFloat()
        is Long -> this.toFloat()
        else -> null
    }

    /**
     * Convert a streaming metadata value (array or scalar) to a FloatArray.
     */
    @Suppress("UNCHECKED_CAST")
    private fun Any?.asFloatArray(): FloatArray? = when (this) {
        is FloatArray -> this
        is List<*> -> FloatArray(size) { i ->
            when (val v = get(i)) {
                is Float -> v
                is Double -> v.toFloat()
                is Number -> v.toFloat()
                else -> return null
            }
        }
        is Float -> floatArrayOf(this)
        is Double -> floatArrayOf(this.toFloat())
        else -> null
    }

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == ApertusTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == ApertusTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == ApertusTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == ApertusTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }
}

// ============== Convenience top-level loaders ==============

/**
 * Load Apertus runtime weights from a GGUF source (sequential).
 */
public suspend fun <T : DType> loadApertusRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    preTransposed: Boolean = false
): ApertusRuntimeWeights<T> {
    val loader = ApertusWeightLoader.fromSource(
        sourceProvider = sourceProvider,
        quantPolicy = quantPolicy,
        preTransposed = preTransposed
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    return ApertusWeightMapper.map(loaded)
}

public suspend fun loadApertusRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    preTransposed: Boolean = false
): ApertusRuntimeWeights<FP32> = loadApertusRuntimeWeights(ctx, sourceProvider, FP32::class, quantPolicy, preTransposed)

/**
 * Load Apertus runtime weights from a GGUF source (streaming, for large files).
 */
public suspend fun <T : DType> loadApertusRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    preTransposed: Boolean = false
): ApertusRuntimeWeights<T> {
    val loader = ApertusWeightLoader.fromRandomAccess(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = quantPolicy,
        preTransposed = preTransposed
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    return ApertusWeightMapper.map(loaded)
}

public suspend fun loadApertusRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    preTransposed: Boolean = false
): ApertusRuntimeWeights<FP32> = loadApertusRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, quantPolicy, preTransposed)
