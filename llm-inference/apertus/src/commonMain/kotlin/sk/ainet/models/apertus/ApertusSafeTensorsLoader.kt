package sk.ainet.models.apertus

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.safetensors.ShardedTensorInfo
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.io.safetensors.StreamingSafeTensorInfo
import sk.ainet.io.safetensors.readTextFile
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Loads Apertus weights from HuggingFace SafeTensors format (sharded).
 *
 * Handles:
 * - HuggingFace → GGUF tensor name mapping
 * - BF16/F16/F32 dequantization to FP32
 * - Shape normalization ([1, dim] norms → [dim])
 * - Scalar tensor extraction for xIELU parameters
 * - Tied word embeddings (output.weight = token_embd.weight)
 */
public class ApertusSafeTensorsLoader(
    private val indexPath: String
) {

    /**
     * Load weights from sharded SafeTensors files.
     */
    public suspend fun <T : DType> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): ApertusWeights<T, Float> {
        require(dtype == FP32::class) {
            "SafeTensors loader currently only supports FP32 (got ${dtype.simpleName})"
        }

        val basePath = indexPath.substringBeforeLast("/")
        val configPath = if (basePath.isEmpty()) "config.json" else "$basePath/config.json"
        val configJson = readTextFile(configPath)
            ?: error("config.json not found at $configPath")

        val metadata = ApertusConfigParser.parse(configJson)

        val reader = StreamingShardedSafeTensorsReader.openFromIndex(indexPath)

        return reader.use {
            loadFromReader(ctx, dtype, it, metadata)
        }
    }

    private fun <T : DType> loadFromReader(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        metadata: ApertusModelMetadata
    ): ApertusWeights<T, Float> {
        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()
        val xieluParams = mutableMapOf<Int, ApertusXIELUParams>()

        val tensorsByHfName = reader.tensors.associateBy { it.name }

        // Load global tensors
        loadGlobalTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName, metadata)

        // Load layer tensors
        for (layer in 0 until metadata.blockCount) {
            loadLayerTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName, xieluParams, layer)
        }

        return ApertusWeights(
            metadata = metadata,
            tensors = tensorsByGgufName,
            xieluParams = xieluParams
        )
    }

    private fun <T : DType> loadGlobalTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        metadata: ApertusModelMetadata
    ) {
        // Token embeddings
        val embedTokens = tensorsByHfName[HF_EMBED_TOKENS]
            ?: error("Missing tensor: $HF_EMBED_TOKENS")
        val embedTensor = loadAndConvertTensor<T>(ctx, dtype, reader, embedTokens, transpose = false)
        tensorsByGgufName[ApertusTensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        val norm = tensorsByHfName[HF_OUTPUT_NORM]
            ?: error("Missing tensor: $HF_OUTPUT_NORM")
        val normTensor = loadAndConvertTensor<T>(ctx, dtype, reader, norm, transpose = false)
        tensorsByGgufName[ApertusTensorNames.OUTPUT_NORM] = normTensor

        // Output weight (may be tied to embeddings)
        val lmHead = tensorsByHfName[HF_LM_HEAD]
        if (lmHead != null) {
            tensorsByGgufName[ApertusTensorNames.OUTPUT_WEIGHT] =
                loadAndConvertTensor(ctx, dtype, reader, lmHead, transpose = false)
        } else if (metadata.tiedEmbeddings) {
            tensorsByGgufName[ApertusTensorNames.OUTPUT_WEIGHT] = embedTensor
            println("  Tied: ${ApertusTensorNames.OUTPUT_WEIGHT} → ${ApertusTensorNames.TOKEN_EMBEDDINGS}")
        } else {
            error("Missing lm_head.weight and tie_word_embeddings is false")
        }
    }

    private fun <T : DType> loadLayerTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        xieluParams: MutableMap<Int, ApertusXIELUParams>,
        layer: Int
    ) {
        // Attention norm (Apertus uses "attention_layernorm")
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnNorm(layer), ApertusTensorNames.attnNorm(layer), transpose = false)

        // QKV + output projections
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnQ(layer), ApertusTensorNames.attnQ(layer))
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnK(layer), ApertusTensorNames.attnK(layer))
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnV(layer), ApertusTensorNames.attnV(layer))
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnO(layer), ApertusTensorNames.attnOut(layer))

        // QK-norm weights
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfQNorm(layer), ApertusTensorNames.attnQNorm(layer), transpose = false)
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfKNorm(layer), ApertusTensorNames.attnKNorm(layer), transpose = false)

        // FFN norm (Apertus uses "feedforward_layernorm")
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfFfnNorm(layer), ApertusTensorNames.ffnNorm(layer), transpose = false)

        // MLP weights (ungated: up + down only, no gate_proj)
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpUp(layer), ApertusTensorNames.ffnUp(layer))
        loadRequiredTensor(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpDown(layer), ApertusTensorNames.ffnDown(layer))

        // xIELU scalar parameters
        val alphaP = loadScalarParam(reader, tensorsByHfName, hfXieluAlphaP(layer))
        val alphaN = loadScalarParam(reader, tensorsByHfName, hfXieluAlphaN(layer))
        val beta = loadScalarParam(reader, tensorsByHfName, hfXieluBeta(layer))
        val eps = loadScalarParam(reader, tensorsByHfName, hfXieluEps(layer))
        xieluParams[layer] = ApertusXIELUParams(
            alphaP = alphaP,
            alphaN = alphaN,
            beta = beta,
            eps = eps
        )
    }

    private fun <T : DType> loadRequiredTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        hfName: String,
        ggufName: String,
        transpose: Boolean = false
    ) {
        val info = tensorsByHfName[hfName]
            ?: error("Missing required tensor: $hfName")
        val tensor = loadAndConvertTensor<T>(ctx, dtype, reader, info, transpose)
        tensorsByGgufName[ggufName] = tensor
        println("  Loaded: $hfName (${info.dtype} ${info.shape}) → $ggufName")
    }

    /**
     * Load a scalar parameter from a tensor with shape [] or [1].
     */
    private fun loadScalarParam(
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        hfName: String
    ): Float {
        val info = tensorsByHfName[hfName]
            ?: error("Missing scalar tensor: $hfName")
        val bytes = reader.loadTensorData(info)
        return when (info.dtype.uppercase()) {
            "BF16" -> dequantBF16FromBytes(bytes)[0]
            "F16" -> dequantF16FromBytes(bytes)[0]
            "F32" -> bytesToFloatArray(bytes)[0]
            else -> error("Unsupported dtype for scalar: ${info.dtype}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> loadAndConvertTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        info: ShardedTensorInfo,
        transpose: Boolean = false
    ): Tensor<T, Float> {
        val bytes = reader.loadTensorData(info)
        val shape = normalizeNormShape(info.shape)

        val floats = when (info.dtype.uppercase()) {
            "BF16" -> dequantBF16FromBytes(bytes)
            "F16" -> dequantF16FromBytes(bytes)
            "F32" -> bytesToFloatArray(bytes)
            else -> error("Unsupported SafeTensors dtype: ${info.dtype}")
        }

        return if (transpose && shape.rank == 2) {
            val rows = shape[0]
            val cols = shape[1]
            val transposed = transposeRowMajor(floats, rows, cols)
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed)
        } else {
            ctx.fromFloatArray<T, Float>(shape, dtype, floats)
        }
    }

    private fun normalizeNormShape(shape: List<Long>): Shape {
        return if (shape.size == 2 && shape[0] == 1L) {
            Shape(shape[1].toInt())
        } else if (shape.isEmpty()) {
            Shape(1) // scalar → [1]
        } else {
            Shape(*shape.map { it.toInt() }.toIntArray())
        }
    }

    // ========== Byte Conversion Helpers ==========

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray = DequantOps.bytesToFloatArray(bytes)
    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantF16FromBytes(bytes)
    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray = DequantOps.dequantBF16FromBytes(bytes)

    private fun transposeRowMajor(data: FloatArray, rows: Int, cols: Int): FloatArray {
        val out = FloatArray(data.size)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[c * rows + r] = data[r * cols + c]
            }
        }
        return out
    }

    // ========== HuggingFace Tensor Name Constants ==========

    private companion object {
        const val HF_EMBED_TOKENS = "model.embed_tokens.weight"
        const val HF_OUTPUT_NORM = "model.norm.weight"
        const val HF_LM_HEAD = "lm_head.weight"

        fun hfAttnNorm(layer: Int) = "model.layers.$layer.attention_layernorm.weight"
        fun hfAttnQ(layer: Int) = "model.layers.$layer.self_attn.q_proj.weight"
        fun hfAttnK(layer: Int) = "model.layers.$layer.self_attn.k_proj.weight"
        fun hfAttnV(layer: Int) = "model.layers.$layer.self_attn.v_proj.weight"
        fun hfAttnO(layer: Int) = "model.layers.$layer.self_attn.o_proj.weight"
        fun hfQNorm(layer: Int) = "model.layers.$layer.self_attn.q_norm.weight"
        fun hfKNorm(layer: Int) = "model.layers.$layer.self_attn.k_norm.weight"
        fun hfFfnNorm(layer: Int) = "model.layers.$layer.feedforward_layernorm.weight"
        fun hfMlpUp(layer: Int) = "model.layers.$layer.mlp.up_proj.weight"
        fun hfMlpDown(layer: Int) = "model.layers.$layer.mlp.down_proj.weight"
        fun hfXieluAlphaP(layer: Int) = "model.layers.$layer.mlp.act_fn.alpha_p"
        fun hfXieluAlphaN(layer: Int) = "model.layers.$layer.mlp.act_fn.alpha_n"
        fun hfXieluBeta(layer: Int) = "model.layers.$layer.mlp.act_fn.beta"
        fun hfXieluEps(layer: Int) = "model.layers.$layer.mlp.act_fn.eps"
    }
}

/**
 * Load Apertus weights from a single (non-sharded) SafeTensors file.
 */
public class ApertusSingleSafeTensorsLoader<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val metadata: ApertusModelMetadata
) {

    /**
     * Load weights into a flat tensor map with GGUF-canonical names.
     */
    public fun loadToMap(randomAccessProvider: () -> RandomAccessSource): ApertusWeights<T, Float> {
        val tensors = mutableMapOf<String, Tensor<T, Float>>()
        val xieluParams = mutableMapOf<Int, ApertusXIELUParams>()

        StreamingSafeTensorsReader.open(randomAccessProvider()).use { reader ->
            val tensorInfoMap = reader.tensors.associateBy { it.name }

            for (info in reader.tensors) {
                val hfName = info.name

                // Check if this is an xIELU scalar parameter
                val xieluMatch = XIELU_PATTERN.matchEntire(hfName)
                if (xieluMatch != null) {
                    val layer = xieluMatch.groupValues[1].toInt()
                    val param = xieluMatch.groupValues[2]
                    val value = loadScalarFromSingle(reader, info)
                    val current = xieluParams.getOrPut(layer) {
                        ApertusXIELUParams(0f, 0f, 0f, 0f)
                    }
                    xieluParams[layer] = when (param) {
                        "alpha_p" -> current.copy(alphaP = value)
                        "alpha_n" -> current.copy(alphaN = value)
                        "beta" -> current.copy(beta = value)
                        "eps" -> current.copy(eps = value)
                        else -> current
                    }
                    continue
                }

                val canonicalName = ApertusHfTensorNameMapper.toCanonical(hfName) ?: continue

                val tensor = loadTensorFromSingle(reader, info)
                if (tensor != null) {
                    tensors[canonicalName] = tensor
                    println("  Loaded: $hfName (${info.dtype} ${info.shape}) → $canonicalName")
                }
            }
        }

        // Handle tied embeddings
        if (metadata.tiedEmbeddings && !tensors.containsKey(ApertusTensorNames.OUTPUT_WEIGHT)) {
            val embedding = tensors[ApertusTensorNames.TOKEN_EMBEDDINGS]
                ?: error("tie_word_embeddings=true but token embedding not found")
            tensors[ApertusTensorNames.OUTPUT_WEIGHT] = embedding
        }

        return ApertusWeights<T, Float>(
            metadata = metadata,
            tensors = tensors,
            xieluParams = xieluParams
        )
    }

    /**
     * Load weights and return structured [ApertusRuntimeWeights].
     */
    public fun load(randomAccessProvider: () -> RandomAccessSource): ApertusRuntimeWeights<T> {
        return ApertusWeightMapper.map(loadToMap(randomAccessProvider))
    }

    private fun loadScalarFromSingle(
        reader: StreamingSafeTensorsReader,
        info: StreamingSafeTensorInfo
    ): Float {
        val bytes = reader.loadTensorData(info)
        return when (info.dataType) {
            sk.ainet.io.model.DataType.BFLOAT16 -> DequantOps.dequantBF16FromBytes(bytes)[0]
            sk.ainet.io.model.DataType.FLOAT16 -> DequantOps.dequantF16FromBytes(bytes)[0]
            sk.ainet.io.model.DataType.FLOAT32 -> DequantOps.bytesToFloatArray(bytes)[0]
            else -> error("Unsupported dtype for scalar: ${info.dtype}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun loadTensorFromSingle(
        reader: StreamingSafeTensorsReader,
        info: StreamingSafeTensorInfo
    ): Tensor<T, Float>? {
        val bytes = reader.loadTensorData(info)
        val floats = when (info.dataType) {
            sk.ainet.io.model.DataType.BFLOAT16 -> DequantOps.dequantBF16FromBytes(bytes)
            sk.ainet.io.model.DataType.FLOAT16 -> DequantOps.dequantF16FromBytes(bytes)
            sk.ainet.io.model.DataType.FLOAT32 -> DequantOps.bytesToFloatArray(bytes)
            else -> {
                println("WARNING: Skipping tensor '${info.name}' with unsupported dtype ${info.dtype}")
                return null
            }
        }
        val shape = if (info.shape.size == 2 && info.shape[0] == 1L) {
            Shape(info.shape[1].toInt())
        } else {
            Shape(*info.shape.map { it.toInt() }.toIntArray())
        }
        return ctx.fromFloatArray<T, Float>(shape, dtype, floats)
    }

    private companion object {
        val XIELU_PATTERN = Regex("""model\.layers\.(\d+)\.mlp\.act_fn\.(alpha_p|alpha_n|beta|eps)""")
    }
}

/**
 * Maps HuggingFace Apertus tensor names to GGUF canonical names.
 */
public object ApertusHfTensorNameMapper {

    private val LAYER_PATTERN = Regex("""model\.layers\.(\d+)\.(.+)""")

    public fun toCanonical(hfName: String): String? {
        return when (hfName) {
            "model.embed_tokens.weight" -> ApertusTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> ApertusTensorNames.OUTPUT_NORM
            "lm_head.weight" -> ApertusTensorNames.OUTPUT_WEIGHT
            else -> {
                val match = LAYER_PATTERN.matchEntire(hfName) ?: return null
                val layer = match.groupValues[1].toInt()
                when (match.groupValues[2]) {
                    "attention_layernorm.weight" -> ApertusTensorNames.attnNorm(layer)
                    "self_attn.q_proj.weight" -> ApertusTensorNames.attnQ(layer)
                    "self_attn.k_proj.weight" -> ApertusTensorNames.attnK(layer)
                    "self_attn.v_proj.weight" -> ApertusTensorNames.attnV(layer)
                    "self_attn.o_proj.weight" -> ApertusTensorNames.attnOut(layer)
                    "self_attn.q_norm.weight" -> ApertusTensorNames.attnQNorm(layer)
                    "self_attn.k_norm.weight" -> ApertusTensorNames.attnKNorm(layer)
                    "feedforward_layernorm.weight" -> ApertusTensorNames.ffnNorm(layer)
                    "mlp.up_proj.weight" -> ApertusTensorNames.ffnUp(layer)
                    "mlp.down_proj.weight" -> ApertusTensorNames.ffnDown(layer)
                    else -> null // Skip xIELU params (handled separately) and unknown tensors
                }
            }
        }
    }
}

// ========== Convenience Functions ==========

/**
 * Load Apertus runtime weights from sharded SafeTensors format.
 */
public suspend fun <T : DType> loadApertusRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String,
    dtype: KClass<T>
): ApertusRuntimeWeights<T> {
    val loader = ApertusSafeTensorsLoader(indexPath)
    val loaded = loader.loadToMap<T>(ctx, dtype)
    return ApertusWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadApertusRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String
): ApertusRuntimeWeights<FP32> = loadApertusRuntimeWeightsFromSafeTensors(ctx, indexPath, FP32::class)
