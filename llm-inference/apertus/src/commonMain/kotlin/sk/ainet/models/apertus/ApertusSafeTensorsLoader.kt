package sk.ainet.models.apertus

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.load
import sk.ainet.io.safetensors.ShardedSafeTensorsParametersLoader
import sk.ainet.io.safetensors.ShardedTensorInfo
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.io.safetensors.StreamingSafeTensorInfo
import sk.ainet.io.safetensors.readTextFile
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Loads Apertus weights from HuggingFace SafeTensors format (sharded).
 *
 * Engine delegation (SKaiNET#1246): reading AND per-tensor materialization ride the engine's
 * [ShardedSafeTensorsParametersLoader]. This class owns only the family policy — which HF
 * tensors are wanted (the `tensorFilter`), the HF → GGUF slot renaming, the `[1, dim]` norm
 * shape normalization, the xIELU scalar extraction, and tied embeddings — while every dtype
 * decision (BF16/F16 widening vs keep-native, F32 passthrough) is the engine's, driven by
 * [dtypePolicy] exactly as on the GGUF lane. No per-family dequant code remains here (#346).
 *
 * @param indexPath path to `model.safetensors.index.json`.
 * @param dtypePolicy narrow-float policy forwarded to
 *   [ShardedSafeTensorsParametersLoader.withPolicy]: `Any` (default) widens BF16/F16 to FP32;
 *   `Require(BF16)` / `Require(FP16)` keep the native encoding under an FP32-typed tensor.
 */
public class ApertusSafeTensorsLoader(
    private val indexPath: String,
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
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

        // Family policy as the engine loader's filter: only the tensors this family maps
        // (anything unmapped never materializes and is exempt from the fail-fast dtype pre-scan).
        val wanted = wantedHfNames(metadata.blockCount)
        val loader = ShardedSafeTensorsParametersLoader.withPolicy(
            indexPath = indexPath,
            policy = dtypePolicy,
            tensorFilter = { info: ShardedTensorInfo -> info.name in wanted },
        )
        val tensorsByHfName = linkedMapOf<String, Tensor<T, Float>>()
        loader.load<T, Float>(ctx, dtype) { name, tensor -> tensorsByHfName[name] = tensor }

        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()
        val xieluParams = mutableMapOf<Int, ApertusXIELUParams>()

        loadGlobalTensors(ctx, dtype, tensorsByHfName, tensorsByGgufName, metadata)
        for (layer in 0 until metadata.blockCount) {
            loadLayerTensors(ctx, dtype, tensorsByHfName, tensorsByGgufName, xieluParams, layer)
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
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        metadata: ApertusModelMetadata
    ) {
        // Token embeddings
        val embedTensor = normalizeNormShape(ctx, dtype, required(tensorsByHfName, HF_EMBED_TOKENS))
        tensorsByGgufName[ApertusTensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        tensorsByGgufName[ApertusTensorNames.OUTPUT_NORM] =
            normalizeNormShape(ctx, dtype, required(tensorsByHfName, HF_OUTPUT_NORM))

        // Output weight (may be tied to embeddings)
        val lmHead = tensorsByHfName[HF_LM_HEAD]
        if (lmHead != null) {
            tensorsByGgufName[ApertusTensorNames.OUTPUT_WEIGHT] = normalizeNormShape(ctx, dtype, lmHead)
        } else if (metadata.tiedEmbeddings) {
            tensorsByGgufName[ApertusTensorNames.OUTPUT_WEIGHT] = embedTensor
        } else {
            error("Missing lm_head.weight and tie_word_embeddings is false")
        }
    }

    private fun <T : DType> loadLayerTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        xieluParams: MutableMap<Int, ApertusXIELUParams>,
        layer: Int
    ) {
        for ((hfName, ggufName) in layerSlots(layer)) {
            tensorsByGgufName[ggufName] = normalizeNormShape(ctx, dtype, required(tensorsByHfName, hfName))
        }

        // xIELU scalar parameters (shape [] or [1]; consumed as plain floats).
        xieluParams[layer] = ApertusXIELUParams(
            alphaP = scalar(tensorsByHfName, hfXieluAlphaP(layer)),
            alphaN = scalar(tensorsByHfName, hfXieluAlphaN(layer)),
            beta = scalar(tensorsByHfName, hfXieluBeta(layer)),
            eps = scalar(tensorsByHfName, hfXieluEps(layer))
        )
    }

    private fun <T : DType> required(tensorsByHfName: Map<String, Tensor<T, Float>>, hfName: String): Tensor<T, Float> =
        tensorsByHfName[hfName] ?: error("Missing required tensor: $hfName")

    private fun <T : DType> scalar(tensorsByHfName: Map<String, Tensor<T, Float>>, hfName: String): Float {
        val tensor = tensorsByHfName[hfName] ?: error("Missing scalar tensor: $hfName")
        return tensor.data.copyToFloatArray()[0]
    }

    /**
     * Some exports store norms as `[1, dim]`; the runtime expects `[dim]`. A widened tensor is
     * re-wrapped over the same buffer (no copy); a keep-native tensor is widened for this one
     * defensive edge case, since the norm is tiny.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> normalizeNormShape(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        tensor: Tensor<T, Float>
    ): Tensor<T, Float> {
        val dims = tensor.shape.dimensions
        if (dims.size != 2 || dims[0] != 1) return tensor
        val newShape = Shape(dims[1])
        val data = tensor.data
        return if (data is FloatArrayTensorData<*>) {
            ctx.wrapFloatArray(newShape, dtype, (data as FloatArrayTensorData<T>).buffer)
        } else {
            ctx.fromFloatArray(newShape, dtype, data.copyToFloatArray())
        }
    }

    // ========== HuggingFace Tensor Name Constants ==========

    private companion object {
        const val HF_EMBED_TOKENS = "model.embed_tokens.weight"
        const val HF_OUTPUT_NORM = "model.norm.weight"
        const val HF_LM_HEAD = "lm_head.weight"

        /** Every HF tensor this family maps, for [blockCount] layers — the engine loader's allowlist. */
        fun wantedHfNames(blockCount: Int): Set<String> = buildSet {
            add(HF_EMBED_TOKENS); add(HF_OUTPUT_NORM); add(HF_LM_HEAD)
            for (layer in 0 until blockCount) {
                addAll(layerSlots(layer).map { it.first })
                add(hfXieluAlphaP(layer)); add(hfXieluAlphaN(layer)); add(hfXieluBeta(layer)); add(hfXieluEps(layer))
            }
        }

        /** Per-layer HF → GGUF slot pairs; every one is required. */
        fun layerSlots(layer: Int): List<Pair<String, String>> = listOf(
            // Attention norm (Apertus uses "attention_layernorm")
            hfAttnNorm(layer) to ApertusTensorNames.attnNorm(layer),
            // QKV + output projections
            hfAttnQ(layer) to ApertusTensorNames.attnQ(layer),
            hfAttnK(layer) to ApertusTensorNames.attnK(layer),
            hfAttnV(layer) to ApertusTensorNames.attnV(layer),
            hfAttnO(layer) to ApertusTensorNames.attnOut(layer),
            // QK-norm weights
            hfQNorm(layer) to ApertusTensorNames.attnQNorm(layer),
            hfKNorm(layer) to ApertusTensorNames.attnKNorm(layer),
            // FFN norm (Apertus uses "feedforward_layernorm")
            hfFfnNorm(layer) to ApertusTensorNames.ffnNorm(layer),
            // MLP weights (ungated: up + down only, no gate_proj)
            hfMlpUp(layer) to ApertusTensorNames.ffnUp(layer),
            hfMlpDown(layer) to ApertusTensorNames.ffnDown(layer),
        )

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
