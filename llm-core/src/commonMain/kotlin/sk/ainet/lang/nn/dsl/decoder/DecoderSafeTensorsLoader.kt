package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.safetensors.StreamingSafeTensorInfo
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.NarrowFloatDenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.coroutines.Continuation
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.coroutines.startCoroutine
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Loads LLaMA-family weights from a HuggingFace SafeTensors file and maps them to
 * the canonical GGUF tensor naming used by `LlamaWeightMapper`.
 *
 * Reading and materialization ride the engine (SKaiNET#1246): the file goes through
 * [SafeTensorsParametersLoader.withPolicy], which owns the BF16/F16 widening and
 * keep-native decisions for [dtypePolicy]. What stays family-side is the policy the
 * engine does not know about:
 *
 * - HuggingFace → GGUF tensor name mapping ([HfTensorNameMapper]); unmapped tensors are dropped
 * - Shape normalization ([1, dim] norms → [dim])
 * - **Input-major relayout of keep-native matmul weights** — see [narrowData]
 * - Tied word embeddings (output.weight = token_embd.weight)
 *
 * One file shape cannot go through the engine yet: the legacy `Q4` + `.qb` companion export
 * (a SKaiNET-specific format the engine has no dtype for), and files that carry non-float or
 * F64 tensors, which the engine's single-file loader would either reject or widen instead of
 * skipping. Those files take [loadLegacy], the pre-#1246 reader kept verbatim for them.
 * Engine follow-up: a `tensorFilter` on the single-file loader (the sharded one has it) would
 * let the standard tensors of such files ride the engine as well.
 *
 * @param dtypePolicy declarative dtype constraint. Default [DTypePolicy.Any]
 *   = widen everything to FP32. `Require(X)` / `Prefer(X)` / `OneOf` containing
 *   X keeps X-encoded source tensors packed, for X in {BF16, FP16}. The two
 *   formats are resolved independently — `Require(BF16)` still widens F16
 *   sources, since neither narrow format can be re-encoded as the other without
 *   a lossy round-trip. This is the engine's own `mapPolicyToNarrow` semantics.
 */
public class DecoderSafeTensorsLoader<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val metadata: GgufDecoderMetadata,
    private val tiedEmbeddings: Boolean = false,
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
) {

    /**
     * Returns `true` iff [dtypePolicy] wants BF16 weights kept in their packed form —
     * used by the legacy path only; the engine resolves this itself on the main path.
     */
    private val keepBf16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, BF16)

    /** As [keepBf16Native], for IEEE binary16 sources. Resolved independently. */
    private val keepFp16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, FP16)

    /**
     * Load weights from SafeTensors file into a flat tensor map with GGUF-canonical names.
     * Useful for feeding into [WeightMapper] with a [WeightNameResolver].
     */
    public fun loadToMap(randomAccessProvider: () -> RandomAccessSource): DecoderGgufWeights<T, Float> {
        // Header-only pass: decide whether the engine loader can consume this file.
        val header = StreamingSafeTensorsReader.open(randomAccessProvider()).use { it.tensors.toList() }
        val tensors = if (engineEligible(header)) {
            loadViaEngine(randomAccessProvider)
        } else {
            loadLegacy(randomAccessProvider)
        }

        // Handle tied embeddings: reuse token_embd as output.weight
        if (tiedEmbeddings && !tensors.containsKey(DecoderTensorNames.OUTPUT_WEIGHT)) {
            val embedding = tensors[DecoderTensorNames.TOKEN_EMBEDDINGS]
                ?: error("tie_word_embeddings=true but token embedding not found")
            tensors[DecoderTensorNames.OUTPUT_WEIGHT] = embedding
            println("  Tied: ${DecoderTensorNames.OUTPUT_WEIGHT} → ${DecoderTensorNames.TOKEN_EMBEDDINGS}")
        }

        return DecoderGgufWeights<T, Float>(
            metadata = metadata,
            tensors = tensors
        )
    }

    // ========== Engine path ==========

    /**
     * The engine's single-file loader delivers every tensor in the file and requires an FP32
     * dtype witness for float sources; it has no filter, so a file is engine-eligible only when
     * every tensor is one the engine materializes the way this loader always did (F32/F16/BF16).
     */
    private fun engineEligible(header: List<StreamingSafeTensorInfo>): Boolean =
        dtype == FP32::class && header.all { it.dataType in ENGINE_FLOAT_TYPES }

    private fun loadViaEngine(randomAccessProvider: () -> RandomAccessSource): MutableMap<String, Tensor<T, Float>> {
        val tensors = linkedMapOf<String, Tensor<T, Float>>()
        val loader = SafeTensorsParametersLoader.withPolicy(randomAccessProvider, dtypePolicy)
        runNonSuspending {
            loader.load<T, Float>(ctx, dtype) { hfName, tensor ->
                val canonicalName = HfTensorNameMapper.toCanonical(hfName) ?: return@load
                val adapted = adapt(canonicalName, tensor)
                tensors[canonicalName] = adapted
                println("  Loaded: $hfName ${tensor.shape} → $canonicalName ${adapted.shape}")
            }
        }
        return tensors
    }

    /**
     * Apply the family-side shape/layout policy to an engine-materialized tensor: norm shape
     * normalization, and the input-major relayout of keep-native matmul weights. Tensors that
     * need neither are returned as delivered (no copy).
     */
    @Suppress("UNCHECKED_CAST")
    private fun adapt(canonicalName: String, tensor: Tensor<T, Float>): Tensor<T, Float> {
        val targetShape = normalizeNormShape(tensor.shape)
        return when (val data = tensor.data) {
            is NarrowFloatDenseTensorData -> {
                val relaid = narrowData(canonicalName, targetShape, data.packedData, data.codec, existing = data)
                if (relaid === data) tensor else ctx.fromData(relaid as TensorData<T, Float>, dtype)
            }
            is FloatArrayTensorData<*> -> {
                if (targetShape == tensor.shape) tensor
                else ctx.wrapFloatArray<T, Float>(targetShape, dtype, data.buffer) as Tensor<T, Float>
            }
            else -> tensor
        }
    }

    /**
     * The engine's single-file `load` is `suspend` by interface contract but never actually
     * suspends (the reader is synchronous), so it can be driven to completion without a
     * coroutine runtime — which keeps [loadToMap] a plain function for the Java consumers
     * (`KLlamaJava`) and the non-suspending network loaders that call it.
     */
    private fun <R> runNonSuspending(block: suspend () -> R): R {
        var outcome: Result<R>? = null
        block.startCoroutine(Continuation(EmptyCoroutineContext) { outcome = it })
        return (outcome ?: error("engine SafeTensors load suspended unexpectedly")).getOrThrow()
    }

    // ========== Legacy path (Q4 + .qb export, non-float tensors, non-FP32 witnesses) ==========

    private fun loadLegacy(randomAccessProvider: () -> RandomAccessSource): MutableMap<String, Tensor<T, Float>> {
        val tensors = mutableMapOf<String, Tensor<T, Float>>()

        StreamingSafeTensorsReader.open(randomAccessProvider()).use { reader ->
            // Index all tensors by name for companion lookup
            val tensorInfoMap = reader.tensors.associateBy { it.name }

            // Process non-qb tensors
            for (info in reader.tensors) {
                if (info.name.endsWith(".qb")) continue  // handled as companion

                val canonicalName = HfTensorNameMapper.toCanonical(info.name) ?: continue

                val tensor = when (info.dataType) {
                    DataType.QUANT4 -> {
                        // Q4 with companion .qb scales → dequant to FP32
                        val qbName = info.name + ".qb"
                        val qbInfo = tensorInfoMap[qbName]
                        val rawQ4 = reader.loadTensorData(info)
                        val qbBytes = qbInfo?.let { reader.loadTensorData(it) }
                        val targetShape = inferShape(canonicalName, info.shape)
                        val floats = dequantQ4(rawQ4, qbBytes, info.shape, qbInfo?.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.BFLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val targetShape = normalizeNormShape(info.shape)
                        if (keepBf16Native) {
                            val data = narrowData(canonicalName, targetShape, bytes, Bf16Codec)
                            @Suppress("UNCHECKED_CAST")
                            ctx.fromData(data as TensorData<T, Float>, dtype) as Tensor<T, Float>
                        } else {
                            val floats = dequantBF16(bytes)
                            @Suppress("UNCHECKED_CAST")
                            ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                        }
                    }
                    DataType.FLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val targetShape = normalizeNormShape(info.shape)
                        if (keepFp16Native) {
                            val data = narrowData(canonicalName, targetShape, bytes, Fp16Codec)
                            @Suppress("UNCHECKED_CAST")
                            ctx.fromData(data as TensorData<T, Float>, dtype) as Tensor<T, Float>
                        } else {
                            val floats = dequantF16(bytes)
                            @Suppress("UNCHECKED_CAST")
                            ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                        }
                    }
                    DataType.FLOAT32 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = bytesToFloatArray(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    else -> {
                        println("WARNING: Skipping tensor '${info.name}' with unsupported dtype ${info.dtype}")
                        continue
                    }
                }

                tensors[canonicalName] = tensor
                println("  Loaded: ${info.name} (${info.dtype} ${info.shape}) → $canonicalName ${tensor.shape}")
            }
        }
        return tensors
    }

    /**
     * Build the KEEP_NATIVE storage for one narrow-float tensor, choosing its byte layout.
     *
     * Matmul weights are relaid **input-major** ([NarrowFloatInputMajorTensorData]). Weights are
     * stored `[out, in]` but the narrow matmul dispatch needs `[in, out]`, so `Linear.onForward`
     * transposes on every forward pass. A row-major narrow tensor has no fast transpose — it
     * widens elementwise through boxed `get()`, which measured 206 ms for a 2048×2048 projection
     * and 4.4 s for 4096×11008, *per weight per token*. Relaying once at load makes that transpose
     * a zero-copy view (engine issue #888), which is what lets the narrow kernel actually run.
     *
     * Two kinds of tensor stay row-major:
     *
     *  - **Rank-1 norms.** Never transposed, never matmul'd — relaying them is undefined and
     *    [NarrowFloatInputMajorTensorData] rejects rank ≠ 2 outright.
     *  - **The token embedding.** Gathered by row, not multiplied. Input-major storage strides
     *    those row reads, so relaying it would trade a win we don't get for a loss we would.
     *    Note this also covers tied embeddings: `output.weight` aliases `token_embd`, so in the
     *    tied case the output projection stays row-major too and forgoes the transpose win. That
     *    is deliberate — one shared buffer cannot be optimal for both access patterns.
     *
     * @param existing the engine-delivered row-major data, returned as-is when it already has
     *   the right layout and shape (so the engine path adds no copy for norms/embeddings).
     */
    private fun narrowData(
        canonicalName: String,
        shape: Shape,
        bytes: ByteArray,
        codec: NarrowFloatCodec,
        existing: NarrowFloatDenseTensorData? = null,
    ): NarrowFloatTensorData {
        val isGatheredEmbedding = canonicalName == DecoderTensorNames.TOKEN_EMBEDDINGS
        return when {
            shape.rank == 2 && !isGatheredEmbedding -> NarrowFloatInputMajorTensorData.fromRowMajor(shape, bytes, codec)
            existing != null && existing.shape == shape -> existing
            else -> NarrowFloatDenseTensorData(shape, bytes, codec)
        }
    }

    /**
     * Infer the target shape for a tensor, normalizing norm shapes.
     * For Q4 tensors, the target shape is the logical shape (not the packed shape).
     */
    private fun inferShape(canonicalName: String, originalShape: List<Long>): Shape {
        // Q4 tensors already have logical shapes in the header
        return Shape(*originalShape.map { it.toInt() }.toIntArray())
    }

    /**
     * Normalize norm shapes: [1, dim] → [dim] for 1D norm weights.
     */
    private fun normalizeNormShape(shape: List<Long>): Shape {
        return if (shape.size == 2 && shape[0] == 1L) {
            Shape(shape[1].toInt())
        } else {
            Shape(*shape.map { it.toInt() }.toIntArray())
        }
    }

    private fun normalizeNormShape(shape: Shape): Shape {
        val dims = shape.dimensions
        return if (dims.size == 2 && dims[0] == 1) Shape(dims[1]) else shape
    }

    // ========== Q4 Dequantization ==========

    /**
     * Dequantize Q4 (4-bit quantized) tensor using companion .qb scale tensor.
     *
     * Q4 packs two 4-bit values per byte. The .qb tensor provides per-group
     * scale factors in FP32. Each group of 32 elements shares one scale factor.
     */
    private fun dequantQ4(
        q4Bytes: ByteArray,
        qbBytes: ByteArray?,
        q4Shape: List<Long>,
        qbShape: List<Long>?
    ): FloatArray {
        val rows = q4Shape[0].toInt()
        val cols = q4Shape[1].toInt()
        val totalElements = rows * cols
        val out = FloatArray(totalElements)

        if (qbBytes == null) {
            // No scale tensor; just unpack nibbles to [-8, 7] range
            for (i in 0 until totalElements) {
                val byteIdx = i / 2
                val nibble = if (i % 2 == 0) {
                    (q4Bytes[byteIdx].toInt() and 0x0F)
                } else {
                    (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                }
                // Signed 4-bit: 0..7 → 0..7, 8..15 → -8..-1
                out[i] = (if (nibble >= 8) nibble - 16 else nibble).toFloat()
            }
            return out
        }

        // Unpack .qb scales from FP32 bytes
        val scales = bytesToFloatArray(qbBytes)

        // Determine group size from shapes: each scale covers a group of elements
        // qb shape is [rows, numGroups] where numGroups = cols / groupSize
        val numGroups = if (qbShape != null && qbShape.size == 2) qbShape[1].toInt() else scales.size / rows
        val groupSize = if (numGroups > 0) cols / numGroups else 32

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val flatIdx = row * cols + col
                val byteIdx = flatIdx / 2
                val nibble = if (flatIdx % 2 == 0) {
                    (q4Bytes[byteIdx].toInt() and 0x0F)
                } else {
                    (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                }
                // Signed 4-bit
                val q = if (nibble >= 8) nibble - 16 else nibble
                // Look up group scale
                val groupIdx = col / groupSize
                val scaleIdx = row * numGroups + groupIdx
                val scale = if (scaleIdx < scales.size) scales[scaleIdx] else 1.0f
                out[flatIdx] = q.toFloat() * scale
            }
        }

        return out
    }

    // ========== BF16/F16 Dequantization (legacy path) ==========

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val bf16Low = bytes[offset].toInt() and 0xFF
            val bf16High = bytes[offset + 1].toInt() and 0xFF
            val bits = (bf16High shl 24) or (bf16Low shl 16)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    private fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            val bits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private companion object {
        /** Source dtypes the engine's single-file loader materializes exactly as this loader always did. */
        val ENGINE_FLOAT_TYPES: Set<DataType> = setOf(DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16)
    }
}

/**
 * Maps HuggingFace tensor names to GGUF canonical names used by [DecoderTensorNames].
 */
public object HfTensorNameMapper {

    private val LAYER_PATTERN = Regex("""model\.layers\.(\d+)\.(.+)""")

    /**
     * Convert a HuggingFace tensor name to its GGUF canonical equivalent.
     * Returns null if the tensor should be skipped (e.g. .qb companions).
     */
    public fun toCanonical(hfName: String): String? {
        // Global tensors
        return when (hfName) {
            "model.embed_tokens.weight" -> DecoderTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> DecoderTensorNames.OUTPUT_NORM
            "lm_head.weight" -> DecoderTensorNames.OUTPUT_WEIGHT
            else -> {
                // Layer tensors
                val match = LAYER_PATTERN.matchEntire(hfName) ?: return null
                val layer = match.groupValues[1].toInt()
                when (match.groupValues[2]) {
                    "input_layernorm.weight" -> DecoderTensorNames.attnNorm(layer)
                    "self_attn.q_proj.weight" -> DecoderTensorNames.attnQ(layer)
                    "self_attn.k_proj.weight" -> DecoderTensorNames.attnK(layer)
                    "self_attn.v_proj.weight" -> DecoderTensorNames.attnV(layer)
                    "self_attn.o_proj.weight" -> DecoderTensorNames.attnOut(layer)
                    "post_attention_layernorm.weight" -> DecoderTensorNames.ffnNorm(layer)
                    "mlp.gate_proj.weight" -> DecoderTensorNames.ffnGate(layer)
                    "mlp.down_proj.weight" -> DecoderTensorNames.ffnDown(layer)
                    "mlp.up_proj.weight" -> DecoderTensorNames.ffnUp(layer)
                    else -> null
                }
            }
        }
    }
}
