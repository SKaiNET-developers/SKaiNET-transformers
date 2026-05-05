package sk.ainet.models.llama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena

/**
 * Post-load converter for the DSL inference path.
 *
 * Counterpart to [MemSegWeightConverter] (which targets the legacy
 * [LlamaRuntimeWeights] format), this one operates directly on
 * [DecoderGgufWeights] — the GGUF-keyed tensor map produced by
 * [DecoderGgufWeightLoader] under [sk.ainet.io.model.QuantPolicy.NATIVE_OPTIMIZED].
 *
 * Behavior per quant type:
 * - **Q4_0 / Q8_0** → wrapped as [Q4MemorySegmentTensorData] /
 *   [Q8MemorySegmentTensorData] with the **logical** matrix shape derived
 *   from metadata. Upstream `DefaultCpuOpsJvm.matmul` and `transpose`
 *   detect the markers and dispatch quant-aware kernels at forward time.
 * - **Q4_K / Q5_K / Q6_K** → dequantized to FP32. The packed K-quant kernels
 *   are MemSeg-only on a hot path the DSL doesn't yet route through, so this
 *   trades memory for correctness. Same trade-off the legacy converter
 *   makes for K-quants.
 * - **token_embd.weight** → always dequantized to FP32 regardless of quant
 *   type. The Embedding layer consumes this via `gather`, not matmul, so it
 *   needs real floats with the logical 2D shape — packed quant bytes would
 *   be misread as FP32 values, and the loader's intermediate Int8 wrapper
 *   stores a 1D byte-count shape that `gather` rejects.
 * - **FP32 (no entry in `quantTypes`)** → passed through unchanged.
 * - **Other quant types** → warning logged, passed through (will fail later
 *   if the model actually hits them via matmul).
 *
 * Why logical shape matters here: the loader stores raw quant bytes via
 * `ctx.fromByteArray(Shape(bytes.size), Int8, bytes)` — a 1D byte-count
 * shape, because the Int8 factory requires `shape.volume == bytes.size`
 * and packed Q4/Q8 have more bytes than logical floats. The Q4/Q8 MemSeg
 * tensor data classes, in contrast, hold the logical shape independently
 * from the byte buffer, which is what `gather` / `transpose` / `matmul`
 * need.
 *
 * Unlike the legacy [MemSegWeightConverter], this one does NOT pre-transpose
 * weights to `[in, out]`. The DSL's [sk.ainet.lang.nn.transformer.linearProject]
 * always calls `ops.transpose(weight)` at forward time; for Q4/Q8 MemSeg
 * tensors that's a free metadata-only swap upstream, so pre-transposing
 * brings no benefit. For dequantized K-quants and FP32 tensors a runtime
 * transpose still has a real cost — addressing it requires a pre-transposed
 * marker on `linearProject`, tracked as a follow-up perf optimization.
 *
 * Caller manages the [Arena] lifecycle. Tying it to the inference
 * `ExecutionContext` lifecycle is the typical pattern.
 */
public object DecoderGgufMemSegConverter {

    /**
     * Return a copy of [weights] with Q4_0/Q8_0 tensors wrapped as MemSeg
     * variants with logical shapes, K-quants dequantized to FP32, and the
     * token embedding always dequantized. No-op if [weights] has no
     * quantized tensors.
     */
    public fun convert(
        weights: DecoderGgufWeights<FP32, Float>,
        ctx: ExecutionContext,
        arena: Arena,
    ): DecoderGgufWeights<FP32, Float> {
        if (weights.quantTypes.isEmpty()) return weights

        val meta = weights.metadata
        val dim = meta.embeddingLength
        val headSize = dim / meta.headCount
        val kvDim = meta.kvHeadCount * headSize
        val ffnDim = meta.feedForwardLength
        val vocab = meta.vocabSize

        val newTensors = LinkedHashMap<String, Tensor<FP32, Float>>(weights.tensors.size)
        for ((name, tensor) in weights.tensors) {
            val quantType = weights.quantTypes[name]
            if (quantType == null) {
                newTensors[name] = tensor
                continue
            }
            val logicalShape = logicalShapeFor(name, dim, kvDim, ffnDim, vocab)
            if (logicalShape == null) {
                println(
                    "WARNING: DecoderGgufMemSegConverter: no logical shape for '$name'; " +
                        "passing through quantized — forward pass may fail.",
                )
                newTensors[name] = tensor
                continue
            }
            newTensors[name] = convertOne(name, tensor, quantType, logicalShape, ctx, arena)
        }

        // Drop quantTypes from the result — tensors are now either packed
        // MemSeg (carry their own marker) or dequantized FP32 (no quant
        // identity). Keeping a stale `quantTypes` map would mislead later
        // consumers into thinking the tensors are still raw bytes.
        return weights.copy(tensors = newTensors, quantTypes = emptyMap())
    }

    private fun logicalShapeFor(
        name: String,
        dim: Int,
        kvDim: Int,
        ffnDim: Int,
        vocab: Int,
    ): Shape? = when {
        name == LlamaTensorNames.TOKEN_EMBEDDINGS -> Shape(vocab, dim)
        name == LlamaTensorNames.OUTPUT_WEIGHT -> Shape(vocab, dim)
        name.endsWith(".attn_q.weight") -> Shape(dim, dim)
        name.endsWith(".attn_k.weight") -> Shape(kvDim, dim)
        name.endsWith(".attn_v.weight") -> Shape(kvDim, dim)
        name.endsWith(".attn_output.weight") -> Shape(dim, dim)
        name.endsWith(".ffn_gate.weight") -> Shape(ffnDim, dim)
        name.endsWith(".ffn_up.weight") -> Shape(ffnDim, dim)
        name.endsWith(".ffn_down.weight") -> Shape(dim, ffnDim)
        else -> null
    }

    private fun convertOne(
        name: String,
        tensor: Tensor<FP32, Float>,
        quantType: GGMLQuantizationType,
        logicalShape: Shape,
        ctx: ExecutionContext,
        arena: Arena,
    ): Tensor<FP32, Float> {
        val bytes = extractBytes(tensor.data)

        // token_embd.weight (and tied output.weight, which holds the same
        // bytes) is consumed by Embedding.gather, not matmul. Packed quant
        // bytes can't be read by gather as floats, so dequantize.
        if (name == LlamaTensorNames.TOKEN_EMBEDDINGS) {
            val floats = DequantOps.dequantFromBytes(bytes, quantType, logicalShape.volume)
            return ctx.fromFloatArray(logicalShape, FP32::class, floats)
        }

        return when (quantType) {
            GGMLQuantizationType.Q4_0 -> {
                val newData = Q4MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
                @Suppress("UNCHECKED_CAST")
                ctx.fromData(newData as TensorData<FP32, Float>, FP32::class)
            }
            GGMLQuantizationType.Q8_0 -> {
                val newData = Q8MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
                @Suppress("UNCHECKED_CAST")
                ctx.fromData(newData as TensorData<FP32, Float>, FP32::class)
            }
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K -> {
                val floats = DequantOps.dequantFromBytes(bytes, quantType, logicalShape.volume)
                ctx.fromFloatArray(logicalShape, FP32::class, floats)
            }
            else -> {
                println(
                    "WARNING: DecoderGgufMemSegConverter: unsupported quant type $quantType for '$name'; " +
                        "passing through unchanged. Forward pass may fail at matmul.",
                )
                tensor
            }
        }
    }

    private fun extractBytes(data: TensorData<*, *>): ByteArray {
        // DecoderGgufWeightLoader with NATIVE_OPTIMIZED stores raw bytes as
        // an IntArrayTensorData of Int8. Mirror MemSegWeightConverter's path.
        if (data is IntArrayTensorData<*>) {
            val buf = data.buffer
            return ByteArray(buf.size) { buf[it].toByte() }
        }
        val size = data.shape.volume
        return ByteArray(size) {
            @Suppress("UNCHECKED_CAST")
            ((data as TensorData<*, Int>)[it]).toByte()
        }
    }
}

/**
 * Convenience wrapper for the DSL inference path: stream a GGUF file with
 * `NATIVE_OPTIMIZED` quant policy, then run [DecoderGgufMemSegConverter] so
 * Q4_0 / Q8_0 tensors are wrapped as `MemorySegment`-backed packed data
 * before binding into a DSL network. Per-architecture loaders compose this
 * with their own `acceptedArchitectures` set and pass the result to
 * `xNetworkLoader.fromWeights(...)`.
 *
 * Caller manages the [arena] lifecycle.
 */
public suspend fun loadDecoderGgufWeightsNative(
    randomAccessProvider: () -> RandomAccessSource,
    acceptedArchitectures: Set<String>,
    ctx: ExecutionContext,
    arena: Arena,
): DecoderGgufWeights<FP32, Float> {
    val loader = DecoderGgufWeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
        acceptedArchitectures = acceptedArchitectures,
    )
    val raw = loader.loadToMapStreaming<FP32, Float>(ctx)
    return DecoderGgufMemSegConverter.convert(raw, ctx, arena)
}
