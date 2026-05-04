package sk.ainet.models.llama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
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
 *   [Q8MemorySegmentTensorData]. Upstream `DefaultCpuOpsJvm.matmul` detects
 *   their markers and dispatches SIMD quant kernels at forward time.
 * - **Q4_K / Q5_K / Q6_K** → dequantized to FP32. The packed K-quant kernels
 *   are MemSeg-only on a hot path the DSL doesn't yet route through, so this
 *   trades memory for correctness. Same trade-off the legacy converter
 *   makes for K-quants.
 * - **FP32 (no entry in `quantTypes`)** → passed through unchanged.
 * - **Other quant types** → warning logged, passed through (will fail later
 *   if the model actually hits them via matmul).
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
     * variants and K-quants dequantized to FP32. No-op if [weights] has no
     * quantized tensors.
     */
    public fun convert(
        weights: DecoderGgufWeights<FP32, Float>,
        ctx: ExecutionContext,
        arena: Arena,
    ): DecoderGgufWeights<FP32, Float> {
        if (weights.quantTypes.isEmpty()) return weights

        val newTensors = LinkedHashMap<String, Tensor<FP32, Float>>(weights.tensors.size)
        for ((name, tensor) in weights.tensors) {
            val quantType = weights.quantTypes[name]
            if (quantType == null) {
                newTensors[name] = tensor
                continue
            }
            newTensors[name] = convertOne(name, tensor, quantType, ctx, arena)
        }

        // Drop quantTypes from the result — tensors are now either packed
        // MemSeg (carry their own marker) or dequantized FP32 (no quant
        // identity). Keeping a stale `quantTypes` map would mislead later
        // consumers into thinking the tensors are still raw bytes.
        return weights.copy(tensors = newTensors, quantTypes = emptyMap())
    }

    private fun convertOne(
        name: String,
        tensor: Tensor<FP32, Float>,
        quantType: GGMLQuantizationType,
        ctx: ExecutionContext,
        arena: Arena,
    ): Tensor<FP32, Float> {
        val bytes = extractBytes(tensor.data)
        val shape = tensor.shape

        return when (quantType) {
            GGMLQuantizationType.Q4_0 -> {
                val newData = Q4MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
                @Suppress("UNCHECKED_CAST")
                ctx.fromData(newData as TensorData<FP32, Float>, FP32::class)
            }
            GGMLQuantizationType.Q8_0 -> {
                val newData = Q8MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
                @Suppress("UNCHECKED_CAST")
                ctx.fromData(newData as TensorData<FP32, Float>, FP32::class)
            }
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K -> {
                val floats = DequantOps.dequantFromBytes(bytes, quantType, shape.volume)
                ctx.fromFloatArray(shape, FP32::class, floats)
            }
            else -> {
                println(
                    "WARNING: DecoderGgufMemSegConverter: unsupported quant type $quantType for '$name'; " +
                        "passing through unchanged. Forward pass may fail at matmul."
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
