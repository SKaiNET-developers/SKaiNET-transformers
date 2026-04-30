package sk.ainet.models.gemma

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * [TensorData] wrapper for the Gemma 4 `embed_tokens_per_layer` table when
 * loaded from a SafeTensors checkpoint. Wraps a memory-mapped region of
 * raw BF16 bytes and provides cheap per-row dequantisation via
 * [dequantRow]; the full logical tensor (`[vocab × num_layers × per_layer_dim]`,
 * ~9.4 GB FP32 on Gemma 4 E2B) is never materialised.
 *
 * The complementary GGUF path is [GemmaPerLayerTokenEmbedTensorData], which
 * wraps Q6_K-packed bytes; both implement [RowDequantSource] so
 * [PerLayerEmbedding.compute] can dispatch uniformly.
 *
 * `get` / `set` are unsupported by design — this class is consumed
 * exclusively via [dequantRow]. If anything else in the pipeline tries
 * to treat it as a normal float tensor it will throw loudly rather than
 * silently producing wrong results.
 *
 * Currently supports BF16 source only (the dtype of the real
 * `google/gemma-4-e2b-it` checkpoint). F16 / F32 sources are easy to add
 * by branching in [dequantRow] on [sourceDtype].
 *
 * @param logicalShape 2-D logical shape `[vocab_size, per_layer_total]`.
 * @param segment memory-mapped, read-only region containing the tensor's
 *   raw bytes laid out row-major along `[vocab_size, per_layer_total]`.
 *   Lifetime is owned by the caller — typically an [java.lang.foreign.Arena]
 *   tied to the chat model's lifecycle.
 * @param sourceDtype on-disk element dtype. The accepted set today is
 *   `"BF16"`; `"F16"` and `"F32"` will be added when a checkpoint surfaces
 *   that needs them.
 */
public class SafeTensorsPerLayerTokenEmbedTensorData(
    logicalShape: Shape,
    private val segment: MemorySegment,
    private val sourceDtype: String,
) : TensorData<FP32, Float>, RowDequantSource {

    override val shape: Shape = logicalShape

    private val rowWidth: Int
    private val bytesPerElement: Long
    private val bytesPerRow: Long

    init {
        require(logicalShape.rank == 2) {
            "SafeTensorsPerLayerTokenEmbedTensorData requires 2-D logical shape, got $logicalShape"
        }
        rowWidth = logicalShape[1]
        bytesPerElement = when (sourceDtype.uppercase()) {
            "BF16", "F16" -> 2L
            "F32" -> 4L
            else -> error("Unsupported SafeTensors PLE dtype: $sourceDtype (expected BF16 / F16 / F32)")
        }
        bytesPerRow = rowWidth.toLong() * bytesPerElement
        val expected = logicalShape[0].toLong() * bytesPerRow
        require(segment.byteSize() >= expected) {
            "Mapped segment is ${segment.byteSize()} bytes but tensor needs $expected " +
                "(shape=$logicalShape dtype=$sourceDtype)"
        }
    }

    override fun dequantRow(rowIdx: Int): FloatArray {
        require(rowIdx in 0 until shape[0]) {
            "rowIdx $rowIdx out of range [0, ${shape[0]})"
        }
        val out = FloatArray(rowWidth)
        val rowBase = rowIdx.toLong() * bytesPerRow
        when (sourceDtype.uppercase()) {
            "BF16" -> {
                // BF16 stores the upper 16 bits of an FP32 value. Reconstruct
                // by zero-extending into the high half of an Int and reading
                // as a float. SafeTensors is little-endian; JVM_SHORT_LE
                // matches that on every platform.
                for (i in 0 until rowWidth) {
                    val bits = segment.get(
                        ValueLayout.JAVA_SHORT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                        rowBase + i.toLong() * 2L
                    ).toInt() and 0xFFFF
                    out[i] = Float.fromBits(bits shl 16)
                }
            }
            "F16" -> {
                for (i in 0 until rowWidth) {
                    val bits = segment.get(
                        ValueLayout.JAVA_SHORT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                        rowBase + i.toLong() * 2L
                    ).toInt() and 0xFFFF
                    out[i] = halfBitsToFloat(bits)
                }
            }
            "F32" -> {
                for (i in 0 until rowWidth) {
                    out[i] = segment.get(
                        ValueLayout.JAVA_FLOAT.withOrder(java.nio.ByteOrder.LITTLE_ENDIAN),
                        rowBase + i.toLong() * 4L
                    )
                }
            }
        }
        return out
    }

    override fun get(vararg indices: Int): Float =
        error(
            "SafeTensorsPerLayerTokenEmbedTensorData.get is unsupported — use dequantRow() " +
                "instead. The full logical table is too large to materialise as a JVM " +
                "FloatArray (> Int.MAX_VALUE elements on Gemma 4 E2B)."
        )

    override fun set(vararg indices: Int, value: Float) {
        error("SafeTensorsPerLayerTokenEmbedTensorData is read-only.")
    }

    private companion object {
        // IEEE 754 half-precision → single-precision. Adapted from the
        // standard reference impl. Branchless on subnormal/inf/nan.
        fun halfBitsToFloat(h: Int): Float {
            val sign = (h ushr 15) and 0x1
            val exp = (h ushr 10) and 0x1F
            val mant = h and 0x3FF
            val fbits = when (exp) {
                0 -> if (mant == 0) sign shl 31 else {
                    // Subnormal: normalise.
                    var m = mant
                    var e = -1
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    val mantissa = (m and 0x3FF) shl 13
                    val biased = (e + 127 + 1) and 0xFF
                    (sign shl 31) or (biased shl 23) or mantissa
                }
                31 -> (sign shl 31) or (0xFF shl 23) or (mant shl 13)
                else -> (sign shl 31) or ((exp + 112) shl 23) or (mant shl 13)
            }
            return Float.fromBits(fbits)
        }
    }
}
