package sk.ainet.models.apertus

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape

/**
 * Holds raw quantized weight bytes with the metadata needed for on-the-fly dequantization.
 *
 * Avoids the 4-8x memory expansion of eagerly dequantizing to FP32 at load time.
 * The runtime dequantizes per-layer as needed, keeping only one layer's FP32 temporaries
 * alive at a time.
 *
 * For a 7B Q4_0 model this reduces resident memory from ~28 GB (FP32) to ~3.5 GB
 * (quantized) plus ~50 MB of per-layer FP32 temporaries.
 */
public class QuantizedTensor(
    /** Raw quantized bytes as loaded from GGUF. */
    public val data: ByteArray,
    /** GGML quantization type (Q4_0, Q4_K, Q8_0, F16, BF16, etc.). */
    public val quantType: GGMLQuantizationType,
    /**
     * Original GGUF shape (column-major for 2D: [out, in]).
     * Used to derive the correct FP32 shape after dequantization.
     */
    public val shape: Shape,
    /** Number of logical elements (needed by block-quantized dequantizers). */
    public val nElements: Int
) {
    /** Size in bytes of the raw quantized data. */
    public val sizeBytes: Int get() = data.size

    /**
     * Dequantize to a flat FP32 array.
     *
     * The caller is responsible for interpreting the layout (column-major from GGUF)
     * and applying any transpose if needed.
     */
    public fun dequantToFloat(): FloatArray = when (quantType) {
        GGMLQuantizationType.F32 -> DequantOps.bytesToFloatArray(data)
        GGMLQuantizationType.F16 -> DequantOps.dequantF16FromBytes(data)
        GGMLQuantizationType.BF16 -> DequantOps.dequantBF16FromBytes(data)
        else -> DequantOps.dequantFromBytes(data, quantType, nElements)
    }

    override fun toString(): String =
        "QuantizedTensor(quantType=$quantType, shape=$shape, nElements=$nElements, sizeBytes=$sizeBytes)"
}
