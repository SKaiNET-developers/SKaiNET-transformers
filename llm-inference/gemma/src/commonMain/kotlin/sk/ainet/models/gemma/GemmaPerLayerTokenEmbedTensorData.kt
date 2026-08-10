package sk.ainet.models.gemma

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.nn.RowDequantSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * [TensorData] wrapper for the Gemma 4 `per_layer_token_embd` weight when
 * loaded from a Q-series GGUF tensor. Keeps the packed quant bytes
 * unmodified and dequants one row (one vocabulary entry) at a time via
 * [dequantRow].
 *
 * Why not just dequant the whole thing to FP32 on load? For Gemma 4 E2B
 * this tensor has shape `[262 144, 8 960]` at Q6_K — 2.35 billion
 * elements, 9.4 GB as FP32. JVM FloatArray tops out at Int.MAX_VALUE
 * elements (2.15 billion), and the memory cost would blow past a normal
 * laptop. Keeping bytes: 262144 × 8960 / 256 × 210 ≈ 1.93 GB. Then
 * [PerLayerEmbedding.compute] pays ~35 KB of dequant work per token
 * at decode time — negligible.
 *
 * The `get` / `set` operations are not supported — this class is meant
 * to be used exclusively via [dequantRow], and [PerLayerEmbedding]
 * branches on its type at compute time. If something else in the
 * pipeline tries to treat it as a normal float tensor, it will throw
 * loudly rather than silently producing wrong results.
 *
 * @param logicalShape 2-D logical shape `[vocabSize, perLayerTotal]`.
 * @param quantType the tensor's GGML quantization type (Q6_K for E2B).
 * @param packedBytes the raw byte payload as it appears in the GGUF
 *   data section, laid out row-major along the logical [rows, cols].
 */
public class GemmaPerLayerTokenEmbedTensorData(
    logicalShape: Shape,
    public val quantType: GGMLQuantizationType,
    public val packedBytes: ByteArray
) : TensorData<FP32, Float>, RowDequantSource {

    override val shape: Shape = logicalShape

    init {
        require(logicalShape.rank == 2) {
            "GemmaPerLayerTokenEmbedTensorData requires 2-D logical shape, got $logicalShape"
        }
        val size = GGML_QUANT_SIZES[quantType]
            ?: error("Unknown quant block size for $quantType")
        val blockElements = size.first
        val bytesPerBlock = size.second
        val perRow = logicalShape[1]
        require(perRow % blockElements == 0) {
            "perLayerTotal ($perRow) must be divisible by ${quantType.name} block size ($blockElements)"
        }
        val blocksPerRow = perRow / blockElements
        val bytesPerRow = blocksPerRow * bytesPerBlock
        val expected = logicalShape[0].toLong() * bytesPerRow.toLong()
        require(packedBytes.size.toLong() == expected) {
            "packedBytes size ${packedBytes.size} != expected $expected for $quantType shape=$logicalShape"
        }
    }

    /** Bytes per row (one vocabulary entry). Cached to avoid per-call map lookups. */
    public val bytesPerRow: Int = run {
        val size = GGML_QUANT_SIZES[quantType]!!
        (shape[1] / size.first) * size.second
    }

    /**
     * Dequant one row (vocabulary entry) to FP32. Returns a fresh
     * `FloatArray` of length `shape[1]`.
     */
    public override fun dequantRow(rowIdx: Int): FloatArray {
        require(rowIdx in 0 until shape[0]) {
            "rowIdx $rowIdx out of range [0, ${shape[0]})"
        }
        val start = rowIdx * bytesPerRow
        val rowBytes = packedBytes.copyOfRange(start, start + bytesPerRow)
        return DequantOps.dequantFromBytes(rowBytes, quantType, shape[1])
    }

    override fun get(vararg indices: Int): Float =
        error(
            "GemmaPerLayerTokenEmbedTensorData.get is unsupported — use dequantRow() " +
                "instead. This class exists because the whole tensor can't be " +
                "materialised as a JVM FloatArray on Gemma 4 E2B (> Int.MAX_VALUE elements)."
        )

    override fun set(vararg indices: Int, value: Float) {
        error("GemmaPerLayerTokenEmbedTensorData is read-only.")
    }
}
