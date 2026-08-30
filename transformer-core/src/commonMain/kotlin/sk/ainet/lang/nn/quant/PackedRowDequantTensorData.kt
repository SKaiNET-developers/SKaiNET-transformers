package sk.ainet.lang.nn.quant

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.RowDequantSource
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A packed 2-D engine weight (heap `Q*BlockTensorData` or mapped
 * `BufferPackedTensorData`) rewrapped as a [RowDequantSource], so [Embedding]
 * dequantizes only the rows a step actually looks up instead of the whole
 * table — the piece that lets a large-vocab `token_embd` honour
 * `WeightForm(residency = MAPPED)` instead of being force-dequantized to a
 * dense FP32 array on the heap (~1 GB for a 152k × 1536 vocabulary).
 *
 * Everything except [dequantRow] forwards to the wrapped data verbatim —
 * including [PackedBlockStorage], so a *tied* output head that aliases this
 * tensor still routes `matmulWeightTransposed` through the packed dispatch
 * chain (`is PackedBlockStorage` → views → KernelDispatch) exactly as the
 * unwrapped tensor would. Note the forwarded [get] keeps the wrapped type's
 * semantics: for mapped packed data that is the RAW quantization code, not
 * the value (the engine's documented source-compat quirk) — row reads must
 * go through [dequantRow], which is precisely why this wrapper exists.
 *
 * [wrapIfRowDequantable] is the constructor to use: it falls through to the
 * original tensor data for anything this wrapper cannot serve (dense
 * delivery, relayouted block order, rows not on block boundaries).
 */
@OptIn(ExperimentalMemoryApi::class)
public class PackedRowDequantTensorData private constructor(
    private val packed: TensorData<FP32, Float>,
    private val blocks: PackedBlockStorage,
) : TensorData<FP32, Float>, PackedBlockStorage, RowDequantSource {

    private val blocksPerRow: Int = shape[1] / blocks.blockSize

    // TensorData — the wrapped tensor's own façade, untouched.
    override val shape: Shape get() = packed.shape
    override val encoding: TensorEncoding get() = blocks.encoding
    override val view: sk.ainet.lang.memory.TensorView? get() = packed.view
    override fun get(vararg indices: Int): Float = packed.get(*indices)
    override fun set(vararg indices: Int, value: Float): Unit = packed.set(*indices, value = value)
    override fun copyToFloatArray(): FloatArray = packed.copyToFloatArray()

    // PackedBlockStorage — forwarded so packed matmul dispatch sees the real storage.
    override val blockCount: Int get() = blocks.blockCount
    override val blockSize: Int get() = blocks.blockSize
    override val packedData: ByteArray get() = blocks.packedData
    override val packedStorage: sk.ainet.lang.memory.Storage get() = blocks.packedStorage
    override val blockOrder: BlockOrder get() = blocks.blockOrder
    override val physicalBytes: Long get() = blocks.physicalBytes
    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int): Unit =
        blocks.dequantizeBlock(blockIdx, output, outputOffset)

    // RowDequantSource — one row = blocksPerRow consecutive canonical blocks.
    override fun dequantRow(rowIdx: Int): FloatArray {
        require(rowIdx in 0 until shape[0]) { "row $rowIdx out of range [0, ${shape[0]})" }
        val out = FloatArray(shape[1])
        val first = rowIdx * blocksPerRow
        for (b in 0 until blocksPerRow) {
            blocks.dequantizeBlock(first + b, out, b * blocks.blockSize)
        }
        return out
    }

    public companion object {
        /**
         * [data] wrapped as a row-dequant source when it is a canonical
         * packed 2-D table whose rows lie on block boundaries; [data] itself
         * otherwise (dense FP32 delivery needs no wrapper, and a relayouted
         * block order or a ragged last block cannot be row-addressed).
         */
        public fun wrapIfRowDequantable(data: TensorData<FP32, Float>): TensorData<FP32, Float> {
            val blocks = data as? PackedBlockStorage ?: return data
            if (data.shape.rank != 2) return data
            if (blocks.blockOrder != BlockOrder.ROW_MAJOR) return data
            if (blocks.blockSize <= 0 || data.shape[1] % blocks.blockSize != 0) return data
            return PackedRowDequantTensorData(data, blocks)
        }
    }
}
