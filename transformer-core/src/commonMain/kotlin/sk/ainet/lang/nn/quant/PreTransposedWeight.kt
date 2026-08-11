package sk.ainet.lang.nn.quant

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_0TensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_0TensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1TensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage

/**
 * Pre-transpose marker (#184 hoist 3, the "Solution C" of the gemma 8B-OOM
 * investigation): tensor data implementing this interface declares that its
 * *logical shape is already the transposed `[in, out]`* a matmul consumes —
 * i.e. the converter that produced it already performed (or absorbed) the
 * `W.t()` that [sk.ainet.lang.nn.transformer.linearProject] would otherwise
 * apply, so `linearProject` dispatches `ops.matmul(x, W)` directly and skips
 * `ops.transpose` entirely.
 *
 * For GGUF block-quant weights this is free: the row-major → block-major
 * relayout ([BlockQuantPacking.relayoutRowMajorToBlockMajor]) already stores
 * the bytes in the kernels' input-block-major order, and the engine's lazy
 * packed `ops.transpose` is a pure logical-shape swap over those same bytes.
 * [BlockQuantPacking.packPreTransposed] performs that shape swap at pack time
 * and attaches this marker, cutting the per-forward transpose wrapper
 * allocation out of every projection.
 *
 * The explicit marker exists because a shape heuristic
 * (`W.shape[0] == x.shape[-1]`) is ambiguous for square projections — see
 * `linearProject`'s kdoc. The engine's own packed `ops.transpose` remains the
 * fallback for unmarked weights, so this is opt-in per tensor.
 *
 * Engine alignment: when the engine grows a first-class pre-transposed flag
 * on its tensor data (the other half of #184 (3)), this interface unifies
 * with it the same way `RowDequantSource` did in #289 (typealias), keeping
 * `linearProject`'s check source-compatible.
 */
public interface PreTransposedWeight

// Internal marked delegating views over the engine's heap-packed block tensor
// data. Each implements the format's *interface* (what every engine dispatch
// site checks: `chooseQuantizedMatmulHeap`, the JVM matmul intercepts, and the
// lazy packed `ops.transpose` all match on `is Q*TensorData`) plus
// [PackedBlockStorage] (what the compile-path `TensorSpecEncoding` matches on),
// so a marked weight behaves byte-for-byte like the unmarked one everywhere
// except the `linearProject` transpose skip. Members declared by both parents
// are disambiguated explicitly to the same delegate.

internal class PreTransposedQ4_K(private val d: Q4_KBlockTensorData) :
    Q4_KTensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ5_K(private val d: Q5_KBlockTensorData) :
    Q5_KTensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ6_K(private val d: Q6_KBlockTensorData) :
    Q6_KTensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ8_0(private val d: Q8_0BlockTensorData) :
    Q8_0TensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ4_0(private val d: Q4_0BlockTensorData) :
    Q4_0TensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ5_0(private val d: Q5_0BlockTensorData) :
    Q5_0TensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}

internal class PreTransposedQ5_1(private val d: Q5_1BlockTensorData) :
    Q5_1TensorData by d, PackedBlockStorage by d, PreTransposedWeight {
    override val shape: Shape get() = d.shape
    override val blockCount: Int get() = d.blockCount
    override val packedData: ByteArray get() = d.packedData
}
