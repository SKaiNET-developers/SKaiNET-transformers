package sk.ainet.lang.nn.transformer

/**
 * A heap view over a cache's K/V prefix, read in place by the fused attention kernels
 * (SKEEP-005). Head `g`, position `s`, element `d` of the keys lives at
 * `keys[g * headStride + s * rowStride + d]` for `s < length`, `d < headDim`; values likewise.
 *
 * `headStride` is the *buffer's* per-head stride: `maxSeqLen * headDim` when the view aliases a
 * [PositionalKVCache] buffer, `length * headDim` for a copied view. `rowStride` is normally
 * `headDim`; a padded shared cache may carry a wider row than the layer's `headDim`.
 *
 * The arrays are read-only for the duration of an attention forward; the coordinator writes the
 * new position before handing the view to any task.
 */
public class KVBufferView(
    public val keys: FloatArray,
    public val values: FloatArray,
    public val length: Int,
    public val headStride: Int,
    public val rowStride: Int,
    public val headDim: Int,
) {
    init {
        require(length >= 0) { "KVBufferView: negative length $length" }
        require(headDim in 1..rowStride) { "KVBufferView: headDim=$headDim must be within rowStride=$rowStride" }
    }

    public companion object {
        /** A view over `[nKVHeads, length, headDim]` heads-first contiguous arrays (the copied form). */
        public fun contiguous(keys: FloatArray, values: FloatArray, length: Int, headDim: Int): KVBufferView =
            KVBufferView(keys, values, length, headStride = length * headDim, rowStride = headDim, headDim = headDim)
    }
}
