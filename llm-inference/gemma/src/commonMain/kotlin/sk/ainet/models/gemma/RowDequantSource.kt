package sk.ainet.models.gemma

/**
 * Marker for [sk.ainet.lang.tensor.data.TensorData] implementations that
 * cannot be materialised as a single dense `FloatArray` (the logical
 * tensor exceeds `Int.MAX_VALUE` elements / 2 GB) and instead expose a
 * cheap per-row dequantisation API.
 *
 * `PerLayerEmbedding.compute` calls [dequantRow] for the small number of
 * rows actually touched per decode step (one per token). Callers that
 * see a tensor implementing this interface MUST use [dequantRow] instead
 * of `copyToFloatArray()`.
 *
 * Implementors:
 *  - [GemmaPerLayerTokenEmbedTensorData] — Q-quantised GGUF source, dequants
 *    Q6_K bytes via the GGUF `DequantOps` table.
 *  - `SafeTensorsPerLayerTokenEmbedTensorData` (JVM-only) — BF16/F16 source,
 *    dequants by reading the half-float for each column out of an mmap'd
 *    file region.
 */
public interface RowDequantSource {
    /**
     * Dequantise one logical row of the embedding table to a fresh
     * `FloatArray` of length equal to the row width (`shape[1]`).
     */
    public fun dequantRow(rowIdx: Int): FloatArray
}
