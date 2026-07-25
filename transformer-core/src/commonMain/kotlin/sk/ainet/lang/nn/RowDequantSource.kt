package sk.ainet.lang.nn

/**
 * Marker for `sk.ainet.lang.tensor.data.TensorData` implementations that should
 * not be materialised as a single dense `FloatArray` — either the logical tensor
 * exceeds `Int.MAX_VALUE` elements / 2 GB, or keeping it packed avoids a large
 * FP32 inflation (e.g. a Q8_0 `token_embd` table) — and instead expose a cheap
 * per-row dequantisation API.
 *
 * Consumers that gather rows (the [layers.Embedding] lookup, gemma's
 * `PerLayerEmbedding`) MUST call [dequantRow] for the few rows actually touched
 * per decode step (one per token) instead of `copyToFloatArray()` on the whole
 * table.
 *
 * (Lives in `llm-core` so the shared `Embedding` layer can honour it; gemma's
 * `GemmaPerLayerTokenEmbedTensorData` / `SafeTensorsPerLayerTokenEmbedTensorData`
 * implement it.)
 */
public interface RowDequantSource {
    /**
     * Dequantise one logical row of the table to a fresh `FloatArray` of length
     * equal to the row width (`shape[1]`).
     */
    public fun dequantRow(rowIdx: Int): FloatArray
}
