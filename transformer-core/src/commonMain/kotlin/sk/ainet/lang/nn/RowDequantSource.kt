package sk.ainet.lang.nn

/**
 * MOVED to the engine: `sk.ainet.lang.tensor.data.RowDequantSource` (present in the SKaiNET
 * engine at the 0.39.0 pin) is the canonical home of the row-dequant contract, and the engine's
 * `ops.gather` consumes it directly — a packed table gathered through any `ops.gather` call (not only the
 * [layers.Embedding] fast path) dequantises only the touched rows.
 *
 * This alias keeps source compatibility for existing implementors (gemma's
 * `GemmaPerLayerTokenEmbedTensorData` / `SafeTensorsPerLayerTokenEmbedTensorData`); implementing
 * either name now implements the engine interface, so both the shared [layers.Embedding] row-dequant
 * path and the engine `ops.gather` row-dequant path recognise the tensor.
 *
 * Binary note: the transformer-core interface `sk/ainet/lang/nn/RowDequantSource` no longer exists
 * as a distinct class — external code compiled against it must recompile (call out in release notes).
 */
@Deprecated(
    message = "Moved to the engine; use sk.ainet.lang.tensor.data.RowDequantSource",
    replaceWith = ReplaceWith("RowDequantSource", "sk.ainet.lang.tensor.data.RowDequantSource"),
)
public typealias RowDequantSource = sk.ainet.lang.tensor.data.RowDequantSource
