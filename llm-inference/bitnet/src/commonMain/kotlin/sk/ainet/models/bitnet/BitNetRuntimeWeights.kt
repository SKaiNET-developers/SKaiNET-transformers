package sk.ainet.models.bitnet

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Canonical GGUF tensor names for the BitNet family (#346's `<F>TensorNames` row).
 *
 * BitNet uses the Llama-family layout (`blk.N.attn_q.weight`, …) — those names come from the
 * shared decoder machinery and are not restated here — plus the two per-layer sub-norm tensors
 * of its own. The module-path→name mapping lives in [BitNetGGUFNameResolver]; these constants
 * are the single place the family's *special* names are written out.
 */
public object BitNetTensorNames {
    public const val TOKEN_EMBEDDING: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"

    /** The lm_head. Absent in tied-embeddings checkpoints (2B4T) — see [BitNetWeightLoader]. */
    public const val OUTPUT: String = "output.weight"

    /** BitNet-specific: RMSNorm on the merged attention output, before o_proj. */
    public fun attnSubNorm(layer: Int): String = "blk.$layer.attn_sub_norm.weight"

    /** BitNet-specific: RMSNorm inside the squared-ReLU FFN, before down_proj. */
    public fun ffnSubNorm(layer: Int): String = "blk.$layer.ffn_sub_norm.weight"
}

/**
 * The BitNet family's runtime-weights container (#346's `<F>RuntimeWeights` row).
 *
 * Deliberate deviation from the Apertus reference shape, documented against #346: BitNet runs
 * through the network DSL + `WeightMapper` (like llama/qwen), not a hand-rolled per-layer loop,
 * so its runtime consumes a *name→tensor map* bound into [bitnetNetwork] — typed per-layer
 * field containers would duplicate the map without a consumer. This container is that map plus
 * the parsed metadata, with the family-specific accessors a caller actually needs.
 */
public data class BitNetRuntimeWeights(
    public val metadata: LlamaModelMetadata,
    public val tensors: Map<String, Tensor<FP32, Float>>,
) {
    /**
     * The lm_head weight in the fused `BITNET_PLANES` format, when the load materialized one
     * (`planesLmHead`, covering both the `output.weight` and tied-2B4T lanes) — the gate for
     * the two-stage decode ([BitNetTwoStageDecode], `generateTwoStage`).
     */
    @OptIn(ExperimentalMemoryApi::class)
    public val planesHead: BitNetPlanesTensorData?
        get() = tensors[BitNetTensorNames.OUTPUT]?.data as? BitNetPlanesTensorData

    /** Bind these weights into a fresh [bitnetNetwork] module. */
    public fun toModule(debug: Boolean = false): Module<FP32, Float> =
        BitNetNetworkLoader.fromWeights(DecoderGgufWeights(metadata, tensors), debug)
}
