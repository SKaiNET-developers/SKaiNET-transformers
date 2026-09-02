package sk.ainet.models.gemma3n

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.models.gemma.PerLayerEmbedding
import kotlin.reflect.KClass

/**
 * Top-level Gemma 3n model — the DSL replacement for the hand-rolled `Gemma3nRuntime`
 * (#377), faithful to HF `Gemma3nTextModel.forward` / `Gemma3nTextDecoderLayer.forward`.
 *
 * Follows the `GemmaModel` wrapper pattern (gemma-4 PLE precedent): the module tree is
 * regular DSL modules (so `WeightMapper` binds every weight by name), but the forward
 * orchestration is bespoke because AltUp threads `numInputs` parallel hidden streams
 * through every layer — inexpressible as a plain Sequential:
 *
 * ```
 * h0       = embed(ids) * sqrt(hidden)
 * ple      = PerLayerEmbedding.compute(ids, h0)          # [B, S, L, pleDim]
 * streams  = altupGlobals.initStreams(h0)                # magnitude-renormed projections
 * per layer:
 *   preds       = altup.predict(streams)
 *   active      = preds[activeIdx]; an = attn_norm(active)
 *   laurel      = laurel(an)                              # an + norm(right(left(an)))
 *   attn        = post_attention_norm( MHA(an) )
 *   attnLaurel  = ((active + attn) + laurel) / √2
 *   ffw         = post_ffw_norm( ffn( ffn_norm(attnLaurel) ) )   # sparsity on first layers
 *   streams     = altup.correct(preds, attnLaurel + ffw)
 *   delta       = perLayerApply( scale(streams[active]), ple[:, :, layer] )
 *   streams[1:] += delta
 * merged   = altupGlobals.mergeStreams(streams)          # renormed mean
 * logits   = lm_head( output_norm(merged) )              # tied embeddings, no softcap
 * ```
 */
public class Gemma3nModel<T : DType, V>(
    public val tokenEmbedding: EmbeddingAdapter<T, V>,
    public val ple: PerLayerEmbedding<T, V>,
    public val altupGlobals: Gemma3nAltUpGlobals<T, V>,
    public val blocks: List<HybridTransformerBlock<T, V>>,
    public val outputNorm: RMSNormalization<T, V>,
    public val lmHead: VoidDense<T, V>,
    public val dtype: KClass<T>,
    public val activeIdx: Int,
    public val embedScale: Float,
    override val name: String = "Gemma3nModel",
) : Module<T, V>() {

    /**
     * Export/runtime injection point: when set, [onForward] uses this tensor
     * (`[batch, seq, numLayers, pleDim]`) as the per-layer inputs and skips [ple].compute —
     * the compiled StableHLO graph takes per_layer_inputs as a second INPUT, computed on
     * the CPU from the packed PLE table at runtime (PLE's whole design point: those
     * parameters stay off the accelerator). Eager decode leaves this null.
     */
    public var externalPerLayerInputs: Tensor<T, V>? = null

    override val modules: List<Module<T, V>> = buildList {
        add(tokenEmbedding)
        add(ple)
        add(altupGlobals)
        addAll(blocks)
        add(outputNorm)
        add(lmHead)
    }

    /** Typed handles into one block's module list (bound by construction in `gemma3nNetwork`). */
    public class BlockRefs<T : DType, V>(
        public val attnNorm: RMSNormalization<T, V>,
        public val mha: MultiHeadAttention<T, V>,
        public val postAttnNorm: RMSNormalization<T, V>,
        public val ffnNorm: RMSNormalization<T, V>,
        public val ffn: Gemma3nSparseGeGluFFN<T, V>,
        public val postFfwNorm: RMSNormalization<T, V>,
        public val laurel: Gemma3nLaurelBlock<T, V>,
        public val altup: Gemma3nAltUpBlock<T, V>,
        public val perLayer: Gemma3nPerLayerApply<T, V>,
    )

    @Suppress("UNCHECKED_CAST")
    private fun refsFor(block: HybridTransformerBlock<T, V>): BlockRefs<T, V> {
        val mods = block.modules
        fun norm(id: String): RMSNormalization<T, V> =
            mods.filterIsInstance<RMSNormalization<T, V>>().firstOrNull { it.name == id }
                ?: error("Gemma3nModel: block ${block.name} has no RMSNorm '$id'")
        return BlockRefs(
            attnNorm = norm("attn_norm"),
            mha = mods.filterIsInstance<MultiHeadAttention<T, V>>().first(),
            postAttnNorm = norm("post_attention_norm"),
            ffnNorm = norm("ffn_norm"),
            ffn = mods.filterIsInstance<Gemma3nSparseGeGluFFN<T, V>>().first(),
            postFfwNorm = norm("post_ffw_norm"),
            laurel = mods.filterIsInstance<Gemma3nLaurelBlock<T, V>>().first(),
            altup = mods.filterIsInstance<Gemma3nAltUpBlock<T, V>>().first(),
            perLayer = mods.filterIsInstance<Gemma3nPerLayerApply<T, V>>().first(),
        )
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val invSqrt2 = 0.70710678f

        // 1 — scaled embedding.
        val rawEmbeds = tokenEmbedding.forward(input, ctx)
        val h0 = if (embedScale != 1f) ops.mulScalar(rawEmbeds, embedScale) else rawEmbeds

        // 2 — per-layer inputs [B, S, L, pleDim] (identical math to gemma-4: token-identity
        // gather * sqrt(pleDim), context projection * hidden^-0.5, norm, sum * 1/sqrt2).
        val perLayerInputs = externalPerLayerInputs ?: run {
            val ids2d = if (input.rank == 1) ops.unsqueeze(input, 0) else input
            val embeds3d = if (h0.rank == 2) ops.unsqueeze(h0, 0) else h0
            ple.compute(ids2d, embeds3d, ctx, dtype)
        }

        // 3 — AltUp stream init.
        var streams = altupGlobals.initStreams(h0, ctx)

        // 4 — per-layer flow.
        for ((layerIdx, block) in blocks.withIndex()) {
            val r = refsFor(block)
            val preds = r.altup.predict(streams, ctx)
            val active = preds[activeIdx]
            val an = r.attnNorm.forward(active, ctx)
            val laurel = r.laurel.forward(an, ctx)
            val attn = r.postAttnNorm.forward(r.mha.forward(an, ctx), ctx)
            val attnGated = ops.add(active, attn)
            val attnLaurel = ops.mulScalar(ops.add(attnGated, laurel), invSqrt2)
            val ffw = r.postFfwNorm.forward(r.ffn.forward(r.ffnNorm.forward(attnLaurel, ctx), ctx), ctx)
            val corrected = r.altup.correct(preds, ops.add(attnLaurel, ffw), ctx)

            // Per-layer input: transform the (scaled) corrected active stream and add the
            // delta to the NON-active streams (HF: `corrected_predictions[1:] += ...`).
            val scaledActive = r.altup.scaleCorrectedOutput(corrected[activeIdx], ctx)
            val pliSlice = perLayerSlice(perLayerInputs, layerIdx, h0.rank, ctx)
            val delta = r.perLayer.computeDelta(scaledActive, pliSlice, ctx)
            streams = List(corrected.size) { i ->
                if (i == 0) corrected[0] else ops.add(corrected[i], delta)
            }
        }

        // 5 — merge streams, final norm, tied lm_head (gemma3n has no final softcap).
        val merged = altupGlobals.mergeStreams(streams, ctx)
        return lmHead.forward(outputNorm.forward(merged, ctx), ctx)
    }

    /** `per_layer_inputs[..., layerIdx, :]` matched to the trunk's working rank. */
    private fun perLayerSlice(
        perLayerInputs: Tensor<T, V>,
        layerIdx: Int,
        workingRank: Int,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        // [B, S, L, pleDim] → narrow L → [B, S, 1, pleDim] → squeeze → [B, S, pleDim]
        val slice = ops.squeeze(ops.narrow(perLayerInputs, dim = 2, start = layerIdx, length = 1), dim = 2)
        return if (workingRank == 2) ops.squeeze(slice, dim = 0) else slice
    }
}
