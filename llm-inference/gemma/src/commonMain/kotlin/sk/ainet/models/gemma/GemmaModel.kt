package sk.ainet.models.gemma

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Top-level Gemma 4 model that orchestrates Per-Layer Embedding (PLE)
 * threading through the decoder layer stack.
 *
 * The generic DSL Sequential wrapper that `dslImpl.create()` produces can't
 * express PLE: each decoder layer needs both the rolling `hidden_states`
 * tensor AND a pre-computed, per-layer auxiliary tensor
 * `per_layer_inputs[:, :, layerIdx, :]`. This wrapper:
 *
 * 1. Runs [embedding] on the raw token IDs to get `inputs_embeds`.
 * 2. If [ple] is non-null, computes `per_layer_inputs` via
 *    [PerLayerEmbedding.compute] using both the raw IDs and `inputs_embeds`.
 * 3. For each block, slices the per-layer tensor, finds that block's
 *    [PerLayerInputBlockHook] in its module list, sets
 *    [PerLayerInputBlockHook.perLayerInput], and calls the block forward.
 *    The hook then consumes the field inside the block's sequential run.
 * 4. Applies final [outputNorm] and [lmHead] projections.
 *
 * When [ple] is null, this degenerates to a plain
 * `embedding → blocks → norm → lm_head` sequence — identical observable
 * behaviour to the old `dslImpl.create()` shape.
 *
 * Not a regular Sequential because the block loop needs to interleave
 * hook-field writes with module forwards. The cost of leaving the generic
 * DSL for one model is small; Gemma 4 is architecturally unique enough
 * that a dedicated top-level is justified.
 */
public class GemmaModel<T : DType, V>(
    public val tokenEmbedding: EmbeddingAdapter<T, V>,
    public val ple: PerLayerEmbedding<T, V>?,
    public val blocks: List<HybridTransformerBlock<T, V>>,
    public val outputNorm: RMSNormalization<T, V>,
    public val lmHead: VoidDense<T, V>,
    public val dtype: KClass<T>,
    override val name: String = "GemmaModel"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = buildList {
        add(tokenEmbedding)
        if (ple != null) add(ple)
        addAll(blocks)
        add(outputNorm)
        add(lmHead)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // Step 1: run main embedding. OptimizedLLMRuntime passes a token-id
        // tensor (Int32, shape [seq] — often [1] for single-token decode);
        // Embedding expands to [seq, hiddenSize].
        val inputsEmbeds = tokenEmbedding.forward(input, ctx)

        // Step 2: compute per_layer_inputs iff PLE is active.
        val perLayerInputs: Tensor<T, V>? = ple?.let { pleModule ->
            val tokenIds2d = ensureRank2Ids(input, ctx)
            val embeds3d = ensureRank3Embeds(inputsEmbeds, ctx)
            pleModule.compute(tokenIds2d, embeds3d, ctx, dtype)
        }

        // Step 3: iterate blocks. For each, pre-populate the hook's
        // perLayerInput with the corresponding slice if PLE is on.
        var hidden = inputsEmbeds
        for ((layerIdx, block) in blocks.withIndex()) {
            if (perLayerInputs != null) {
                val hook = findHook(block)
                    ?: error("GemmaModel: PLE is active but blk.$layerIdx has no PerLayerInputBlockHook")
                val slice3d = perLayerInputSliceFor(perLayerInputs, layerIdx, ctx)
                // Hook expects [batch, seq, perLayerDim] matching the
                // block's [seq, hiddenSize] working shape. Squeeze batch
                // dim if the block is operating in rank-2 mode.
                hook.perLayerInput = if (hidden.rank == 2) {
                    ctx.ops.squeeze(slice3d, dim = 0)
                } else {
                    slice3d
                }
            }
            hidden = block.forward(hidden, ctx)
        }

        // Step 4: final norm + lm_head.
        hidden = outputNorm.forward(hidden, ctx)
        return lmHead.forward(hidden, ctx)
    }

    @Suppress("UNCHECKED_CAST")
    private fun findHook(block: HybridTransformerBlock<T, V>): PerLayerInputBlockHook<T, V>? =
        block.modules.firstOrNull { it is PerLayerInputBlockHook<*, *> }
            as? PerLayerInputBlockHook<T, V>

    /** Extract `per_layer_inputs[..., layerIdx, :]` as a `[batch, seq, perLayerDim]` tensor. */
    private fun perLayerInputSliceFor(
        perLayerInputs: Tensor<T, V>,
        layerIdx: Int,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        val ops = ctx.ops
        // perLayerInputs shape: [batch, seq, numLayers, perLayerDim]
        // narrow on dim=2 from `layerIdx` for 1 step, then squeeze dim=2.
        val narrowed = ops.narrow(perLayerInputs, dim = 2, start = layerIdx, length = 1)
        return ops.squeeze(narrowed, dim = 2)
    }

    /** OptimizedLLMRuntime may pass a rank-1 token-id tensor `[seq]`
     *  (typically `[1]` for single-token decode). PLE requires
     *  `[batch, seq]` — unsqueeze a leading batch dim. */
    private fun ensureRank2Ids(ids: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = when (ids.rank) {
        2 -> ids
        1 -> ctx.ops.unsqueeze(ids, dim = 0)
        else -> error("GemmaModel: expected token-id tensor of rank 1 or 2, got ${ids.rank}")
    }

    /** inputsEmbeds may come in as `[seq, hidden]` (rank 2) when the
     *  runtime decodes single-token, or `[batch, seq, hidden]` (rank 3)
     *  when batched. PLE needs rank 3 (`[batch, seq, hidden]`) to combine
     *  with the `[batch, seq, numLayers, perLayerDim]` PLE tensor. */
    private fun ensureRank3Embeds(embeds: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = when (embeds.rank) {
        3 -> embeds
        2 -> ctx.ops.unsqueeze(embeds, dim = 0)
        else -> error("GemmaModel: expected inputs_embeds of rank 2 or 3, got ${embeds.rank}")
    }

    public companion object {
        /**
         * Walk [stageModules] and return the first [PerLayerInputBlockHook]
         * found, or `null`. Package-internal helper for tests that want to
         * poke at a block's hook without reflection.
         */
        @Suppress("UNCHECKED_CAST")
        public fun <T : DType, V> findHookIn(
            stageModules: List<Module<T, V>>
        ): PerLayerInputBlockHook<T, V>? =
            stageModules.firstOrNull { it is PerLayerInputBlockHook<*, *> }
                as? PerLayerInputBlockHook<T, V>
    }
}
