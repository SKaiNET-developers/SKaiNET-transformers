package sk.ainet.models.gemma

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.embedding
import sk.ainet.lang.nn.dsl.geGluFFN
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.dsl.rmsNorm
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.transformer.AppendKVCache
import sk.ainet.lang.nn.transformer.KVCache
import sk.ainet.lang.nn.transformer.RoPEScaling
import sk.ainet.lang.nn.transformer.SharedKVCache
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType

/**
 * Gemma 4 architecture defined via the network DSL.
 *
 * Phase 5b: every Gemma 4 architectural feature the DSL has primitives for
 * is now expressed here, walking [Gemma4ModelMetadata.layerTypes] to pick
 * per-layer configuration:
 *
 * - **Full-attention layers** use proportional (NTK-aware) RoPE with the
 *   global base / scaling factor, the global `head_dim`, and
 *   `partialRotaryFactor = 0.5` (Gemma 4 convention).
 * - **Sliding-attention layers** use the standard RoPE base, the sliding
 *   `head_dim`, full-rotation, and a sliding-window mask of
 *   [Gemma4ModelMetadata.slidingWindow] tokens on the attention step.
 * - **Per-layer FFN width** via [Gemma4ModelMetadata.getIntermediateSize].
 * - **Shared KV cache** across the trailing `kvSharedLayers` layers: an
 *   [AppendKVCache] is built on the owner layer, and each follower gets a
 *   [SharedKVCache] that forwards to it. Follower shape must match the
 *   owner's, which holds for Gemma 4 because shared layers all fall in the
 *   same global/sliding group as the owner.
 *
 * Gemma-specific FFN: GELU-gated (via [geGluFFN]) instead of SwiGLU.
 *
 * @param metadata the parsed Gemma 4 model metadata.
 * @param maxInferenceLen upper bound on sequence length the cos/sin tables
 *   are sized for (and the KV cache capacity). Default caps to 4 096 so
 *   long-context (131 072) checkpoints don't precompute gigabytes of RoPE
 *   tables up front.
 */
public inline fun <reified T : DType, V> gemmaNetwork(
    metadata: Gemma4ModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096)
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val eps = 1e-6f

    // Gemma 4 rotates only half of head_dim on global (p-RoPE) layers; the
    // metadata field is per-scheme, but the two schemes share this fraction
    // in current checkpoints so we read it off the full params.
    val partialRotaryFactor = metadata.ropeParametersFull.partialRotaryFactor
    val scalingFactor = metadata.ropeParametersFull.factor

    return sequential<T, V> {
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        dslImpl.embedding(vocabSize, dim, id = "token_embd")

        val nnCtx = DefaultNeuralNetworkExecutionContext()
        // Owner layers' AppendKVCaches, indexed by their layer number. Follower
        // layers wrap these with SharedKVCache.
        val ownerCaches = mutableMapOf<Int, KVCache<T, V>>()

        for (layer in 0 until nLayers) {
            val layerHeadDim = metadata.getHeadDim(layer)
            val ropeBase = metadata.getRopeBase(layer)
            val isGlobal = metadata.getLayerType(layer) == LayerType.GLOBAL
            val ropeScaling = if (isGlobal && metadata.ropeParametersFull.ropeType == "proportional") {
                RoPEScaling.PROPORTIONAL
            } else {
                RoPEScaling.NONE
            }
            // Sliding layers get a bounded attention window; global layers see everything.
            val slidingWindow = if (isGlobal) null else metadata.slidingWindow
            val ffnDim = metadata.getIntermediateSize(layer)
            val cacheOwnerLayer = metadata.getCacheLayerIndex(layer)

            val stage = StageImpl<T, V>(nnCtx, "blk.$layer", T::class)
            stage.rmsNorm(dim, eps, id = "attn_norm")
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                nKVHeads = nKVHeads,
                causal = true,
                id = "attn",
                slidingWindow = slidingWindow
            ) {
                rope(
                    headDim = layerHeadDim,
                    maxSeqLen = seqLen,
                    base = ropeBase,
                    scaling = ropeScaling,
                    scalingFactor = if (ropeScaling == RoPEScaling.PROPORTIONAL) scalingFactor else 1.0f,
                    partialRotaryFactor = if (isGlobal) partialRotaryFactor else 1.0f
                )
                if (cacheOwnerLayer == layer) {
                    val owner = AppendKVCache<T, V>(
                        maxSeqLen = seqLen,
                        nKVHeads = nKVHeads,
                        headDim = layerHeadDim,
                        name = "blk.$layer.attn.kv_cache"
                    )
                    ownerCaches[layer] = owner
                    kvCache(owner)
                } else {
                    val owner = ownerCaches[cacheOwnerLayer]
                        ?: error("Gemma: layer $layer expects to share KV with layer $cacheOwnerLayer, but owner hasn't been built yet")
                    kvCache(SharedKVCache(owner, name = "blk.$layer.attn.kv_cache"))
                }
            }
            stage.residual()

            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.geGluFFN(dim, ffnDim, id = "ffn")
            stage.residual()

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
        }

        dslImpl.rmsNorm(dim, eps, id = "output_norm")
        // Void placeholder for output projection — Gemma 4 E2B vocab is 262 144
        // and dense(vocabSize) would eagerly allocate ~1.5 GB of zeros before
        // WeightMapper runs.
        dslImpl.modules += VoidDense<T, V>("output", vocabSize, dim)
    }
}
