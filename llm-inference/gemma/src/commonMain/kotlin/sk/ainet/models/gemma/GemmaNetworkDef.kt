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
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType

/**
 * Gemma architecture defined via the network DSL — simplified first pass.
 *
 * Unlocks the DSL → ComputeGraph (DAG) → ComputeGraphExecutor → CPU-on-JVM
 * execution path (Phase 5a of PLAN-unified-pipeline.md). Accuracy parity with
 * the hand-coded [Gemma4Runtime] is not the goal of this pass; the goal is to
 * close the DSL/loader/weight-mapping loop.
 *
 * Key differences from Llama already expressed here:
 * - GELU-gated FFN via [geGluFFN] (instead of SwiGLU/SiLU)
 *
 * Gemma features deliberately *not* expressed yet (deferred to Phase 5b, need
 * new DSL primitives):
 * - Proportional RoPE (p-RoPE) for global layers — uses standard RoPE here
 * - Sliding-window attention on a subset of layers — all layers are full attention
 * - Per-layer varying head_dim (global vs sliding) — uses one uniform head_dim
 * - Per-layer varying FFN intermediate size — uses the scalar intermediateSize
 * - Shared KV cache across the last `kvSharedLayers` layers — one cache per layer
 *
 * Architecture: Embedding → N × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 *               RMSNorm → GeGLUFFN → Residual) → RMSNorm → Dense(vocab)
 */
public inline fun <reified T : DType, V> gemmaNetwork(
    metadata: Gemma4ModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096)
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.intermediateSize
    // Cap sequence length to avoid RoPE precomputing gigabytes of cos/sin tables per layer
    // on long-context checkpoints (Gemma 4 E2B advertises 131 072).
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    // Phase 5a: treat all layers as global. Use globalHeadDim (falls back to headDim when equal).
    val headDim = if (metadata.globalHeadDim > 0) metadata.globalHeadDim else metadata.headDim
    val ropeBase = metadata.ropeParametersFull.base
    val eps = 1e-6f

    return sequential<T, V> {
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        dslImpl.embedding(vocabSize, dim, id = "token_embd")

        val nnCtx = DefaultNeuralNetworkExecutionContext()
        for (layer in 0 until nLayers) {
            val stage = StageImpl<T, V>(nnCtx, "blk.$layer", T::class)
            stage.rmsNorm(dim, eps, id = "attn_norm")
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                nKVHeads = nKVHeads,
                causal = true,
                id = "attn"
            ) {
                rope(headDim, seqLen, base = ropeBase)
                kvCache(seqLen, nKVHeads, headDim)
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
