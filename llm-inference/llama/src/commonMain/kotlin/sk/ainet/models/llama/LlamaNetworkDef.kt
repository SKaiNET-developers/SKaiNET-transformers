package sk.ainet.models.llama

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.embedding
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.dsl.rmsNorm
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.swiGluFFN
import sk.ainet.lang.types.DType

/**
 * Llama architecture defined via the network DSL.
 *
 * Replaces the hand-coded [LlamaRuntime] with a declarative definition that:
 * - Builds a Module<T,V> tree (for direct execution and weight loading)
 * - Can be traced into a GraphProgram (DAG) for optimization
 *
 * Architecture: Embedding → N × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 *               RMSNorm → SwiGLU FFN → Residual) → RMSNorm → Dense
 *
 * Each transformer layer uses [TransformerBlock] (not the generic MLP) so that
 * [ResidualAdd] modules receive the correct skip-connection input.
 */
public inline fun <reified T : DType, V> llamaNetwork(
    metadata: LlamaModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096)
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.feedForwardLength
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val headDim = metadata.ropeDimensionCount ?: (dim / nHeads)
    val eps = 1e-5f

    return sequential<T, V> {
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        dslImpl.embedding(vocabSize, dim, id = "token_embd")

        // Build each transformer layer via the DSL helpers, but wrap in
        // TransformerBlock instead of MLP so residual connections work.
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
                rope(headDim, seqLen)
                kvCache(seqLen, nKVHeads, headDim)
            }
            stage.residual()

            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.swiGluFFN(dim, ffnDim, id = "ffn")
            stage.residual()

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
        }

        dslImpl.rmsNorm(dim, eps, id = "output_norm")
        // Use void placeholder for output projection to avoid allocating [vocabSize, dim] zeros.
        // Weights are loaded by WeightMapper.
        dslImpl.modules += VoidDenseModule<T, V>("output", vocabSize, dim)
    }
}

/**
 * Lightweight dense (linear) module with void placeholder weight.
 * No bias. Used for large projections (lm_head with vocab=131K) to avoid
 * allocating gigabytes of zero-initialized memory before weights are loaded.
 * Follows the same pattern as [SwiGLUFFN] for void weight creation.
 */
@Suppress("UNCHECKED_CAST")
@PublishedApi
internal class VoidDenseModule<T : DType, V>(
    override val name: String,
    outDim: Int,
    inDim: Int
) : sk.ainet.lang.nn.Module<T, V>(), sk.ainet.lang.nn.topology.ModuleParameters<T, V> {

    override val params: List<sk.ainet.lang.nn.topology.ModuleParameter<T, V>> = listOf(
        sk.ainet.lang.nn.topology.ModuleParameter.WeightParameter(
            "$name.weight",
            sk.ainet.lang.tensor.VoidOpsTensor(
                object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                    override val shape = sk.ainet.lang.tensor.Shape(outDim, inDim)
                    override fun get(vararg indices: Int): V = 0.0f as V
                    override fun set(vararg indices: Int, value: V) {}
                },
                Any::class as kotlin.reflect.KClass<T>
            )
        ),
        sk.ainet.lang.nn.topology.ModuleParameter.BiasParameter(
            "$name.bias",
            sk.ainet.lang.tensor.VoidOpsTensor(
                object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                    override val shape = sk.ainet.lang.tensor.Shape(outDim)
                    override fun get(vararg indices: Int): V = 0.0f as V
                    override fun set(vararg indices: Int, value: V) {}
                },
                Any::class as kotlin.reflect.KClass<T>
            )
        )
    )

    override val modules: List<sk.ainet.lang.nn.Module<T, V>> = emptyList()

    override fun onForward(input: sk.ainet.lang.tensor.Tensor<T, V>, ctx: sk.ainet.context.ExecutionContext): sk.ainet.lang.tensor.Tensor<T, V> {
        val ops = ctx.ops
        val weight = params[0].value
        // output = input @ weight^T (no bias — LLaMA lm_head has no bias)
        // Weight shape: [vocabSize, dim]. Input: [1, dim]. Result: [1, vocabSize].
        // Use matmul(input, transpose(weight)) but for very large vocab, transpose
        // alone allocates too much memory. Check if weight is already transposed-friendly.
        val wShape = weight.shape
        if (wShape.rank == 2 && wShape[0] > wShape[1]) {
            // Weight is [outDim, inDim] — typical case. Transpose needed.
            return ops.matmul(input, ops.transpose(weight))
        }
        // Weight is [inDim, outDim] — already transposed
        return ops.matmul(input, weight)
    }
}
