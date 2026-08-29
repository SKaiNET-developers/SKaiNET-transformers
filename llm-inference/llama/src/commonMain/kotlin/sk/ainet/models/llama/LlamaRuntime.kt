package sk.ainet.models.llama

import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Unified LLaMA decoder runtime with pluggable attention backend.
 *
 * The attention strategy (CPU vs GPU) is injected via [AttentionBackend].
 * All other logic (embedding, norms, projections, FFN, sampling, generate loop)
 * is shared.
 *
 * Extends [DecoderRuntime] for shared forward/generate/sample logic.
 * Adds batch-prefill optimization for long prompts.
 *
 * @param ctx ExecutionContext for tensor operations
 * @param weights LLaMA model weights
 * @param attentionBackend Strategy for attention computation (RoPE + KV cache + attention)
 * @param eps Epsilon for RMS normalization
 * @param random Random generator for sampling
 */
@Deprecated(
    message = "Use OptimizedLLMRuntime with llamaNetwork() instead. " +
        "See docs/optimizable-LLM-NNs-DAG.md for migration guide.",
    replaceWith = ReplaceWith(
        "OptimizedLLMRuntime.create(llamaNetwork(config), tensors, resolver, ctx)",
        "sk.ainet.apps.llm.OptimizedLLMRuntime"
    )
)
public class LlamaRuntime<T : DType>(
    private val ctx: ExecutionContext,
    val weights: LlamaRuntimeWeights<T>,
    private val attentionBackend: AttentionBackend<T>,
    private val dtype: KClass<T>,
    private val eps: Float = 1e-5f,
    random: Random = Random.Default,
) : DecoderRuntime<T>(random), LlamaRuntimeInterface<T> {

    private companion object {
        const val DEFAULT_BOS_TOKEN: Int = 1
    }

    // NOTE: weights are transposed on-the-fly during forward pass rather than
    // pre-transposed at init. This halves peak memory (~31GB saved for 8B models)
    // at the cost of per-token transpose allocations that the GC reclaims.
    // Quantized weights (Q4_K) skip transpose entirely — their matmul kernel
    // handles the [out, in] layout directly.

    /**
     * Linear projection: y = x @ W.
     *
     * When weights are pre-transposed to [in, out] by MemSegWeightConverter
     * (Q4_K, Q6_K, FP32 via NATIVE_OPTIMIZED), uses direct matmul.
     * Otherwise falls back to .t() for non-converted weights (tests, DEQUANTIZE_TO_FP32).
     */
    private fun linearProject(x: Tensor<T, Float>, w: Tensor<T, Float>): Tensor<T, Float> {
        val xCols = if (x.shape.rank >= 2) x.shape[x.shape.rank - 1] else x.shape[0]
        val wRows = w.shape[0]
        return if (wRows == xCols) {
            // Weight is [in, out] — already transposed, direct matmul
            x.matmul(w)
        } else {
            // Weight is [out, in] — needs transpose (legacy path)
            x.matmul(w.t())
        }
    }

    // ---- DecoderRuntime abstract properties ----
    override val dim: Int = weights.metadata.embeddingLength
    override val seqLen: Int = weights.metadata.contextLength
    override val vocabSize: Int = weights.metadata.vocabSize
    override val nLayers: Int = weights.layers.size
    override val bosToken: Int = weights.metadata.bosTokenId

    private val nHeads: Int = weights.metadata.headCount
    private val nKvHeads: Int = weights.metadata.kvHeadCount
    private val headDim: Int = dim / nHeads
    private val hasQKNorm: Boolean = weights.layers.firstOrNull()?.qNorm != null

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val outputNormLayer = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "output_norm",
        initWeight = weights.outputNorm
    )

    private val attnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.attn_norm",
            initWeight = layer.attnNorm
        )
    }

    private val ffnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.ffn_norm",
            initWeight = layer.ffnNorm
        )
    }

    override val currentPosition: Int
        get() = position

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        val layer = weights.layers[layerIdx]

        val attnNorm = attnNorms[layerIdx].forward(x, ctx)
        var q = linearProject(attnNorm, layer.wq)
        var k = linearProject(attnNorm, layer.wk)
        val v = linearProject(attnNorm, layer.wv)

        // QK-norm (Qwen3, Apertus-style): per-head RMSNorm on Q and K before RoPE
        if (hasQKNorm) {
            q = applyPerHeadRMSNorm(q, nHeads, headDim, layer.qNorm!!)
            k = applyPerHeadRMSNorm(k, nKvHeads, headDim, layer.kNorm!!)
        }

        // Delegate attention (RoPE + KV cache + scoring) to backend
        val attnOut = attentionBackend.attention(q, k, v, layerIdx, position)

        // Output projection + residual
        val afterAttn = x + linearProject(attnOut, layer.wo)

        val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)
        val gate = linearProject(ffnNorm, layer.ffnGate).silu()
        val up = linearProject(ffnNorm, layer.ffnUp)
        val ffnOut = linearProject(gate * up, layer.ffnDown)
        return afterAttn + ffnOut
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> =
        outputNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> =
        linearProject(x, weights.outputWeight)

    override fun resetState() {
        attentionBackend.reset()
    }

    // ---- QK-norm (per-head RMSNorm on Q/K) ----

    private fun applyPerHeadRMSNorm(
        x: Tensor<T, Float>,
        numHeads: Int,
        headDim: Int,
        weight: Tensor<T, Float>
    ): Tensor<T, Float> {
        val buf = x.expectFloatBuffer().copyOf()
        val w = weight.expectFloatBuffer()
        val totalDim = numHeads * headDim
        val batchSize = if (x.shape.rank == 2) x.shape[0] else 1

        for (b in 0 until batchSize) {
            val batchOffset = b * totalDim
            for (h in 0 until numHeads) {
                val headOffset = batchOffset + h * headDim
                var sumSq = 0f
                for (i in 0 until headDim) {
                    val v = buf[headOffset + i]
                    sumSq += v * v
                }
                val rms = sqrt(sumSq / headDim + eps)
                for (i in 0 until headDim) {
                    buf[headOffset + i] = (buf[headOffset + i] / rms) * w[i]
                }
            }
        }
        return ctx.fromFloatArray<T, Float>(x.shape, dtype, buf)
    }

    // ---- LLaMA-specific batch optimization ----

    /**
     * Override generate to add batch-prefill optimization for long prompts.
     */
    override fun generate(prompt: IntArray, steps: Int, temperature: Float, onToken: (Int) -> Unit) {
        require(steps > 0) { "steps must be > 0" }

        val fullPrompt = if (prompt.isNotEmpty() && prompt[0] != bosToken) {
            intArrayOf(bosToken) + prompt
        } else if (prompt.isEmpty()) {
            intArrayOf(bosToken)
        } else {
            prompt
        }

        // Phase 1: Batch-process prompt tokens in chunks (only for longer prompts)
        val batchChunkSize = 256

        if (fullPrompt.size > batchChunkSize) {
            // Process all prompt tokens except the last one in batches
            val promptTokens = fullPrompt.sliceArray(0 until fullPrompt.size - 1)
            var promptPos = 0
            while (promptPos < promptTokens.size) {
                val chunkEnd = minOf(promptPos + batchChunkSize, promptTokens.size)
                val chunk = promptTokens.sliceArray(promptPos until chunkEnd)
                batchForward(chunk, promptPos)
                promptPos = chunkEnd
            }
            // Phase 2: Auto-regressive generation starting from last prompt token
            var token = fullPrompt[fullPrompt.size - 1]
            var generatedCount = 0
            while (generatedCount < steps) {
                val logits = forward(token)
                val next = sample(logits, temperature)
                onToken(next)
                generatedCount++
                token = next
            }
        } else {
            // Short prompts: use the base class sequential path
            super.generate(fullPrompt, steps, temperature, onToken)
        }
    }

    /**
     * Batch-forward a chunk of tokens through all layers.
     *
     * Embeds all tokens at once, runs each layer with batch-aware attention
     * (if the backend supports it, otherwise falls back to sequential),
     * and returns the logits for the last token.
     *
     * @param tokenIds Array of token IDs to process
     * @param startPos Starting position in the sequence
     * @return Logits tensor for the **last** token in the batch
     */
    public fun batchForward(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        require(tokenIds.isNotEmpty()) { "tokenIds must not be empty" }
        require(startPos + tokenIds.size <= seqLen) {
            "Batch would exceed context: startPos=$startPos, batchSize=${tokenIds.size}, seqLen=$seqLen"
        }

        // Try the full batch path; batchForwardFull falls back to sequential
        // if the attention backend returns null from batchAttention.
        if (tokenIds.size > 1) {
            return batchForwardFull(tokenIds, startPos)
        }

        // Single token: use regular forward
        position = startPos
        return forward(tokenIds[0])
    }

    private fun batchForwardFull(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        // Embed all tokens: produces [batchSize, dim]
        var x = embedding.forward(tokenIds, ctx)

        weights.layers.forEachIndexed { layerIdx, layer ->
            val attnNorm = attnNorms[layerIdx].forward(x, ctx)
            var q = linearProject(attnNorm, layer.wq)
            var k = linearProject(attnNorm, layer.wk)
            val v = linearProject(attnNorm, layer.wv)

            if (hasQKNorm) {
                q = applyPerHeadRMSNorm(q, nHeads, headDim, layer.qNorm!!)
                k = applyPerHeadRMSNorm(k, nKvHeads, headDim, layer.kNorm!!)
            }

            val attnOut = attentionBackend.batchAttention(q, k, v, layerIdx, startPos)
                ?: return batchForwardFallback(tokenIds, startPos)

            val afterAttn = x + linearProject(attnOut, layer.wo)

            val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)
            val gate = linearProject(ffnNorm, layer.ffnGate).silu()
            val up = linearProject(ffnNorm, layer.ffnUp)
            val ffnOut = linearProject(gate * up, layer.ffnDown)
            x = afterAttn + ffnOut
        }

        val norm = outputNormLayer.forward(x, ctx)
        val logits = linearProject(norm, weights.outputWeight)
        position = startPos + tokenIds.size
        return logits
    }

    private fun batchForwardFallback(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        position = startPos
        var logits: Tensor<T, Float>? = null
        for (tokenId in tokenIds) {
            logits = forward(tokenId)
        }
        return logits!!
    }
}
