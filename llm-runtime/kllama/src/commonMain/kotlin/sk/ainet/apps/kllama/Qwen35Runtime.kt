package sk.ainet.apps.kllama

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.apps.llm.HeapKvCache
import sk.ainet.apps.llm.applyRopeRotation
import sk.ainet.apps.llm.softmaxInPlace
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata
import kotlin.reflect.KClass

/**
 * Qwen3.5 hybrid runtime supporting both DeltaNet (linear attention + SSM)
 * and full attention layers.
 *
 * Qwen3.5 alternates between DeltaNet layers and full attention layers
 * with a configurable interval (default: every 4th layer is full attention).
 *
 * DeltaNet layers use a linear recurrence with gated state updates:
 *   h_t = decay * h_{t-1} + beta_t * outer(k_t, q_t)
 *   y_t = h_t @ q_t
 *
 * Full attention layers use standard GQA with QK-norm and RoPE.
 */
public class Qwen35Runtime<T : DType>(
    private val ctx: ExecutionContext,
    private val metadata: LlamaModelMetadata,
    private val tensors: Map<String, Tensor<T, Float>>,
    private val dtype: KClass<T>,
    private val fullAttentionInterval: Int = 4,
    private val ssmStateSize: Int = 128,
    private val ssmConvKernel: Int = 4,
    private val ropeFreqBase: Float = 10_000_000f,
    maxContextLength: Int? = null,
    random: Random = Random.Default
) : DecoderRuntime<T>(random) {

    override val dim: Int = metadata.embeddingLength
    override val vocabSize: Int = metadata.vocabSize
    override val seqLen: Int = maxContextLength?.let { minOf(it, metadata.contextLength) }
        ?: minOf(metadata.contextLength, 8192) // cap for safety
    override val nLayers: Int = metadata.blockCount
    override val bosToken: Int = metadata.bosTokenId

    // Qwen3.5 does not use BOS — skip the BOS prepend from DecoderRuntime
    override fun generate(
        prompt: IntArray,
        steps: Int,
        temperature: Float,
        onToken: (Int) -> Unit
    ) {
        require(steps > 0) { "steps must be > 0" }
        // Use prompt as-is — no BOS prepend
        val fullPrompt = if (prompt.isEmpty()) intArrayOf(bosToken) else prompt
        var token = fullPrompt[0]
        var pos = 0
        var generatedCount = 0
        while (generatedCount < steps) {
            val logits = forward(token)
            val next = if (pos + 1 < fullPrompt.size) {
                fullPrompt[pos + 1]
            } else {
                sample(logits, temperature)
            }
            if (pos + 1 >= fullPrompt.size) {
                onToken(next)
                generatedCount++
            }
            token = next
            pos++
        }
    }

    // (diagnostic init block placed after all property declarations below)

    private val eps: Float = metadata.rmsNormEps

    // Derive head dimensions from norm weight shapes (more reliable than GGUF metadata
    // which may report halved head counts for Qwen3.5 hybrid architecture)
    private val headDim: Int = run {
        val kNorm = tensors["blk.${firstFullAttnLayer()}.attn_k_norm.weight"]
        if (kNorm != null) kNorm.shape[0]
        else metadata.ropeDimensionCount ?: (dim / metadata.headCount)
    }
    private val nKvHeads: Int = run {
        val kWeight = tensors["blk.${firstFullAttnLayer()}.attn_k.weight"]
        if (kWeight != null) kWeight.shape[0] / headDim else metadata.kvHeadCount
    }
    private val nHeads: Int = run {
        val oWeight = tensors["blk.${firstFullAttnLayer()}.attn_output.weight"]
        if (oWeight != null) oWeight.shape[1] / headDim else metadata.headCount
    }
    private val kvDim: Int = nKvHeads * headDim
    private val nHeadsPerKv: Int = nHeads / nKvHeads

    // DeltaNet parameters
    private val ssmInner: Int = dim
    private val ssmNumHeads: Int = ssmInner / ssmStateSize
    private val ropeDim: Int = metadata.ropeDimensionCount ?: 64

    // Full attention Q has doubled head dim (e.g. 512 vs 256 for K/V)
    private val fullAttnQHeadDim: Int = get("blk.${firstFullAttnLayer()}.attn_q.weight")
        .let { it.shape[0] / nHeads }
    private val fullAttnKHeadDim: Int = headDim

    // Embedding
    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = get("token_embd.weight"),
        name = "token_embd"
    )

    // Output
    private val outputNormLayer = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "output_norm",
        initWeight = get("output_norm.weight")
    )
    private val outputWeight = get("output.weight")

    // Per-layer norms
    private val attnNorms = (0 until nLayers).map { i ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "blk_$i.attn_norm",
            initWeight = get("blk.$i.attn_norm.weight")
        )
    }
    private val postAttnNorms = (0 until nLayers).map { i ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "blk_$i.post_attention_norm",
            initWeight = get("blk.$i.post_attention_norm.weight")
        )
    }

    // FFN weights
    private data class FFNWeights<T : DType>(
        val gate: Tensor<T, Float>,
        val down: Tensor<T, Float>,
        val up: Tensor<T, Float>
    )

    private val ffnWeights = (0 until nLayers).map { i ->
        FFNWeights(
            gate = get("blk.$i.ffn_gate.weight"),
            down = get("blk.$i.ffn_down.weight"),
            up = get("blk.$i.ffn_up.weight")
        )
    }

    // Full attention weights (only for full attention layers)
    private data class FullAttnWeights<T : DType>(
        val wq: Tensor<T, Float>,
        val wk: Tensor<T, Float>,
        val wv: Tensor<T, Float>,
        val wo: Tensor<T, Float>,
        val qNorm: Tensor<T, Float>,
        val kNorm: Tensor<T, Float>
    )

    private val fullAttnWeights: Map<Int, FullAttnWeights<T>> = (0 until nLayers)
        .filter { isFullAttentionLayer(it) }
        .associateWith { i ->
            FullAttnWeights(
                wq = get("blk.$i.attn_q.weight"),
                wk = get("blk.$i.attn_k.weight"),
                wv = get("blk.$i.attn_v.weight"),
                wo = get("blk.$i.attn_output.weight"),
                qNorm = get("blk.$i.attn_q_norm.weight"),
                kNorm = get("blk.$i.attn_k_norm.weight")
            )
        }

    // DeltaNet weights
    private data class DeltaNetWeights<T : DType>(
        val qkv: Tensor<T, Float>,
        val gate: Tensor<T, Float>,
        val ssmA: FloatArray,
        val ssmAlpha: Tensor<T, Float>,
        val ssmBeta: Tensor<T, Float>,
        val ssmConv1d: FloatArray,
        val ssmDtBias: FloatArray,
        val ssmNorm: FloatArray,
        val ssmOut: Tensor<T, Float>
    )

    private val deltaNetWeights: Map<Int, DeltaNetWeights<T>> = (0 until nLayers)
        .filter { !isFullAttentionLayer(it) }
        .associateWith { i ->
            DeltaNetWeights(
                qkv = get("blk.$i.attn_qkv.weight"),
                gate = get("blk.$i.attn_gate.weight"),
                ssmA = get("blk.$i.ssm_a").toFloatBuffer(),
                ssmAlpha = get("blk.$i.ssm_alpha.weight"),
                ssmBeta = get("blk.$i.ssm_beta.weight"),
                ssmConv1d = get("blk.$i.ssm_conv1d.weight").toFloatBuffer(),
                ssmDtBias = get("blk.$i.ssm_dt.bias").toFloatBuffer(),
                ssmNorm = get("blk.$i.ssm_norm.weight").toFloatBuffer(),
                ssmOut = get("blk.$i.ssm_out.weight")
            )
        }

    // KV cache for full attention layers
    private val fullAttnLayerIndices = (0 until nLayers).filter { isFullAttentionLayer(it) }
    private val fullAttnLayerToSlot = fullAttnLayerIndices.withIndex().associate { (slot, layer) -> layer to slot }
    private val kvCache = HeapKvCache(fullAttnLayerIndices.size, seqLen, kvDim)
    private val scoreBuffer = FloatArray(seqLen)

    // DeltaNet recurrent state
    private val deltaNetLayerIndices = (0 until nLayers).filter { !isFullAttentionLayer(it) }
    private val deltaNetLayerToSlot = deltaNetLayerIndices.withIndex().associate { (slot, layer) -> layer to slot }
    private val deltaNetStates = Array(deltaNetLayerIndices.size) {
        FloatArray(ssmNumHeads * ssmStateSize * ssmStateSize)
    }

    // DeltaNet conv1d state
    private val convStates = Array(deltaNetLayerIndices.size) {
        FloatArray(ssmConvKernel * ssmInner * 2)
    }
    private val convPositions = IntArray(deltaNetLayerIndices.size)

    init {
        println("Qwen35Runtime: dim=$dim nHeads=$nHeads nKvHeads=$nKvHeads headDim=$headDim ropeDim=$ropeDim")
        println("  fullAttnQHeadDim=$fullAttnQHeadDim fullAttnKHeadDim=$fullAttnKHeadDim")
        println("  ssmInner=$ssmInner ssmNumHeads=$ssmNumHeads ssmStateSize=$ssmStateSize")
        println("  nLayers=$nLayers fullAttnInterval=$fullAttentionInterval ropeFreqBase=$ropeFreqBase")
    }

    private fun get(name: String): Tensor<T, Float> =
        tensors[name] ?: error("Missing tensor: $name (available: ${tensors.keys.take(5)}...)")

    private fun Tensor<T, Float>.toFloatBuffer(): FloatArray {
        val data = this.data
        if (data is FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }

    private fun isFullAttentionLayer(layer: Int): Boolean =
        (layer + 1) % fullAttentionInterval == 0

    private fun firstFullAttnLayer(): Int =
        (0 until nLayers).first { isFullAttentionLayer(it) }

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        val normed = attnNorms[layerIdx].forward(x, ctx)

        val attnOut = if (isFullAttentionLayer(layerIdx)) {
            fullAttentionForward(layerIdx, normed, x)
        } else {
            deltaNetForward(layerIdx, normed, x)
        }

        // Post-attention norm → FFN → residual
        val ffnInput = postAttnNorms[layerIdx].forward(attnOut, ctx)
        val ffn = ffnWeights[layerIdx]
        val gate = matmulTransposed(ffnInput, ffn.gate).silu()
        val up = matmulTransposed(ffnInput, ffn.up)
        val ffnOut = matmulTransposed(gate * up, ffn.down)
        return attnOut + ffnOut
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> =
        outputNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> =
        matmulTransposed(x, outputWeight)

    override fun resetState() {
        kvCache.reset()
        for (state in deltaNetStates) state.fill(0f)
        for (state in convStates) state.fill(0f)
        convPositions.fill(0)
    }

    // ---- Full Attention Forward ----

    private fun fullAttentionForward(
        layerIdx: Int,
        normed: Tensor<T, Float>,
        residual: Tensor<T, Float>
    ): Tensor<T, Float> {
        val weights = fullAttnWeights[layerIdx]!!

        // Q/K/V projections
        var qTensor = matmulTransposed(normed, weights.wq)
        var kTensor = matmulTransposed(normed, weights.wk)
        val vTensor = matmulTransposed(normed, weights.wv)

        // Per-head QK-norm
        qTensor = applyPerHeadRMSNorm(qTensor, nHeads, fullAttnQHeadDim, weights.qNorm)
        kTensor = applyPerHeadRMSNorm(kTensor, nKvHeads, fullAttnKHeadDim, weights.kNorm)

        val qBuf = qTensor.toFloatBuffer()
        val kBuf = kTensor.toFloatBuffer()
        val vBuf = vTensor.toFloatBuffer()

        // Apply RoPE to the first ropeDim elements of each head
        applyRopeRotation(qBuf, nHeads, fullAttnQHeadDim, ropeDim, position, ropeFreqBase)
        applyRopeRotation(kBuf, nKvHeads, fullAttnKHeadDim, ropeDim, position, ropeFreqBase)

        // KV cache
        val cacheSlot = fullAttnLayerToSlot[layerIdx]!!
        kvCache.store(cacheSlot, position, kBuf, 0, vBuf, 0)

        // GQA attention — score uses first fullAttnKHeadDim dims of each Q head
        val out = FloatArray(nHeads * fullAttnKHeadDim)
        val scale = 1f / sqrt(fullAttnKHeadDim.toDouble()).toFloat()
        val scores = scoreBuffer

        for (h in 0 until nHeads) {
            val qOffset = h * fullAttnQHeadDim
            val kvHeadIdx = h / nHeadsPerKv
            val kvOffset = kvHeadIdx * fullAttnKHeadDim
            val outOffset = h * fullAttnKHeadDim

            for (t in 0..position) {
                var score = 0f
                for (i in 0 until fullAttnKHeadDim) {
                    score += qBuf[qOffset + i] * kvCache.getKey(cacheSlot, t, kvOffset, i)
                }
                scores[t] = score * scale
            }

            softmaxInPlace(scores, position + 1)

            for (t in 0..position) {
                val weight = scores[t]
                for (i in 0 until fullAttnKHeadDim) {
                    out[outOffset + i] += weight * kvCache.getValue(cacheSlot, t, kvOffset, i)
                }
            }
        }

        val attnOutTensor = ctx.fromFloatArray<T, Float>(Shape(1, nHeads * fullAttnKHeadDim), dtype, out)
        return residual + matmulTransposed(attnOutTensor, weights.wo)
    }

    // ---- DeltaNet Forward ----

    private fun deltaNetForward(
        layerIdx: Int,
        normed: Tensor<T, Float>,
        residual: Tensor<T, Float>
    ): Tensor<T, Float> {
        val w = deltaNetWeights[layerIdx]!!
        val slot = deltaNetLayerToSlot[layerIdx]!!
        val state = deltaNetStates[slot]
        val convState = convStates[slot]

        // 1. Project to QK: [1, dim] @ [dim, 2*ssmInner] → [1, 2*ssmInner]
        val qkTensor = matmulTransposed(normed, w.qkv)
        val qkBuf = qkTensor.toFloatBuffer()

        // 2. Causal conv1d on concatenated QK
        val qkConved = applyCausalConv1d(qkBuf, convState, w.ssmConv1d, slot)

        // 3. Split into Q and K, apply SiLU
        val q = FloatArray(ssmInner)
        val k = FloatArray(ssmInner)
        for (i in 0 until ssmInner) {
            q[i] = siluScalar(qkConved[i])
            k[i] = siluScalar(qkConved[ssmInner + i])
        }

        // 4. Compute alpha and beta
        val alphaTensor = matmulTransposed(normed, w.ssmAlpha)
        val betaTensor = matmulTransposed(normed, w.ssmBeta)
        val alphaRaw = alphaTensor.toFloatBuffer()
        val betaRaw = betaTensor.toFloatBuffer()

        // 5. Run DeltaNet recurrence per head
        val output = FloatArray(ssmInner)
        for (h in 0 until ssmNumHeads) {
            val qOffset = h * ssmStateSize
            val stateOffset = h * ssmStateSize * ssmStateSize

            val aVal = softplus(w.ssmA[h])
            val alphaVal = sigmoid(alphaRaw[h] + w.ssmDtBias[h])
            val betaVal = sigmoid(betaRaw[h])
            val decay = exp(-aVal * alphaVal)

            // State update: state = decay * state + beta * outer(k, q)
            for (i in 0 until ssmStateSize) {
                for (j in 0 until ssmStateSize) {
                    val idx = stateOffset + i * ssmStateSize + j
                    state[idx] = decay * state[idx] + betaVal * k[qOffset + i] * q[qOffset + j]
                }
            }

            // Output: y_h = state @ q_h, then RMS normalize
            var sumSq = 0f
            for (i in 0 until ssmStateSize) {
                var sum = 0f
                for (j in 0 until ssmStateSize) {
                    sum += state[stateOffset + i * ssmStateSize + j] * q[qOffset + j]
                }
                output[qOffset + i] = sum
                sumSq += sum * sum
            }

            val rms = sqrt(sumSq / ssmStateSize + eps)
            for (i in 0 until ssmStateSize) {
                output[qOffset + i] = (output[qOffset + i] / rms) * w.ssmNorm[i]
            }
        }

        // 6. Output projection: [1, ssmInner] @ ssmOut → [1, dim]
        val outputTensor = ctx.fromFloatArray<T, Float>(Shape(1, ssmInner), dtype, output)
        val projected = matmulTransposed(outputTensor, w.ssmOut)

        // 7. Gating: gate = silu(input @ gate_weight)
        val gateTensor = matmulTransposed(normed, w.gate).silu()
        val gated = projected * gateTensor

        return residual + gated
    }

    // ---- Causal Conv1d ----

    private fun applyCausalConv1d(
        input: FloatArray,
        convState: FloatArray,
        convWeight: FloatArray,
        slot: Int
    ): FloatArray {
        val innerDim = ssmInner * 2
        val pos = convPositions[slot]

        // Store input in circular buffer
        val stateOffset = (pos % ssmConvKernel) * innerDim
        input.copyInto(convState, stateOffset, 0, innerDim)
        convPositions[slot] = pos + 1

        // Apply depthwise conv1d — weight layout is [channels, kernel] in row-major
        val output = FloatArray(innerDim)
        for (d in 0 until innerDim) {
            var sum = 0f
            for (k in 0 until ssmConvKernel) {
                val stateIdx = ((pos + 1 - ssmConvKernel + k + ssmConvKernel * ssmConvKernel) % ssmConvKernel)
                sum += convState[stateIdx * innerDim + d] * convWeight[k * innerDim + d]
            }
            output[d] = sum
        }
        return output
    }

    // ---- Per-head RMS Norm ----

    private fun applyPerHeadRMSNorm(
        x: Tensor<T, Float>,
        numHeads: Int,
        headDim: Int,
        weight: Tensor<T, Float>
    ): Tensor<T, Float> {
        val buf = x.toFloatBuffer().copyOf()
        val w = weight.toFloatBuffer()

        for (h in 0 until numHeads) {
            val offset = h * headDim
            var sumSq = 0f
            for (i in 0 until headDim) {
                val v = buf[offset + i]
                sumSq += v * v
            }
            val rms = sqrt(sumSq / headDim + eps)
            for (i in 0 until headDim) {
                buf[offset + i] = (buf[offset + i] / rms) * w[i % w.size]
            }
        }
        return ctx.fromFloatArray<T, Float>(x.shape, dtype, buf)
    }

    // ---- Math utilities ----

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
    private fun softplus(x: Float): Float = ln(1f + exp(x))
    private fun siluScalar(x: Float): Float = x * sigmoid(x)

    // ---- Apply RoPE ----

    private fun applyRopeRotation(
        buf: FloatArray, nHeads: Int, headSize: Int,
        ropeDim: Int, pos: Int, base: Float
    ) {
        applyRopeRotation(buf, nHeads, headSize, ropeDim, pos, base, null, null, 0)
    }

    // ---- Memory-efficient matmul ----

    private val dequantCache = HashMap<Tensor<T, Float>, FloatArray>()

    private fun getWeightBuffer(w: Tensor<T, Float>): FloatArray {
        return dequantCache.getOrPut(w) { w.toFloatBuffer() }
    }

    /**
     * Compute y = x @ W^T where W is stored as [outDim, inDim].
     */
    private fun matmulTransposed(x: Tensor<T, Float>, w: Tensor<T, Float>): Tensor<T, Float> {
        val xBuf = x.toFloatBuffer()
        val wBuf = getWeightBuffer(w)
        val outDim = w.shape[0]
        val inDim = w.shape[1]
        val result = FloatArray(outDim)

        for (o in 0 until outDim) {
            var sum = 0f
            val wOffset = o * inDim
            for (i in 0 until inDim) {
                sum += xBuf[i] * wBuf[wOffset + i]
            }
            result[o] = sum
        }
        return ctx.fromFloatArray<T, Float>(Shape(1, outDim), dtype, result)
    }
}
