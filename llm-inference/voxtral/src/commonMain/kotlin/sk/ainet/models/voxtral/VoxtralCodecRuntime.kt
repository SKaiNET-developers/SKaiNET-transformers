package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Runtime for the Voxtral audio codec decoder.
 *
 * Converts semantic + acoustic token representations into a 24kHz audio waveform.
 *
 * **Architecture (8 flat decoder blocks):**
 * ```
 * Semantic embedding (8192 x 256) ++ Acoustic FSQ (36 raw values in [-1,1])
 *   -> concat -> [seqLen, 292]
 *   -> Block 0: CausalConv1d(292 -> 1024, k=3, s=1)
 *   -> Block 1: 2x TransformerLayer (window=2)
 *   -> Block 2: CausalConvTranspose1d(1024, k=4, s=2)  [upsample 2x]
 *   -> Block 3: 2x TransformerLayer (window=4)
 *   -> Block 4: CausalConvTranspose1d(1024, k=4, s=2)  [upsample 2x]
 *   -> Block 5: 2x TransformerLayer (window=8)
 *   -> Block 6: CausalConvTranspose1d(1024, k=4, s=2)  [upsample 2x]
 *   -> Block 7: 2x TransformerLayer (window=16)
 *   -> Output: CausalConv1d(1024 -> 240, k=7, s=1)
 *   -> reshape to audio waveform
 * ```
 *
 * Transformer layers: RMSNorm -> MHA (sliding window, QK-norm) -> LayerScale -> Residual
 *                   + RMSNorm -> SwiGLU FFN -> LayerScale -> Residual
 *
 * Convolutions use weight normalization: effective_weight = v * (g / ||v||).
 */
public class VoxtralCodecRuntime<T : DType>(
    private val weights: Map<String, Tensor<T, Float>>,
    private val metadata: VoxtralCodecMetadata,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>
) {
    private val ops get() = ctx.ops

    // Precompute weight-normalized convolution weights at construction time
    private val precomputedConvWeights: Map<String, Tensor<T, Float>> = buildPrecomputedWeights()

    /**
     * Decode semantic and acoustic codes to audio waveform.
     *
     * @param semanticCodes Semantic token IDs per frame, IntArray of length seqLen
     * @param acousticCodes Acoustic codes per frame, IntArray of length seqLen * nCodebooks
     *   (laid out as [frame0_cb0, frame0_cb1, ..., frame0_cb35, frame1_cb0, ...])
     * @return Audio samples as FloatArray, normalized to [-1, 1], at 24kHz
     */
    public fun decode(
        semanticCodes: IntArray,
        acousticCodes: IntArray
    ): FloatArray {
        val seqLen = semanticCodes.size
        val nCodebooks = metadata.acousticDim

        require(acousticCodes.size == seqLen * nCodebooks) {
            "Expected ${seqLen * nCodebooks} acoustic codes, got ${acousticCodes.size}"
        }

        // 1. Look up semantic embeddings: [seqLen, semanticDim]
        val semanticEmb = lookupEmbeddings(semanticCodes, seqLen)

        // 2. Map acoustic codes to raw FSQ values: [seqLen, acousticDim]
        val acousticRaw = mapFSQCodes(acousticCodes, seqLen, nCodebooks)

        // 3. Concatenate: [seqLen, semanticDim + acousticDim] = [seqLen, 292]
        var hidden = ops.concat(listOf(semanticEmb, acousticRaw), dim = 1)

        // 4. Reshape for 1D convolutions: [1, inputDim, seqLen]
        hidden = ops.reshape(hidden, Shape(1, seqLen, metadata.inputDim))
        hidden = permuteToConv(hidden) // [1, inputDim, seqLen]

        // 5. Run through 8 decoder blocks (alternating conv / transformer)
        var currentLength = seqLen
        val numStages = metadata.decoderConvsStrides.size
        for (stage in 0 until numStages) {
            val convBlock = stage * 2        // blocks 0, 2, 4, 6
            val transformerBlock = stage * 2 + 1  // blocks 1, 3, 5, 7
            val stride = metadata.decoderConvsStrides[stage]
            val kernelSize = metadata.decoderConvsKernels[stage]
            val nLayers = metadata.decoderTransformerLengths[stage]
            val windowSize = metadata.decoderWindowSizes[stage]

            // Convolution (block 0 is regular conv1d, blocks 2,4,6 are transposed conv)
            hidden = if (stage == 0) {
                causalConv1d(hidden, convBlock, kernelSize)
            } else {
                causalConvTranspose1d(hidden, convBlock, kernelSize, stride)
            }

            // Snake activation
            hidden = snakeActivation(hidden)

            // Update length after potential upsampling
            if (stage > 0) {
                currentLength *= stride
            }

            // Transformer layers
            for (layer in 0 until nLayers) {
                hidden = codecTransformerLayer(transformerBlock, layer, hidden, windowSize)
            }
        }

        // 6. Output projection: CausalConv1d(1024 -> 240, k=7, s=1)
        hidden = causalConv1dOutput(hidden)

        // 7. Extract and interleave audio samples
        // Shape: [1, patchSize, outLen] -> flatten to [outLen * patchSize] audio samples
        val data = hidden.data.copyToFloatArray()
        val outChannels = metadata.pretransformPatchSize
        val outLen = data.size / outChannels
        val audio = FloatArray(outLen * outChannels)

        // Interleave: for each position, write patchSize samples
        for (pos in 0 until outLen) {
            for (ch in 0 until outChannels) {
                audio[pos * outChannels + ch] = data[ch * outLen + pos]
            }
        }

        // Clamp to [-1, 1]
        for (i in audio.indices) {
            audio[i] = audio[i].coerceIn(-1.0f, 1.0f)
        }

        return audio
    }

    // ========== Embedding & Input ==========

    /**
     * Look up semantic embeddings from codebook.
     */
    @Suppress("UNCHECKED_CAST")
    private fun lookupEmbeddings(codes: IntArray, seqLen: Int): Tensor<T, Float> {
        val embDim = metadata.semanticDim
        val codebook = weights[VoxtralTensorNames.CODEC_SEMANTIC_CODEBOOK]
        if (codebook == null) {
            return ctx.fromFloatArray<T, Float>(Shape(seqLen, embDim), dtype, FloatArray(seqLen * embDim))
                as Tensor<T, Float>
        }

        val cbData = codebook.data
        val result = FloatArray(seqLen * embDim)
        for (i in 0 until seqLen) {
            val codeId = codes[i].coerceIn(0, metadata.semanticCodebookSize - 1)
            for (d in 0 until embDim) {
                result[i * embDim + d] = cbData.get(codeId, d) as Float
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(seqLen, embDim), dtype, result)
            as Tensor<T, Float>
    }

    /**
     * Map acoustic FSQ codes to raw values in [-1, 1].
     * Each code (0..levels-1) is linearly mapped: val = code * 2 / (levels - 1) - 1.
     */
    @Suppress("UNCHECKED_CAST")
    private fun mapFSQCodes(codes: IntArray, seqLen: Int, nCodebooks: Int): Tensor<T, Float> {
        val levels = metadata.acousticCodebookSize
        val result = FloatArray(seqLen * nCodebooks)
        for (frame in 0 until seqLen) {
            for (cb in 0 until nCodebooks) {
                val code = codes[frame * nCodebooks + cb].coerceIn(0, levels - 1)
                result[frame * nCodebooks + cb] = code.toFloat() * 2.0f / (levels - 1).toFloat() - 1.0f
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(seqLen, nCodebooks), dtype, result)
            as Tensor<T, Float>
    }

    // ========== Convolution Operations ==========

    /**
     * Causal Conv1d: left-pad by (kernel-1) so each output only depends on past+current inputs.
     * Uses weight normalization if available, otherwise falls back to pre-composed weight.
     */
    @Suppress("UNCHECKED_CAST")
    private fun causalConv1d(
        input: Tensor<T, Float>,
        block: Int,
        kernelSize: Int
    ): Tensor<T, Float> {
        val weight = getConvWeight(block) ?: return input
        val bias = weights[VoxtralTensorNames.codecBlockConvBias(block)]

        // Weight shape: [outChannels, inChannels, kernelSize]
        val outChannels = weight.shape[0]
        val inChannels = weight.shape[1]
        val kSize = weight.shape[2]

        // Input shape: [1, inChannels, length]
        val length = input.shape[2]
        val padLeft = kSize - 1
        val outLength = length  // stride=1, output same length

        val inputData = input.data.copyToFloatArray()
        val weightData = weight.data.copyToFloatArray()
        val biasData = bias?.data?.copyToFloatArray()
        val output = FloatArray(outChannels * outLength)

        for (oc in 0 until outChannels) {
            for (pos in 0 until outLength) {
                var sum = biasData?.get(oc) ?: 0.0f
                for (ic in 0 until inChannels) {
                    for (k in 0 until kSize) {
                        val inputPos = pos + k - padLeft
                        if (inputPos in 0 until length) {
                            sum += weightData[oc * inChannels * kSize + ic * kSize + k] *
                                inputData[ic * length + inputPos]
                        }
                    }
                }
                output[oc * outLength + pos] = sum
            }
        }

        return ctx.fromFloatArray<T, Float>(Shape(1, outChannels, outLength), dtype, output)
            as Tensor<T, Float>
    }

    /**
     * Causal ConvTranspose1d for upsampling.
     */
    private fun causalConvTranspose1d(
        input: Tensor<T, Float>,
        block: Int,
        kernelSize: Int,
        stride: Int
    ): Tensor<T, Float> {
        val weight = getConvWeight(block) ?: return input
        val bias = weights[VoxtralTensorNames.codecBlockConvBias(block)]

        // Use ops.convTranspose1d, then trim for causal alignment
        val padding = (kernelSize - 1) / 2
        var result = ops.convTranspose1d(
            input = input,
            weight = weight,
            bias = bias,
            stride = stride,
            padding = padding
        )

        // For causal: trim trailing samples to get correct output length
        val inputLen = input.shape[2]
        val expectedLen = inputLen * stride
        val actualLen = result.shape[2]
        if (actualLen > expectedLen) {
            result = ops.narrow(result, 2, 0, expectedLen)
        }

        return result
    }

    /**
     * Output projection: CausalConv1d(1024 -> 240, k=7, s=1)
     */
    @Suppress("UNCHECKED_CAST")
    private fun causalConv1dOutput(input: Tensor<T, Float>): Tensor<T, Float> {
        val weight = precomputedConvWeights["output_proj"]
            ?: weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ]
            ?: return input
        val bias = weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ_BIAS]

        val outChannels = weight.shape[0]
        val inChannels = weight.shape[1]
        val kSize = weight.shape[2]
        val length = input.shape[2]
        val padLeft = kSize - 1
        val outLength = length

        val inputData = input.data.copyToFloatArray()
        val weightData = weight.data.copyToFloatArray()
        val biasData = bias?.data?.copyToFloatArray()
        val output = FloatArray(outChannels * outLength)

        for (oc in 0 until outChannels) {
            for (pos in 0 until outLength) {
                var sum = biasData?.get(oc) ?: 0.0f
                for (ic in 0 until inChannels) {
                    for (k in 0 until kSize) {
                        val inputPos = pos + k - padLeft
                        if (inputPos in 0 until length) {
                            sum += weightData[oc * inChannels * kSize + ic * kSize + k] *
                                inputData[ic * length + inputPos]
                        }
                    }
                }
                output[oc * outLength + pos] = sum
            }
        }

        return ctx.fromFloatArray<T, Float>(Shape(1, outChannels, outLength), dtype, output)
            as Tensor<T, Float>
    }

    /**
     * Get convolution weight for a block, preferring precomputed weight-normalized version.
     */
    private fun getConvWeight(block: Int): Tensor<T, Float>? {
        return precomputedConvWeights["block_$block"]
            ?: weights[VoxtralTensorNames.codecBlockConvWeight(block)]
    }

    // ========== Weight Normalization ==========

    /**
     * Precompute weight-normalized convolution weights at construction time.
     * Weight norm: effective = v * (g / ||v||_per_output_channel)
     */
    @Suppress("UNCHECKED_CAST")
    private fun buildPrecomputedWeights(): Map<String, Tensor<T, Float>> {
        val result = mutableMapOf<String, Tensor<T, Float>>()

        // Decoder block convolutions
        for (block in listOf(0, 2, 4, 6)) {
            val g = weights[VoxtralTensorNames.codecBlockConvG(block)]
            val v = weights[VoxtralTensorNames.codecBlockConvV(block)]
            if (g != null && v != null) {
                result["block_$block"] = computeWeightNorm(g, v)
            }
        }

        // Output projection
        val outG = weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ_G]
        val outV = weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ_V]
        if (outG != null && outV != null) {
            result["output_proj"] = computeWeightNorm(outG, outV)
        }

        return result
    }

    /**
     * Compute weight normalization: effective = v * (g / ||v||_per_output_channel).
     * g shape: [outChannels, 1, 1] or [outChannels]
     * v shape: [outChannels, inChannels, kernelSize]
     */
    @Suppress("UNCHECKED_CAST")
    private fun computeWeightNorm(g: Tensor<T, Float>, v: Tensor<T, Float>): Tensor<T, Float> {
        val vData = v.data.copyToFloatArray()
        val gData = g.data.copyToFloatArray()
        val outChannels = v.shape[0]
        val innerSize = vData.size / outChannels
        val result = FloatArray(vData.size)

        for (oc in 0 until outChannels) {
            // Compute L2 norm of v for this output channel
            var normSq = 0.0f
            val offset = oc * innerSize
            for (i in 0 until innerSize) {
                normSq += vData[offset + i] * vData[offset + i]
            }
            val norm = sqrt(normSq.toDouble()).toFloat()
            val scale = if (norm > 0) gData[oc] / norm else 0.0f

            for (i in 0 until innerSize) {
                result[offset + i] = vData[offset + i] * scale
            }
        }

        return ctx.fromFloatArray<T, Float>(v.shape, dtype, result)
            as Tensor<T, Float>
    }

    // ========== Transformer Layer ==========

    /**
     * Full codec transformer layer with sliding window attention, QK-norm, and layer scale.
     *
     * 1. RMSNorm -> MHA (sliding window, QK-norm) -> LayerScale -> Residual
     * 2. RMSNorm -> SwiGLU FFN -> LayerScale -> Residual
     */
    private fun codecTransformerLayer(
        block: Int,
        layer: Int,
        input: Tensor<T, Float>,
        windowSize: Int
    ): Tensor<T, Float> {
        // Check if transformer weights exist for this block/layer
        val attnNormW = weights[VoxtralTensorNames.codecTransformerAttnNorm(block, layer)]
            ?: return input  // No weights -> pass through

        // Input shape: [1, dim, length] (conv format)
        // Transpose to [length, dim] for transformer ops
        val length = input.shape[2]
        val dim = input.shape[1]
        var x = permuteFromConv(input) // [1, length, dim]
        x = ops.reshape(x, Shape(length, dim))

        // ---- Attention block ----
        val residual1 = x

        // RMSNorm
        x = rmsNorm(x, attnNormW, metadata.normEps)

        // QKV projections: [length, dim] -> [length, dim] each
        val wq = weights[VoxtralTensorNames.codecTransformerAttnQ(block, layer)] ?: return input
        val wk = weights[VoxtralTensorNames.codecTransformerAttnK(block, layer)] ?: return input
        val wv = weights[VoxtralTensorNames.codecTransformerAttnV(block, layer)] ?: return input
        val wo = weights[VoxtralTensorNames.codecTransformerAttnOut(block, layer)] ?: return input

        var q = ops.matmul(x, ops.transpose(wq))  // [length, dim]
        var k = ops.matmul(x, ops.transpose(wk))
        val v = ops.matmul(x, ops.transpose(wv))

        // QK-norm (per-head RMSNorm on Q and K)
        if (metadata.qkNorm) {
            val qNormW = weights[VoxtralTensorNames.codecTransformerQNorm(block, layer)]
            val kNormW = weights[VoxtralTensorNames.codecTransformerKNorm(block, layer)]
            if (qNormW != null) q = perHeadRmsNorm(q, qNormW, metadata.nHeads, metadata.headDim, metadata.qkNormEps)
            if (kNormW != null) k = perHeadRmsNorm(k, kNormW, metadata.nHeads, metadata.headDim, metadata.qkNormEps)
        }

        // Reshape for multi-head attention: [length, dim] -> [1, nHeads, length, headDim]
        val nHeads = metadata.nHeads
        val headDim = metadata.headDim
        val qHeads = ops.reshape(q, Shape(1, length, nHeads, headDim))
        val kHeads = ops.reshape(k, Shape(1, length, nHeads, headDim))
        val vHeads = ops.reshape(v, Shape(1, length, nHeads, headDim))

        // Transpose to [1, nHeads, length, headDim]
        val qT = permuteHeads(qHeads)
        val kT = permuteHeads(kHeads)
        val vT = permuteHeads(vHeads)

        // Build sliding window mask
        val mask = buildSlidingWindowMask(length, windowSize)

        // Scaled dot-product attention
        val scale = 1.0f / sqrt(headDim.toFloat())
        val attnOut = ops.scaledDotProductAttention(
            query = qT,
            key = kT,
            value = vT,
            mask = mask,
            scale = scale,
            causal = false  // causal is handled by the mask
        )

        // Transpose back: [1, nHeads, length, headDim] -> [length, dim]
        val attnMerged = unpermuteHeads(attnOut)
        var attnResult = ops.reshape(attnMerged, Shape(length, dim))

        // Output projection
        attnResult = ops.matmul(attnResult, ops.transpose(wo))

        // Layer scale
        val attnScale = weights[VoxtralTensorNames.codecTransformerAttnScale(block, layer)]
        if (attnScale != null) {
            attnResult = ops.multiply(attnResult, attnScale)
        }

        // Residual
        x = ops.add(residual1, attnResult)

        // ---- FFN block ----
        val ffnNormW = weights[VoxtralTensorNames.codecTransformerFfnNorm(block, layer)]
        if (ffnNormW != null) {
            val residual2 = x

            // RMSNorm
            x = rmsNorm(x, ffnNormW, metadata.normEps)

            // SwiGLU: out = w2(silu(w1(x)) * w3(x))
            val w1 = weights[VoxtralTensorNames.codecTransformerFfnGate(block, layer)]
            val w2 = weights[VoxtralTensorNames.codecTransformerFfnDown(block, layer)]
            val w3 = weights[VoxtralTensorNames.codecTransformerFfnUp(block, layer)]

            if (w1 != null && w2 != null && w3 != null) {
                val gate = ops.silu(ops.matmul(x, ops.transpose(w1)))
                val up = ops.matmul(x, ops.transpose(w3))
                val gated = ops.multiply(gate, up)
                x = ops.matmul(gated, ops.transpose(w2))

                // Layer scale
                val ffnScale = weights[VoxtralTensorNames.codecTransformerFfnScale(block, layer)]
                if (ffnScale != null) {
                    x = ops.multiply(x, ffnScale)
                }

                // Residual
                x = ops.add(residual2, x)
            }
        }

        // Reshape back to conv format: [length, dim] -> [1, dim, length]
        x = ops.reshape(x, Shape(1, length, dim))
        x = permuteToConv(x) // [1, dim, length]

        return x
    }

    // ========== RMSNorm ==========

    /**
     * RMSNorm: x / sqrt(mean(x^2) + eps) * weight
     */
    private fun rmsNorm(input: Tensor<T, Float>, weight: Tensor<T, Float>, eps: Float): Tensor<T, Float> {
        val xSquared = ops.multiply(input, input)
        val meanSquared = ops.mean(xSquared, -1)
        val meanPlusEps = ops.addScalar(meanSquared, eps)
        val rms = ops.sqrt(meanPlusEps)
        val normalized = ops.divide(input, rms)
        return ops.multiply(normalized, weight)
    }

    /**
     * Per-head RMSNorm for QK normalization.
     * Input: [seqLen, dim], weight: [headDim]
     * Applies RMSNorm independently to each head's portion of the vector.
     */
    @Suppress("UNCHECKED_CAST")
    private fun perHeadRmsNorm(
        input: Tensor<T, Float>,
        weight: Tensor<T, Float>,
        nHeads: Int,
        headDim: Int,
        eps: Float
    ): Tensor<T, Float> {
        val seqLen = input.shape[0]
        val dim = input.shape[1]
        val data = input.data.copyToFloatArray()
        val wData = weight.data.copyToFloatArray()
        val result = FloatArray(data.size)

        for (s in 0 until seqLen) {
            for (h in 0 until nHeads) {
                val offset = s * dim + h * headDim
                // Compute RMS for this head
                var sumSq = 0.0f
                for (d in 0 until headDim) {
                    sumSq += data[offset + d] * data[offset + d]
                }
                val rms = sqrt((sumSq / headDim + eps).toDouble()).toFloat()
                // Normalize and scale
                for (d in 0 until headDim) {
                    result[offset + d] = data[offset + d] / rms * wData[d]
                }
            }
        }

        return ctx.fromFloatArray<T, Float>(input.shape, dtype, result)
            as Tensor<T, Float>
    }

    // ========== Attention Mask ==========

    /**
     * Build a causal sliding window attention mask.
     * Position i can attend to positions max(0, i - window + 1)..i.
     * Returns a [1, 1, seqLen, seqLen] mask tensor with 0 for allowed and -inf for blocked.
     */
    @Suppress("UNCHECKED_CAST")
    private fun buildSlidingWindowMask(seqLen: Int, windowSize: Int): Tensor<T, Float> {
        val mask = FloatArray(seqLen * seqLen)
        val negInf = -1e9f

        for (i in 0 until seqLen) {
            for (j in 0 until seqLen) {
                mask[i * seqLen + j] = if (j <= i && j >= i - windowSize + 1) 0.0f else negInf
            }
        }

        return ctx.fromFloatArray<T, Float>(Shape(1, 1, seqLen, seqLen), dtype, mask)
            as Tensor<T, Float>
    }

    // ========== Tensor Permutation Helpers ==========

    /**
     * Permute [1, length, dim] -> [1, dim, length] (to conv format).
     */
    @Suppress("UNCHECKED_CAST")
    private fun permuteToConv(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val length = tensor.shape[1]
        val dim = tensor.shape[2]
        val data = tensor.data.copyToFloatArray()
        val result = FloatArray(data.size)
        for (l in 0 until length) {
            for (d in 0 until dim) {
                result[d * length + l] = data[l * dim + d]
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(1, dim, length), dtype, result)
            as Tensor<T, Float>
    }

    /**
     * Permute [1, dim, length] -> [1, length, dim] (from conv format).
     */
    @Suppress("UNCHECKED_CAST")
    private fun permuteFromConv(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val dim = tensor.shape[1]
        val length = tensor.shape[2]
        val data = tensor.data.copyToFloatArray()
        val result = FloatArray(data.size)
        for (d in 0 until dim) {
            for (l in 0 until length) {
                result[l * dim + d] = data[d * length + l]
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(1, length, dim), dtype, result)
            as Tensor<T, Float>
    }

    /**
     * Permute [1, seqLen, nHeads, headDim] -> [1, nHeads, seqLen, headDim]
     */
    @Suppress("UNCHECKED_CAST")
    private fun permuteHeads(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val seqLen = tensor.shape[1]
        val nHeads = tensor.shape[2]
        val headDim = tensor.shape[3]
        val data = tensor.data.copyToFloatArray()
        val result = FloatArray(data.size)
        for (s in 0 until seqLen) {
            for (h in 0 until nHeads) {
                for (d in 0 until headDim) {
                    result[h * seqLen * headDim + s * headDim + d] =
                        data[s * nHeads * headDim + h * headDim + d]
                }
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(1, nHeads, seqLen, headDim), dtype, result)
            as Tensor<T, Float>
    }

    /**
     * Permute [1, nHeads, seqLen, headDim] -> [1, seqLen, nHeads, headDim]
     */
    @Suppress("UNCHECKED_CAST")
    private fun unpermuteHeads(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val nHeads = tensor.shape[1]
        val seqLen = tensor.shape[2]
        val headDim = tensor.shape[3]
        val data = tensor.data.copyToFloatArray()
        val result = FloatArray(data.size)
        for (h in 0 until nHeads) {
            for (s in 0 until seqLen) {
                for (d in 0 until headDim) {
                    result[s * nHeads * headDim + h * headDim + d] =
                        data[h * seqLen * headDim + s * headDim + d]
                }
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(1, seqLen, nHeads, headDim), dtype, result)
            as Tensor<T, Float>
    }

    // ========== Activation ==========

    /**
     * Snake activation: x + sin²(x)
     */
    private fun snakeActivation(input: Tensor<T, Float>): Tensor<T, Float> {
        val sinX = ops.sin(input)
        val sin2X = ops.multiply(sinX, sinX)
        return ops.add(input, sin2X)
    }
}
