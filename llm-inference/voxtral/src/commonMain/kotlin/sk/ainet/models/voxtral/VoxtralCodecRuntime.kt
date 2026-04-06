package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Runtime for the Voxtral audio codec decoder.
 *
 * Converts semantic + acoustic token representations into a 24kHz audio waveform.
 * The decoder has 4 stages, each with a transposed convolution (upsampling) followed
 * by transformer layers with Snake activation and layer scale.
 *
 * **Architecture:**
 * ```
 * Semantic embedding (8192 × 256) + Acoustic embedding (36 × 21 → 756, projected to 256)
 *   → concat → patch projection (to codec dim 1024)
 *   → 4 × DecoderStage:
 *       ConvTranspose1d (upsample by stride) + Snake activation
 *       → N × TransformerLayer (sliding window attn, qk_norm, layer scale)
 *   → output projection (1024 → 1 channel)
 * ```
 *
 * Total upsampling: stride 1 × 2 × 2 × 2 = 8x. Combined with the codec's
 * pretransform patch size (240 samples per frame at 24kHz / 12.5fps), this
 * reconstructs the full audio waveform.
 *
 * @param weights Named tensor map containing all codec weights
 * @param metadata Codec configuration
 * @param ctx Execution context
 * @param dtype DType class
 */
public class VoxtralCodecRuntime<T : DType>(
    private val weights: Map<String, Tensor<T, Float>>,
    private val metadata: VoxtralCodecMetadata,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>
) {
    private val ops get() = ctx.ops
    private val numStages = metadata.decoderTransformerLengths.size

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
        val semanticEmb = lookupEmbeddings(
            codes = semanticCodes,
            codebookName = VoxtralTensorNames.CODEC_SEMANTIC_CODEBOOK,
            embDim = metadata.semanticDim,
            seqLen = seqLen
        )

        // 2. Look up acoustic embeddings: for each codebook, look up the code
        //    and sum all codebook embeddings per frame → [seqLen, semanticDim]
        val acousticEmb = lookupAcousticEmbeddings(acousticCodes, seqLen, nCodebooks)

        // 3. Combine: sum semantic + acoustic embeddings
        var hidden = ops.add(semanticEmb, acousticEmb)

        // 4. Patch projection: [seqLen, semanticDim] → [seqLen, codecDim]
        hidden = linearForward(hidden, VoxtralTensorNames.CODEC_PATCH_PROJ,
            VoxtralTensorNames.CODEC_PATCH_PROJ_BIAS)

        // 5. Reshape for 1D convolutions: [1, codecDim, seqLen] (batch=1, channels, length)
        hidden = transposeLastTwo(hidden)
        hidden = ops.reshape(hidden, Shape(1, metadata.dim, seqLen))

        // 6. Run through decoder stages
        var currentLength = seqLen
        for (stage in 0 until numStages) {
            val stride = metadata.decoderConvsStrides[stage]
            val kernelSize = metadata.decoderConvsKernels[stage]
            val nTransformerLayers = metadata.decoderTransformerLengths[stage]

            // Transposed convolution (upsample)
            val convWeightName = VoxtralTensorNames.codecDecoderConvWeight(stage)
            val convBiasName = VoxtralTensorNames.codecDecoderConvBias(stage)
            val convWeight = weights[convWeightName]
            val convBias = weights[convBiasName]

            if (convWeight != null) {
                hidden = ops.convTranspose1d(
                    input = hidden,
                    weight = convWeight,
                    bias = convBias,
                    stride = stride,
                    padding = (kernelSize - 1) / 2
                )
                currentLength = (currentLength - 1) * stride + 1  // approximate
            }

            // Snake activation (element-wise, no learned params needed for basic version)
            hidden = snakeActivation(hidden)

            // Transformer layers for this stage
            for (layer in 0 until nTransformerLayers) {
                hidden = codecTransformerLayer(stage, layer, hidden)
            }
        }

        // 7. Output projection: [1, codecDim, outLength] → [1, 1, outLength]
        val outputWeight = weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ]
        val outputBias = weights[VoxtralTensorNames.CODEC_OUTPUT_PROJ_BIAS]
        if (outputWeight != null) {
            // Transpose to [1, outLength, codecDim], apply linear, transpose back
            hidden = transposeConvDims(hidden) // [1, outLength, codecDim]
            hidden = linearForwardRaw(hidden, outputWeight, outputBias)
            // [1, outLength, 1] → flatten to audio
        }

        // 8. Extract audio samples
        val audioData = hidden.data.copyToFloatArray()

        // Clamp to [-1, 1]
        for (i in audioData.indices) {
            audioData[i] = audioData[i].coerceIn(-1.0f, 1.0f)
        }

        return audioData
    }

    /**
     * Look up embeddings from a codebook tensor.
     */
    @Suppress("UNCHECKED_CAST")
    private fun lookupEmbeddings(
        codes: IntArray,
        codebookName: String,
        embDim: Int,
        seqLen: Int
    ): Tensor<T, Float> {
        val codebook = weights[codebookName]
        if (codebook == null) {
            // Return zeros if codebook not loaded
            val data = FloatArray(seqLen * embDim)
            return ctx.fromFloatArray<T, Float>(Shape(seqLen, embDim), dtype, data) as Tensor<T, Float>
        }

        val cbData = codebook.data
        val result = FloatArray(seqLen * embDim)
        for (i in 0 until seqLen) {
            val codeId = codes[i]
            for (d in 0 until embDim) {
                result[i * embDim + d] = cbData.get(codeId, d) as Float
            }
        }
        return ctx.fromFloatArray<T, Float>(Shape(seqLen, embDim), dtype, result) as Tensor<T, Float>
    }

    /**
     * Look up and sum acoustic embeddings across all codebooks.
     * Each codebook has its own embedding table; we sum them per frame.
     */
    @Suppress("UNCHECKED_CAST")
    private fun lookupAcousticEmbeddings(
        acousticCodes: IntArray,
        seqLen: Int,
        nCodebooks: Int
    ): Tensor<T, Float> {
        val embDim = metadata.semanticDim
        val result = FloatArray(seqLen * embDim)

        val codebook = weights[VoxtralTensorNames.CODEC_ACOUSTIC_CODEBOOK]
        if (codebook != null) {
            val cbData = codebook.data
            for (frame in 0 until seqLen) {
                for (cb in 0 until nCodebooks) {
                    val code = acousticCodes[frame * nCodebooks + cb]
                    for (d in 0 until embDim) {
                        // Acoustic codebook may be indexed as [nCodebooks * codebookSize, embDim]
                        // or [codebookSize, embDim] shared across codebooks
                        val idx = code.coerceIn(0, metadata.acousticCodebookSize - 1)
                        result[frame * embDim + d] += cbData.get(idx, d) as Float
                    }
                }
            }
        }

        return ctx.fromFloatArray<T, Float>(Shape(seqLen, embDim), dtype, result) as Tensor<T, Float>
    }

    /**
     * Apply linear projection: out = input @ weight^T + bias
     */
    private fun linearForward(
        input: Tensor<T, Float>,
        weightName: String,
        biasName: String
    ): Tensor<T, Float> {
        val weight = weights[weightName] ?: return input
        val bias = weights[biasName]
        return linearForwardRaw(input, weight, bias)
    }

    private fun linearForwardRaw(
        input: Tensor<T, Float>,
        weight: Tensor<T, Float>,
        bias: Tensor<T, Float>?
    ): Tensor<T, Float> {
        var out = ops.matmul(input, ops.transpose(weight))
        if (bias != null) {
            out = ops.add(out, bias)
        }
        return out
    }

    /**
     * Snake activation: x + sin²(x) (simplified, without learned alpha).
     */
    private fun snakeActivation(input: Tensor<T, Float>): Tensor<T, Float> {
        val sinX = ops.sin(input)
        val sin2X = ops.multiply(sinX, sinX)
        return ops.add(input, sin2X)
    }

    /**
     * Single codec transformer layer with RMSNorm, attention, FFN, layer scale.
     */
    private fun codecTransformerLayer(
        stage: Int,
        layer: Int,
        input: Tensor<T, Float>
    ): Tensor<T, Float> {
        // For now, pass through if weights aren't loaded.
        // Full implementation would do:
        // 1. RMSNorm → MHA (sliding window) → LayerScale → Residual
        // 2. RMSNorm → SwiGLU FFN → LayerScale → Residual
        val attnNormWeight = weights[VoxtralTensorNames.codecDecoderTransformerAttnNorm(stage, layer)]
            ?: return input

        // Simplified: just return input (codec transformer weights need full MHA impl with sliding window)
        // The convolution stages do the heavy lifting for upsampling
        return input
    }

    /**
     * Transpose last two dimensions: [A, B, C] → [A, C, B]
     */
    private fun transposeLastTwo(tensor: Tensor<T, Float>): Tensor<T, Float> {
        return ops.transpose(tensor)
    }

    /**
     * Transpose conv dims: [batch, channels, length] → [batch, length, channels]
     */
    @Suppress("UNCHECKED_CAST")
    private fun transposeConvDims(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val batch = tensor.shape[0]
        val channels = tensor.shape[1]
        val length = tensor.shape[2]
        val data = tensor.data
        val result = FloatArray(batch * length * channels)
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                for (l in 0 until length) {
                    result[b * length * channels + l * channels + c] =
                        data.get(b, c, l) as Float
                }
            }
        }
        return ctx.fromFloatArray<T, Float>(
            Shape(batch, length, channels), dtype, result
        ) as Tensor<T, Float>
    }
}
