package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Runtime for the Voxtral acoustic flow-matching pipeline.
 *
 * The acoustic model generates 36 continuous acoustic values per frame from
 * backbone hidden states via iterative flow-matching denoising.
 *
 * **Per-frame velocity computation (3-token sequence):**
 * 1. Token 0: `inputProj(x_t)` — noised acoustic sample projected to model dim
 * 2. Token 1: `timeProj(sinusoidal_embed(t))` — timestep encoding
 * 3. Token 2: `llmProj(backbone_hidden)` — backbone conditioning
 * 4. Forward 3-token sequence through 3-layer **bidirectional** transformer
 * 5. Extract first token → RMSNorm → `outputProj` → velocity in acoustic space
 *
 * **Classifier-Free Guidance (CFG):**
 * Two passes per step: conditioned (real backbone hidden) + unconditioned (zeros).
 * Combined: `v = alpha * v_cond + (1 - alpha) * v_uncond` with alpha=1.2.
 *
 * After all Euler steps, continuous values are quantized to FSQ codes [0, levels-1].
 */
public class VoxtralAcousticRuntime<T : DType>(
    private val weights: Map<String, Tensor<T, Float>>,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val nCodebooks: Int = 36,
    private val codebookLevels: Int = 21,
    private val dim: Int = 3072,
    private val nLayers: Int = 3,
    private val nHeads: Int = 32,
    private val nKVHeads: Int = 8,
    private val ffnDim: Int = 9216,
    private val normEps: Float = 1e-5f,
    private val cfgAlpha: Float = 1.2f
) {
    private val ops get() = ctx.ops
    private val headDim = dim / nHeads
    private val kvDim = nKVHeads * headDim
    private val acousticDim = nCodebooks  // 36: one continuous value per codebook
    private val flowMatching = VoxtralFlowMatching()

    // Projection weights
    private val inputProj = weights[VoxtralTensorNames.ACOUSTIC_INPUT_PROJ]   // [dim, acousticDim]=[3072,36]
    private val outputProj = weights[VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ]  // [acousticDim, dim]=[36,3072]
    private val llmProj = weights[VoxtralTensorNames.ACOUSTIC_LLM_PROJ]       // [dim, dim]=[3072,3072]
    private val timeProj = weights[VoxtralTensorNames.ACOUSTIC_TIME_PROJ]     // [dim, dim]=[3072,3072]
    private val outputNorm = weights[VoxtralTensorNames.ACOUSTIC_NORM]        // [dim]=[3072]

    /**
     * Generate acoustic codes from backbone hidden states.
     *
     * @param backboneHidden [seqLen, dim] — backbone's last hidden states
     * @param numSteps Euler ODE steps (default: 8)
     * @return IntArray of length seqLen * nCodebooks, codes in [0, codebookLevels-1]
     */
    public fun generate(
        backboneHidden: Tensor<T, Float>,
        numSteps: Int = 8,
        method: String = "euler",
        random: Random = Random.Default
    ): IntArray {
        val seqLen = backboneHidden.shape[0]

        // Pre-project backbone hidden states through llmProj (shared across all steps)
        val conditioned = if (llmProj != null) {
            ops.matmul(backboneHidden, ops.transpose(llmProj))
        } else {
            backboneHidden
        }

        // Zero conditioning for CFG unconditional pass
        @Suppress("UNCHECKED_CAST")
        val unconditioned = ctx.fromFloatArray<T, Float>(
            Shape(seqLen, dim), dtype, FloatArray(seqLen * dim)
        ) as Tensor<T, Float>

        // Initialize x from Gaussian noise [seqLen, acousticDim]
        val noiseData = FloatArray(seqLen * acousticDim)
        val rng = random
        for (i in noiseData.indices step 2) {
            // Box-Muller
            val u1 = rng.nextFloat().coerceIn(1e-7f, 1.0f)
            val u2 = rng.nextFloat()
            val r = sqrt(-2.0f * ln(u1.toDouble()).toFloat())
            val theta = (2.0 * PI * u2).toFloat()
            noiseData[i] = r * cos(theta.toDouble()).toFloat()
            if (i + 1 < noiseData.size) noiseData[i + 1] = r * sin(theta.toDouble()).toFloat()
        }

        val x = FloatArray(seqLen * acousticDim)
        noiseData.copyInto(x)

        // Euler ODE solver
        val dt = 1.0f / numSteps
        for (step in 0 until numSteps) {
            val t = step.toFloat() / numSteps

            // Conditioned velocity
            val vCond = computeVelocityPerFrame(x, conditioned, t, seqLen)
            // Unconditioned velocity (CFG)
            val vUncond = computeVelocityPerFrame(x, unconditioned, t, seqLen)

            // CFG: v = alpha * v_cond + (1 - alpha) * v_uncond
            for (i in x.indices) {
                val v = cfgAlpha * vCond[i] + (1.0f - cfgAlpha) * vUncond[i]
                x[i] += v * dt
            }
        }

        // Quantize: clamp to [-1, 1], scale to [0, levels-1]
        val codes = IntArray(seqLen * nCodebooks)
        for (i in x.indices) {
            val clamped = x[i].coerceIn(-1.0f, 1.0f)
            codes[i] = ((clamped + 1.0f) / 2.0f * (codebookLevels - 1)).toInt().coerceIn(0, codebookLevels - 1)
        }
        return codes
    }

    /**
     * Compute velocity for all frames at timestep t.
     * Processes each frame independently with a 3-token sequence.
     */
    @Suppress("UNCHECKED_CAST")
    private fun computeVelocityPerFrame(
        x: FloatArray,
        conditioning: Tensor<T, Float>,
        t: Float,
        seqLen: Int
    ): FloatArray {
        val velocity = FloatArray(seqLen * acousticDim)

        // Compute time embedding once (shared across frames)
        val timeEmb = computeTimeEmbedding(t)

        // Project time embedding
        val timeEmbTensor = ctx.fromFloatArray<T, Float>(Shape(1, dim), dtype, timeEmb) as Tensor<T, Float>
        val timeProjResult = if (timeProj != null) {
            ops.matmul(timeEmbTensor, ops.transpose(timeProj)).data.copyToFloatArray()
        } else {
            timeEmb
        }

        // Process each frame
        val condData = conditioning.data.copyToFloatArray()

        for (frame in 0 until seqLen) {
            // Build 3-token input: [3, dim]
            val tokens = FloatArray(3 * dim)

            // Token 0: inputProj(x_t) — [acousticDim] → [dim]
            if (inputProj != null) {
                val ipData = inputProj.data.copyToFloatArray()
                // matmul: x_t[acousticDim] × inputProj^T[acousticDim, dim] → [dim]
                // inputProj shape: [dim, acousticDim], so transposed: [acousticDim, dim]
                for (d in 0 until dim) {
                    var sum = 0.0f
                    for (a in 0 until acousticDim) {
                        sum += x[frame * acousticDim + a] * ipData[d * acousticDim + a]
                    }
                    tokens[0 * dim + d] = sum
                }
            }

            // Token 1: timeProj(timeEmbed(t)) — already computed
            timeProjResult.copyInto(tokens, 1 * dim, 0, dim)

            // Token 2: llmProj(hidden) — already in conditioning
            condData.copyInto(tokens, 2 * dim, frame * dim, (frame + 1) * dim)

            // Forward through 3-layer bidirectional transformer
            val output = transformerForward(tokens, seqLen = 3)

            // RMSNorm on first token output
            val normed = rmsNormArray(output, 0, dim, outputNorm, normEps)

            // Output projection: [dim] → [acousticDim]
            if (outputProj != null) {
                val opData = outputProj.data.copyToFloatArray()
                // matmul: normed[dim] × outputProj^T[dim, acousticDim] → [acousticDim]
                // outputProj shape: [acousticDim, dim], transposed: [dim, acousticDim]
                for (a in 0 until acousticDim) {
                    var sum = 0.0f
                    for (d in 0 until dim) {
                        sum += normed[d] * opData[a * dim + d]
                    }
                    velocity[frame * acousticDim + a] = sum
                }
            }
        }

        return velocity
    }

    /**
     * Sinusoidal time embedding: cos/sin with log-space frequencies.
     * Same as positional encoding but with timestep t instead of position.
     */
    private fun computeTimeEmbedding(t: Float): FloatArray {
        val emb = FloatArray(dim)
        val halfDim = dim / 2
        for (i in 0 until halfDim) {
            val invFreq = exp(-ln(10000.0) * i.toDouble() / halfDim).toFloat()
            emb[i] = cos(t.toDouble() * invFreq).toFloat()
            emb[halfDim + i] = sin(t.toDouble() * invFreq).toFloat()
        }
        return emb
    }

    /**
     * Forward pass through the 3-layer bidirectional transformer.
     * Input: tokens [seqLen * dim] flat array (seqLen=3 for acoustic model).
     * Output: tokens [seqLen * dim] flat array (modified in-place style).
     */
    private fun transformerForward(tokens: FloatArray, seqLen: Int): FloatArray {
        var current = tokens.copyOf()

        for (layer in 0 until nLayers) {
            // ---- Attention block ----
            val attnNormW = weights[VoxtralTensorNames.acousticAttnNorm(layer)]
                ?: continue

            // RMSNorm
            val normed = FloatArray(seqLen * dim)
            for (s in 0 until seqLen) {
                val n = rmsNormArray(current, s * dim, dim, attnNormW, normEps)
                n.copyInto(normed, s * dim)
            }

            // QKV projections
            val wq = weights[VoxtralTensorNames.acousticAttnQ(layer)] ?: continue
            val wk = weights[VoxtralTensorNames.acousticAttnK(layer)] ?: continue
            val wv = weights[VoxtralTensorNames.acousticAttnV(layer)] ?: continue
            val wo = weights[VoxtralTensorNames.acousticAttnOut(layer)] ?: continue

            val wqData = wq.data.copyToFloatArray()
            val wkData = wk.data.copyToFloatArray()
            val wvData = wv.data.copyToFloatArray()
            val woData = wo.data.copyToFloatArray()

            val qDim = wq.shape[0]  // may differ from dim for GQA
            val q = matmulFlat(normed, seqLen, dim, wqData, qDim, dim)
            val k = matmulFlat(normed, seqLen, dim, wkData, kvDim, dim)
            val v = matmulFlat(normed, seqLen, dim, wvData, kvDim, dim)

            // Bidirectional multi-head attention
            val attnOut = bidirectionalMHA(q, k, v, seqLen, nHeads, nKVHeads, headDim)

            // Output projection
            val projected = matmulFlat(attnOut, seqLen, qDim, woData, dim, qDim)

            // Residual
            for (i in current.indices) current[i] += projected[i]

            // ---- FFN block ----
            val ffnNormW = weights[VoxtralTensorNames.acousticFfnNorm(layer)] ?: continue
            val w1 = weights[VoxtralTensorNames.acousticFfnGate(layer)] ?: continue
            val w2 = weights[VoxtralTensorNames.acousticFfnDown(layer)] ?: continue
            val w3 = weights[VoxtralTensorNames.acousticFfnUp(layer)] ?: continue

            val normedFfn = FloatArray(seqLen * dim)
            for (s in 0 until seqLen) {
                val n = rmsNormArray(current, s * dim, dim, ffnNormW, normEps)
                n.copyInto(normedFfn, s * dim)
            }

            val w1Data = w1.data.copyToFloatArray()
            val w2Data = w2.data.copyToFloatArray()
            val w3Data = w3.data.copyToFloatArray()

            val gate = matmulFlat(normedFfn, seqLen, dim, w1Data, ffnDim, dim)
            val up = matmulFlat(normedFfn, seqLen, dim, w3Data, ffnDim, dim)

            // SiLU(gate) * up
            for (i in gate.indices) {
                val sigmoid = 1.0f / (1.0f + exp((-gate[i]).toDouble()).toFloat())
                gate[i] = gate[i] * sigmoid * up[i]
            }

            val ffnOut = matmulFlat(gate, seqLen, ffnDim, w2Data, dim, ffnDim)

            // Residual
            for (i in current.indices) current[i] += ffnOut[i]
        }

        return current
    }

    /**
     * Bidirectional multi-head attention (no causal mask).
     */
    private fun bidirectionalMHA(
        q: FloatArray, k: FloatArray, v: FloatArray,
        seqLen: Int, nH: Int, nKVH: Int, hDim: Int
    ): FloatArray {
        val qDim = nH * hDim
        val scale = 1.0f / sqrt(hDim.toFloat())
        val headsPerGroup = nH / nKVH
        val output = FloatArray(seqLen * qDim)

        for (h in 0 until nH) {
            val kvHead = h / headsPerGroup

            // Compute attention scores [seqLen, seqLen]
            val scores = FloatArray(seqLen * seqLen)
            for (i in 0 until seqLen) {
                for (j in 0 until seqLen) {
                    var dot = 0.0f
                    for (d in 0 until hDim) {
                        dot += q[i * qDim + h * hDim + d] * k[j * nKVH * hDim + kvHead * hDim + d]
                    }
                    scores[i * seqLen + j] = dot * scale
                }
            }

            // Softmax per row
            for (i in 0 until seqLen) {
                var maxVal = scores[i * seqLen]
                for (j in 1 until seqLen) {
                    if (scores[i * seqLen + j] > maxVal) maxVal = scores[i * seqLen + j]
                }
                var sumExp = 0.0f
                for (j in 0 until seqLen) {
                    scores[i * seqLen + j] = exp((scores[i * seqLen + j] - maxVal).toDouble()).toFloat()
                    sumExp += scores[i * seqLen + j]
                }
                for (j in 0 until seqLen) {
                    scores[i * seqLen + j] /= sumExp
                }
            }

            // Weighted sum of values
            for (i in 0 until seqLen) {
                for (d in 0 until hDim) {
                    var sum = 0.0f
                    for (j in 0 until seqLen) {
                        sum += scores[i * seqLen + j] * v[j * nKVH * hDim + kvHead * hDim + d]
                    }
                    output[i * qDim + h * hDim + d] = sum
                }
            }
        }

        return output
    }

    // ========== Utility ==========

    /**
     * Flat matmul: input[rows, inDim] × weight^T[outDim, inDim]^T → [rows, outDim]
     * Weight stored as [outDim, inDim] (row-major), transposed for matmul.
     */
    private fun matmulFlat(
        input: FloatArray, rows: Int, inDim: Int,
        weight: FloatArray, outDim: Int, wInDim: Int
    ): FloatArray {
        val output = FloatArray(rows * outDim)
        for (r in 0 until rows) {
            for (o in 0 until outDim) {
                var sum = 0.0f
                for (k in 0 until inDim) {
                    sum += input[r * inDim + k] * weight[o * wInDim + k]
                }
                output[r * outDim + o] = sum
            }
        }
        return output
    }

    /**
     * RMSNorm on a slice of a flat array.
     */
    private fun rmsNormArray(
        data: FloatArray, offset: Int, size: Int,
        weight: Tensor<T, Float>?, eps: Float
    ): FloatArray {
        val result = FloatArray(size)
        var sumSq = 0.0f
        for (i in 0 until size) {
            sumSq += data[offset + i] * data[offset + i]
        }
        val rms = sqrt((sumSq / size + eps).toDouble()).toFloat()
        val wData = weight?.data?.copyToFloatArray()
        for (i in 0 until size) {
            val normalized = data[offset + i] / rms
            result[i] = if (wData != null) normalized * wData[i] else normalized
        }
        return result
    }
}
