package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Global AltUp weights shared across all layers.
 *
 * @param projWeight Projects embedding into (numInputs-1) additional states [hiddenSize, hiddenSize, numInputs-1]
 * @param unembdProjWeight Projects back for output combination [hiddenSize, hiddenSize, numInputs-1]
 */
public data class AltUpGlobalWeights<T : DType>(
    val projWeight: Tensor<T, Float>,
    val unembdProjWeight: Tensor<T, Float>
)

/**
 * Per-layer AltUp weights.
 *
 * @param predictCoef Prediction coefficients [numInputs, numInputs * numInputs]
 * @param correctCoef Correction coefficients [numInputs, numInputs]
 * @param correctScale Per-element scaling for correction [hiddenSize]
 * @param routerWeight Router projection [hiddenSize, numInputs]
 * @param routerNorm Router normalization [hiddenSize]
 */
public data class AltUpLayerWeights<T : DType>(
    val predictCoef: Tensor<T, Float>,
    val correctCoef: Tensor<T, Float>,
    val correctScale: Tensor<T, Float>,
    val routerWeight: Tensor<T, Float>,
    val routerNorm: Tensor<T, Float>
)

/**
 * AltUp (Alternating Updates) implementation for Gemma 3n E4B.
 *
 * AltUp maintains multiple parallel hidden states (E4B: 4) but only routes
 * the "active" state through expensive transformer layers. The other states
 * are cheaply predicted/corrected using learned per-layer coefficients.
 *
 * Architecture (from GGUF inspection):
 * - Global: altup_proj [2048,2048,3] creates 3 extra states from embedding
 * - Per-layer: router projects hidden to routing logits, predict_coef/correct_coef
 *   control state updates, correct_scale modulates corrections element-wise
 * - Global: altup_unembd_proj [2048,2048,3] recombines states for output
 *
 * @param ctx ExecutionContext for tensor operations
 * @param dtype Data type class
 * @param numInputs Number of parallel inputs (E4B: 4)
 * @param activeIdx Index of the active input (0)
 * @param hiddenSize Model hidden dimension
 * @param globalWeights Global projection/unprojection weights
 * @param layerWeights Per-layer AltUp weights
 */
public class AltUp<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val numInputs: Int,
    public val activeIdx: Int,
    private val hiddenSize: Int,
    private val globalWeights: AltUpGlobalWeights<T>,
    private val layerWeights: List<AltUpLayerWeights<T>>
) {

    private val numExtra = numInputs - 1  // 3 for E4B

    /**
     * Initialize AltUp states from a single embedding vector.
     *
     * The active state (idx 0) is the embedding itself.
     * Additional states are created by projecting the embedding through altup_proj slices.
     *
     * @param embedding The token embedding [hiddenSize]
     * @return List of [numInputs] state tensors
     */
    public fun initialize(embedding: Tensor<T, Float>): List<Tensor<T, Float>> {
        val states = mutableListOf(embedding)

        // Project embedding into additional states using altup_proj [hiddenSize, hiddenSize, numExtra]
        val projBuf = globalWeights.projWeight.expectFloatBuffer()
        val embBuf = embedding.expectFloatBuffer()
        val h = hiddenSize

        for (k in 0 until numExtra) {
            val out = FloatArray(h)
            val offset = k * h * h
            for (i in 0 until h) {
                var sum = 0f
                for (j in 0 until h) {
                    sum += projBuf[offset + i * h + j] * embBuf[j]
                }
                out[i] = sum
            }
            states.add(ctx.fromFloatArray<T, Float>(embedding.shape, dtype, out))
        }

        return states
    }

    /**
     * Predict phase: generate predictions for all states using per-layer coefficients.
     *
     * Uses the router to compute routing logits, then applies predict_coef to
     * create weighted combinations of states.
     *
     * @param layerIdx Layer index to get per-layer weights
     * @param states Current parallel states
     * @return Predicted states
     */
    public fun predict(layerIdx: Int, states: List<Tensor<T, Float>>): List<Tensor<T, Float>> {
        val lw = layerWeights[layerIdx]
        val coeffBuf = lw.predictCoef.expectFloatBuffer()
        // predict_coef shape: [numInputs, numInputs * numInputs]
        // For each output state i, coefficients for combining input states
        val n = numInputs

        return List(n) { i ->
            var result = states[i]
            for (j in 0 until n) {
                if (i != j) {
                    // Use coefficient from the flattened matrix
                    val coeff = coeffBuf[i * n + j]
                    if (coeff != 0f) {
                        result = addScaled(result, states[j], coeff)
                    }
                }
            }
            result
        }
    }

    /**
     * Correct phase: update all states after the active state passes through the layer.
     *
     * innovation = layerOutput - predictions[activeIdx]
     * corrected[i] = predictions[i] + coeff[i, activeIdx] * (correctScale * innovation)
     *
     * @param layerIdx Layer index
     * @param layerOutput Output of the transformer layer for the active state
     * @param predictions Predicted states from [predict]
     * @return Corrected states
     */
    public fun correct(
        layerIdx: Int,
        layerOutput: Tensor<T, Float>,
        predictions: List<Tensor<T, Float>>
    ): List<Tensor<T, Float>> {
        val lw = layerWeights[layerIdx]
        val innovation = addScaled(layerOutput, predictions[activeIdx], -1f)

        // Apply element-wise scale to innovation
        val scaleBuf = lw.correctScale.expectFloatBuffer()
        val innBuf = innovation.expectFloatBuffer()
        val scaledInnovation = FloatArray(innBuf.size) { innBuf[it] * scaleBuf[it % scaleBuf.size] }
        val scaledInnovationTensor = ctx.fromFloatArray<T, Float>(innovation.shape, dtype, scaledInnovation)

        val coeffBuf = lw.correctCoef.expectFloatBuffer()
        val n = numInputs

        return List(n) { i ->
            if (i == activeIdx) {
                layerOutput
            } else {
                val coeff = coeffBuf[i * n + activeIdx]
                addScaled(predictions[i], scaledInnovationTensor, coeff)
            }
        }
    }

    /**
     * Finalize: combine all states into a single output using altup_unembd_proj.
     *
     * output = states[activeIdx] + sum over k of (unembd_proj[k] @ states[k+1])
     *
     * @param states Final parallel states after all layers
     * @return Combined output tensor
     */
    public fun finalize(states: List<Tensor<T, Float>>): Tensor<T, Float> {
        val unprojBuf = globalWeights.unembdProjWeight.expectFloatBuffer()
        val h = hiddenSize
        var result = states[activeIdx].expectFloatBuffer().copyOf()

        // Add projected extra states
        for (k in 0 until numExtra) {
            val stateBuf = states[k + 1].expectFloatBuffer()
            val offset = k * h * h
            for (i in 0 until h) {
                var sum = 0f
                for (j in 0 until h) {
                    sum += unprojBuf[offset + i * h + j] * stateBuf[j]
                }
                result[i] += sum
            }
        }

        return ctx.fromFloatArray<T, Float>(states[activeIdx].shape, dtype, result)
    }

    private fun addScaled(a: Tensor<T, Float>, b: Tensor<T, Float>, bScale: Float): Tensor<T, Float> {
        val aBuf = a.expectFloatBuffer()
        val bBuf = b.expectFloatBuffer()
        val out = FloatArray(aBuf.size) { aBuf[it] + bScale * bBuf[it] }
        return ctx.fromFloatArray<T, Float>(a.shape, dtype, out)
    }

    private fun Tensor<T, Float>.expectFloatBuffer(): FloatArray {
        val data = this.data
        if (data is sk.ainet.lang.tensor.data.FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }
}
