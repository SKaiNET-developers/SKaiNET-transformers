package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Runtime for the Voxtral acoustic flow-matching pipeline.
 *
 * Generates acoustic codebook values from backbone hidden states using iterative
 * flow-matching denoising through a 3-layer acoustic transformer.
 *
 * **Pipeline per ODE step:**
 * 1. Project noised acoustic representation to model dim: `x_proj = input_proj(x_t)`
 * 2. Add timestep-scaled conditioning: `combined = condition + x_proj * (1-t) + condition * t`
 *    (simplified: `combined = condition + x_proj`)
 * 3. Forward through 3-layer acoustic transformer
 * 4. Project output to acoustic dim: `v = output_proj(transformer_out)`
 * 5. Update sample: `x_{t+dt} = x_t + v * dt`
 *
 * After all steps, apply FSQ quantization to get discrete codes (36 codebooks × 21 levels).
 *
 * @param T The DType for model weights
 * @param acousticTransformer The 3-layer acoustic transformer module (built via DSL, weights loaded)
 * @param inputProj Input projection weight [dim, acousticDim]
 * @param outputProj Output projection weight [acousticDim, dim]
 * @param ctx Execution context for tensor operations
 * @param dtype KClass for tensor creation
 * @param nCodebooks Number of acoustic codebooks (default: 36)
 * @param codebookLevels FSQ levels per codebook (default: 21)
 * @param dim Model hidden dimension (default: 3072)
 */
public class VoxtralAcousticRuntime<T : DType>(
    private val acousticTransformer: Module<T, Float>,
    private val inputProj: Tensor<T, Float>,
    private val outputProj: Tensor<T, Float>,
    private val inputProjBias: Tensor<T, Float>? = null,
    private val outputProjBias: Tensor<T, Float>? = null,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val nCodebooks: Int = 36,
    private val codebookLevels: Int = 21,
    private val dim: Int = 3072
) {
    private val acousticDim: Int = nCodebooks * codebookLevels
    private val flowMatching = VoxtralFlowMatching()

    /**
     * Generate acoustic codes from backbone hidden states.
     *
     * @param backboneHidden The backbone's last hidden state [seqLen, dim].
     *   This is the output of the backbone transformer BEFORE the output norm + lm_head projection.
     * @param numSteps Number of ODE solver steps (default: 16). More = higher quality, slower.
     * @param method ODE solver method: "euler" or "midpoint" (default: "euler")
     * @param random Random generator for initial noise
     * @return Acoustic codes as IntArray of length seqLen * nCodebooks.
     *   Codes are in [0, codebookLevels-1] and laid out as
     *   [frame0_cb0, frame0_cb1, ..., frame0_cb35, frame1_cb0, ...].
     */
    public fun generate(
        backboneHidden: Tensor<T, Float>,
        numSteps: Int = 16,
        method: String = "euler",
        random: Random = Random.Default
    ): IntArray {
        val seqLen = backboneHidden.shape[0]

        // The velocity function for flow matching: v(x_t, t) → velocity
        val velocityFn: (Tensor<T, Float>, Float) -> Tensor<T, Float> = { xt, t ->
            computeVelocity(backboneHidden, xt, t)
        }

        // Run ODE solver
        val continuous = when (method) {
            "midpoint" -> flowMatching.sampleMidpoint(
                ctx, dtype, seqLen, acousticDim, numSteps, velocityFn, random
            )
            else -> flowMatching.sampleEuler(
                ctx, dtype, seqLen, acousticDim, numSteps, velocityFn, random
            )
        }

        // Argmax over levels per codebook to get discrete codes
        return flowMatching.quantizeFSQ(continuous, nCodebooks, codebookLevels)
    }

    /**
     * Compute the velocity field v(x_t, t | condition) at a single ODE step.
     *
     * 1. Project x_t from acoustic space to model dim
     * 2. Add backbone conditioning
     * 3. Forward through acoustic transformer
     * 4. Project back to acoustic space
     */
    private fun computeVelocity(
        condition: Tensor<T, Float>,
        xt: Tensor<T, Float>,
        t: Float
    ): Tensor<T, Float> {
        val ops = ctx.ops

        // 1. Input projection: x_t [seqLen, acousticDim] → [seqLen, dim]
        var projected = ops.matmul(xt, ops.transpose(inputProj))
        if (inputProjBias != null) {
            projected = ops.add(projected, inputProjBias)
        }

        // 2. Combine with conditioning (sum, as per Voxtral config)
        val combined = ops.add(condition, projected)

        // 3. Forward through acoustic transformer
        // Reset KV caches before each full-sequence forward pass
        resetKVCaches(acousticTransformer)
        val transformerOut = acousticTransformer.forward(combined, ctx)

        // 4. Output projection: [seqLen, dim] → [seqLen, acousticDim]
        var velocity = ops.matmul(transformerOut, ops.transpose(outputProj))
        if (outputProjBias != null) {
            velocity = ops.add(velocity, outputProjBias)
        }

        return velocity
    }

    /**
     * Reset all KV caches in the module tree. Required before each full-sequence
     * forward pass since flow matching is non-autoregressive.
     */
    private fun resetKVCaches(module: Module<*, *>) {
        for (child in module.modules) {
            resetKVCaches(child)
        }
        if (module is sk.ainet.lang.nn.transformer.KVCache<*, *>) {
            module.reset()
        }
    }
}
