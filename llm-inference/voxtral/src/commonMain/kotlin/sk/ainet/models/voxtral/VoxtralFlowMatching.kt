package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.roundToInt
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Flow-matching ODE sampler for the Voxtral acoustic model.
 *
 * Flow matching learns a velocity field v(x_t, t) that transports samples from
 * a noise distribution (t=0) to the data distribution (t=1) along straight paths.
 * At inference time, we solve the ODE: dx/dt = v(x_t, t) from t=0 to t=1.
 *
 * Voxtral uses this to generate 36 acoustic codebooks (each with 21 FSQ levels)
 * from the backbone's hidden-state conditioning.
 *
 * @param sigma Minimum noise level (default: 1e-5)
 * @param sigmaMax Maximum noise level for the initial distribution (default: 1.0)
 */
public class VoxtralFlowMatching(
    private val sigma: Float = 1e-5f,
    private val sigmaMax: Float = 1.0f
) {
    /**
     * Solve the flow-matching ODE using the Euler method.
     *
     * Starting from Gaussian noise, iteratively applies the learned velocity field
     * to transport samples to the data distribution.
     *
     * @param T The tensor dtype
     * @param ctx Execution context for tensor operations
     * @param dtype KClass for tensor creation
     * @param seqLen Number of audio frames (sequence positions)
     * @param acousticDim Output dimension per frame (nCodebooks * codebookSize, e.g. 36 * 21 = 756)
     * @param numSteps Number of ODE solver steps (more steps = higher quality, slower)
     * @param velocityFn Function that computes the velocity v(x_t, t) given the current
     *   noised sample and timestep. The function receives (x_t: Tensor[seqLen, acousticDim], t: Float)
     *   and returns a velocity tensor of the same shape.
     * @param random Random generator for initial noise
     * @return Final denoised sample of shape [seqLen, acousticDim]
     */
    public fun <T : DType> sampleEuler(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        seqLen: Int,
        acousticDim: Int,
        numSteps: Int,
        velocityFn: (Tensor<T, Float>, Float) -> Tensor<T, Float>,
        random: Random = Random.Default
    ): Tensor<T, Float> {
        val ops = ctx.ops

        // x_0 ~ N(0, sigmaMax^2 * I)
        var x: Tensor<T, Float> = createNoise(ctx, dtype, Shape(seqLen, acousticDim), random)
        if (sigmaMax != 1.0f) {
            x = ops.mulScalar(x, sigmaMax)
        }

        val dt = 1.0f / numSteps

        for (step in 0 until numSteps) {
            val t = step.toFloat() / numSteps

            // v = velocityFn(x_t, t)
            val velocity = velocityFn(x, t)

            // x_{t+dt} = x_t + v * dt
            val scaledV: Tensor<T, Float> = ops.mulScalar(velocity, dt)
            x = ops.add(x, scaledV)
        }

        return x
    }

    /**
     * Solve the flow-matching ODE using the midpoint method (2nd order).
     *
     * More accurate than Euler for the same number of steps, at 2x the compute cost.
     *
     * @see sampleEuler for parameter descriptions
     */
    public fun <T : DType> sampleMidpoint(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        seqLen: Int,
        acousticDim: Int,
        numSteps: Int,
        velocityFn: (Tensor<T, Float>, Float) -> Tensor<T, Float>,
        random: Random = Random.Default
    ): Tensor<T, Float> {
        val ops = ctx.ops

        var x: Tensor<T, Float> = createNoise(ctx, dtype, Shape(seqLen, acousticDim), random)
        if (sigmaMax != 1.0f) {
            x = ops.mulScalar(x, sigmaMax)
        }

        val dt = 1.0f / numSteps

        for (step in 0 until numSteps) {
            val t = step.toFloat() / numSteps
            val tMid = t + dt / 2.0f

            // k1 = v(x_t, t)
            val k1 = velocityFn(x, t)

            // x_mid = x_t + k1 * dt/2
            val halfStep: Tensor<T, Float> = ops.mulScalar(k1, dt / 2.0f)
            val xMid: Tensor<T, Float> = ops.add(x, halfStep)

            // k2 = v(x_mid, t_mid)
            val k2 = velocityFn(xMid, tMid)

            // x_{t+dt} = x_t + k2 * dt
            val fullStep: Tensor<T, Float> = ops.mulScalar(k2, dt)
            x = ops.add(x, fullStep)
        }

        return x
    }

    /**
     * Apply Finite Scalar Quantization (FSQ) to convert continuous acoustic output to discrete codes.
     *
     * The input tensor has shape [seqLen, nCodebooks * levels]. This is reshaped to
     * Two modes depending on acoustic dimension:
     *
     * **Continuous mode** (acousticDim == nCodebooks): Each value is a continuous scalar
     * that maps to a discrete FSQ level via nearest-level quantization.
     * value in [-1, 1] → code in [0, levels-1].
     *
     * **Logits mode** (acousticDim == nCodebooks * levels): Output is reshaped to
     * [seqLen, nCodebooks, levels] and argmax is taken per codebook.
     *
     * @param continuous The continuous output from flow matching [seqLen, acousticDim]
     * @param nCodebooks Number of acoustic codebooks (e.g. 36)
     * @param levels Number of quantization levels per codebook (e.g. 21)
     * @return IntArray of quantized codes, length = seqLen * nCodebooks, each in [0, levels-1]
     */
    public fun <T : DType> quantizeFSQ(
        continuous: Tensor<T, Float>,
        nCodebooks: Int,
        levels: Int = 21
    ): IntArray {
        val data = continuous.data.copyToFloatArray()
        val seqLen = continuous.shape[0]
        val acousticDim = continuous.shape[1]

        val codes = IntArray(seqLen * nCodebooks)

        if (acousticDim == nCodebooks) {
            // Continuous mode: one scalar per codebook, map [-1, 1] → [0, levels-1]
            for (frame in 0 until seqLen) {
                for (cb in 0 until nCodebooks) {
                    val value = data[frame * nCodebooks + cb]
                    // Clamp to [-1, 1], then map to [0, levels-1]
                    val clamped = value.coerceIn(-1.0f, 1.0f)
                    val code = ((clamped + 1.0f) / 2.0f * (levels - 1)).toInt().coerceIn(0, levels - 1)
                    codes[frame * nCodebooks + cb] = code
                }
            }
        } else {
            // Logits mode: argmax over levels per codebook
            require(acousticDim == nCodebooks * levels) {
                "Expected acousticDim=$acousticDim to equal nCodebooks ($nCodebooks) or nCodebooks*levels (${nCodebooks * levels})"
            }
            for (frame in 0 until seqLen) {
                for (cb in 0 until nCodebooks) {
                    val offset = frame * acousticDim + cb * levels
                    var bestLevel = 0
                    var bestVal = data[offset]
                    for (l in 1 until levels) {
                        if (data[offset + l] > bestVal) {
                            bestVal = data[offset + l]
                            bestLevel = l
                        }
                    }
                    codes[frame * nCodebooks + cb] = bestLevel
                }
            }
        }
        return codes
    }

    private fun <T : DType> createNoise(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape,
        random: Random
    ): Tensor<T, Float> {
        // Box-Muller transform for Gaussian noise
        val n = shape.volume
        val values = FloatArray(n)
        var i = 0
        while (i < n - 1) {
            val u1 = random.nextFloat().coerceIn(1e-7f, 1.0f)
            val u2 = random.nextFloat()
            val mag = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1))
            val angle = (2.0 * Math.PI * u2).toFloat()
            values[i] = mag * kotlin.math.cos(angle)
            values[i + 1] = mag * kotlin.math.sin(angle)
            i += 2
        }
        if (i < n) {
            val u1 = random.nextFloat().coerceIn(1e-7f, 1.0f)
            val u2 = random.nextFloat()
            values[i] = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1)) * kotlin.math.cos((2.0 * Math.PI * u2).toFloat())
        }
        @Suppress("UNCHECKED_CAST")
        val result = ctx.fromFloatArray<T, Float>(shape, dtype, values)
        return result as Tensor<T, Float>
    }
}
