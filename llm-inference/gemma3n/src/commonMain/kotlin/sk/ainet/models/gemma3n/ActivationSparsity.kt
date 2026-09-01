package sk.ainet.models.gemma3n

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Activation sparsity via Gaussian top-k selection.
 *
 * Used in Gemma 3n E4B to zero out a fraction of FFN activations,
 * reducing effective computation while preserving output quality.
 *
 * The threshold is computed assuming activations follow a Gaussian
 * distribution: the (1-sparsityRate) quantile of N(mean, std) is used
 * as a cutoff, and values with |x - mean| below that threshold are zeroed.
 */
public object ActivationSparsity {

    /**
     * Apply Gaussian top-k sparsity to activation values.
     *
     * Keeps only the top (1 - sparsityRate) fraction of activations
     * by magnitude (relative to the distribution), zeroing the rest.
     *
     * @param values Activation values (modified in-place for efficiency)
     * @param sparsityRate Fraction of values to zero out (0.0 to 1.0). E.g., 0.95 keeps top 5%.
     * @return The same array with sparse values zeroed
     */
    public fun applyGaussianTopK(values: FloatArray, sparsityRate: Float): FloatArray {
        if (sparsityRate <= 0f || values.isEmpty()) return values
        if (sparsityRate >= 1f) {
            values.fill(0f)
            return values
        }

        // Compute mean and std
        var sum = 0.0
        for (v in values) sum += v
        val mean = (sum / values.size).toFloat()

        var variance = 0.0
        for (v in values) {
            val d = (v - mean).toDouble()
            variance += d * d
        }
        val std = sqrt(variance / values.size).toFloat()

        if (std < 1e-10f) return values

        // Compute threshold: the z-score corresponding to keeping top (1 - sparsityRate) by magnitude
        // We want the quantile at (1 + sparsityRate) / 2 for two-tailed
        val z = inverseNormalCDF((1.0 + sparsityRate) / 2.0).toFloat()
        val threshold = z * std

        // Zero out values with |x - mean| < threshold
        for (i in values.indices) {
            if (abs(values[i] - mean) < threshold) {
                values[i] = 0f
            }
        }

        return values
    }

    /**
     * Approximation of the inverse normal CDF (probit function)
     * using Abramowitz & Stegun rational approximation.
     *
     * Accurate to ~4.5e-4 for p in (0, 1).
     */
    internal fun inverseNormalCDF(p: Double): Double {
        if (p <= 0.0) return Double.NEGATIVE_INFINITY
        if (p >= 1.0) return Double.POSITIVE_INFINITY

        return if (p < 0.5) {
            -rationalApprox(sqrt(-2.0 * ln(p)))
        } else {
            rationalApprox(sqrt(-2.0 * ln(1.0 - p)))
        }
    }

    // Abramowitz & Stegun constants
    private const val C0 = 2.515517
    private const val C1 = 0.802853
    private const val C2 = 0.010328
    private const val D1 = 1.432788
    private const val D2 = 0.189269
    private const val D3 = 0.001308

    private fun rationalApprox(t: Double): Double {
        return t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t)
    }
}
