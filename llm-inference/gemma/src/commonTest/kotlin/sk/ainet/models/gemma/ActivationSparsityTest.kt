package sk.ainet.models.gemma

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActivationSparsityTest {

    @Test
    fun `zero sparsity returns values unchanged`() {
        val values = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val result = ActivationSparsity.applyGaussianTopK(values.copyOf(), 0f)
        assertTrue(values.contentEquals(result))
    }

    @Test
    fun `full sparsity zeros all values`() {
        val values = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val result = ActivationSparsity.applyGaussianTopK(values, 1f)
        assertTrue(result.all { it == 0f })
    }

    @Test
    fun `high sparsity zeros most values`() {
        val n = 1000
        val values = FloatArray(n) { i ->
            (i - n / 2).toFloat() / (n / 6).toFloat()
        }
        val original = values.copyOf()
        val result = ActivationSparsity.applyGaussianTopK(values, 0.95f)
        val zeroed = result.count { it == 0f }
        // Most values should be zeroed
        assertTrue(zeroed > n / 2, "Expected most values zeroed, but only $zeroed out of $n were zeroed")
        // Values that remain non-zero should be the extreme ones (large magnitude)
        for (i in result.indices) {
            if (result[i] != 0f) {
                assertEquals(original[i], result[i], "Non-zero values should be unchanged")
            }
        }
    }

    @Test
    fun `empty array handled`() {
        val result = ActivationSparsity.applyGaussianTopK(floatArrayOf(), 0.5f)
        assertEquals(0, result.size)
    }

    @Test
    fun `inverse normal CDF known values`() {
        // Φ⁻¹(0.5) = 0
        assertTrue(abs(ActivationSparsity.inverseNormalCDF(0.5)) < 0.01)
        // Φ⁻¹(0.975) ≈ 1.96
        assertTrue(abs(ActivationSparsity.inverseNormalCDF(0.975) - 1.96) < 0.01)
        // Φ⁻¹(0.025) ≈ -1.96
        assertTrue(abs(ActivationSparsity.inverseNormalCDF(0.025) + 1.96) < 0.01)
    }
}
