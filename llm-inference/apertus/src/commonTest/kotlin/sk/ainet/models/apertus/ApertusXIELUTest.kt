package sk.ainet.models.apertus

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals

class ApertusXIELUTest {

    private val tolerance = 1e-5f

    @Test
    fun testSoftplus() {
        // softplus(0) = ln(2)
        assertClose(ln(2f), softplus(0f), tolerance)

        // softplus(large) ≈ x
        assertClose(100f, softplus(100f), 0.01f)

        // softplus(-large) ≈ 0
        assertClose(0f, softplus(-100f), 1e-10f)

        // softplus(1) = ln(1 + e)
        assertClose(ln(1f + exp(1f)), softplus(1f), tolerance)
    }

    @Test
    fun testXieluPositiveInputs() {
        // For x > 0: alpha_p_eff * x^2 + beta * x
        val params = ApertusXIELUParams(
            alphaP = 0.0f,   // softplus(0) = ln(2)
            alphaN = 0.0f,
            beta = 1.0f,
            eps = -10.0f
        )
        val alphaPEff = softplus(0f) // ln(2) ≈ 0.6931

        val buf = floatArrayOf(1f, 2f, 3f)
        xielu(buf, params)

        // x=1: ln(2)*1 + 1*1 = ln(2) + 1
        assertClose(alphaPEff * 1f + 1f * 1f, buf[0], tolerance)
        // x=2: ln(2)*4 + 1*2 = 4*ln(2) + 2
        assertClose(alphaPEff * 4f + 1f * 2f, buf[1], tolerance)
        // x=3: ln(2)*9 + 1*3 = 9*ln(2) + 3
        assertClose(alphaPEff * 9f + 1f * 3f, buf[2], tolerance)
    }

    @Test
    fun testXieluNegativeInputs() {
        // For x <= 0: (expm1(min(x, eps)) - x) * alpha_n_eff + beta * x
        val params = ApertusXIELUParams(
            alphaP = 0.0f,
            alphaN = 0.0f,   // softplus(0) = ln(2), alpha_n_eff = beta + ln(2)
            beta = 1.0f,
            eps = -10.0f
        )
        val alphaNEff = 1.0f + softplus(0f) // 1 + ln(2)

        val x = -1f
        val buf = floatArrayOf(x)
        xielu(buf, params)

        // x=-1, eps=-10, min(-1, -10) = -10: expm1(-10) ≈ -0.99995
        // Actually min(x, eps) means min(-1, -10) = -10
        // Wait, eps is -10 here. min(-1, -10) = -10
        // expm1(-10) ≈ exp(-10) - 1 ≈ -0.999955
        val clamped = -10f
        val expm1Val = kotlin.math.expm1(clamped.toDouble()).toFloat()
        val expected = (expm1Val - x) * alphaNEff + 1.0f * x
        assertClose(expected, buf[0], 1e-3f)
    }

    @Test
    fun testXieluZeroInput() {
        val params = ApertusXIELUParams(
            alphaP = 1.0f,
            alphaN = 1.0f,
            beta = 1.0f,
            eps = -10.0f
        )

        val buf = floatArrayOf(0f)
        xielu(buf, params)

        // x=0 goes to the else branch: (expm1(min(0, -10)) - 0) * alpha_n_eff + beta * 0
        val alphaNEff = 1.0f + softplus(1.0f)
        val clamped = -10f
        val expm1Val = kotlin.math.expm1(clamped.toDouble()).toFloat()
        val expected = (expm1Val - 0f) * alphaNEff + 1.0f * 0f
        assertClose(expected, buf[0], 1e-3f)
    }

    @Test
    fun testXieluTypicalParams() {
        // Simulate typical learned Apertus params
        val params = ApertusXIELUParams(
            alphaP = -0.5f,
            alphaN = -0.3f,
            beta = 0.8f,
            eps = -5.0f
        )

        val input = floatArrayOf(-2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f)
        xielu(input, params)

        // Just verify no NaN/Inf
        for (v in input) {
            assertTrue(v.isFinite(), "xIELU output should be finite, got $v")
        }

        // Positive side should be monotonically increasing for typical params
        assertTrue(input[5] < input[6], "xIELU should be increasing for positive inputs")
    }

    @Test
    fun testXieluEmptyBuffer() {
        val params = ApertusXIELUParams(0f, 0f, 1f, -10f)
        val buf = floatArrayOf()
        xielu(buf, params) // should not crash
        assertEquals(0, buf.size)
    }

    private fun assertClose(expected: Float, actual: Float, tol: Float) {
        assertTrue(
            abs(expected - actual) < tol,
            "Expected $expected but got $actual (diff=${abs(expected - actual)}, tol=$tol)"
        )
    }
}
