package sk.ainet.apps.llm

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * RoPE rotation tests covering both conventions emitted by GGUF tooling.
 *
 * Llama family is INTERLEAVED (`(buf[2i], buf[2i+1])`); Qwen 2/3 / Phi /
 * Falcon are HALF_SPLIT (`(buf[i], buf[i+ropeDim/2])`). Mismatching the
 * convention vs. how the model was trained silently corrupts attention —
 * see ISSUE-74 (Qwen3 runtime degenerate output).
 */
class RopeUtilsTest {

    private fun assertCloseTo(expected: Float, actual: Float, tol: Float = 1e-5f) {
        assertTrue(abs(expected - actual) < tol,
            "expected $expected, got $actual (delta ${abs(expected - actual)})")
    }

    @Test
    fun interleavedRotationMatchesAdjacentPairFormula() {
        // Single head, headSize = ropeDim = 4 → 2 pairs at indices (0,1) and (2,3).
        // pos = 1, base = 10000.
        val buf = floatArrayOf(1f, 2f, 3f, 4f)
        val pos = 1
        val ropeDim = 4
        val base = 10000f
        applyRopeRotation(buf, nHeads = 1, headSize = ropeDim, ropeDim = ropeDim, pos = pos, base = base)

        // Reference: rotate (b[0], b[1]) by freq(pair=0), (b[2], b[3]) by freq(pair=1).
        val f0 = pos / base.toDouble().pow(0.0).toFloat()
        val f1 = pos / base.toDouble().pow(0.5).toFloat()  // 2*1/4 = 0.5
        val expected = floatArrayOf(
            1f * cos(f0) - 2f * sin(f0),
            1f * sin(f0) + 2f * cos(f0),
            3f * cos(f1) - 4f * sin(f1),
            3f * sin(f1) + 4f * cos(f1)
        )
        for (i in buf.indices) assertCloseTo(expected[i], buf[i])
    }

    @Test
    fun halfSplitRotationPairsFirstHalfWithSecondHalf() {
        // Single head, headSize = ropeDim = 4 → pairs are (b[0], b[2]) and (b[1], b[3]).
        // Distinct from interleaved which would pair (b[0], b[1]) and (b[2], b[3]).
        val buf = floatArrayOf(1f, 2f, 3f, 4f)
        val pos = 1
        val ropeDim = 4
        val base = 10000f
        applyRopeRotation(
            buf, nHeads = 1, headSize = ropeDim, ropeDim = ropeDim,
            pos = pos, base = base, ropeType = RopeType.HALF_SPLIT
        )

        val f0 = pos / base.toDouble().pow(0.0).toFloat()
        val f1 = pos / base.toDouble().pow(0.5).toFloat()
        // Pair 0: (b[0]=1, b[2]=3) rotated by f0 → goes back to (b[0], b[2])
        // Pair 1: (b[1]=2, b[3]=4) rotated by f1 → goes back to (b[1], b[3])
        val expected = floatArrayOf(
            1f * cos(f0) - 3f * sin(f0),  // b[0]
            2f * cos(f1) - 4f * sin(f1),  // b[1]
            1f * sin(f0) + 3f * cos(f0),  // b[2]
            2f * sin(f1) + 4f * cos(f1)   // b[3]
        )
        for (i in buf.indices) assertCloseTo(expected[i], buf[i])
    }

    @Test
    fun halfSplitDoesNotEqualInterleavedForSameInput() {
        // Regression guard: two conventions must produce *different* outputs on
        // the same input (so a wiring mistake is observable).
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val b = a.copyOf()
        val pos = 3
        applyRopeRotation(a, nHeads = 1, headSize = 8, ropeDim = 8, pos = pos, base = 10000f, ropeType = RopeType.INTERLEAVED)
        applyRopeRotation(b, nHeads = 1, headSize = 8, ropeDim = 8, pos = pos, base = 10000f, ropeType = RopeType.HALF_SPLIT)

        var anyDiff = false
        for (i in a.indices) if (abs(a[i] - b[i]) > 1e-4f) anyDiff = true
        assertTrue(anyDiff, "INTERLEAVED and HALF_SPLIT must produce different rotations: a=${a.toList()} b=${b.toList()}")
    }

    @Test
    fun defaultRopeTypeIsInterleavedForBackwardsCompat() {
        // Callers that don't pass ropeType (existing Llama path) must keep the
        // pre-fix behavior, i.e. interleaved rotation.
        val withDefault = floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f)
        val explicitInterleaved = withDefault.copyOf()
        applyRopeRotation(withDefault, nHeads = 1, headSize = 4, ropeDim = 4, pos = 2, base = 10000f)
        applyRopeRotation(
            explicitInterleaved, nHeads = 1, headSize = 4, ropeDim = 4,
            pos = 2, base = 10000f, ropeType = RopeType.INTERLEAVED
        )
        for (i in withDefault.indices) assertEquals(explicitInterleaved[i], withDefault[i])
    }

    @Test
    fun halfSplitMultipleHeadsAreIndependent() {
        // Each head is rotated independently — the half-split partitioning is
        // within the head, not across heads.
        val buf = floatArrayOf(
            1f, 2f, 3f, 4f,   // head 0
            5f, 6f, 7f, 8f    // head 1
        )
        val pos = 2
        applyRopeRotation(
            buf, nHeads = 2, headSize = 4, ropeDim = 4,
            pos = pos, base = 10000f, ropeType = RopeType.HALF_SPLIT
        )
        val f0 = pos / 1f
        val f1 = pos / 100f
        // Head 0: (1,3) rotated by f0, (2,4) rotated by f1
        assertCloseTo(1f * cos(f0) - 3f * sin(f0), buf[0])
        assertCloseTo(2f * cos(f1) - 4f * sin(f1), buf[1])
        assertCloseTo(1f * sin(f0) + 3f * cos(f0), buf[2])
        assertCloseTo(2f * sin(f1) + 4f * cos(f1), buf[3])
        // Head 1: (5,7) rotated by f0, (6,8) rotated by f1 — same freqs, different inputs
        assertCloseTo(5f * cos(f0) - 7f * sin(f0), buf[4])
        assertCloseTo(6f * cos(f1) - 8f * sin(f1), buf[5])
        assertCloseTo(5f * sin(f0) + 7f * cos(f0), buf[6])
        assertCloseTo(6f * sin(f1) + 8f * cos(f1), buf[7])
    }
}

private fun Double.pow(exp: Double): Double = kotlin.math.exp(exp * kotlin.math.ln(this))
