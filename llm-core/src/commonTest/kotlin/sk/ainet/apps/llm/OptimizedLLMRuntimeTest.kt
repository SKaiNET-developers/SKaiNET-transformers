package sk.ainet.apps.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for [OutputEquivalenceChecker] — validates the comparison and assertion
 * utilities used for DIRECT vs OPTIMIZED mode equivalence testing.
 *
 * Full end-to-end runtime equivalence tests (DIRECT vs OPTIMIZED with real model
 * weights) live in the platform-specific test suites that have access to
 * [DirectCpuExecutionContext] and model files.
 */
class OutputEquivalenceCheckerTest {

    @Test
    fun identicalLogitsPass() {
        val checker = OutputEquivalenceChecker(tolerance = 1e-4f)
        val logits = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val result = checker.compareLogits(logits, logits.copyOf(), "identical")
        assertTrue(result.passed)
        assertEquals(0f, result.maxAbsDiff)
        assertEquals(0, result.mismatchCount)
    }

    @Test
    fun withinTolerancePasses() {
        val checker = OutputEquivalenceChecker(tolerance = 1e-4f)
        val ref = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val cand = floatArrayOf(1.00005f, 2.00003f, 3.00001f, 4.00009f)
        val result = checker.compareLogits(ref, cand, "within-tol")
        assertTrue(result.passed)
        assertEquals(0, result.mismatchCount)
    }

    @Test
    fun exceedingToleranceFails() {
        val checker = OutputEquivalenceChecker(tolerance = 1e-4f, maxMismatchFraction = 0f)
        val ref = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val cand = floatArrayOf(1.0f, 2.1f, 3.0f, 4.0f) // element [1] differs by 0.1
        val result = checker.compareLogits(ref, cand, "exceed-tol")
        assertFalse(result.passed)
        assertEquals(1, result.mismatchCount)
    }

    @Test
    fun assertLogitsMatchThrowsOnFailure() {
        val checker = OutputEquivalenceChecker(tolerance = 1e-6f, maxMismatchFraction = 0f)
        val ref = floatArrayOf(1.0f, 2.0f)
        val cand = floatArrayOf(1.0f, 3.0f)
        assertFails {
            checker.assertLogitsMatch(ref, cand, "should-fail")
        }
    }

    @Test
    fun shapeMismatchThrows() {
        val checker = OutputEquivalenceChecker()
        assertFails {
            checker.compareLogits(floatArrayOf(1f, 2f), floatArrayOf(1f), "size-mismatch")
        }
    }

    @Test
    fun sequenceMatchPassesForIdentical() {
        val checker = OutputEquivalenceChecker()
        // Should not throw
        checker.assertSequenceMatch(listOf(1, 2, 3), listOf(1, 2, 3), "identical-seq")
    }

    @Test
    fun sequenceMatchFailsOnDifference() {
        val checker = OutputEquivalenceChecker()
        assertFails {
            checker.assertSequenceMatch(listOf(1, 2, 3), listOf(1, 4, 3), "diff-seq")
        }
    }

    @Test
    fun forwardPassComparisonTracksSteps() {
        val checker = OutputEquivalenceChecker(tolerance = 1e-4f)
        val tokens = intArrayOf(1, 2, 3)
        val results = checker.compareForwardPass(
            tokens = tokens,
            referenceForward = { floatArrayOf(it.toFloat(), it.toFloat() * 2) },
            candidateForward = { floatArrayOf(it.toFloat(), it.toFloat() * 2) },
            label = "test"
        )
        assertEquals(3, results.size)
        assertTrue(results.all { it.passed })
    }

    @Test
    fun emptyLogitsPass() {
        val checker = OutputEquivalenceChecker()
        val result = checker.compareLogits(floatArrayOf(), floatArrayOf(), "empty")
        assertTrue(result.passed)
        assertEquals(0, result.totalElements)
    }
}
