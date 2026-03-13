package sk.ainet.apps.llm

import kotlin.math.abs
import kotlin.test.assertTrue

/**
 * Test utility for validating output equivalence between runtime implementations.
 *
 * Compares token-level logits and generated sequences between:
 * - Old hand-coded runtimes (LlamaRuntime, BertRuntime, ApertusRuntime)
 * - New OptimizedLLMRuntime in DIRECT mode
 * - New OptimizedLLMRuntime in OPTIMIZED mode
 *
 * Usage:
 * ```kotlin
 * val checker = OutputEquivalenceChecker(tolerance = 1e-4f)
 * checker.assertLogitsMatch(
 *     reference = oldRuntime.forward(tokenId),
 *     candidate = newRuntime.forward(tokenId),
 *     label = "Llama token=42"
 * )
 * ```
 */
class OutputEquivalenceChecker(
    /** Maximum allowed absolute difference per logit element. */
    val tolerance: Float = 1e-4f,
    /** Maximum allowed fraction of elements exceeding tolerance. */
    val maxMismatchFraction: Float = 0.001f
) {

    data class ComparisonResult(
        val label: String,
        val maxAbsDiff: Float,
        val meanAbsDiff: Float,
        val mismatchCount: Int,
        val totalElements: Int,
        val passed: Boolean
    ) {
        val mismatchFraction: Float get() = if (totalElements > 0) mismatchCount.toFloat() / totalElements else 0f

        override fun toString(): String = buildString {
            val status = if (passed) "PASS" else "FAIL"
            append("[$status] $label: ")
            append("maxDiff=%.6f, meanDiff=%.6f, ".format(maxAbsDiff, meanAbsDiff))
            append("mismatches=$mismatchCount/$totalElements (%.4f%%)".format(mismatchFraction * 100))
        }
    }

    /**
     * Compare two float arrays element-wise.
     */
    fun compareLogits(
        reference: FloatArray,
        candidate: FloatArray,
        label: String = "logits"
    ): ComparisonResult {
        require(reference.size == candidate.size) {
            "Shape mismatch: reference=${reference.size} vs candidate=${candidate.size}"
        }

        var maxDiff = 0f
        var sumDiff = 0f
        var mismatches = 0

        for (i in reference.indices) {
            val diff = abs(reference[i] - candidate[i])
            if (diff > maxDiff) maxDiff = diff
            sumDiff += diff
            if (diff > tolerance) mismatches++
        }

        val meanDiff = if (reference.isNotEmpty()) sumDiff / reference.size else 0f
        val mismatchFrac = if (reference.isNotEmpty()) mismatches.toFloat() / reference.size else 0f

        return ComparisonResult(
            label = label,
            maxAbsDiff = maxDiff,
            meanAbsDiff = meanDiff,
            mismatchCount = mismatches,
            totalElements = reference.size,
            passed = mismatchFrac <= maxMismatchFraction
        )
    }

    /**
     * Assert that two logit arrays are equivalent within tolerance.
     */
    fun assertLogitsMatch(
        reference: FloatArray,
        candidate: FloatArray,
        label: String = "logits"
    ) {
        val result = compareLogits(reference, candidate, label)
        assertTrue(result.passed, result.toString())
    }

    /**
     * Assert that two token sequences are identical.
     */
    fun assertSequenceMatch(
        reference: List<Int>,
        candidate: List<Int>,
        label: String = "sequence"
    ) {
        if (reference == candidate) return

        val firstDiff = reference.zip(candidate).indexOfFirst { (a, b) -> a != b }
        val message = buildString {
            appendLine("[$label] Token sequence mismatch at position $firstDiff")
            appendLine("  reference: ${reference.take(20)}${if (reference.size > 20) "..." else ""}")
            appendLine("  candidate: ${candidate.take(20)}${if (candidate.size > 20) "..." else ""}")
            if (firstDiff >= 0) {
                appendLine("  first diff: ref[${firstDiff}]=${reference[firstDiff]} vs cand[${firstDiff}]=${candidate[firstDiff]}")
            }
            if (reference.size != candidate.size) {
                appendLine("  length: ref=${reference.size} vs cand=${candidate.size}")
            }
        }
        assertTrue(false, message)
    }

    /**
     * Run a multi-token forward pass comparison.
     *
     * Feeds the same token sequence through both runtimes and compares
     * logits at each step.
     *
     * @param tokens Token IDs to feed through both runtimes
     * @param referenceForward Forward function for the reference runtime
     * @param candidateForward Forward function for the candidate runtime
     * @param label Description for diagnostics
     * @return List of per-step comparison results
     */
    fun compareForwardPass(
        tokens: IntArray,
        referenceForward: (Int) -> FloatArray,
        candidateForward: (Int) -> FloatArray,
        label: String = "forward"
    ): List<ComparisonResult> {
        return tokens.mapIndexed { step, tokenId ->
            val refLogits = referenceForward(tokenId)
            val candLogits = candidateForward(tokenId)
            compareLogits(refLogits, candLogits, "$label step=$step token=$tokenId")
        }
    }

    /**
     * Assert all steps in a forward pass comparison match.
     */
    fun assertForwardPassMatch(
        tokens: IntArray,
        referenceForward: (Int) -> FloatArray,
        candidateForward: (Int) -> FloatArray,
        label: String = "forward"
    ) {
        val results = compareForwardPass(tokens, referenceForward, candidateForward, label)
        val failures = results.filter { !it.passed }
        if (failures.isNotEmpty()) {
            val message = buildString {
                appendLine("Forward pass equivalence check failed (${failures.size}/${results.size} steps):")
                failures.forEach { appendLine("  $it") }
            }
            assertTrue(false, message)
        }
    }
}
