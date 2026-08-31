package sk.ainet.models.bitnet

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.ScoredToken
import sk.ainet.apps.llm.sampleFromCandidates
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData

/**
 * transformers#337: the two-stage lm_head must return the EXACT top-k — the bound-based
 * candidate selection makes stage-1 truncation invisible to the sampler. Oracle: brute-force
 * exact scores over all rows.
 */
@OptIn(ExperimentalMemoryApi::class)
class BitNetTwoStageDecodeTest {

    private fun weight(n: Int, k: Int, seed: Int): BitNetPlanesTensorData {
        val rng = Random(seed)
        return BitNetPlanesTensorData.fromFloats(
            Shape(n, k),
            FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f },
        )
    }

    private fun exactAll(w: BitNetPlanesTensorData, h: FloatArray): FloatArray =
        FloatArray(w.rows) { BitNetTwoStageDecode.exactScore(w, h, it) }

    @Test
    fun topKEqualsTheBruteForceExactTopK() {
        val n = 64; val k = 24
        val w = weight(n, k, seed = 5)
        val rng = Random(9)
        repeat(5) { trial ->
            val h = FloatArray(k) { rng.nextFloat() - 0.5f }
            val exact = exactAll(w, h)
            val expected = exact.withIndex().sortedByDescending { it.value }.take(8).map { it.index }
            val got = BitNetTwoStageDecode.topK(w, h, k = 8).map { it.token }
            assertEquals(expected, got, "trial $trial: two-stage top-8 must equal exact top-8")
        }
    }

    @Test
    fun stage1IsWithinTheDocumentedBoundOfExact() {
        val n = 32; val k = 24
        val w = weight(n, k, seed = 7)
        val rng = Random(11)
        val h = FloatArray(k) { rng.nextFloat() - 0.5f }
        var hAbs = 0f
        for (v in h) hAbs += abs(v)
        val tail = (1f / 81f) + (1f / 243f) + (1f / 729f) + (1f / 2187f)
        val stage1 = BitNetTwoStageDecode.stage1Scores(w, h)
        for (r in 0 until n) {
            val bound = TernaryCodec.planesRowScale(w.packedData, n, k, r) * tail * hAbs + 1e-5f
            val err = abs(stage1[r] - BitNetTwoStageDecode.exactScore(w, h, r))
            assertTrue(err <= bound, "row $r: |stage1 - exact| = $err > bound $bound")
        }
    }

    @Test
    fun nativeStage1MatchesTheKotlinReference() {
        // The fused lmhead_stage1 kernel and the portable loop implement the same contract
        // (#358); skip quietly where the bundled native library is not available.
        if (!sk.ainet.exec.kernel.NativeTernaryLmheadKernel.isAvailable()) return
        val n = 48; val k = 32
        val w = weight(n, k, seed = 13)
        val rng = Random(17)
        repeat(3) { trial ->
            val h = FloatArray(k) { rng.nextFloat() - 0.5f }
            val ref = BitNetTwoStageDecode.stage1Scores(w, h)
            val native = BitNetTwoStageDecode.stage1Scores(
                w, h,
                native = BitNetStage1Kernel(sk.ainet.exec.kernel.NativeTernaryLmheadKernel::lmheadStage1),
            )
            for (r in 0 until n) {
                assertTrue(
                    abs(ref[r] - native[r]) <= 1e-4f * maxOf(1f, abs(ref[r])),
                    "trial $trial row $r: native ${native[r]} vs reference ${ref[r]}",
                )
            }
        }
    }

    @Test
    fun sampleFromCandidatesIsGreedyAtZeroTemperatureAndStaysInTheList() {
        val candidates = listOf(ScoredToken(3, 1.0f), ScoredToken(17, 2.5f), ScoredToken(9, -0.5f))
        assertEquals(17, sampleFromCandidates(candidates, temperature = 0f))
        val allowed = candidates.map { it.token }.toSet()
        val rng = Random(3)
        repeat(50) {
            assertTrue(sampleFromCandidates(candidates, temperature = 0.8f, random = rng) in allowed)
        }
    }
}
