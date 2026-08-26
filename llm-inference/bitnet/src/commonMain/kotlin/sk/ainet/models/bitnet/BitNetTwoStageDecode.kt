package sk.ainet.models.bitnet

import sk.ainet.apps.llm.ScoredToken
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * NeoGPU's two-stage lm_head (transformers#337, upstream `hs_ml_infer.c` Stage 1/Stage 2): score
 * the full vocabulary cheaply with **planes 0–3** of a [TensorEncoding.BITNET_PLANES] weight,
 * keep the top candidates, and rescore only those rows **exactly** (all 8 planes).
 *
 * Why this is sound: plane `p` carries weight `1/3^p`, so the stage-1 truncation error of row `r`
 * is bounded by `rowScale(r) · Σ_{p=4..7} 3^{-p} · Σ_c |h(c)|` — computable per row, which lets
 * [topK] select candidates with a **guarantee**: every row whose stage-1 upper bound reaches the
 * k-th best stage-1 lower bound is rescored, so the returned top-k equals the exact full-matmul
 * top-k. No margin heuristics.
 *
 * This is deliberately a *sampling-level* utility, not a dispatch citizen: `KernelDispatch`'s
 * matmul over a `BITNET_PLANES` weight is always the exact 8-plane product (the invariant
 * matmul == decoded matmul), and approximation is an application decision — made here, visibly,
 * and handed to [sk.ainet.apps.llm.sampleFromCandidates].
 */
@OptIn(ExperimentalMemoryApi::class)
public object BitNetTwoStageDecode {

    /** NeoGPU's Stage-2 candidate count (`LMH_CANDIDATES`). */
    public const val DEFAULT_CANDIDATES: Int = 200

    private const val STAGE1_PLANES = 4

    /** `Σ_{p=4..7} 3^{-p}` — the per-unit residual weight the stage-1 scan ignores. */
    private const val TAIL_WEIGHT: Float =
        (1f / 81f) + (1f / 243f) + (1f / 729f) + (1f / 2187f)

    /**
     * Stage-1 scores — planes 0–3 with the FP16 row scales applied — for all [weight] rows
     * against [hidden]. This is exactly what the fused NEON kernel computes in one pass; the
     * Kotlin loop here is the portable reference of the same contract.
     */
    public fun stage1Scores(weight: BitNetPlanesTensorData, hidden: FloatArray): FloatArray {
        val n = weight.rows
        val k = weight.cols
        require(hidden.size == k) { "hidden has ${hidden.size} elements, weight expects $k" }
        val bytes = weight.packedData
        val planeStride = TensorEncoding.BITNET_PLANES.planeStrideBytes(n, k)
        val rowBytes = k / 4
        val out = FloatArray(n)
        for (r in 0 until n) {
            var acc = 0f
            var w = 1f
            for (p in 0 until STAGE1_PLANES) {
                val base = p * planeStride + r * rowBytes
                var dot = 0f
                for (c in 0 until k) {
                    val code = ((bytes[base + c / 4].toInt() and 0xFF) shr ((c % 4) * 2)) and 3
                    dot += (code - 1) * hidden[c]
                }
                acc += dot * w
                w /= 3f
            }
            out[r] = acc * TernaryCodec.planesRowScale(bytes, n, k, r)
        }
        return out
    }

    /** The exact (all-8-plane) score of row [row] — a decoded-row dot product. */
    public fun exactScore(weight: BitNetPlanesTensorData, hidden: FloatArray, row: Int): Float {
        val k = weight.cols
        val decoded = FloatArray(k)
        TernaryCodec.decodeBitNetPlanesRow(weight.packedData, weight.rows, k, row, decoded, 0)
        var acc = 0f
        for (c in 0 until k) acc += decoded[c] * hidden[c]
        return acc
    }

    /**
     * The exact top-[k] tokens of `weight · hidden`, found the two-stage way: a stage-1 scan over
     * all rows, then exact rescoring of every row whose error bound could still promote it into
     * the top-k. Returns candidates sorted by exact score, descending.
     *
     * The guarantee costs rescoring a few extra rows when scores are tightly packed; on real
     * lm_head distributions the rescored set stays near [k]. Cap the exact work with
     * [maxCandidates] (NeoGPU rescores a flat 200) — when the bound wants more than that, the
     * top-[maxCandidates] stage-1 rows are rescored and the guarantee becomes NeoGPU's heuristic.
     */
    public fun topK(
        weight: BitNetPlanesTensorData,
        hidden: FloatArray,
        k: Int = 8,
        maxCandidates: Int = DEFAULT_CANDIDATES,
    ): List<ScoredToken> {
        val n = weight.rows
        require(k in 1..n) { "k=$k outside 1..$n" }
        val stage1 = stage1Scores(weight, hidden)

        // Per-row error bound: |exact - stage1| <= rowScale * TAIL_WEIGHT * Σ|h|.
        var hAbs = 0f
        for (h in hidden) hAbs += if (h >= 0f) h else -h
        val bytes = weight.packedData

        // k-th best stage-1 LOWER bound.
        val lower = FloatArray(n) { r ->
            stage1[r] - TernaryCodec.planesRowScale(bytes, n, weight.cols, r) * TAIL_WEIGHT * hAbs
        }
        val kthLower = lower.sortedDescending()[k - 1]

        // Every row whose UPPER bound reaches it is a candidate.
        val candidates = (0 until n).filter { r ->
            stage1[r] + TernaryCodec.planesRowScale(bytes, n, weight.cols, r) * TAIL_WEIGHT * hAbs >= kthLower
        }
        val rescored = (
            if (candidates.size <= maxCandidates) candidates
            else candidates.sortedByDescending { stage1[it] }.take(maxCandidates)
            ).map { r -> ScoredToken(r, exactScore(weight, hidden, r)) }

        return rescored.sortedByDescending { it.score }.take(k)
    }
}
