package sk.ainet.apps.llm

import kotlin.math.exp
import kotlin.random.Random

/**
 * Numerically stable in-place softmax over the first [length] elements of [values].
 *
 * After this call, `values[0..length-1]` will sum to 1.0 (approximately).
 * Uses the max-subtraction trick to avoid overflow.
 */
public fun softmaxInPlace(values: FloatArray, length: Int = values.size) {
    var maxVal = Float.NEGATIVE_INFINITY
    for (i in 0 until length) {
        if (values[i] > maxVal) maxVal = values[i]
    }
    if (maxVal.isInfinite()) maxVal = 0f
    var sum = 0f
    for (i in 0 until length) {
        val e = exp((values[i] - maxVal).toDouble()).toFloat()
        values[i] = e
        sum += e
    }
    if (sum == 0f) return
    val inv = 1f / sum
    for (i in 0 until length) {
        values[i] *= inv
    }
}

/**
 * Sample a token ID from a logits array.
 *
 * **Warning:** This function mutates [logits] in-place (temperature scaling + softmax).
 * Pass a copy if you need to preserve the original values.
 *
 * - temperature <= 1e-6 -> greedy (argmax)
 * - otherwise -> temperature-scaled softmax + categorical sample
 *
 * @param logits Raw logits array (will be mutated for temperature > 0)
 * @param temperature Sampling temperature
 * @param random Random generator for sampling
 * @return The sampled index
 */
public fun sampleFromLogits(
    logits: FloatArray,
    temperature: Float,
    random: Random = Random.Default
): Int {
    // Greedy (argmax) for near-zero temperature
    if (temperature <= 1e-6f) {
        var best = 0
        var bestVal = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestVal) {
                bestVal = logits[i]
                best = i
            }
        }
        return best
    }

    // Temperature scaling
    var maxLogit = Float.NEGATIVE_INFINITY
    for (i in logits.indices) {
        val v = logits[i] / temperature
        logits[i] = v
        if (v > maxLogit) maxLogit = v
    }

    // Softmax + categorical sample (combined to skip normalization)
    var sum = 0f
    for (i in logits.indices) {
        val e = exp((logits[i] - maxLogit).toDouble()).toFloat()
        logits[i] = e
        sum += e
    }

    val r = random.nextFloat() * sum
    var acc = 0f
    for (i in logits.indices) {
        acc += logits[i]
        if (acc >= r) return i
    }
    return logits.lastIndex
}

/**
 * Sample a token ID from a logits array with top-k / top-p (nucleus) filtering —
 * the sampling triple small instruction-tuned checkpoints ship as their
 * recommended settings (FunctionGemma 270M: temperature 1.0, topK 64, topP 0.95).
 *
 * **Warning:** mutates [logits] in-place like [sampleFromLogits].
 *
 * - `temperature <= 1e-6` → greedy (argmax); topK/topP are irrelevant
 * - `topK <= 0` disables the top-k cut; `topP >= 1` disables the nucleus cut
 * - both cuts active: top-k first, then nucleus over the survivors — the
 *   llama.cpp/HF ordering
 */
public fun sampleFromLogits(
    logits: FloatArray,
    temperature: Float,
    topK: Int,
    topP: Float,
    random: Random = Random.Default
): Int {
    if (temperature <= 1e-6f) return sampleFromLogits(logits, temperature, random)
    val kActive = topK in 1 until logits.size
    val pActive = topP < 1f
    if (!kActive && !pActive) return sampleFromLogits(logits, temperature, random)

    // Select the top-k indices by logit (k is small — insertion into a k-sized
    // scratch beats sorting a 256k vocab).
    val k = if (kActive) topK else logits.size
    val idx = IntArray(k) { -1 }
    val v = FloatArray(k) { Float.NEGATIVE_INFINITY }
    for (i in logits.indices) {
        val x = logits[i]
        if (x <= v[k - 1]) continue
        var j = k - 1
        while (j > 0 && v[j - 1] < x) {
            v[j] = v[j - 1]; idx[j] = idx[j - 1]; j--
        }
        v[j] = x; idx[j] = i
    }
    var count = 0
    while (count < k && idx[count] >= 0) count++

    // Temperature-scaled softmax over the survivors only.
    val probs = FloatArray(count) { v[it] / temperature }
    softmaxInPlace(probs)

    // Nucleus cut: survivors are sorted desc, keep the smallest prefix with
    // cumulative probability >= topP (always at least one token).
    var cut = count
    if (pActive) {
        var acc = 0f
        for (i in 0 until count) {
            acc += probs[i]
            if (acc >= topP) { cut = i + 1; break }
        }
    }
    var sum = 0f
    for (i in 0 until cut) sum += probs[i]
    var r = random.nextFloat() * sum
    for (i in 0 until cut) {
        r -= probs[i]
        if (r <= 0f) return idx[i]
    }
    return idx[cut - 1]
}

/**
 * A token with a score — the currency of candidate-based sampling (transformers#337): a two-stage
 * lm_head (cheap approximate scoring over the full vocab, exact rescoring of the survivors) hands
 * the sampler a *candidate list* instead of a full-vocab logits array.
 */
public data class ScoredToken(public val token: Int, public val score: Float)

/**
 * Sample a token from a candidate list — the candidate-shaped sibling of [sampleFromLogits],
 * with the same temperature semantics:
 *
 * - `temperature <= 1e-6` → greedy (highest score)
 * - otherwise → temperature-scaled softmax over the **candidates** + categorical sample
 *
 * The probability mass outside [candidates] is treated as zero — that is the two-stage contract:
 * whoever built the list already decided the tail cannot win. Pair with a rescorer that selects
 * candidates conservatively (e.g. `BitNetTwoStageDecode.topK`).
 */
public fun sampleFromCandidates(
    candidates: List<ScoredToken>,
    temperature: Float,
    random: Random = Random.Default,
): Int {
    require(candidates.isNotEmpty()) { "sampleFromCandidates needs at least one candidate" }
    if (temperature <= 1e-6f) {
        var best = candidates[0]
        for (i in 1 until candidates.size) if (candidates[i].score > best.score) best = candidates[i]
        return best.token
    }
    val scores = FloatArray(candidates.size) { candidates[it].score / temperature }
    softmaxInPlace(scores)
    var r = random.nextFloat()
    for (i in scores.indices) {
        r -= scores[i]
        if (r <= 0f) return candidates[i].token
    }
    return candidates.last().token
}
