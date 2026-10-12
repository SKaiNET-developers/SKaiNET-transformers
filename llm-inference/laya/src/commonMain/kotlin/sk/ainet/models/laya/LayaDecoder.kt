package sk.ainet.models.laya

import kotlin.math.exp
import kotlin.math.ln

/** A decoded answer; `probabilities` are in the caller's option order. */
public sealed class LayaAnswer {
    public abstract val probabilities: DoubleArray

    /** The reference `confidence`: `1 − H(p) / log k` for choice and score, `max(p, 1 − p)` for noul. */
    public abstract val confidence: Double

    /** `max(p)`: the mass on the reported answer (the reference `answer_confidence`). */
    public val answerConfidence: Double get() = if (probabilities.isEmpty()) 1.0 else probabilities.max().coerceIn(0.0, 1.0)

    public class Choice(public val choice: String, public val keys: List<String>, override val probabilities: DoubleArray, override val confidence: Double) : LayaAnswer() {
        public val byKey: Map<String, Double> get() = keys.indices.associate { keys[it] to probabilities[it] }
    }

    public class Score(public val score: Double, override val probabilities: DoubleArray, override val confidence: Double) : LayaAnswer()

    public class Noul(public val noul: Double, override val probabilities: DoubleArray, override val confidence: Double) : LayaAnswer()
}

/**
 * Logits → probabilities → typed answer, as the reference `Agent` decodes (REQ-DEC): temperature
 * from `temperature_by_options[type:bucket]`, else `temperature[type]`, each clamped to
 * `[0.5, 5.0]`; softmax; slots mapped back to options (`unpermute_probs`); `choice` = argmax,
 * `score` = expected level, `noul` = the probability of the `true` slot.
 */
public class LayaDecoder(public val config: LayaAgentConfig) {

    public fun temperature(type: LayaQuestionType, optionCount: Int): Double {
        val raw = config.temperatureByOptions[bucket(type, optionCount)] ?: config.temperature[type.index]
        return raw.coerceIn(TEMP_MIN, TEMP_MAX)
    }

    /** Probabilities in option order for slot-ordered [logits] produced with [optionOrder]. */
    public fun probabilities(logits: FloatArray, type: LayaQuestionType, optionOrder: IntArray): DoubleArray {
        val k = logits.size
        val t = temperature(type, k)
        val z = DoubleArray(k) { logits[it] / t }
        val max = z.max()
        var sum = 0.0
        val e = DoubleArray(k) { exp(z[it] - max).also { v -> sum += v } }
        val slotProbs = DoubleArray(k) { e[it] / sum }
        if (optionOrder.size != k) return slotProbs
        val canonical = DoubleArray(k)
        for (slot in 0 until k) canonical[optionOrder[slot]] = slotProbs[slot]
        return canonical
    }

    public fun decode(question: LayaQuestion, logits: FloatArray, optionOrder: IntArray): LayaAnswer {
        val p = probabilities(logits, question.type, optionOrder)
        val confidence = entropyConfidence(p)
        return when (question) {
            is LayaQuestion.Choice -> LayaAnswer.Choice(question.keys[argmax(p)], question.keys, p, confidence)
            is LayaQuestion.Score -> LayaAnswer.Score(p.indices.sumOf { it * p[it] }, p, confidence)
            // The reference reports max(p_true, 1 − p_true) for noul, not the entropy form.
            is LayaQuestion.Noul -> LayaAnswer.Noul(p[1], p, maxOf(p[1], 1.0 - p[1]))
        }
    }

    public companion object {
        public const val TEMP_MIN: Double = 0.5
        public const val TEMP_MAX: Double = 5.0

        public fun bucket(type: LayaQuestionType, optionCount: Int): String {
            val size = when {
                optionCount <= 2 -> "2"
                optionCount <= 5 -> "3-5"
                optionCount <= 10 -> "6-10"
                else -> "11+"
            }
            return "${type.wire}:$size"
        }

        public fun entropyConfidence(p: DoubleArray): Double {
            val k = p.size
            if (k < 2) return 1.0
            var entropy = 0.0
            for (v in p) entropy -= v * ln(v.coerceIn(1e-12, 1.0))
            return (1.0 - entropy / ln(k.toDouble())).coerceIn(0.0, 1.0)
        }

        private fun argmax(p: DoubleArray): Int {
            var best = 0
            for (i in 1 until p.size) if (p[i] > p[best]) best = i
            return best
        }
    }
}
