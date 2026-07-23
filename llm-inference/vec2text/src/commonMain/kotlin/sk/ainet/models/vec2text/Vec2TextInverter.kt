package sk.ainet.models.vec2text

import sk.ainet.lang.tensor.Tensor
import sk.ainet.models.t5.GtrEmbedder
import sk.ainet.lang.types.DType
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Tokenizer contract the inverter needs. Both operations use the T5 SentencePiece tokenizer
 * (t5-base); [encodeForEmbedder] truncates to the embedder's `max_seq_length` and appends the
 * EOS `</s>` exactly as vec2text's `embedder_tokenizer` does. Wire this to the concrete
 * SentencePiece tokenizer in the consuming app.
 */
public interface Vec2TextTokenizer {
    /** Encode text to embedder token ids (truncated + EOS, no padding for batch-1). */
    public fun encodeForEmbedder(text: String): IntArray

    /** Decode generated T5 token ids to text, skipping special tokens. */
    public fun decode(ids: IntArray): String
}

/** One correction step: the hypothesis text and its cosine similarity to the target embedding. */
public data class Vec2TextStep(val step: Int, val text: String, val cosine: Float)

/** Result of an inversion: the best reconstruction, its cosine score, and the full step trace. */
public data class Vec2TextResult(
    val text: String,
    val cosine: Float,
    val trace: List<Vec2TextStep>,
)

/**
 * Drives the vec2text iterative correction loop (port of `vec2text/trainers/corrector.py`
 * with `sequence_beam_width = 1`, i.e. greedy): initial hypothesis from the inversion model,
 * then repeatedly re-embed the hypothesis text and run the corrector, keeping the hypothesis
 * with the highest cosine similarity to the target embedding. Early-stops when the best score
 * plateaus (`|Δ| < 1e-3`).
 */
public class Vec2TextInverter<T : DType>(
    private val embedder: GtrEmbedder<T>,
    private val inversion: InversionModel<T>,
    private val corrector: CorrectorModel<T>,
    private val tokenizer: Vec2TextTokenizer,
) {
    /**
     * Invert [text]'s embedding back to text using [numSteps] correction rounds.
     *
     * With [sequenceBeamWidth] and [tokenBeams] both 1 (default) this is the greedy path.
     * Raising them enables beam search (vec2text's main quality lever):
     * - [tokenBeams] — token-level beam inside each T5 generation.
     * - [sequenceBeamWidth] — keep this many hypotheses across correction rounds, ranked by
     *   cosine similarity to the target embedding (the "oracle" that beam search exploits).
     * (Set [numSteps] = 0 for single-shot inversion — the hypothesizer only.)
     */
    public fun invert(
        text: String,
        numSteps: Int = 20,
        maxLength: Int = 128,
        sequenceBeamWidth: Int = 1,
        tokenBeams: Int = 1,
    ): Vec2TextResult =
        invertEmbedding(embedder.embed(tokenizer.encodeForEmbedder(text)), numSteps, maxLength, sequenceBeamWidth, tokenBeams)

    /** As [invert], but starting from a raw target embedding (e.g. an interpolated vector). */
    public fun invertEmbedding(
        target: Tensor<T, Float>,
        numSteps: Int = 20,
        maxLength: Int = 128,
        sequenceBeamWidth: Int = 1,
        tokenBeams: Int = 1,
    ): Vec2TextResult =
        if (sequenceBeamWidth <= 1 && tokenBeams <= 1) invertGreedy(target, numSteps, maxLength)
        else invertBeam(target, numSteps, maxLength, sequenceBeamWidth.coerceAtLeast(1), tokenBeams.coerceAtLeast(1))

    private fun invertGreedy(target: Tensor<T, Float>, numSteps: Int, maxLength: Int): Vec2TextResult {
        var hypIds = inversion.invert(target, maxLength)
        var hypText = tokenizer.decode(hypIds)
        var cos = cosineToTarget(target, hypText)

        val trace = ArrayList<Vec2TextStep>()
        trace.add(Vec2TextStep(0, hypText, cos))
        var bestText = hypText
        var bestCos = cos

        for (step in 1..numSteps) {
            val hypEmb = embedder.embed(tokenizer.encodeForEmbedder(hypText))
            hypIds = corrector.correct(target, hypEmb, hypIds, maxLength)
            hypText = tokenizer.decode(hypIds)
            val newCos = cosineToTarget(target, hypText)
            trace.add(Vec2TextStep(step, hypText, newCos))

            val improvement = newCos - bestCos
            if (newCos > bestCos) {
                bestCos = newCos
                bestText = hypText
            }
            cos = newCos
            if (abs(improvement) < EARLY_STOP_ATOL) break
        }
        return Vec2TextResult(bestText, bestCos, trace)
    }

    /** Sequence-level beam: keep [beamWidth] hypotheses per round, ranked by cosine to [target]. */
    private fun invertBeam(
        target: Tensor<T, Float>,
        numSteps: Int,
        maxLength: Int,
        beamWidth: Int,
        tokenBeams: Int,
    ): Vec2TextResult {
        var beams = rankByCosine(target, inversion.invertBeam(target, maxOf(beamWidth, tokenBeams), maxLength)).take(beamWidth)
        val trace = ArrayList<Vec2TextStep>()
        var best = beams.first()
        trace.add(Vec2TextStep(0, best.text, best.cos))

        for (step in 1..numSteps) {
            val pool = ArrayList<IntArray>()
            for (b in beams) {
                val hypEmb = embedder.embed(tokenizer.encodeForEmbedder(b.text))
                pool += corrector.correctBeam(target, hypEmb, b.ids, tokenBeams, maxLength)
            }
            beams = rankByCosine(target, pool).take(beamWidth)
            val stepBest = beams.first()
            trace.add(Vec2TextStep(step, stepBest.text, stepBest.cos))

            val improvement = stepBest.cos - best.cos
            if (stepBest.cos > best.cos) best = stepBest
            if (abs(improvement) < EARLY_STOP_ATOL) break
        }
        return Vec2TextResult(best.text, best.cos, trace)
    }

    /** Decode + re-embed each candidate, dedup by text, and sort by cosine to [target] (best first). */
    private fun rankByCosine(target: Tensor<T, Float>, idsList: List<IntArray>): List<Candidate> =
        idsList.asSequence()
            .map { ids -> tokenizer.decode(ids) to ids }
            .distinctBy { it.first }
            .map { (text, ids) -> Candidate(ids, text, cosineToTarget(target, text)) }
            .sortedByDescending { it.cos }
            .toList()

    private class Candidate(val ids: IntArray, val text: String, val cos: Float)

    private fun cosineToTarget(target: Tensor<T, Float>, hypText: String): Float {
        val hypEmb = embedder.embed(tokenizer.encodeForEmbedder(hypText))
        return cosine(target, hypEmb)
    }

    public companion object {
        public const val EARLY_STOP_ATOL: Float = 1e-3f

        /** Cosine similarity of two rank-1 embedding tensors. */
        public fun <T : DType> cosine(a: Tensor<T, Float>, b: Tensor<T, Float>): Float {
            val av = a.data.copyToFloatArray()
            val bv = b.data.copyToFloatArray()
            var dot = 0.0
            var na = 0.0
            var nb = 0.0
            for (i in av.indices) {
                dot += av[i] * bv[i]
                na += av[i] * av[i]
                nb += bv[i] * bv[i]
            }
            val denom = sqrt(na) * sqrt(nb)
            return if (denom > 0.0) (dot / denom).toFloat() else 0.0f
        }
    }
}
