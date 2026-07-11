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
     * (Set [numSteps] = 0 for single-shot inversion — the hypothesizer only.)
     */
    public fun invert(text: String, numSteps: Int = 20, maxLength: Int = 128): Vec2TextResult {
        val target = embedder.embed(tokenizer.encodeForEmbedder(text))

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
