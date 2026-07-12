package sk.ainet.models.vec2text

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.tokenizer.SentencePieceTokenizer
import sk.ainet.models.t5.GtrEmbedder
import sk.ainet.models.t5.T5Config
import sk.ainet.models.t5.T5Runtime
import sk.ainet.models.t5.loadT5Weights
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * End-to-end embedding-inversion round-trip against the real gtr-base checkpoints.
 *
 * Skipped unless `VEC2TEXT_MODELS_DIR` holds `tokenizer.json`, `gtr_encoder.safetensors`,
 * `inversion.safetensors`, and `corrector.safetensors` (produced by
 * EmbeddingInversion/scripts/fetch-vec2text-models.sh). Loads ~1.1 GB and runs T5 decode
 * on CPU, so it is opt-in / slow. Prints the per-step hypothesis + cosine trace.
 */
class Vec2TextRoundTripTest {

    private val dir: File? = System.getenv("VEC2TEXT_MODELS_DIR")?.let(::File)

    /** SentencePiece-backed [Vec2TextTokenizer]: appends EOS (</s>=1), truncates to 32, strips specials. */
    private class T5Codec(
        private val sp: SentencePieceTokenizer,
        private val maxSeq: Int,
        private val eosId: Int,
    ) : Vec2TextTokenizer {
        override fun encodeForEmbedder(text: String): IntArray {
            val ids = sp.encode(text).take(maxSeq - 1).toMutableList()
            ids.add(eosId)
            return ids.toIntArray()
        }

        override fun decode(ids: IntArray): String =
            sp.decode(ids.filter { it != 0 && it != eosId }.toIntArray())
    }

    private suspend fun buildInverter(d: File): Vec2TextInverter<FP32> {
        val ctx = DirectCpuExecutionContext()
        val cfg = T5Config()
        fun loader(name: String) =
            SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(File(d, name).toString()) })
        val gtrWeights = loadT5Weights(loader("gtr_encoder.safetensors"), ctx, FP32::class, cfg, "", withDecoder = false)
        val embedder = GtrEmbedder(T5Runtime(ctx, gtrWeights, FP32::class))
        val inversion = InversionModel(ctx, Vec2TextWeightLoader.loadInversion(loader("inversion.safetensors"), ctx, FP32::class, cfg), FP32::class)
        val corrector = CorrectorModel(ctx, Vec2TextWeightLoader.loadCorrector(loader("corrector.safetensors"), ctx, FP32::class, cfg), FP32::class)
        val sp = SentencePieceTokenizer.fromTokenizerJson(Json.parseToJsonElement(File(d, "tokenizer.json").readText()).jsonObject)
        return Vec2TextInverter(embedder, inversion, corrector, T5Codec(sp, cfg.maxSeqLength, cfg.eosTokenId))
    }

    private fun modelsOrSkip(): File? {
        val d = dir
        val required = listOf("tokenizer.json", "gtr_encoder.safetensors", "inversion.safetensors", "corrector.safetensors")
        if (d == null || required.any { !File(d, it).exists() }) {
            println("SKIP: set VEC2TEXT_MODELS_DIR to the exported models dir")
            return null
        }
        return d
    }

    @Test
    fun invert_roundTrip() = runBlocking {
        val d = modelsOrSkip() ?: return@runBlocking
        val inverter = buildInverter(d)
        val text = "jack morris is a phd student at cornell tech in new york city"
        val result = inverter.invert(text, numSteps = 3, maxLength = 32)

        println("original:      $text")
        println("reconstructed: ${result.text}  (cos=${result.cosine})")
        result.trace.forEach { println("  step ${it.step}: cos=${it.cosine}  \"${it.text}\"") }
    }

    /** Beam search should reconstruct at least as well as greedy at matched steps. */
    @Test
    fun invert_beamBeatsGreedy() = runBlocking {
        val d = modelsOrSkip() ?: return@runBlocking
        val inverter = buildInverter(d)
        val text = "jack morris is a phd student at cornell tech in new york city"

        val greedy = inverter.invert(text, numSteps = 1, maxLength = 32)
        val beam = inverter.invert(text, numSteps = 1, maxLength = 32, sequenceBeamWidth = 3, tokenBeams = 3)

        println("greedy: cos=${greedy.cosine}  \"${greedy.text}\"")
        println("beam:   cos=${beam.cosine}  \"${beam.text}\"")
        // Beam explores more candidates and picks by cosine, so it should not do worse
        // (small fp tolerance for ties).
        assertTrue(beam.cosine >= greedy.cosine - 1e-4f, "beam ${beam.cosine} < greedy ${greedy.cosine}")
    }
}
