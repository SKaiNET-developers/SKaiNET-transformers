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

    @Test
    fun invert_roundTrip() = runBlocking {
        val d = dir
        val required = listOf("tokenizer.json", "gtr_encoder.safetensors", "inversion.safetensors", "corrector.safetensors")
        if (d == null || required.any { !File(d, it).exists() }) {
            println("SKIP Vec2TextRoundTripTest: set VEC2TEXT_MODELS_DIR to the exported models dir")
            return@runBlocking
        }

        val ctx = DirectCpuExecutionContext()
        val cfg = T5Config()

        fun loader(name: String) =
            SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(File(d, name).toString()) })

        val gtrWeights = loadT5Weights(loader("gtr_encoder.safetensors"), ctx, FP32::class, cfg, "", withDecoder = false)
        val embedder = GtrEmbedder(T5Runtime(ctx, gtrWeights, FP32::class))
        val inversion = InversionModel(ctx, Vec2TextWeightLoader.loadInversion(loader("inversion.safetensors"), ctx, FP32::class, cfg), FP32::class)
        val corrector = CorrectorModel(ctx, Vec2TextWeightLoader.loadCorrector(loader("corrector.safetensors"), ctx, FP32::class, cfg), FP32::class)

        val tokenizerJson = Json.parseToJsonElement(File(d, "tokenizer.json").readText()).jsonObject
        val sp = SentencePieceTokenizer.fromTokenizerJson(tokenizerJson)
        val codec = T5Codec(sp, cfg.maxSeqLength, cfg.eosTokenId)

        val inverter = Vec2TextInverter(embedder, inversion, corrector, codec)
        val text = "jack morris is a phd student at cornell tech in new york city"
        val result = inverter.invert(text, numSteps = 3, maxLength = 32)

        println("original:      $text")
        println("reconstructed: ${result.text}  (cos=${result.cosine})")
        result.trace.forEach { println("  step ${it.step}: cos=${it.cosine}  \"${it.text}\"") }
    }
}
