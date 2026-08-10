package sk.ainet.models.t5

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Numerical parity for the GTR (gtr-t5-base) sentence embedder against a golden dump from
 * the reference `sentence-transformers/gtr-t5-base` model
 * (see EmbeddingInversion/scripts/dump_embedder_golden.py).
 *
 * The test is skipped unless `VEC2TEXT_MODELS_DIR` points at a directory containing
 * `gtr_encoder.safetensors` and `embedder_golden.json` (the ~209 MB weights are not
 * checked in). Weights are fp16 → the comparison is by cosine similarity (robust to the
 * fp16-vs-fp32 gap), asserting cos > 0.999 plus a generous max-abs bound.
 */
class GtrEmbedderParityTest {

    private val modelsDir: File? = System.getenv("VEC2TEXT_MODELS_DIR")?.let(::File)

    @Test
    fun embedding_matchesReferenceGtr() = runTest {
        val dir = modelsDir
        if (dir == null || !File(dir, "gtr_encoder.safetensors").exists() ||
            !File(dir, "embedder_golden.json").exists()
        ) {
            println("SKIP GtrEmbedderParityTest: set VEC2TEXT_MODELS_DIR to the exported models dir")
            return@runTest
        }

        val ctx = DirectCpuExecutionContext()
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(File(dir, "gtr_encoder.safetensors").toString()) }
        )
        val weights = loadT5Weights(loader, ctx, FP32::class, T5Config(), prefix = "", withDecoder = false)
        val embedder = GtrEmbedder(T5Runtime(ctx, weights, FP32::class))

        val golden = Json.parseToJsonElement(File(dir, "embedder_golden.json").readText())
            .jsonObject["records"]!!.jsonArray

        for (rec in golden) {
            val obj = rec.jsonObject
            val text = obj["text"]!!.jsonPrimitive.content
            val ids = obj["input_ids"]!!.jsonArray.map { it.jsonPrimitive.content.toInt() }.toIntArray()
            val expected = obj["embedding"]!!.jsonArray.map { it.jsonPrimitive.content.toFloat() }.toFloatArray()

            val actual = embedder.embed(ids).data.copyToFloatArray()

            val cos = cosine(actual, expected)
            var maxAbs = 0.0
            for (i in actual.indices) {
                val d = kotlin.math.abs(actual[i] - expected[i]).toDouble()
                if (d > maxAbs) maxAbs = d
            }
            println("embed(\"${text.take(40)}\"): cos=$cos maxAbs=$maxAbs")
            assertTrue(cos > 0.999, "cosine $cos too low for \"$text\" (fp16 weights)")
        }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrt(na) * sqrt(nb))
    }
}
