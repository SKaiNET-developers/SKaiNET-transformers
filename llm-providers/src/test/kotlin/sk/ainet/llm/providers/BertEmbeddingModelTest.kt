package sk.ainet.llm.providers

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.nio.file.Path
import kotlin.io.path.isDirectory
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * One-call factory tests on a real LEAF snapshot. Both tests self-skip when
 * no model is available; [fromHuggingFace_downloadsAndEmbeds] additionally
 * needs network on a cold cache, so it is integration-tagged.
 */
@Tag("integration")
class BertEmbeddingModelTest {

    private fun localSnapshot(): Path? {
        System.getenv("LEAF_MODEL_DIR")?.let { p ->
            Path.of(p).takeIf { it.isDirectory() }?.let { return it }
        }
        val home = System.getProperty("user.home") ?: return null
        return Path.of(home, ".cache", "skainet", "models", "MongoDB_mdbr-leaf-mt")
            .takeIf { it.isDirectory() }
    }

    private fun norm(v: FloatArray) = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot / (norm(a) * norm(b))
    }

    @Test
    fun fromSafeTensors_embedsAndRanksSemantically() {
        val dir = localSnapshot()
        assumeTrue(dir != null, "No local LEAF snapshot — skipping")
        val model = BertEmbeddingModel.fromSafeTensors(dir!!)

        val query = model.embed("How do I reset my password?")
        assertEquals(model.dimensions, query.size)
        assertTrue(abs(norm(query) - 1f) < 0.01f, "embedding must be L2-normalized")

        val related = model.embed("Steps to change a forgotten password")
        val unrelated = model.embed("The quick brown fox jumps over the lazy dog")
        val simRelated = cosine(query, related)
        val simUnrelated = cosine(query, unrelated)
        assertTrue(
            simRelated > simUnrelated,
            "semantic ranking broken: related=$simRelated unrelated=$simUnrelated"
        )
    }

    @Test
    fun fromHuggingFace_downloadsAndEmbeds() {
        // Uses the shared skainet model cache: instant when warm, downloads
        // (~90 MB) when cold. Self-skips only if the machine is fully offline
        // AND the cache is cold — surfaced as a DataSourceException then.
        val model = BertEmbeddingModel.fromHuggingFace("MongoDB/mdbr-leaf-mt")
        val v = model.embed("hello world")
        assertEquals(model.dimensions, v.size)
        assertEquals(1024, v.size, "mdbr-leaf-mt projects to 1024 dims")
        assertTrue(abs(norm(v) - 1f) < 0.01f)
    }
}
