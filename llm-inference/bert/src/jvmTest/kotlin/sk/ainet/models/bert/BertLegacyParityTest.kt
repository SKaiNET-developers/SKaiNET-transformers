@file:Suppress("DEPRECATION")

package sk.ainet.models.bert

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.types.FP32

/**
 * Numerical parity between the legacy eager [BertRuntime] (validated against
 * PyTorch at 5e-4 upstream) and the DSL-path [BertEncoderRuntime], on a real
 * LEAF checkpoint. Passing transfers the PyTorch validation to the new path.
 *
 * This test is deleted together with [BertRuntime] once the migration is
 * complete — [BertNumericalAccuracyTest] (golden vectors) remains the
 * long-term numerics gate.
 *
 * Model resolution: `LEAF_MODEL_DIR` env → `~/.cache/skainet/models/MongoDB_mdbr-leaf-mt`
 * → HF hub cache (`models--MongoDB--mdbr-leaf-ir`). Self-skips when absent.
 */
@Tag("integration")
class BertLegacyParityTest {

    private val ctx = DirectCpuExecutionContext()

    private fun resolveModelDir(): Path? {
        System.getenv("LEAF_MODEL_DIR")?.let { p ->
            Path.of(p).takeIf { it.isDirectory() }?.let { return it }
        }
        val home = System.getProperty("user.home") ?: return null
        Path.of(home, ".cache", "skainet", "models", "MongoDB_mdbr-leaf-mt")
            .takeIf { it.isDirectory() }?.let { return it }
        val snapshots = Path.of(home, ".cache", "huggingface", "hub", "models--MongoDB--mdbr-leaf-ir", "snapshots")
        if (!snapshots.isDirectory()) return null
        return snapshots.toFile().listFiles()?.firstOrNull { it.isDirectory }?.toPath()
    }

    /** Token-id sequences shaped like WordPiece output: [CLS] … [SEP]. */
    private val sequences = listOf(
        intArrayOf(101, 7592, 2088, 102),
        intArrayOf(101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102),
        intArrayOf(101, 2129, 2079, 1045, 25141, 2026, 20786, 102),
    )

    @Test
    fun newRuntimeMatchesLegacyRuntimeOnRealModel() = runTest {
        val modelDir = resolveModelDir()
        assumeTrue(modelDir != null, "No LEAF model available — skipping parity test")
        modelDir!!
        val mainFile = modelDir.resolve("model.safetensors")
        assumeTrue(mainFile.exists(), "model.safetensors not found in $modelDir")

        val denseFile = modelDir.resolve("2_Dense").resolve("model.safetensors")
        val denseConfig = modelDir.resolve("2_Dense").resolve("config.json")
        val config = BertConfigParser.parse(
            modelDir.resolve("config.json").readText(),
            if (denseConfig.exists()) denseConfig.readText() else null,
        )

        fun loaderFor(path: Path) = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(path.toString()) }
        )

        // Encoder-only comparison: the legacy runtime cannot apply LEAF's
        // bias-free projection (it required weight AND bias, silently skipping
        // otherwise), so old-vs-new parity is meaningful only up to pooling.
        // The projection itself is verified below by independent recomputation.
        val encoderConfig = config.copy(projectionDim = null)
        val mainLoaders = listOf(loaderFor(mainFile))

        val legacyWeights = loadBertWeights(mainLoaders, ctx, FP32::class, encoderConfig)
        val legacy = BertRuntime(ctx, legacyWeights, FP32::class)

        val encoderTensors = BertNetworkLoader.loadWeightTensors(mainLoaders, ctx, FP32::class)
        val dslEncoder = createBertEncoderRuntime(encoderConfig, encoderTensors, ctx)

        for (tokens in sequences) {
            // Full-sequence hidden states
            val legacyHidden = legacy.forward(tokens)
            val dslHidden = dslEncoder.forward(tokens)
            assertEquals(legacyHidden.shape.dimensions.toList(), dslHidden.shape.dimensions.toList())
            var maxDiff = 0.0
            for (i in 0 until legacyHidden.shape[0]) {
                for (j in 0 until legacyHidden.shape[1]) {
                    val d = abs((legacyHidden.data.get(i, j) - dslHidden.data.get(i, j)).toDouble())
                    if (d > maxDiff) maxDiff = d
                }
            }
            assertTrue(maxDiff < 1e-4, "hidden-state parity: max diff $maxDiff for ${tokens.size} tokens")

            // Pooled + normalized embedding (projection-less), with mask
            val mask = IntArray(tokens.size) { 1 }
            val legacyEmb = legacy.encode(tokens, mask)
            val dslEmb = dslEncoder.encode(tokens, mask)
            assertEquals(legacyEmb.volume, dslEmb.size)
            var embDiff = 0.0
            for (i in dslEmb.indices) {
                val d = abs((legacyEmb.data[i] - dslEmb[i]).toDouble())
                if (d > embDiff) embDiff = d
            }
            assertTrue(embDiff < 1e-4, "embedding parity: max diff $embDiff for ${tokens.size} tokens")
            println("parity ok (${tokens.size} tokens): hidden=$maxDiff embedding=$embDiff")
        }

        // Projected pipeline: verify encode() == plain-Kotlin recomputation of
        // mean-pool → W·pooled (bias-free) → L2 norm from the same hidden states.
        assumeTrue(denseFile.exists(), "2_Dense/model.safetensors not present — projection check skipped")
        val fullLoaders = listOf(loaderFor(mainFile), loaderFor(denseFile))
        val fullTensors = BertNetworkLoader.loadWeightTensors(fullLoaders, ctx, FP32::class)
        val dslFull = createBertEncoderRuntime(config, fullTensors, ctx)
        val projDim = config.projectionDim
        assumeTrue(projDim != null, "config has no projectionDim — projection check skipped")
        projDim!!
        val projW = fullTensors.first { it.name == BertNetworkLoader.PROJECTION_WEIGHT }.tensor

        for (tokens in sequences) {
            val hidden = dslFull.forward(tokens) // [L, h]
            val l = hidden.shape[0]
            val h = hidden.shape[1]
            val pooled = FloatArray(h)
            for (j in 0 until h) {
                var s = 0f
                for (i in 0 until l) s += hidden.data.get(i, j)
                pooled[j] = s / l
            }
            val projected = FloatArray(projDim)
            for (o in 0 until projDim) {
                var s = 0f
                for (j in 0 until h) s += projW.data.get(o, j) * pooled[j]
                projected[o] = s
            }
            var norm = 0.0
            for (v in projected) norm += (v * v).toDouble()
            val invNorm = (1.0 / kotlin.math.sqrt(norm + 1e-12)).toFloat()
            val expected = FloatArray(projDim) { projected[it] * invNorm }

            val actual = dslFull.encode(tokens)
            assertEquals(projDim, actual.size)
            var d = 0.0
            for (i in actual.indices) {
                val diff = abs((actual[i] - expected[i]).toDouble())
                if (diff > d) d = diff
            }
            assertTrue(d < 1e-4, "projected-pipeline recomputation: max diff $d for ${tokens.size} tokens")
            println("projection ok (${tokens.size} tokens): max diff $d, dim=$projDim")
        }
    }
}
