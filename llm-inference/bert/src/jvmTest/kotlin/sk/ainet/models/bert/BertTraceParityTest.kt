package sk.ainet.models.bert

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * DIRECT vs OPTIMIZED parity: the traced/fused ComputeGraph must reproduce
 * the eager module execution — on a synthetic tiny model (always) and on the
 * real LEAF checkpoint (integration, self-skipping).
 */
class BertTraceParityTest {

    private val ctx = DirectCpuExecutionContext()

    private fun maxDiff(a: FloatArray, b: FloatArray): Double {
        assertEquals(a.size, b.size)
        var d = 0.0
        for (i in a.indices) {
            val diff = abs((a[i] - b[i]).toDouble())
            if (diff > d) d = diff
        }
        return d
    }

    @Test
    fun syntheticModel_optimizedMatchesDirect() {
        val cfg = BertSyntheticFixtures.tinyConfig(layers = 2)
        val tensors = BertSyntheticFixtures.weightTensors(ctx, cfg)
        val direct = createBertEncoderRuntime(cfg, tensors, ctx, mode = BertExecutionMode.DIRECT)
        val optimized = createBertEncoderRuntime(cfg, tensors, ctx, mode = BertExecutionMode.OPTIMIZED)

        val sequences = listOf(
            intArrayOf(1, 2, 3),
            intArrayOf(4, 5, 6, 7, 8),
            intArrayOf(9, 0, 1), // same length as the first — exercises the compiled cache hit
        )
        for (tokens in sequences) {
            val d = maxDiff(direct.encode(tokens), optimized.encode(tokens))
            assertTrue(d < 1e-4, "DIRECT vs OPTIMIZED encode diff $d for ${tokens.size} tokens")
        }
    }

    @Test
    fun syntheticModel_compileReportsSingleInput() {
        val cfg = BertSyntheticFixtures.tinyConfig(layers = 1)
        val tensors = BertSyntheticFixtures.weightTensors(ctx, cfg)
        val optimized = createBertEncoderRuntime(cfg, tensors, ctx, mode = BertExecutionMode.OPTIMIZED)
        val diagnostics = optimized.compile(sampleSeqLen = 4)
        assertTrue(diagnostics.any { it.startsWith("Traced graph:") }, "missing trace diagnostics: $diagnostics")
        assertTrue(diagnostics.any { it.startsWith("Input node:") }, "missing input-node diagnostics: $diagnostics")
    }

    @Test
    @Tag("integration")
    fun realModel_optimizedMatchesDirect() = runTest {
        val modelDir = resolveModelDir()
        assumeTrue(modelDir != null, "No LEAF model available — skipping")
        modelDir!!
        val mainFile = modelDir.resolve("model.safetensors")
        assumeTrue(mainFile.exists(), "model.safetensors not found")

        val denseFile = modelDir.resolve("2_Dense").resolve("model.safetensors")
        val denseConfig = modelDir.resolve("2_Dense").resolve("config.json")
        val config = BertConfigParser.parse(
            modelDir.resolve("config.json").readText(),
            if (denseConfig.exists()) denseConfig.readText() else null,
        )

        val loaders = buildList {
            add(SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(mainFile.toString()) }))
            if (denseFile.exists()) {
                add(SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(denseFile.toString()) }))
            }
        }
        val tensors = BertNetworkLoader.loadWeightTensors(loaders, ctx, FP32::class)
        val direct = createBertEncoderRuntime(config, tensors, ctx, mode = BertExecutionMode.DIRECT)
        val optimized = createBertEncoderRuntime(config, tensors, ctx, mode = BertExecutionMode.OPTIMIZED)

        val sequences = listOf(
            intArrayOf(101, 7592, 2088, 102),
            intArrayOf(101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102),
        )
        for (tokens in sequences) {
            val d = maxDiff(direct.encode(tokens), optimized.encode(tokens))
            assertTrue(d < 1e-4, "real-model DIRECT vs OPTIMIZED encode diff $d for ${tokens.size} tokens")
            println("trace parity ok (${tokens.size} tokens): max diff $d")
        }
    }

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
}
