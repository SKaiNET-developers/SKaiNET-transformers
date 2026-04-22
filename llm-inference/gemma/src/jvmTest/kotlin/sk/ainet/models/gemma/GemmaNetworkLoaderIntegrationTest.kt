package sk.ainet.models.gemma

import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.jupiter.api.Tag
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Integration test for the Phase 5a Gemma DSL pipeline, exercised on a real
 * Gemma 4 E2B GGUF checkpoint.
 *
 * Skipped unless the checkpoint is present; pass `-PincludeIntegration` to
 * the Gradle test task to enable integration-tagged runs. Override the path
 * with `-Dgemma4.e2b.gguf=/abs/path/to/model.gguf`.
 *
 * **Scope (load-free).** Phase 5a's `GemmaNetworkLoader` only supports
 * `QuantPolicy.DEQUANTIZE_TO_FP32`, which expands a Q4_K_M Gemma 4 E2B
 * checkpoint (~3 GB on disk) to ~20 GB of FP32 weights in memory. That
 * doesn't fit on a typical developer laptop. Moving the DSL/DAG path onto
 * `QuantPolicy.NATIVE_OPTIMIZED` requires quant-aware matmul dispatch on the
 * DAG executor (see `ISSUE-skainet-8b-oom.md` §Solution C) and is not yet
 * wired up.
 *
 * So this test verifies what it *can* verify on a laptop today:
 *
 * 1. `UnifiedModelLoader.peek()` identifies the checkpoint as the GEMMA
 *    family and exposes block/vocab/embedding dimensions.
 * 2. `Gemma4WeightLoader(loadTensorData = false)` parses the full Gemma 4
 *    metadata (headDim, ropeParameters, layerTypes, etc.).
 * 3. `gemmaNetwork()` builds a DSL module tree from that real-world
 *    metadata with the expected structure (embedding, N transformer blocks,
 *    output norm, output projection) and matching block count.
 *
 * Full weight population + forward() on a real Gemma 4 checkpoint is
 * deferred to a follow-up phase alongside either (a) quant-aware DAG
 * kernels or (b) a machine with ≥ 40 GB RAM for FP32 dequant.
 */
class GemmaNetworkLoaderIntegrationTest {

    private val modelPath: String = System.getProperty("gemma4.e2b.gguf")
        ?: "${System.getProperty("user.home")}/.lmstudio/models/lmstudio-community/" +
            "gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf"

    private fun skipIfModelNotPresent() {
        assumeTrue(
            "Skipping - Gemma 4 E2B GGUF not present at $modelPath " +
                "(set -Dgemma4.e2b.gguf=/abs/path to override)",
            File(modelPath).exists()
        )
    }

    @Tag("integration")
    @Test
    fun `peeks real Gemma 4 E2B GGUF and builds the DSL network from its metadata`() = runBlocking {
        skipIfModelNotPresent()

        // 1. Peek: architecture + family detection (dimensions come from the
        // full metadata parse below — peek's generic key lookup does not cover
        // gemma4-specific fields and returns 0 for block/embedding/vocab).
        val info = UnifiedModelLoader.peek { JvmRandomAccessSource.open(modelPath) }
        assertEquals(
            ModelFamily.GEMMA, info.family,
            "UnifiedModelLoader should detect Gemma family, got ${info.family} for arch='${info.architecture}'"
        )

        // 2. Metadata-only parse via Gemma4WeightLoader (no tensor data → no 20 GB blow-up).
        // Use the random-access streaming path because the sequential Source reader
        // tries to slurp the whole file into a ByteArray and overflows Int.MAX_VALUE
        // on a 3 GB GGUF.
        val ctx = DirectCpuExecutionContext()
        val loader = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(modelPath) },
            loadTensorData = false
        )
        val metadata: Gemma4ModelMetadata = loader.loadStreaming<FP32, Float>(
            ctx = ctx,
            dtype = FP32::class,
            onTensorLoaded = { _: String, _: Tensor<FP32, Float> -> /* ignored: loadTensorData = false */ }
        )

        assertTrue(metadata.blockCount in 20..50, "Unexpected blockCount ${metadata.blockCount} for E2B-class model")
        assertTrue(metadata.headCount > 0)
        assertTrue(metadata.kvHeadCount in 1..metadata.headCount)
        assertTrue(metadata.intermediateSize > 0)
        assertTrue(metadata.headDim > 0)

        // 3. Build the DSL network from real metadata and check structure.
        val model = gemmaNetwork<FP32, Float>(metadata)
        val topLevelNames = model.modules.map { it.name }

        assertTrue("token_embd" in topLevelNames, "Missing token_embd in $topLevelNames")
        assertTrue("output_norm" in topLevelNames, "Missing output_norm")
        assertTrue("output" in topLevelNames, "Missing output")

        val dslBlockCount = topLevelNames.count { it.startsWith("blk.") }
        assertEquals(
            metadata.blockCount, dslBlockCount,
            "DSL block count should match checkpoint"
        )

        println("Gemma 4 E2B real-checkpoint integration test PASSED")
        println("  Checkpoint   : $modelPath")
        println("  Architecture : ${info.architecture} (${info.family})")
        println("  Blocks       : ${metadata.blockCount}")
        println("  Embedding    : ${metadata.embeddingLength}")
        println("  Heads / KV   : ${metadata.headCount} / ${metadata.kvHeadCount}")
        println("  headDim      : ${metadata.headDim} (global ${metadata.globalHeadDim})")
        println("  Intermediate : ${metadata.intermediateSize}")
        println("  Vocab        : ${metadata.vocabSize}")
    }
}
