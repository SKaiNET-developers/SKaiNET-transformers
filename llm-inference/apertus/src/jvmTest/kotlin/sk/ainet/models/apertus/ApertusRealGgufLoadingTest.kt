package sk.ainet.models.apertus

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.model.QuantPolicy
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration test against a real Apertus-8B-Instruct-2509 GGUF (Q4_K_S) downloaded
 * from `unsloth/Apertus-8B-Instruct-2509-GGUF` on Hugging Face.
 *
 * Skips silently when the GGUF is not present, so CI without network/cache stays green.
 *
 * Path resolution order:
 *  - `APERTUS_GGUF_PATH` env var
 *  - HF cache: `~/.cache/huggingface/hub/models--unsloth--Apertus-8B-Instruct-2509-GGUF/snapshots/.../Apertus-8B-Instruct-2509-Q4_K_S.gguf`
 */
class ApertusRealGgufLoadingTest {

    private val modelFile: File? = locateModel()

    @Test
    fun `peek detects apertus architecture and reads metadata fields`() {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found; set APERTUS_GGUF_PATH or download Q4_K_S into HF cache.")
            return
        }

        val info = UnifiedModelLoader.peek { JvmRandomAccessSource.open(file) }

        assertEquals("apertus", info.architecture, "GGUF should report apertus arch")
        assertEquals(ModelFamily.APERTUS, info.family, "ModelRegistry must classify as APERTUS")

        // Apertus-8B-Instruct-2509: 32 layers, 4096 hidden, 32k context, 131k vocab.
        assertTrue(info.blockCount > 0, "blockCount must be populated (got ${info.blockCount})")
        assertTrue(info.embeddingLength > 0, "embeddingLength must be populated (got ${info.embeddingLength})")
        assertTrue(info.contextLength > 0, "contextLength must be populated (got ${info.contextLength})")
        assertTrue(info.vocabSize > 0, "vocabSize must be populated (got ${info.vocabSize})")

        println("[real-load peek] arch=${info.architecture} layers=${info.blockCount} dim=${info.embeddingLength} ctx=${info.contextLength} vocab=${info.vocabSize}")
    }

    @Test
    fun `streaming reader exposes every tensor required by the apertus loader`() {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return
        }

        val source = JvmRandomAccessSource.open(file)
        StreamingGGUFReader.open(source).use { reader ->
            val present = reader.tensors.map { it.name }.toSet()
            val blockCount = (reader.fields["apertus.block_count"] as? Number)?.toInt()
                ?: (reader.fields["apertus.block_count"] as? UInt)?.toInt()
                ?: error("apertus.block_count missing")

            val required = buildList {
                add(ApertusTensorNames.TOKEN_EMBEDDINGS)
                add(ApertusTensorNames.OUTPUT_NORM)
                add(ApertusTensorNames.OUTPUT_WEIGHT)
                repeat(blockCount) { layer ->
                    add(ApertusTensorNames.attnNorm(layer))
                    add(ApertusTensorNames.attnQ(layer))
                    add(ApertusTensorNames.attnK(layer))
                    add(ApertusTensorNames.attnV(layer))
                    add(ApertusTensorNames.attnOut(layer))
                    add(ApertusTensorNames.attnQNorm(layer))
                    add(ApertusTensorNames.attnKNorm(layer))
                    add(ApertusTensorNames.ffnNorm(layer))
                    add(ApertusTensorNames.ffnDown(layer))
                    add(ApertusTensorNames.ffnUp(layer))
                }
            }

            val missing = required.filter { it !in present }
            assertTrue(missing.isEmpty(), "Tensors required by ApertusWeightLoader are absent from real GGUF:\n  ${missing.joinToString("\n  ")}")
        }
    }

    @Test
    fun `loadQuantized fully populates ApertusQuantizedWeights from real GGUF`() = runBlocking {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return@runBlocking
        }
        // Token-embedding dequant to FP32 alone is ~2 GB (4096 × 131072 floats); the
        // raw quant bytes for the rest add another ~5 GB. Need ≥ 8 GB heap to fit.
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("[skip] heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return@runBlocking
        }

        val ctx = DirectCpuExecutionContext.create()
        val loader = ApertusWeightLoader.fromRandomAccess(
            randomAccessProvider = { JvmRandomAccessSource.open(file) },
            quantPolicy = QuantPolicy.RAW_BYTES
        )

        val weights = loader.loadQuantized(ctx)
        val md = weights.metadata

        // Apertus-8B reference dimensions (from HF config.json).
        assertTrue(md.blockCount in 24..40, "Unexpected blockCount=${md.blockCount}")
        assertEquals(4096, md.embeddingLength, "Unexpected embeddingLength=${md.embeddingLength}")
        assertTrue(md.headCount > 0, "headCount=${md.headCount}")
        assertTrue(md.kvHeadCount in 1..md.headCount, "kvHeadCount=${md.kvHeadCount}")
        assertTrue(md.vocabSize > 100_000, "vocabSize=${md.vocabSize}")

        // FP32 small tensors (norms, token embedding) must be present.
        assertNotNull(weights.fp32Tensors[ApertusTensorNames.TOKEN_EMBEDDINGS],
            "${ApertusTensorNames.TOKEN_EMBEDDINGS} must be loaded as FP32")
        assertNotNull(weights.fp32Tensors[ApertusTensorNames.OUTPUT_NORM],
            "${ApertusTensorNames.OUTPUT_NORM} must be loaded as FP32")
        repeat(md.blockCount) { layer ->
            assertNotNull(weights.fp32Tensors[ApertusTensorNames.attnNorm(layer)],
                "${ApertusTensorNames.attnNorm(layer)} must be FP32")
            assertNotNull(weights.fp32Tensors[ApertusTensorNames.ffnNorm(layer)],
                "${ApertusTensorNames.ffnNorm(layer)} must be FP32")
            assertNotNull(weights.fp32Tensors[ApertusTensorNames.attnQNorm(layer)],
                "${ApertusTensorNames.attnQNorm(layer)} must be FP32")
            assertNotNull(weights.fp32Tensors[ApertusTensorNames.attnKNorm(layer)],
                "${ApertusTensorNames.attnKNorm(layer)} must be FP32")
        }

        // Large quantized projection matrices must be present.
        repeat(md.blockCount) { layer ->
            assertNotNull(weights.quantizedTensors[ApertusTensorNames.attnQ(layer)],
                "${ApertusTensorNames.attnQ(layer)} must be quantized")
            assertNotNull(weights.quantizedTensors[ApertusTensorNames.ffnDown(layer)],
                "${ApertusTensorNames.ffnDown(layer)} must be quantized")
        }

        // xIELU params must be populated for every layer.
        assertEquals(md.blockCount, weights.xieluParams.size,
            "xieluParams (${weights.xieluParams.size}) must match blockCount (${md.blockCount})")

        println("[real-load loadQuantized] fp32=${weights.fp32Tensors.size} quant=${weights.quantizedTensors.size} xielu-layers=${weights.xieluParams.size}")
    }

    @Test
    fun `ApertusNetworkLoader fromGguf builds module from real Q4_K_S GGUF`() = runBlocking {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return@runBlocking
        }
        // Network construction allocates raw quant bytes (~5 GB) plus the placeholder
        // metadata for ~27 GB of unrealized FP32 zero tensors. With upstream
        // SKaiNET#587 (lazy zero-init in NetworkBuilder), the placeholders never
        // materialize because WeightMapper substitutes them before any read.
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("[skip] heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return@runBlocking
        }

        val ctx = sk.ainet.context.DirectCpuExecutionContext.create()
        val loader = ApertusNetworkLoader.fromGguf(
            randomAccessProvider = { JvmRandomAccessSource.open(file) },
            quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
        )

        val model = loader.load<sk.ainet.lang.types.FP32, Float>(ctx)
        assertNotNull(model, "ApertusNetworkLoader.load must return a Module")
        assertTrue(model.modules.isNotEmpty(), "loaded module must have submodules")

        val topNames = model.modules.map { it.name }.toSet()
        assertTrue("token_embd" in topNames, "module tree missing token_embd: $topNames")
        assertTrue("output_norm" in topNames, "module tree missing output_norm: $topNames")
        assertTrue("output" in topNames, "module tree missing output: $topNames")

        println("[real-load fromGguf NATIVE_OPTIMIZED] top-modules=${topNames.size}")
    }

    /**
     * End-to-end inference (forward / generate / tool calling) is intentionally
     * NOT covered here.
     *
     * `ApertusNetworkLoader.fromGguf().load()` succeeds end-to-end (verified by
     * the test above), and the embedding lookup works after the
     * `loadStreamingTensor` token-embd dequant special case. But the rest of
     * the forward pass — Q/K/V/O projections, FFN matmuls — relies on the
     * standard `linearProject(ops, input, weight) = ops.matmul(input, ops.transpose(weight))`
     * helper, which assumes a logical rank-2 weight. Under
     * `QuantPolicy.NATIVE_OPTIMIZED` the loader stores quantized weights as
     * raw byte-level rank-1 `Int8` tensors so the native FFM kernels can
     * address the block layout directly — but `ops.transpose(byteShape)` then
     * fails.
     *
     * Gemma's Q4_K end-to-end test works because Gemma's loader uses
     * `Q4_KBlockTensorData(logicalShape, blockMajorBytes)` with a lazy
     * `transpose` override and a quant-aware `matmul` dispatch (see
     * `GemmaDslQ4KTest`, `relayoutQ4_KRowMajorToBlockMajor`). Apertus's
     * loader stores raw Int8 bytes instead, so `linearProject` blows up at
     * the first attention projection.
     *
     * Tracking issue: see the upstream / transformers follow-up — the
     * Apertus loader needs per-quant-type tensor-data wrappers
     * (`Q4_KBlockTensorData` / `Q5_KBlockTensorData` / `Q6_KBlockTensorData`)
     * with row-major → block-major relayout, mirroring Gemma's path.
     */

    private fun locateModel(): File? {
        System.getenv("APERTUS_GGUF_PATH")?.let { p ->
            val f = File(p)
            if (f.isFile) return f
        }
        val home = System.getProperty("user.home")
        val snapshotsDir = File("$home/.cache/huggingface/hub/models--unsloth--Apertus-8B-Instruct-2509-GGUF/snapshots")
        if (!snapshotsDir.isDirectory) return null
        return snapshotsDir.listFiles()?.asSequence()
            ?.flatMap { it.listFiles()?.asSequence() ?: emptySequence() }
            ?.firstOrNull { it.name == "Apertus-8B-Instruct-2509-Q4_K_S.gguf" }
    }
}
