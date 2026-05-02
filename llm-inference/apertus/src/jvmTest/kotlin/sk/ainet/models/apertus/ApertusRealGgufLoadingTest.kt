package sk.ainet.models.apertus

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import java.io.File
import java.lang.foreign.Arena
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

    @Test
    fun `forward pass on real Apertus produces finite logits of vocab size`() = runBlocking {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return@runBlocking
        }
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("[skip] heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return@runBlocking
        }

        val ctx = DirectCpuExecutionContext.create()
        val info = UnifiedModelLoader.peek { JvmRandomAccessSource.open(file) }

        // load → convert → fromWeights — the converter wraps each quantized
        // weight in the right block-major TensorData (Q4_K / Q6_K / Q4_0 / Q8_0)
        // so the SIMD matmul kernels can dispatch directly. Without it the
        // first attention projection blows up with "Transpose requires at least
        // 2 dimensions" because NATIVE_OPTIMIZED stores raw byte-shape rank-1
        // tensors that the standard transpose+matmul path can't consume.
        Arena.ofShared().use { arena ->
            val raw = ApertusWeightLoader.fromRandomAccess(
                randomAccessProvider = { JvmRandomAccessSource.open(file) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            ).loadToMap<FP32, Float>(ctx)

            println("[real-forward weights] tensors=${raw.tensors.size} quantTypes=${raw.quantTypes.size} xielu=${raw.xieluParams.size}")
            raw.xieluParams[0]?.let { p ->
                println("[real-forward xielu0] alpha_p=${p.alphaP} alpha_n=${p.alphaN} beta=${p.beta} eps=${p.eps}")
            }
            raw.tensors[ApertusTensorNames.TOKEN_EMBEDDINGS]?.let { t ->
                println("[real-forward token_embd shape] ${t.shape}")
            }

            val converted = convertApertusWeightsToMemSeg(raw, ctx, arena)
            val model = ApertusNetworkLoader.fromWeights(ctx, converted)

            val runtime = OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class
            )

            val bosTokenId = (info.fields["tokenizer.ggml.bos_token_id"] as? Number)?.toInt()
                ?: (info.fields["tokenizer.ggml.bos_token_id"] as? UInt)?.toInt()
                ?: 1
            val logits = runtime.forward(bosTokenId)

            assertEquals(info.vocabSize, logits.shape[logits.shape.rank - 1],
                "last logit dim must equal vocabSize=${info.vocabSize}, got shape ${logits.shape}")

            val buf = logits.data.copyToFloatArray()
            var nonZero = 0
            var nan = 0
            var inf = 0
            var max = Float.NEGATIVE_INFINITY
            var min = Float.POSITIVE_INFINITY
            for (v in buf) {
                if (v.isNaN()) nan++
                else if (!v.isFinite()) inf++
                else {
                    if (v != 0f) nonZero++
                    if (v > max) max = v
                    if (v < min) min = v
                }
            }
            println("[real-forward] vocab=${info.vocabSize} nan=$nan inf=$inf non-zero=$nonZero/${buf.size} min=$min max=$max")
            assertEquals(0, nan, "logits must not contain NaN")
            assertEquals(0, inf, "logits must not contain ±Inf")
            assertTrue(nonZero > buf.size / 2,
                "expected a broad logit distribution; got $nonZero/${buf.size} non-zero (min=$min max=$max)")
            assertTrue(max > min,
                "logits look constant (min=$min max=$max) — model didn't actually run")
        }
    }

    @Test
    fun `greedy generate on real Apertus produces in-vocab token sequence`() = runBlocking {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return@runBlocking
        }
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("[skip] heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return@runBlocking
        }

        val ctx = DirectCpuExecutionContext.create()
        val info = UnifiedModelLoader.peek { JvmRandomAccessSource.open(file) }

        Arena.ofShared().use { arena ->
            val raw = ApertusWeightLoader.fromRandomAccess(
                randomAccessProvider = { JvmRandomAccessSource.open(file) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            ).loadToMap<FP32, Float>(ctx)
            val converted = convertApertusWeightsToMemSeg(raw, ctx, arena)
            val model = ApertusNetworkLoader.fromWeights(ctx, converted)

            val bosTokenId = (info.fields["tokenizer.ggml.bos_token_id"] as? Number)?.toInt()
                ?: (info.fields["tokenizer.ggml.bos_token_id"] as? UInt)?.toInt()
                ?: 1
            val runtime = OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
                bos = bosTokenId
            )

            val steps = 4
            val generated = mutableListOf<Int>()
            runtime.generate(
                prompt = intArrayOf(bosTokenId),
                steps = steps,
                temperature = 0f
            ) { tokenId -> generated.add(tokenId) }

            assertEquals(steps, generated.size, "must emit exactly $steps tokens")
            for (tokenId in generated) {
                assertTrue(tokenId in 0 until info.vocabSize,
                    "generated token $tokenId out of [0, ${info.vocabSize})")
            }
            // Greedy on real weights should not collapse to the same token every step.
            assertTrue(generated.toSet().size > 1,
                "greedy collapsed to a single token across $steps steps: $generated")

            println("[real-generate greedy] tokens=$generated")
        }
    }

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
