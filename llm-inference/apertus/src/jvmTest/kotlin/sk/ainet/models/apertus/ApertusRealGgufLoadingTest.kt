package sk.ainet.models.apertus

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
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
    fun `engine loader keeps projection weights packed and dequantizes the token embedding`() = runBlocking {
        val file = modelFile ?: run {
            println("[skip] Apertus GGUF not found.")
            return@runBlocking
        }
        // Packed weights are ~5 GB; the FP32 token embedding adds ~2 GB
        // (4096 × 131072 floats). Need ≥ 8 GB heap to fit.
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("[skip] heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return@runBlocking
        }

        val ctx = DirectCpuExecutionContext.create()
        val loader = ApertusWeightLoader.fromRandomAccess(
            randomAccessProvider = { JvmRandomAccessSource.open(file) }
        )

        val weights = loader.loadToMap<FP32, Float>(ctx)
        val md = weights.metadata

        // Apertus-8B reference dimensions (from HF config.json).
        assertTrue(md.blockCount in 24..40, "Unexpected blockCount=${md.blockCount}")
        assertEquals(4096, md.embeddingLength, "Unexpected embeddingLength=${md.embeddingLength}")
        assertTrue(md.headCount > 0, "headCount=${md.headCount}")
        assertTrue(md.kvHeadCount in 1..md.headCount, "kvHeadCount=${md.kvHeadCount}")
        assertTrue(md.vocabSize > 100_000, "vocabSize=${md.vocabSize}")

        // Token embedding must be a dense rank-2 [vocab, dim] tensor —
        // Embedding.gather() needs real element access.
        val tokenEmbd = weights.tensors[ApertusTensorNames.TOKEN_EMBEDDINGS]
        assertNotNull(tokenEmbd, "${ApertusTensorNames.TOKEN_EMBEDDINGS} must be loaded")
        assertEquals(2, tokenEmbd.shape.rank, "token embedding must be rank-2, got ${tokenEmbd.shape}")
        assertEquals(md.vocabSize, tokenEmbd.shape[0], "token embedding dim 0 must be vocab")
        assertEquals(md.embeddingLength, tokenEmbd.shape[1], "token embedding dim 1 must be embedding length")
        assertTrue(tokenEmbd.data !is PackedBlockStorage, "token embedding must be dequantized to dense data")

        // Quantized projection matrices keep their stored block encoding with
        // logical [out, in] shapes — no rank-1 byte tensors anywhere (the old
        // NATIVE_OPTIMIZED failure mode, transformers#100).
        repeat(md.blockCount) { layer ->
            val wq = weights.tensors[ApertusTensorNames.attnQ(layer)]
            assertNotNull(wq, "${ApertusTensorNames.attnQ(layer)} must be loaded")
            assertEquals(2, wq.shape.rank, "attn_q layer $layer must have a logical rank-2 shape, got ${wq.shape}")
            val ffnDown = weights.tensors[ApertusTensorNames.ffnDown(layer)]
            assertNotNull(ffnDown, "${ApertusTensorNames.ffnDown(layer)} must be loaded")
            assertEquals(2, ffnDown.shape.rank, "ffn_down layer $layer must have a logical rank-2 shape, got ${ffnDown.shape}")
        }
        val packedCount = weights.tensors.values.count { it.data is PackedBlockStorage }
        assertTrue(packedCount > 0, "expected packed block tensors from a Q4_K_S model, got none")

        // xIELU params must be populated for every layer.
        assertEquals(md.blockCount, weights.xieluParams.size,
            "xieluParams (${weights.xieluParams.size}) must match blockCount (${md.blockCount})")

        println("[real-load engine] tensors=${weights.tensors.size} packed=$packedCount xielu-layers=${weights.xieluParams.size}")
    }

    @Test
    fun `ApertusNetworkLoader fromGguf builds module from real Q4_K_S GGUF`() = runBlocking {
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
        val loader = ApertusNetworkLoader.fromGguf(
            randomAccessProvider = { JvmRandomAccessSource.open(file) }
        )

        val model = loader.load<FP32, Float>(ctx)
        assertNotNull(model, "ApertusNetworkLoader.load must return a Module")
        assertTrue(model.modules.isNotEmpty(), "loaded module must have submodules")

        val topNames = model.modules.map { it.name }.toSet()
        assertTrue("token_embd" in topNames, "module tree missing token_embd: $topNames")
        assertTrue("output_norm" in topNames, "module tree missing output_norm: $topNames")
        assertTrue("output" in topNames, "module tree missing output: $topNames")

        println("[real-load fromGguf engine] top-modules=${topNames.size}")
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

        // The engine loader delivers packed block tensors with logical [out, in]
        // shapes; the packed matmul kernels dispatch on the TensorData directly,
        // so no post-load conversion step is needed.
        val raw = ApertusWeightLoader.fromRandomAccess(
            randomAccessProvider = { JvmRandomAccessSource.open(file) }
        ).loadToMap<FP32, Float>(ctx)

        println("[real-forward weights] tensors=${raw.tensors.size} xielu=${raw.xieluParams.size}")
        raw.xieluParams[0]?.let { p ->
            println("[real-forward xielu0] alpha_p=${p.alphaP} alpha_n=${p.alphaN} beta=${p.beta} eps=${p.eps}")
        }
        raw.tensors[ApertusTensorNames.TOKEN_EMBEDDINGS]?.let { t ->
            println("[real-forward token_embd shape] ${t.shape}")
        }

        val model = ApertusNetworkLoader.fromWeights(ctx, raw)

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

        val raw = ApertusWeightLoader.fromRandomAccess(
            randomAccessProvider = { JvmRandomAccessSource.open(file) }
        ).loadToMap<FP32, Float>(ctx)
        val model = ApertusNetworkLoader.fromWeights(ctx, raw)

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
