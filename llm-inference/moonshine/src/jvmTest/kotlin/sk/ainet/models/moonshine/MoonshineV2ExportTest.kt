package sk.ainet.models.moonshine

import kotlinx.io.asSink
import kotlinx.io.buffered
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The whole export, end to end, on a synthetic checkpoint small enough for CI: a Hugging Face-shaped snapshot
 * (`config.json`, `model.safetensors`, `tokenizer.json`) is generated from the DSL model's own parameter list,
 * exported, and the outputs are checked. Byte-parity with real checkpoints is verified outside CI.
 */
class MoonshineV2ExportTest {

    private val options = MoonshineV2ExportOptions(frontendSamples = 1600, encoderFrames = 8, maxMemoryFrames = 16)

    @Test
    fun configIsReadFromTheSnapshot() {
        val cfg = MoonshineV2Checkpoint.configFromHf(configJson(splitWidths = true))
        assertEquals(24, cfg.dim); assertEquals(48, cfg.ffnDim)
        assertEquals(20, cfg.decoderDim); assertEquals(40, cfg.decoderFfnDim)
        assertEquals(listOf(3 to 1, 3 to 0), cfg.slidingWindows)
        assertEquals(listOf(4, 4), (0 until 2).map(cfg::slidingWindowForLayer))   // HF left context + 1
        assertEquals(0.8f, cfg.partialRotaryFactor)
        assertFailsWith<IllegalStateException> { MoonshineV2Checkpoint.configFromHf("""{"hidden_size":20}""") }
    }

    @Test
    fun exportsTheFullContractAndUsesEveryTensor() {
        for (splitWidths in listOf(false, true)) {
            val snapshot = syntheticSnapshot(splitWidths)
            val out = File(snapshot, "out")
            MoonshineV2Checkpoint(snapshot).use { checkpoint ->
                val harness = MoonshineV2ExportHarness(checkpoint, options)
                val files = harness.export(out) + harness.writeEmbeddings(out) + harness.writeVocab(out)
                assertEquals(emptyList(), checkpoint.unconsumed(), "every checkpoint tensor must be used")
                harness.writeManifest(out, files)

                for (graph in MoonshineV2ExportHarness.Graph.entries) {
                    val mlir = File(out, graph.file).readText()
                    assertTrue("@${graph.function}" in mlir, "${graph.file} must define @${graph.function}")
                    assertTrue("stablehlo.constant" in mlir, "${graph.file}: weights fold to constants")
                }
                assertEquals(1, entryArgs(File(out, "frontend.mlir")), "frontend: audio")
                assertEquals(1, entryArgs(File(out, "encoder.mlir")), "encoder: features")
                assertEquals(2, entryArgs(File(out, "adapter.mlir")), "adapter: memory + positions")
                assertEquals(3, entryArgs(File(out, "prefill.mlir")), "prefill: embedding + memory + mask")
                // token embedding, cos, sin, 4 caches per layer, mask
                assertEquals(3 + 4 * checkpoint.config.decoderLayers + 1, entryArgs(File(out, "with_past.mlir")))
                assertTrue("?" in File(out, "with_past.mlir").readText().substringBefore(") ->"), "the self-attention cache is dynamic")

                val table = checkpoint.floats(MoonshineV2HfWeightMap.EMBED_TOKENS)
                val written = ByteBuffer.wrap(File(out, "dec_embed.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                assertContentEquals(table, FloatArray(written.remaining()).also(written::get))

                assertEquals(PIECES, readVocab(File(out, "vocab.bin")))
                assertTrue("\"function\": \"moonshine_v2_decoder_with_past\"" in File(out, "manifest.json").readText())
            }
        }
    }

    @Test
    fun aStaticCacheLengthCanBeRequested() {
        val snapshot = syntheticSnapshot(splitWidths = false)
        MoonshineV2Checkpoint(snapshot).use { checkpoint ->
            val out = File(snapshot, "out")
            MoonshineV2ExportHarness(checkpoint, options.copy(staticPast = 3)).export(out, setOf(MoonshineV2ExportHarness.Graph.WITH_PAST))
            assertTrue("?" !in File(out, "with_past.mlir").readText().substringBefore(") ->"))
        }
    }

    @Test
    fun anUnmappedCheckpointTensorIsReported() {
        val snapshot = syntheticSnapshot(splitWidths = false, extra = "model.decoder.something_new.weight")
        MoonshineV2Checkpoint(snapshot).use { checkpoint ->
            val harness = MoonshineV2ExportHarness(checkpoint, options)
            harness.export(File(snapshot, "out")); harness.writeEmbeddings(File(snapshot, "out"))
            assertEquals(listOf("model.decoder.something_new.weight"), checkpoint.unconsumed())
        }
    }

    // ---- fixture ----

    private fun configJson(splitWidths: Boolean): String {
        val encDim = if (splitWidths) 24 else 20
        return """
            {"model_type":"moonshine_streaming","hidden_size":20,"intermediate_size":40,"num_hidden_layers":2,
             "num_attention_heads":2,"head_dim":10,"vocab_size":${PIECES.size + 2},"max_position_embeddings":32,
             "rope_parameters":{"partial_rotary_factor":0.8,"rope_theta":10000.0,"rope_type":"default"},
             "encoder_config":{"hidden_size":$encDim,"intermediate_size":${encDim * 2},"num_hidden_layers":2,
               "num_attention_heads":2,"head_dim":10,"sliding_windows":[[3,1],[3,0]]}}
        """.trimIndent()
    }

    /** Writes a snapshot whose tensors are exactly what the weight map asks for, in checkpoint layout. */
    private fun syntheticSnapshot(splitWidths: Boolean, extra: String? = null): File {
        val dir = Files.createTempDirectory("moonshine-v2-snapshot").toFile()
        File(dir, "config.json").writeText(configJson(splitWidths))
        File(dir, "tokenizer.json").writeText(
            """{"model":{"type":"Unigram","vocab":[${PIECES.joinToString(",") { "[\"$it\",0.0]" }}]}}""",
        )
        val cfg = MoonshineV2Checkpoint.configFromHf(configJson(splitWidths))
        val tensors = LinkedHashMap<String, IntArray>()
        fun collect(m: Module<*, *>, map: (String) -> MoonshineV2WeightRef?) {
            for (p in m.params) {
                val ref = map(p.name) ?: error("unmapped ${p.name}")
                val shape = p.value.shape.dimensions
                ref.hfName?.let { tensors[it] = if (ref.transform == MoonshineV2WeightRef.Transform.TRANSPOSE) shape.reversedArray() else shape }
            }
            m.modules.forEach { collect(it, map) }
        }
        collect(moonshineV2Frontend<FP32, Float>(FP32::class, cfg.dim), MoonshineV2HfWeightMap::frontend)
        collect(moonshineV2Encoder<FP32, Float>(cfg, FP32::class), MoonshineV2HfWeightMap::encoder)
        collect(MoonshineV2Adapter<FP32, Float>(cfg, maxFrames = 32, dtype = FP32::class), MoonshineV2HfWeightMap::encoder)
        collect(moonshineV2Decoder<FP32, Float>(cfg, FP32::class)) { MoonshineV2HfWeightMap.decoder(it) }
        extra?.let { tensors[it] = intArrayOf(2) }

        var seed = 1
        File(dir, "model.safetensors").outputStream().asSink().buffered().use { sink ->
            SafeTensorsWriter.write(sink) {
                for ((name, shape) in tensors) {
                    val n = shape.fold(1) { a, b -> a * b }
                    tensorF32(name, shape, FloatArray(n) { ((it * 31 + seed * 17) % 97 - 48) / 480f })
                    seed++
                }
            }
        }
        return dir
    }

    private fun entryArgs(mlir: File): Int =
        Regex("""%arg\d+""").findAll(mlir.readText().substringBefore(") ->")).map { it.value }.toSet().size

    private fun readVocab(file: File): List<String> {
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return List(buf.getInt()) { ByteArray(buf.getShort().toInt() and 0xFFFF).also(buf::get).decodeToString() }
    }

    private companion object {
        val PIECES = listOf("<unk>", "<s>", "</s>", "▁hallo", "▁wärme", "▁世界")
    }
}
