package sk.ainet.models.whisper

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** `config.json` → [WhisperConfig] and the manifest prompt, without a checkpoint. */
class WhisperHfConfigTest {
    // The fields the export reads, as a Hugging Face whisper-tiny `config.json` carries them.
    private fun config(vocab: Int = 51865, dModel: Int = 384, heads: Int = 6, layers: Int = 4, ffn: Int = 1536) = """
        {"model_type": "whisper", "num_mel_bins": 80, "d_model": $dModel,
         "encoder_attention_heads": $heads, "decoder_attention_heads": $heads,
         "encoder_layers": $layers, "decoder_layers": $layers,
         "encoder_ffn_dim": $ffn, "decoder_ffn_dim": $ffn,
         "vocab_size": $vocab, "max_target_positions": 448, "max_source_positions": 1500,
         "activation_function": "gelu"}
    """.trimIndent()

    @Test
    fun tinyMultilingualMatchesTheDefaultConfig() {
        assertEquals(WhisperConfig(), WhisperExportHarness.configFromHf(config(), audioCtx = 200))
    }

    @Test
    fun geometryComesFromTheCheckpoint() {
        val base = WhisperExportHarness.configFromHf(config(dModel = 512, heads = 8, layers = 6, ffn = 2048), audioCtx = 1500)
        assertEquals(512, base.dim); assertEquals(8, base.nHeads); assertEquals(64, base.headDim)
        assertEquals(6, base.encoderLayers); assertEquals(6, base.decoderLayers); assertEquals(2048, base.ffnDim)
        assertEquals(3000, base.melFrames)
    }

    @Test
    fun rejectsOtherModelsAndTooLongWindows() {
        assertFailsWith<IllegalArgumentException> { WhisperExportHarness.configFromHf(config().replace("\"whisper\"", "\"moonshine\"")) }
        assertFailsWith<IllegalArgumentException> { WhisperExportHarness.configFromHf(config(), audioCtx = 1501) }
    }

    @Test
    fun promptFollowsTheVocabulary() {
        val multi = WhisperExportHarness(WhisperConfig(), WhisperExportOptions(language = "de"))
        assertContentEquals(intArrayOf(50258, 50261, 50359, 50363), multi.promptIds())
        val english = WhisperExportHarness(WhisperConfig(vocabSize = 51864), WhisperExportOptions(promptLength = 2))
        assertContentEquals(intArrayOf(50257, 50362), english.promptIds())
        assertFailsWith<IllegalArgumentException> { WhisperExportHarness(WhisperConfig(vocabSize = 51864)).manifestJson() }
    }
}
