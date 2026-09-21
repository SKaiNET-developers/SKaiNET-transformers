package sk.ainet.models.moonshine

import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MoonshineV2HfWeightMapTest {

    private val tiny = MoonshineV2Config(dim = 20, encoderLayers = 2, nHeads = 2, headDim = 10, ffnDim = 40, vocabSize = 32, decoderLayers = 2)
    private val split = tiny.copy(dim = 24, ffnDim = 48, decoderDim = 20, decoderFfnDim = 40)

    private fun names(m: Module<*, *>): List<String> = m.params.map { it.name } + m.modules.flatMap(::names)

    @Test
    fun everyParameterOfEveryModuleHasAMapping() {
        for (cfg in listOf(tiny, split)) {
            for (n in names(moonshineV2Frontend<FP32, Float>(FP32::class, cfg.dim))) assertNotNull(MoonshineV2HfWeightMap.frontend(n), n)
            for (n in names(moonshineV2Encoder<FP32, Float>(cfg, FP32::class))) assertNotNull(MoonshineV2HfWeightMap.encoder(n), n)
            for (n in names(MoonshineV2Adapter<FP32, Float>(cfg, maxFrames = 8, dtype = FP32::class))) assertNotNull(MoonshineV2HfWeightMap.encoder(n), n)
            for (n in names(moonshineV2Decoder<FP32, Float>(cfg, FP32::class))) assertNotNull(MoonshineV2HfWeightMap.decoder(n), n)
        }
    }

    @Test
    fun onlySplitWidthCheckpointsCarryTheAdapterProjection() {
        assertTrue(names(MoonshineV2Adapter<FP32, Float>(tiny, 8, FP32::class)).none { it.startsWith("v2_adapter.proj") })
        assertTrue(names(MoonshineV2Adapter<FP32, Float>(split, 8, FP32::class)).any { it == "v2_adapter.proj.weight" })
    }

    @Test
    fun theThreeThingsThatAreNotAPlainCopy() {
        assertEquals(MoonshineV2WeightRef.Transform.TRANSPOSE, MoonshineV2HfWeightMap.frontend("fe_filterbank.weight")!!.transform)
        // Encoder norms are zero-centred in the checkpoint, decoder norms are not.
        assertEquals(
            MoonshineV2WeightRef("model.encoder.layers.1.input_layernorm.gamma", MoonshineV2WeightRef.Transform.PLUS_ONE),
            MoonshineV2HfWeightMap.encoder("enc.1.attn_norm.weight"),
        )
        assertEquals(MoonshineV2WeightRef("model.decoder.layers.1.input_layernorm.weight"), MoonshineV2HfWeightMap.decoder("dec.1.self_attn_norm.weight"))
        // Bias-free in the model, present in the DSL module: zeros.
        assertNull(MoonshineV2HfWeightMap.encoder("enc.0.attn_norm.bias")!!.hfName)
        assertNull(MoonshineV2HfWeightMap.decoder("lm_head.bias")!!.hfName)
    }

    @Test
    fun crossAttentionAndTheOutputHead() {
        assertEquals("model.decoder.layers.0.encoder_attn.k_proj.weight", MoonshineV2HfWeightMap.decoder("dec.0.cross_attn.k_proj.weight")!!.hfName)
        assertEquals(MoonshineV2HfWeightMap.EMBED_TOKENS, MoonshineV2HfWeightMap.decoder("lm_head.weight")!!.hfName)
        assertEquals("proj_out.weight", MoonshineV2HfWeightMap.decoder("lm_head.weight", outputHead = "proj_out.weight")!!.hfName)
        assertNull(MoonshineV2HfWeightMap.decoder("dec.0.unknown"))
    }
}
