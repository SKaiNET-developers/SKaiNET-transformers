package sk.ainet.models.moonshine

import kotlin.test.Test
import kotlin.test.assertEquals

class MoonshineV2ConfigTest {

    @Test
    fun defaultsDescribeTheEnglishTinyCheckpoint() {
        val cfg = MoonshineV2Config()
        // (16,4) on the first and last two layers, causal in between; one DSL window for all layers.
        assertEquals(listOf(4, 4, 0, 0, 4, 4), (0 until cfg.encoderLayers).map(cfg::rightContextForLayer))
        assertEquals(List(cfg.encoderLayers) { 17 }, (0 until cfg.encoderLayers).map(cfg::slidingWindowForLayer))
        // Equal widths: the adapter needs no projection.
        assertEquals(cfg.dim, cfg.decoderDim)
        assertEquals(cfg.ffnDim, cfg.decoderFfnDim)
    }

    @Test
    fun explicitSlidingWindowsOverrideTheEdgeLayerRule() {
        // HF `encoder_config.sliding_windows` of the German tiny checkpoint: the middle layers look one frame
        // ahead, which the edge-layer rule cannot express.
        val de = MoonshineV2Config(
            vocabSize = 12288,
            slidingWindows = listOf(17 to 5, 17 to 5, 17 to 1, 17 to 1, 17 to 5, 17 to 5),
        )
        assertEquals(listOf(5, 5, 1, 1, 5, 5), (0 until de.encoderLayers).map(de::rightContextForLayer))
        // HF left context 17 ⇒ DSL window 18 (the band j ∈ [i−left, i+right] needs w = left + 1).
        assertEquals(List(de.encoderLayers) { 18 }, (0 until de.encoderLayers).map(de::slidingWindowForLayer))
    }

    @Test
    fun splitWidthsAreIndependentOfTheEncoderWidth() {
        val small = MoonshineV2Config(dim = 620, ffnDim = 2480, headDim = 64, decoderDim = 512, decoderFfnDim = 2048)
        assertEquals(620, small.dim)
        assertEquals(512, small.decoderDim)
        assertEquals(2048, small.decoderFfnDim)
    }
}
