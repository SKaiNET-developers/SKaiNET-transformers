package sk.ainet.transformers.iree.android

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/** `manifest.json` → [IreeKvSpec]: the fields the native session sizes buffers by, with the historical defaults. */
class IreeKvSpecTest {
    @Test fun stockDefaults() {
        val s = IreeKvSpec.functionGemma270m()
        assertEquals(IreeKvSpec.DEFAULT_VOCAB_SIZE, s.vocabSize)
        assertEquals(640, s.hiddenSize); assertEquals(4, s.nHeads); assertEquals(512, s.slidingWindow); assertEquals(32, s.chunk)
    }

    @Test fun manifestWithAddedTokensOverridesTheVocabulary() {
        val json = """{ "contractVersion": 1, "nLayers": 18, "headDim": 256, "nKvHeads": 1, "nHeads": 4, "hiddenSize": 640, "vocabSize": 262170, "slidingWindow": 512, "chunk": 32, "slidingRopeBase": 10000.0, "globalRopeBase": 1000000.0, "globalLayerPeriod": 6 }"""
        val s = IreeKvSpec.fromManifest(json)
        assertEquals(262170, s.vocabSize)
        assertEquals(32, s.chunk)
        assertEquals(64, IreeKvSpec.fromManifest(json, chunkOverride = 64).chunk)
    }

    @Test fun manifestWithoutTheGeometryKeepsTheStockValues() {
        val s = IreeKvSpec.fromManifest("""{ "contractVersion": 1, "nLayers": 18 }""")
        assertEquals(IreeKvSpec.DEFAULT_VOCAB_SIZE, s.vocabSize)
        assertEquals(IreeKvSpec.functionGemma270m(vocabSize = 262170).vocabSize, 262170)
    }
}
