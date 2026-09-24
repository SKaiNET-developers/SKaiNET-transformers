package sk.ainet.transformers.iree.android

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

/**
 * The GQA / qwen-kv-v1 spec facts this class's JNI counterpart (`iree_kv_jni.c`) relies on:
 * `nHeads % nKvHeads == 0` (any nKvHeads, not just 1), `globalLayerPeriod == 1` meaning every
 * layer is "global" (no sliding-side graph inputs), and [IreeKvSpec.fromManifest] parsing those
 * fields correctly out of a `QwenKvContract.manifestJson`-shaped string.
 */
class IreeKvSpecTest {
    @Test fun functionGemma270mIsPlainMultiHead() {
        val s = IreeKvSpec.functionGemma270m()
        assertEquals(1, s.nKvHeads)
        assertEquals(4, s.nHeads)
        assertEquals(6, s.globalLayerPeriod)
        assertEquals(512, s.slidingWindow)
    }

    @Test fun qwen25_05bInstructIsGqaNoSliding() {
        val s = IreeKvSpec.qwen25_05bInstruct()
        assertEquals(24, s.nLayers)
        assertEquals(64, s.headDim)
        assertEquals(2, s.nKvHeads)
        assertEquals(14, s.nHeads)
        assertEquals(0, s.nHeads % s.nKvHeads)
        assertEquals(1, s.globalLayerPeriod)
        assertEquals(0, s.slidingWindow)
        assertEquals(1_000_000f, s.globalRopeBase)
        assertEquals(s.globalRopeBase, s.slidingRopeBase)
        assertEquals(32, s.chunk)
    }

    @Test fun qwen3_06bIsGqaNoSliding() {
        val s = IreeKvSpec.qwen3_06b(chunk = 64)
        assertEquals(28, s.nLayers)
        assertEquals(128, s.headDim)
        assertEquals(8, s.nKvHeads)
        assertEquals(16, s.nHeads)
        assertEquals(0, s.nHeads % s.nKvHeads)
        assertEquals(1, s.globalLayerPeriod)
        assertEquals(0, s.slidingWindow)
        assertEquals(64, s.chunk)
    }

    @Test fun fromManifestParsesAQwenKvV1Manifest() {
        // Shape of QwenKvContract.manifestJson (SKaiNET-transformers#411, not yet written) --
        // this pins the field names IreeKvSpec.fromManifest must keep reading.
        val json = """
            {
              "contract": "qwen-kv-v1",
              "nLayers": 28, "headDim": 128, "nKvHeads": 8, "nHeads": 16,
              "hiddenSize": 1024, "vocabSize": 151936,
              "slidingWindow": 0, "globalLayerPeriod": 1, "chunk": 32,
              "slidingRopeBase": 1000000.0, "globalRopeBase": 1000000.0
            }
        """.trimIndent()
        val s = IreeKvSpec.fromManifest(json)
        assertEquals(28, s.nLayers)
        assertEquals(128, s.headDim)
        assertEquals(8, s.nKvHeads)
        assertEquals(16, s.nHeads)
        assertEquals(1024, s.hiddenSize)
        assertEquals(151936, s.vocabSize)
        assertEquals(0, s.slidingWindow)
        assertEquals(1, s.globalLayerPeriod)
        assertEquals(32, s.chunk)
        assertEquals(1_000_000f, s.globalRopeBase)
    }

    @Test fun fromManifestChunkOverrideWins() {
        val json = """{"nLayers":28,"chunk":32}"""
        assertEquals(64, IreeKvSpec.fromManifest(json, chunkOverride = 64).chunk)
    }

    @Test fun fromManifestMissingFieldsFallBackToFunctionGemmaDefaults() {
        val s = IreeKvSpec.fromManifest("{}")
        val d = IreeKvSpec.functionGemma270m()
        assertEquals(d.nLayers, s.nLayers)
        assertEquals(d.nKvHeads, s.nKvHeads)
        assertEquals(d.globalLayerPeriod, s.globalLayerPeriod)
    }

    @Test fun maskHeads_defaultsToPerHeadForFunctionGemmaAndOneForQwen() {
        assertEquals(0, IreeKvSpec.functionGemma270m().maskHeads)
        assertEquals(1, IreeKvSpec.qwen25_05bInstruct().maskHeads)
        assertEquals(1, IreeKvSpec.qwen3_06b().maskHeads)
        assertEquals(1, IreeKvSpec.fromManifest("""{"nHeads": 16, "nKvHeads": 8, "maskHeads": 1}""").maskHeads)
        assertEquals(0, IreeKvSpec.fromManifest("""{"nHeads": 4, "nKvHeads": 1}""").maskHeads)
        // the pre-existing 11-arg constructor still exists (binary compatibility via @JvmOverloads)
        val publicArities = IreeKvSpec::class.java.constructors
            .filter { c -> c.parameterTypes.none { it.name.endsWith("DefaultConstructorMarker") } }
            .map { it.parameterCount }.toSet()
        assertEquals(setOf(11, 12), publicArities)
    }
}
