package sk.ainet.models.qwen

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pure-Kotlin gate for [QwenKvContract] (no checkpoint needed): arg/output counts and order must
 * match exactly what `iree_kv_jni.c`'s `ins[]`/`outs[]` assembly builds (see that file's
 * `nativePrefill`/`nativeChunk`/`nativeStep`, and [QwenKvContract]'s own class doc for the
 * side-by-side derivation), and [QwenKvContract.manifestJson] must carry every key
 * `IreeKvSpec.fromManifest` (llm-runtime/iree-android) reads.
 */
class QwenKvContractTest {
    private val qwen25 = QwenKvArch.qwen25_05bInstruct()
    private val qwen3 = QwenKvArch.qwen3_06b()

    @Test fun prefillAtArgsAndOutputsCounts() {
        assertEquals(listOf("tokens", "emb", "select"), QwenKvContract.prefillAtArgs())
        for (arch in listOf(qwen25, qwen3)) {
            val outs = QwenKvContract.prefillAtOutputs(arch)
            assertEquals(2 * arch.nLayers + 1, outs.size)
            assertEquals("token", outs.last())
            assertEquals("l0.k", outs.first())
        }
    }

    @Test fun withPastArgsAndOutputsMatchNativeStepFormula() {
        for (arch in listOf(qwen25, qwen3)) {
            val args = QwenKvContract.withPastArgs(arch)
            // native nIn = 2 + 2 (no sliding side) + 2*nLayers
            assertEquals(4 + 2 * arch.nLayers, args.size)
            assertEquals(listOf("token", "emb", "cos", "sin"), args.take(4))
            assertEquals("l0.k", args[4]); assertEquals("l0.v", args[5])
            assertEquals("l${arch.nLayers - 1}.v", args.last())

            val outs = QwenKvContract.withPastOutputs(arch)
            assertEquals(2 * arch.nLayers + 1, outs.size)
            assertEquals("token", outs.last())
        }
    }

    @Test fun prefillWithPastArgsPlacesMaskRightAfterFirstLayer() {
        for (arch in listOf(qwen25, qwen3)) {
            val args = QwenKvContract.prefillWithPastArgs(arch)
            // native nIn = 2 + 2 (no sliding side) + 2*nLayers + 1 (mask) + 1 (select)
            assertEquals(6 + 2 * arch.nLayers, args.size)
            assertEquals(listOf("tokens", "emb", "cos", "sin", "l0.k", "l0.v", "mask"), args.take(7))
            assertEquals("select", args.last())
            // mask appears exactly once, only after the first layer's k/v
            assertEquals(1, args.count { it == "mask" })

            val outs = QwenKvContract.prefillWithPastOutputs(arch)
            assertEquals(2 * arch.nLayers + 1, outs.size)
            assertEquals("token", outs.last())
        }
    }

    @Test fun kFirstInOutputOrderHoldsForEveryLayer() {
        for (arch in listOf(qwen25, qwen3)) {
            val outs = QwenKvContract.withPastOutputs(arch)
            for (l in 0 until arch.nLayers) {
                assertEquals("l$l.k", outs[2 * l])
                assertEquals("l$l.v", outs[2 * l + 1])
            }
        }
    }

    @Test fun manifestJsonCarriesEveryFieldIreeKvSpecReads() {
        val json = QwenKvContract.manifestJson(qwen3, chunk = 32)
        val requiredKeys = listOf(
            "nLayers", "headDim", "nKvHeads", "nHeads", "hiddenSize", "vocabSize",
            "slidingWindow", "globalLayerPeriod", "chunk", "slidingRopeBase", "globalRopeBase",
        )
        for (key in requiredKeys) {
            assertTrue("\"$key\"" in json, "manifest missing key \"$key\": $json")
        }
        assertTrue("\"slidingWindow\": 0" in json, "qwen-kv-v1 must always report slidingWindow=0")
        assertTrue("\"globalLayerPeriod\": 1" in json, "qwen-kv-v1 must always report globalLayerPeriod=1")
        assertTrue("\"contract\": \"qwen-kv-v1\"" in json)
        assertTrue("\"nKvHeads\": ${qwen3.nKvHeads}" in json)
        assertTrue("\"qkNorm\": true" in json)
        assertTrue("\"attnBias\": false" in json)
    }

    @Test fun manifestReflectsQwen25sBiasAndNoQkNorm() {
        val json = QwenKvContract.manifestJson(qwen25)
        assertTrue("\"qkNorm\": false" in json)
        assertTrue("\"attnBias\": true" in json)
        assertTrue("\"nKvHeads\": 2" in json)
    }

    @Test fun qualifiedPrefixesWithModuleDot() {
        assertEquals("module.qwen_with_past", QwenKvContract.qualified(QwenKvContract.FN_WITH_PAST))
        assertEquals("module.qwen_with_past", QwenKvContract.qualified("module.qwen_with_past"))
    }

    @Test fun archRejectsInconsistentHeadCounts() {
        var threw = false
        try {
            QwenKvArch(
                architecture = "bad", nLayers = 1, headDim = 1, nKvHeads = 3, nHeads = 5,
                hiddenSize = 8, vocabSize = 16, ropeBase = 1f, rmsNormEps = 1e-6f,
                qkNorm = false, attnBias = false, eos = 0, eot = 0,
            )
        } catch (_: IllegalArgumentException) {
            threw = true
        }
        assertTrue(threw, "nHeads not a multiple of nKvHeads must be rejected")
    }
}
