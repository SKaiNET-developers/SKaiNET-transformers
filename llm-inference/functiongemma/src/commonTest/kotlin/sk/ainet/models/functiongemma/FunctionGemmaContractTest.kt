package sk.ainet.models.functiongemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pure structural test of [FunctionGemmaContract] emission — no checkpoint, no I/O. Pins the
 * contract shapes/orders so a board runtime consuming `manifest.json` mechanically (D3 — the
 * whisper contract-v2 pattern) can trust the arg/result orders without re-deriving them from the
 * MLIR text. The real-checkpoint dump tests ([FunctionGemmaExportDumpTest]) cross-check these
 * orders against the emitted StableHLO signatures.
 */
class FunctionGemmaContractTest {

    private val spec = FunctionGemmaSpec(gguf = "unused-for-this-test")

    @Test
    fun prefillArgsOutputs_tokensThenPerLayerKvThenTokensLast() {
        assertEquals(listOf("tokens"), FunctionGemmaContract.prefillArgs())
        val outs = FunctionGemmaContract.prefillOutputs(spec)
        assertEquals(spec.nLayers * 2 + 1, outs.size)
        assertEquals("tokens", outs.last(), "argMax token ids must be the LAST prefill result")
        assertEquals("l0.k", outs[0]); assertEquals("l0.v", outs[1])
        assertEquals("l${spec.nLayers - 1}.v", outs[outs.size - 2])
    }

    @Test
    fun withPastArgs_tokenFirst_thenRopeBasesIntroducedOnFirstUse_thenPerLayerKV() {
        val args = FunctionGemmaContract.withPastArgs(spec)
        assertEquals("token", args.first())
        // Layer 0 is NOT global (period=6 -> global at index 5), so sliding cos/sin appear
        // right after "token", before any layer K/V.
        assertEquals("cosSliding", args[1]); assertEquals("sinSliding", args[2])
        assertEquals("l0.k", args[3]); assertEquals("l0.v", args[4])
        // Global cos/sin introduced at the first global layer (index period-1 = 5).
        val globalIdx = args.indexOf("cosGlobal")
        assertTrue(globalIdx > 0, "cosGlobal must appear once introduced")
        assertEquals("sinGlobal", args[globalIdx + 1])
        // Each RoPE base name appears exactly once (introduced-on-first-use, not per layer).
        assertEquals(1, args.count { it == "cosSliding" })
        assertEquals(1, args.count { it == "cosGlobal" })
        // Every layer contributes exactly one K + one V arg.
        assertEquals(spec.nLayers, args.count { it.endsWith(".k") })
        assertEquals(spec.nLayers, args.count { it.endsWith(".v") })
    }

    @Test
    fun withPastOutputs_perLayerKvThenTokenLast() {
        val outs = FunctionGemmaContract.withPastOutputs(spec)
        assertEquals(spec.nLayers * 2 + 1, outs.size)
        assertEquals("token", outs.last(), "the next token id must be the LAST with_past result")
    }

    @Test
    fun manifestJson_roundTripsTheCoreContractFields() {
        val json = FunctionGemmaContract.manifestJson(spec)
        assertTrue(json.contains("\"contractVersion\": ${FunctionGemmaContract.CONTRACT_VERSION}"))
        assertTrue(json.contains("\"redecode\": \"${FunctionGemmaContract.FN_REDECODE}\""))
        assertTrue(json.contains("\"prefill\": \"${FunctionGemmaContract.FN_PREFILL}\""))
        assertTrue(json.contains("\"withPast\": \"${FunctionGemmaContract.FN_WITH_PAST}\""))
        assertTrue(json.contains("\"nLayers\": ${spec.nLayers}"))
        assertTrue(json.contains("\"headDim\": ${spec.headDim}"))
        assertTrue(json.contains("\"eot\": ${spec.eot}"))
        assertTrue(json.contains("\"kFirstInOutput\": true"))
        // Tool map round-trips (default v10 vocabulary), including the explicit no-op null.
        assertTrue(json.contains("\"0\": \"set_lights\""))
        assertTrue(json.contains("\"none\": null"))
    }

    @Test
    fun customToolMap_overridesDefault() {
        val custom = spec.copy(toolMap = mapOf("6" to "open_gripper"))
        val json = FunctionGemmaContract.manifestJson(custom)
        assertTrue(json.contains("\"6\": \"open_gripper\""))
        assertTrue(!json.contains("set_lights"))
    }

    @Test
    fun manifestJson_carriesTheEmbeddingGeometryTheBoardRuntimeSizesBy() {
        val stock = FunctionGemmaContract.manifestJson(spec)
        assertTrue(stock.contains("\"vocabSize\": ${FunctionGemmaSpec.DEFAULT_VOCAB_SIZE}"), stock)
        assertTrue(stock.contains("\"hiddenSize\": 640") && stock.contains("\"nHeads\": 4") && stock.contains("\"slidingWindow\": 512"), stock)
        // a fine-tune that added 26 special tokens
        val grown = FunctionGemmaContract.manifestJson(spec.copy(vocabSize = FunctionGemmaSpec.DEFAULT_VOCAB_SIZE + 26))
        assertTrue(grown.contains("\"vocabSize\": 262170"), grown)
        val parsed = Regex("\"vocabSize\"\\s*:\\s*(\\d+)").find(grown)?.groupValues?.get(1)?.toInt()
        assertEquals(262170, parsed, "the runtime's regex parser (IreeKvSpec.fromManifest) must find the value")
    }

    @Test
    fun vocabSize_belowTheGemmaVocabularyIsRejected() {
        val r = runCatching { spec.copy(vocabSize = 1000) }
        assertTrue(r.isFailure, "a vocabSize smaller than the tokenizer cannot describe a Gemma checkpoint")
    }
}
