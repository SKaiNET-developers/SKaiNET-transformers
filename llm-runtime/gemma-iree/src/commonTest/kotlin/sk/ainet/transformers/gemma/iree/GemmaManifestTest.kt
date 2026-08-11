package sk.ainet.transformers.gemma.iree

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * [GemmaManifest] parses `manifest.json` as PLAIN JSON text (D3 — this module does not depend on
 * `:llm-inference:functiongemma`, so the sample text below is a hand-written literal shaped like
 * `sk.ainet.models.functiongemma.FunctionGemmaContract.manifestJson`'s output, not a shared type).
 * A structural mismatch between the two would only surface at runtime (a real manifest.json fails
 * to parse); [FunctionGemmaVmfbParityTest] in the functiongemma module's docker-driven vmfb parity
 * run is the end-to-end check that the two sides actually agree.
 */
class GemmaManifestTest {

    private val sample = """
        {
          "contractVersion": 1,
          "model": "functiongemma-270m",
          "functions": { "redecode": "gemma", "prefill": "gemma_prefill", "withPast": "gemma_with_past" },
          "nLayers": 18,
          "headDim": 256,
          "nKvHeads": 1,
          "seq": 24,
          "maxPositions": 1024,
          "eot": 106,
          "slidingRopeBase": 10000.0,
          "globalRopeBase": 1000000.0,
          "globalLayerPeriod": 6,
          "kFirstInOutput": true,
          "parameterScope": "model",
          "parameters": { "redecode": "gemma-gen.irpa", "prefill": "gemma-prefill.irpa", "withPast": "gemma-with-past.irpa" },
          "redecodeArgs": ["tokens"],
          "redecodeOutputs": ["tokens"],
          "prefillArgs": ["tokens"],
          "prefillOutputs": ["l0.k","l0.v"],
          "withPastArgs": ["token","cosSliding","sinSliding","l0.k","l0.v"],
          "withPastOutputs": ["l0.k","l0.v","token"],
          "toolMap": { "0": "set_lights", "1": "play_buzzer", "none": null }
        }
    """.trimIndent()

    @Test
    fun parse_readsArchitectureConstantsAndOrders() {
        val m = GemmaManifest.parse(sample)
        assertEquals(1, m.contractVersion)
        assertEquals(18, m.nLayers)
        assertEquals(256, m.headDim)
        assertEquals(1, m.nKvHeads)
        assertEquals(24, m.seq)
        assertEquals(106, m.eot)
        assertEquals(10_000f, m.slidingRopeBase)
        assertEquals(1_000_000f, m.globalRopeBase)
        assertEquals(6, m.globalLayerPeriod)
        assertEquals(true, m.kFirstInOutput)
        assertEquals("model", m.parameterScope)
        assertEquals("gemma-with-past.irpa", m.parameters.withPast)
        assertEquals(listOf("token", "cosSliding", "sinSliding", "l0.k", "l0.v"), m.withPastArgs)
        assertEquals(listOf("l0.k", "l0.v", "token"), m.withPastOutputs)
    }

    @Test
    fun parse_readsToolMapIncludingExplicitNoOp() {
        val m = GemmaManifest.parse(sample)
        assertEquals("set_lights", m.toolMap["0"])
        assertEquals("play_buzzer", m.toolMap["1"])
        assertNull(m.toolMap.getValue("none"))
    }

    @Test
    fun parse_ignoresUnknownFields() {
        val withExtra = sample.replace("\"contractVersion\": 1,", "\"contractVersion\": 1, \"someFutureField\": 42,")
        val m = GemmaManifest.parse(withExtra)
        assertEquals(1, m.contractVersion)
    }

    @Test
    fun compactToolCodec_fromManifest_usesManifestToolMap() {
        val m = GemmaManifest.parse(sample)
        val codec = CompactToolCodec.fromManifest(m)
        val calls = codec.parse("""<tool_0>(state="on")<end>""")
        assertEquals(1, calls.size)
        assertEquals("set_lights", calls.single().tool)
        assertEquals("on", calls.single().args["state"])

        // "5" isn't in this manifest's tool map (unlike the DEFAULT_TOKEN_TO_NAME vocabulary) -> dropped.
        val unmapped = codec.parse("""<tool_5>(message="hi")<end>""")
        assertEquals(0, unmapped.size, "tokens outside the manifest's toolMap must not fall back to defaults")
    }
}
