package sk.ainet.transformers.iree.android

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

/** `manifest.json` of `WhisperExportCli` → [IreeWhisperSession.Contract], the values the native session sizes its buffers by. */
class IreeWhisperContractTest {
    // as WhisperExportHarness.manifestJson writes it for a multilingual tiny checkpoint, 4 s window
    private val manifest = """
        {
          "contractVersion": 2,
          "functions": { "encoder": "whisper_encoder", "prefill": "whisper_prefill", "step": "whisper_step" },
          "nLayers": 4,
          "dModel": 384,
          "audioCtx": 200,
          "maxPositions": 48,
          "vocab": 51865,
          "eot": 50257,
          "promptIds": [50258,50261,50359,50363],
          "melFrames": 400,
          "parameters": "params.irpa",
          "parameterScope": "model",
          "prefillArgs": ["promptIds","feat"],
          "prefillOutputs": ["logits","l0.selfK","l0.selfV","l0.crossK","l0.crossV"],
          "stepArgs": ["tok","pos","addMask","wf","l0.selfK","l0.selfV","l0.crossK","l0.crossV"],
          "stepOutputs": ["logits","l0.selfK","l0.selfV"]
        }
    """.trimIndent()

    @Test fun readsTheTinyMultilingualContract() {
        val c = IreeWhisperSession.Contract.fromManifest(manifest)
        assertEquals("whisper_encoder", c.encoderFunction)
        assertEquals("whisper_prefill", c.prefillFunction)
        assertEquals("whisper_step", c.stepFunction)
        assertEquals("model", c.parameterScope)
        assertArrayEquals(intArrayOf(4, 384, 200, 80, 400, 48, 51865), c.geometry())
        assertEquals(50257, c.eot)
        assertArrayEquals(intArrayOf(50258, 50261, 50359, 50363), c.promptIds)
    }

    @Test fun readsAnEnglishOnlyPrompt() {
        val en = manifest.replace("[50258,50261,50359,50363]", "[50257, 50362]").replace("\"eot\": 50257", "\"eot\": 50256")
        val c = IreeWhisperSession.Contract.fromManifest(en)
        assertArrayEquals(intArrayOf(50257, 50362), c.promptIds)
        assertEquals(50256, c.eot)
    }

    @Test fun rejectsAnotherContractVersion() {
        assertThrows(IllegalArgumentException::class.java) {
            IreeWhisperSession.Contract.fromManifest(manifest.replace("\"contractVersion\": 2", "\"contractVersion\": 1"))
        }
    }

    @Test fun rejectsAPromptThatDoesNotFitTheCache() {
        assertThrows(IllegalArgumentException::class.java) {
            IreeWhisperSession.Contract.fromManifest(manifest.replace("\"maxPositions\": 48", "\"maxPositions\": 4"))
        }
    }
}
