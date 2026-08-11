package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Tests for [ToolCallingSupportResolver.resolveWithDiagnostics] (#42/#44):
 * the selection reason and reported [ToolCallingMode] must be deterministic
 * so demo/agent output can explain why a provider was chosen.
 */
class ResolutionDiagnosticsTest {

    @Test
    fun explicitSelectionReportsOverrideReason() {
        val result = ToolCallingSupportResolver.resolveWithDiagnostics(
            metadata = ModelMetadata(family = "gemma"),
            explicitFamily = "llama3"
        )
        assertIs<Llama3ToolCallingSupport>(result.provider)
        assertEquals(ToolCallingMode.NATIVE, result.mode)
        assertTrue("explicit" in result.reason && "llama3" in result.reason)
    }

    @Test
    fun autoDetectionReportsMetadataReason() {
        val result = ToolCallingSupportResolver.resolveWithDiagnostics(
            metadata = ModelMetadata(family = "qwen", architecture = "qwen2")
        )
        assertIs<QwenToolCallingSupport>(result.provider)
        assertEquals(ToolCallingMode.NATIVE, result.mode)
        assertTrue("auto-detected" in result.reason)
        assertTrue("qwen" in result.reason)
    }

    @Test
    fun unknownModelFallsBackToGenericWithGenericMode() {
        val result = ToolCallingSupportResolver.resolveWithDiagnostics(
            metadata = ModelMetadata(family = "totally-unknown", architecture = "mystery")
        )
        assertIs<GenericToolCallingSupport>(result.provider)
        assertEquals(ToolCallingMode.GENERIC, result.mode)
        assertTrue("fallback" in result.reason)
    }

    @Test
    fun metadataFromGgufExtractionCarriesThroughDiagnostics() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf("general.architecture" to "qwen2")
        )
        val result = ToolCallingSupportResolver.resolveWithDiagnostics(metadata)
        assertEquals("qwen", result.provider.family)
        assertEquals(ToolCallingMode.NATIVE, result.mode)
    }
}
