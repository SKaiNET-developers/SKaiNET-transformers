package sk.ainet.apps.llm.validation

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertEquals

class ShapeValidatorTest {

    @Test
    fun `pipeline validator passes with matching frame counts`() {
        val pv = PipelineShapeValidator()
        // Acoustic noise xt has dim=36, inputProj [3072, 36] transposed = [36, 3072]
        // matmul([seqLen, 36], [36, 3072]) → input dim = 36 matches
        pv.stage("acoustic.noise", listOf(9, 36))
        pv.projection("acoustic.inputProj", listOf(3072, 36), transpose = true)
        pv.matchCount("semanticTokens", 9, "acousticFrames", 9)

        val result = pv.validate()
        assertTrue(result.isValid, "Should pass with matching dims and counts: ${result.errors}")
    }

    @Test
    fun `pipeline validator detects frame count mismatch`() {
        val pv = PipelineShapeValidator()
        pv.matchCount("semanticTokens", 8, "acousticFrames", 9)

        val result = pv.validate()
        assertFalse(result.isValid, "Should fail with mismatched counts")
        assertTrue(result.errors.any { it.contains("Frame count mismatch") })
    }

    @Test
    fun `pipeline validator detects projection dim mismatch`() {
        val pv = PipelineShapeValidator()
        pv.stage("backbone.hidden", listOf(9, 3072))
        // inputProj [756, 3072] transposed = [3072, 756], input dim = 3072 ✓
        // But if backbone outputs dim=2048 and proj expects 3072:
        pv.stage("wrong_backbone", listOf(9, 2048))
        pv.projection("proj", listOf(3072, 36), transpose = true)

        val result = pv.validate()
        assertFalse(result.isValid, "Should fail with dim mismatch")
        assertTrue(result.errors.any { it.contains("matmul dim mismatch") })
    }

    @Test
    fun `pipeline validator produces trace`() {
        val pv = PipelineShapeValidator()
        pv.stage("backbone", listOf(9, 3072))
        pv.projection("inputProj", listOf(3072, 36), transpose = true)
        pv.matchCount("tokens", 9, "frames", 9)

        val result = pv.validate()
        assertTrue(result.trace.isNotEmpty(), "Trace should have entries")
        assertTrue(result.trace.size == 2, "Should have projection + count trace entries")
    }

    @Test
    fun `pipeline validator handles empty pipeline`() {
        val pv = PipelineShapeValidator()
        val result = pv.validate()
        assertTrue(result.isValid, "Empty pipeline should pass")
    }

    @Test
    fun `shape inference registry has rules for standard modules`() {
        // Verify built-in rules are registered
        val rmsRule = ShapeInferenceRegistry.getRule(
            sk.ainet.lang.nn.normalization.RMSNormalization::class
        )
        assertTrue(rmsRule != null, "RMSNormalization should have a shape rule")

        val mhaRule = ShapeInferenceRegistry.getRule(
            sk.ainet.lang.nn.transformer.MultiHeadAttention::class
        )
        assertTrue(mhaRule != null, "MultiHeadAttention should have a shape rule")

        val ffnRule = ShapeInferenceRegistry.getRule(
            sk.ainet.lang.nn.transformer.SwiGLUFFN::class
        )
        assertTrue(ffnRule != null, "SwiGLUFFN should have a shape rule")
    }
}
