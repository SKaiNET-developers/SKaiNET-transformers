package sk.ainet.apps.llm

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Pins the capability model behind [DTypePolicyValidation].
 *
 * A `Require` naming a narrow float is a promise that the weights reach the kernel packed, so it
 * may only be accepted by a chain that actually implements KEEP_NATIVE for *that* format. The
 * previous BF16-only boolean could not express "keeps FP16 but not BF16", nor the empty case
 * (Gemma / Apertus) — which it papered over by accepting `Require(BF16)` and ignoring it.
 */
class DTypePolicyValidationTest {

    private val bothNarrow = setOf(BF16, FP16)

    @Test
    fun `Require(FP32) is always accepted — every chain produces FP32`() {
        DTypePolicyValidation.validate(DTypePolicy.Require(FP32), "test", keepNative = emptySet())
        DTypePolicyValidation.validate(DTypePolicy.Require(FP32), "test", keepNative = bothNarrow)
    }

    @Test
    fun `soft policies never raise, whatever they name`() {
        for (keepNative in listOf(emptySet(), bothNarrow)) {
            DTypePolicyValidation.validate(DTypePolicy.Any, "test", keepNative)
            DTypePolicyValidation.validate(DTypePolicy.Prefer(BF16), "test", keepNative)
            DTypePolicyValidation.validate(DTypePolicy.Prefer(FP16), "test", keepNative)
            DTypePolicyValidation.validate(DTypePolicy.Prefer(Int8), "test", keepNative)
            DTypePolicyValidation.validate(DTypePolicy.OneOf(setOf(FP32, BF16)), "test", keepNative)
        }
    }

    @Test
    fun `a chain that keeps nothing packed rejects both narrow Requires`() {
        // The Gemma / Apertus position: their weight chains widen every narrow float, so a
        // Require they cannot honor must fail loudly rather than be silently ignored.
        assertFailsWith<IllegalArgumentException> {
            DTypePolicyValidation.validate(DTypePolicy.Require(BF16), "test", keepNative = emptySet())
        }
        assertFailsWith<IllegalArgumentException> {
            DTypePolicyValidation.validate(DTypePolicy.Require(FP16), "test", keepNative = emptySet())
        }
    }

    @Test
    fun `the two narrow formats are tracked independently`() {
        // Keeps BF16 only.
        DTypePolicyValidation.validate(DTypePolicy.Require(BF16), "test", keepNative = setOf(BF16))
        assertFailsWith<IllegalArgumentException> {
            DTypePolicyValidation.validate(DTypePolicy.Require(FP16), "test", keepNative = setOf(BF16))
        }

        // Keeps FP16 only — the mirror image, which the old boolean flag could not express.
        DTypePolicyValidation.validate(DTypePolicy.Require(FP16), "test", keepNative = setOf(FP16))
        assertFailsWith<IllegalArgumentException> {
            DTypePolicyValidation.validate(DTypePolicy.Require(BF16), "test", keepNative = setOf(FP16))
        }
    }

    @Test
    fun `a dtype no chain produces is rejected even when both narrow floats are kept`() {
        assertFailsWith<IllegalArgumentException> {
            DTypePolicyValidation.validate(DTypePolicy.Require(Int8), "test", keepNative = bothNarrow)
        }
    }

    @Test
    fun `keepsNative names one format at a time`() {
        assertFalse(DTypePolicyValidation.keepsNative(DTypePolicy.Any, BF16))
        assertFalse(DTypePolicyValidation.keepsNative(DTypePolicy.Any, FP16))

        // Require / Prefer keep exactly the format they name — never the sibling, which would
        // mean reinterpreting one 16-bit layout as the other and decoding to silent garbage.
        assertTrue(DTypePolicyValidation.keepsNative(DTypePolicy.Require(BF16), BF16))
        assertFalse(DTypePolicyValidation.keepsNative(DTypePolicy.Require(BF16), FP16))
        assertTrue(DTypePolicyValidation.keepsNative(DTypePolicy.Prefer(FP16), FP16))
        assertFalse(DTypePolicyValidation.keepsNative(DTypePolicy.Prefer(FP16), BF16))

        // OneOf may admit both at once.
        val oneOf = DTypePolicy.OneOf(setOf(BF16, FP16))
        assertTrue(DTypePolicyValidation.keepsNative(oneOf, BF16))
        assertTrue(DTypePolicyValidation.keepsNative(oneOf, FP16))

        assertFalse(DTypePolicyValidation.keepsNative(DTypePolicy.Require(FP32), BF16))
    }
}
