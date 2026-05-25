package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import sk.ainet.lang.tensor.Shape

/**
 * Pins the per-tensor logical shape that
 * [DecoderGgufMemSegConverter.logicalShapeFor] hands to `convertOne` for the
 * standard Llama-family tensor names.
 *
 * The motivating regression is **Qwen3-0.6B** (issue #148): for that
 * checkpoint `head_dim != hidden_size / num_attention_heads`, so the
 * previous `dim / meta.headCount` shortcut produced a half-sized Q / O
 * projection and the forward pass crashed in
 * `MultiHeadAttention.attentionImpl` with a `Reshape volume mismatch`.
 *
 * Each test case feeds `(dim, qDim, kvDim, ffnDim, vocab)` corresponding to
 * a real model and asserts every named slot.
 */
class DecoderGgufMemSegConverterLogicalShapeTest {

    @Test
    fun `Qwen3-0_6B shape map honours head_dim != hidden_size  num_heads`() {
        // hidden=1024, n_heads=16, n_kv_heads=8, head_dim=128
        //   → qDim = 16 * 128 = 2048
        //   → kvDim = 8 * 128 = 1024
        // ffn=3072, vocab=151_936
        val dim = 1024
        val qDim = 2048
        val kvDim = 1024
        val ffnDim = 3072
        val vocab = 151_936

        assertEquals(Shape(vocab, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.TOKEN_EMBEDDINGS, dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(vocab, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.OUTPUT_WEIGHT, dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(qDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnQ(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(kvDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnK(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(kvDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnV(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(dim, qDim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnOut(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(ffnDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.ffnGate(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(ffnDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.ffnUp(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(dim, ffnDim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.ffnDown(0), dim, qDim, kvDim, ffnDim, vocab))
    }

    @Test
    fun `Llama-3_2-1B shape map back-compat (head_dim == hidden_size  n_heads)`() {
        // hidden=2048, n_heads=32, n_kv_heads=8, head_dim=64
        //   → qDim = 32 * 64 = 2048 = hidden
        //   → kvDim = 8 * 64 = 512
        // ffn=8192, vocab=128_256
        val dim = 2048
        val qDim = 2048
        val kvDim = 512
        val ffnDim = 8192
        val vocab = 128_256

        assertEquals(Shape(qDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnQ(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(kvDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnK(0), dim, qDim, kvDim, ffnDim, vocab))
        assertEquals(Shape(kvDim, dim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnV(0), dim, qDim, kvDim, ffnDim, vocab))
        // qDim == dim for Llama-3.2-1B, so attn_output's two shape entries
        // happen to coincide — but the new code path is exercised.
        assertEquals(Shape(dim, qDim), DecoderGgufMemSegConverter.logicalShapeFor(LlamaTensorNames.attnOut(0), dim, qDim, kvDim, ffnDim, vocab))
    }

    @Test
    fun `unknown tensor names return null`() {
        val result = DecoderGgufMemSegConverter.logicalShapeFor(
            "blk.0.attn_norm.weight", dim = 1024, qDim = 2048, kvDim = 1024,
            ffnDim = 3072, vocab = 151_936,
        )
        assertNull(result, "norm weights aren't routed through the converter; logicalShapeFor should return null")
    }
}
