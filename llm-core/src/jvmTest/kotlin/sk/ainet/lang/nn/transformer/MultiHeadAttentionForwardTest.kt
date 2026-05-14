package sk.ainet.lang.nn.transformer

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Forward-pass tests for [MultiHeadAttention] focused on the encoder-decoder
 * cross-attention path added alongside this test file
 * (`forward(input, encoderMemory, ctx)`).
 *
 * Pins three contracts:
 *  - self-attention output is bit-identical whether reached via the inherited
 *    `forward(input, ctx)` (today's API) or the new 3-arg entry with
 *    `encoderMemory = null` (the compatibility path);
 *  - cross-attention accepts asymmetric Q vs K/V sequence lengths;
 *  - cross-attention ignores `causal` (no temporal ordering between decoder
 *    queries and encoder memory frames);
 *  - cross-attention rejects a non-null kvCache (caching is a runtime
 *    concern for cross-attn).
 */
class MultiHeadAttentionForwardTest {

    private val ctx = DirectCpuExecutionContext()

    // Small but non-degenerate shapes: 4-dim model with 2 heads → headDim=2.
    private val dim = 4
    private val nHeads = 2
    private val headDim = dim / nHeads

    private fun buildMha(
        causal: Boolean = false,
        kvCache: KVCache<FP32, Float>? = null,
    ): MultiHeadAttention<FP32, Float> {
        val mha = MultiHeadAttention<FP32, Float>(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nHeads,
            causal = causal,
            bias = false,
            qkNorm = false,
            kvCache = kvCache,
            name = "test_mha",
        )
        // Replace the void placeholder weights with deterministic real ones.
        // Layout: [out, in]. Different per-projection so we can tell them apart.
        mha.params[0].value = wMatrix(dim, dim, seed = 1)  // q_proj
        mha.params[1].value = wMatrix(dim, dim, seed = 2)  // k_proj
        mha.params[2].value = wMatrix(dim, dim, seed = 3)  // v_proj
        mha.params[3].value = wMatrix(dim, dim, seed = 4)  // o_proj
        return mha
    }

    /** Deterministic `[out, in]` weight matrix with values in a small range. */
    private fun wMatrix(out: Int, inDim: Int, seed: Int): Tensor<FP32, Float> {
        // Pseudo-random but reproducible. sin(seed + idx) keeps values in [-1, 1].
        val values = FloatArray(out * inDim) { idx ->
            (kotlin.math.sin((seed * 1000 + idx).toFloat()) * 0.3f)
        }
        return ctx.fromFloatArray(Shape(out, inDim), FP32::class, values)
    }

    private fun inputTensor(seqLen: Int, seed: Int): Tensor<FP32, Float> {
        val values = FloatArray(seqLen * dim) { idx ->
            (kotlin.math.cos((seed * 1000 + idx).toFloat()) * 0.5f)
        }
        return ctx.fromFloatArray(Shape(seqLen, dim), FP32::class, values)
    }

    @Test
    fun selfAttentionUnchangedWhenEncoderMemoryIsNull() {
        val mha = buildMha(causal = false)
        val input = inputTensor(seqLen = 3, seed = 7)

        // Path 1: inherited 2-arg forward (today's API).
        val viaOldEntry = mha.forward(input, ctx).data.copyToFloatArray()

        // Path 2: new 3-arg forward with encoderMemory = null (compat path).
        val viaNewEntry = mha.forward(input, null, ctx).data.copyToFloatArray()

        assertContentEquals(
            viaOldEntry, viaNewEntry,
            "passing encoderMemory = null must be bit-identical to the inherited self-attention path",
        )
    }

    @Test
    fun crossAttentionProducesOutputShapedLikeQuery() {
        val mha = buildMha(causal = false)
        val query = inputTensor(seqLen = 2, seed = 11)
        val memory = inputTensor(seqLen = 5, seed = 23)

        val out = mha.forward(query, memory, ctx)

        // Output sequence length follows Q, not K/V — that's the cross-attn contract.
        assertEquals(2, out.shape[0], "output seqLen must equal query seqLen")
        assertEquals(dim, out.shape[1], "output dim must equal model dim")
    }

    @Test
    fun crossAttentionDoesNotApplyCausalMask() {
        // Build MHA with causal=true. In self-attention that masks the future;
        // in cross-attention the new branch forces causal off.
        val mhaCausal = buildMha(causal = true)
        val mhaNonCausal = buildMha(causal = false)

        // Use the SAME query and SAME memory for both. Cross-attn output must
        // agree regardless of the `causal` flag — proves the mask is bypassed.
        val query = inputTensor(seqLen = 3, seed = 41)
        val memory = inputTensor(seqLen = 4, seed = 53)

        val outCausal = mhaCausal.forward(query, memory, ctx).data.copyToFloatArray()
        val outNonCausal = mhaNonCausal.forward(query, memory, ctx).data.copyToFloatArray()

        // Same projection weights → same output (modulo float jitter) when the
        // mask isn't applied. The two MHAs have identical seeded weights, so
        // exact equality is the right assertion.
        assertContentEquals(
            outCausal, outNonCausal,
            "cross-attn output must not depend on the `causal` constructor flag",
        )
    }

    @Test
    fun crossAttentionRejectsKvCache() {
        val cache = AppendKVCache<FP32, Float>(maxSeqLen = 8, nKVHeads = nHeads, headDim = headDim)
        val mha = buildMha(causal = true, kvCache = cache)
        val query = inputTensor(seqLen = 1, seed = 61)
        val memory = inputTensor(seqLen = 3, seed = 67)

        val err = assertFailsWith<IllegalArgumentException> {
            mha.forward(query, memory, ctx)
        }
        assertTrue(
            err.message?.contains("kvCache", ignoreCase = true) == true,
            "error message should mention kvCache; got: ${err.message}",
        )
    }

    @Test
    fun moduleTreeUnchanged() {
        val mha = buildMha(causal = false)
        // No RoPE, no QK-norm, no KV cache → modules list must be empty (the
        // pre-refactor snapshot for this configuration). Adding a submodule
        // here would surface accidental extra state.
        assertEquals(emptyList(), mha.modules, "modules list must be empty for vanilla self-attn config")
        // Params: q/k/v/o weights only, no biases.
        val paramNames = mha.params.map { it.name }
        assertContentEquals(
            listOf("test_mha.q_proj.weight", "test_mha.k_proj.weight", "test_mha.v_proj.weight", "test_mha.o_proj.weight"),
            paramNames,
            "param names + order must match the pre-refactor snapshot",
        )
    }
}
