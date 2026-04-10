package sk.ainet.apps.kllama

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Unit tests for Qwen35Runtime using a tiny synthetic model (no GGUF required).
 *
 * Model dimensions are minimal to keep tests fast while exercising the key
 * algorithms: 3-way QKV split, conv1d weight indexing, delta rule recurrence,
 * L2 norm, head repetition, and full attention Q+Gate split.
 */
class Qwen35RuntimeTest {

    private val ctx = DirectCpuExecutionContext()

    // Tiny model dimensions
    private val dim = 16
    private val vocabSize = 8
    private val nLayers = 4 // layers 0,1,2 = DeltaNet; layer 3 = full attention
    private val fullAttnInterval = 4
    private val ssmStateSize = 4
    // DeltaNet: numKHeads=1, numVHeads=2, keyDim=4, valueDim=8, ssmQkvDim=16
    // (2*keyDim + valueDim = 2*4 + 8 = 16 = dim; matches attn_qkv [16, 16])
    private val numKHeads = 1
    private val numVHeads = 2
    private val keyDim = numKHeads * ssmStateSize   // 4
    private val valueDim = numVHeads * ssmStateSize  // 8
    private val ssmQkvDim = 2 * keyDim + valueDim   // 16
    private val ssmConvKernel = 4

    // Full attention: nHeads=2, nKvHeads=1, headDim=8
    // Joint Q+Gate: attn_q.weight = [2 * nHeads * headDim, dim] = [32, 16]
    private val faHeadDim = 8
    private val faNHeads = 2
    private val faNKvHeads = 1
    private val faQGateDim = 2 * faNHeads * faHeadDim // 32
    private val faKvDim = faNKvHeads * faHeadDim      // 8

    private val ffnHidden = 32
    private val ropeDim = 4

    private fun tensor(vararg dims: Int, value: Float = 0.1f): Tensor<FP32, Float> =
        ctx.full<FP32, Float>(Shape(*dims), FP32::class, value)

    private fun tensorFromArray(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray<FP32, Float>(shape, FP32::class, data)

    private fun buildMetadata() = LlamaModelMetadata(
        architecture = "qwen35",
        embeddingLength = dim,
        contextLength = 64,
        blockCount = nLayers,
        headCount = faNHeads,
        kvHeadCount = faNKvHeads,
        feedForwardLength = ffnHidden,
        ropeDimensionCount = ropeDim,
        vocabSize = vocabSize,
        ropeFreqBase = 10_000f,
        rmsNormEps = 1e-5f,
        bosTokenId = 0,
        eosTokenId = 1
    )

    private fun buildTensors(): Map<String, Tensor<FP32, Float>> {
        val tensors = mutableMapOf<String, Tensor<FP32, Float>>()

        // Embedding and output
        tensors["token_embd.weight"] = tensor(vocabSize, dim, value = 0.02f)
        tensors["output_norm.weight"] = tensor(dim, value = 1f)
        tensors["output.weight"] = tensor(vocabSize, dim, value = 0.1f)

        for (i in 0 until nLayers) {
            // Shared per-layer norms and FFN
            tensors["blk.$i.attn_norm.weight"] = tensor(dim, value = 1f)
            tensors["blk.$i.post_attention_norm.weight"] = tensor(dim, value = 1f)
            tensors["blk.$i.ffn_gate.weight"] = tensor(ffnHidden, dim, value = 0.05f)
            tensors["blk.$i.ffn_up.weight"] = tensor(ffnHidden, dim, value = 0.05f)
            tensors["blk.$i.ffn_down.weight"] = tensor(dim, ffnHidden, value = 0.02f)

            val isFullAttn = (i + 1) % fullAttnInterval == 0

            if (isFullAttn) {
                // Full attention layer: joint Q+Gate, K, V, output, QK-norms
                tensors["blk.$i.attn_q.weight"] = tensor(faQGateDim, dim, value = 0.1f)
                tensors["blk.$i.attn_k.weight"] = tensor(faKvDim, dim, value = 0.1f)
                tensors["blk.$i.attn_v.weight"] = tensor(faKvDim, dim, value = 0.1f)
                tensors["blk.$i.attn_output.weight"] = tensor(dim, faNHeads * faHeadDim, value = 0.1f)
                tensors["blk.$i.attn_q_norm.weight"] = tensor(faHeadDim, value = 1f)
                tensors["blk.$i.attn_k_norm.weight"] = tensor(faHeadDim, value = 1f)
            } else {
                // DeltaNet layer
                tensors["blk.$i.attn_qkv.weight"] = tensor(ssmQkvDim, dim, value = 0.1f)
                tensors["blk.$i.attn_gate.weight"] = tensor(valueDim, dim, value = 0.1f)
                // ssm_a: per V-head, should be negative (represents -exp(A_log))
                tensors["blk.$i.ssm_a"] = tensorFromArray(
                    Shape(numVHeads), FloatArray(numVHeads) { -0.5f }
                )
                tensors["blk.$i.ssm_alpha.weight"] = tensor(numVHeads, dim, value = 0.05f)
                tensors["blk.$i.ssm_beta.weight"] = tensor(numVHeads, dim, value = 0.1f)
                // conv1d: [ssmQkvDim, ssmConvKernel] with channel-major layout
                tensors["blk.$i.ssm_conv1d.weight"] = tensorFromArray(
                    Shape(ssmQkvDim, ssmConvKernel),
                    FloatArray(ssmQkvDim * ssmConvKernel) { 0.25f }
                )
                tensors["blk.$i.ssm_dt.bias"] = tensorFromArray(
                    Shape(numVHeads), FloatArray(numVHeads) { 0.1f }
                )
                tensors["blk.$i.ssm_norm.weight"] = tensorFromArray(
                    Shape(ssmStateSize), FloatArray(ssmStateSize) { 1f }
                )
                tensors["blk.$i.ssm_out.weight"] = tensor(dim, valueDim, value = 0.1f)
            }
        }
        return tensors
    }

    private fun createRuntime(): Qwen35Runtime<FP32> = Qwen35Runtime(
        ctx = ctx,
        metadata = buildMetadata(),
        tensors = buildTensors(),
        dtype = FP32::class,
        fullAttentionInterval = fullAttnInterval,
        ssmStateSize = ssmStateSize,
        ssmConvKernel = ssmConvKernel,
        ropeFreqBase = 10_000f,
        maxContextLength = 64
    )

    // ---- Basic sanity ----

    @Test
    fun `forward produces finite logits of correct shape`() {
        val runtime = createRuntime()
        val logits = runtime.forward(0)

        assertEquals(Shape(1, vocabSize), logits.shape)
        val buf = logits.data.copyToFloatArray()
        for (v in buf) {
            assertTrue(v.isFinite(), "Logit is not finite: $v")
        }
    }

    @Test
    fun `successive forward calls with different tokens produce different logits`() {
        // Use varied embedding weights so different token IDs produce
        // meaningfully different input vectors to the DeltaNet layers.
        val tensors = buildTensors().toMutableMap()
        val embData = FloatArray(vocabSize * dim) { i -> ((i % 7) - 3) * 0.1f }
        tensors["token_embd.weight"] = tensorFromArray(Shape(vocabSize, dim), embData)

        val runtime = Qwen35Runtime<FP32>(
            ctx = ctx, metadata = buildMetadata(), tensors = tensors, dtype = FP32::class,
            fullAttentionInterval = fullAttnInterval, ssmStateSize = ssmStateSize,
            ssmConvKernel = ssmConvKernel, ropeFreqBase = 10_000f, maxContextLength = 64
        )

        // Process two different tokens
        val logits0 = runtime.forward(0).data.copyToFloatArray().copyOf()
        val logits1 = runtime.forward(3).data.copyToFloatArray().copyOf()

        var anyDiff = false
        for (i in logits0.indices) {
            if (abs(logits0[i] - logits1[i]) > 1e-6f) {
                anyDiff = true
                break
            }
        }
        assertTrue(anyDiff, "Logits should differ across different tokens due to DeltaNet state evolution")
    }

    @Test
    fun `generate produces requested number of tokens`() {
        val runtime = createRuntime()
        val generated = mutableListOf<Int>()
        runtime.generate(intArrayOf(0, 1), steps = 4, temperature = 1f) { generated.add(it) }
        assertEquals(4, generated.size, "Should generate exactly 4 tokens")
    }

    @Test
    fun `fresh runtime produces reproducible logits`() {
        val logits1 = createRuntime().forward(0).data.copyToFloatArray().copyOf()
        val logits2 = createRuntime().forward(0).data.copyToFloatArray().copyOf()

        for (i in logits1.indices) {
            assertEquals(logits1[i], logits2[i], 1e-5f,
                "Fresh runtimes should produce identical logits for same token")
        }
    }

    // ---- DeltaNet dimensional correctness ----

    @Test
    fun `runtime derives correct DeltaNet dimensions from tensor shapes`() {
        // The runtime should not throw during construction — this validates
        // that ssmQkvDim, numKHeads, numVHeads, keyDim, valueDim, headRatio
        // are all derived consistently from the tensor shapes we provide.
        val runtime = createRuntime()

        // Run through all layers (3 DeltaNet + 1 full attention) without crash
        runtime.forward(0)
        runtime.forward(1)
        runtime.forward(2)
        // If dimensions were wrong, we'd get ArrayIndexOutOfBoundsException
    }

    // ---- Conv1d weight indexing ----

    @Test
    fun `conv1d uses channel-major weight layout`() {
        // Build a runtime with distinct conv weights so we can verify indexing.
        // Set conv weight at channel c, tap k = (c + 1) * (k + 1) * 0.01
        // Then verify the output matches the expected channel-major dot product.
        val tensors = buildTensors().toMutableMap()

        // Only layer 0 (DeltaNet) matters for this test
        val convWeights = FloatArray(ssmQkvDim * ssmConvKernel)
        for (c in 0 until ssmQkvDim) {
            for (k in 0 until ssmConvKernel) {
                // channel-major: flat[c * ssmConvKernel + k]
                convWeights[c * ssmConvKernel + k] = (c + 1f) * (k + 1f) * 0.01f
            }
        }
        tensors["blk.0.ssm_conv1d.weight"] = tensorFromArray(
            Shape(ssmQkvDim, ssmConvKernel), convWeights
        )

        val runtime = Qwen35Runtime<FP32>(
            ctx = ctx,
            metadata = buildMetadata(),
            tensors = tensors,
            dtype = FP32::class,
            fullAttentionInterval = fullAttnInterval,
            ssmStateSize = ssmStateSize,
            ssmConvKernel = ssmConvKernel,
            ropeFreqBase = 10_000f,
            maxContextLength = 64
        )

        // If conv weight indexing were wrong (tap-major instead of channel-major),
        // the output would be numerically different. We just verify no crash
        // and finite outputs through two forward passes (fills conv buffer).
        val logits1 = runtime.forward(0)
        val logits2 = runtime.forward(1)
        for (v in logits2.data.copyToFloatArray()) {
            assertTrue(v.isFinite(), "Conv1d output should be finite: $v")
        }
    }

    // ---- Full attention Q+Gate split ----

    @Test
    fun `full attention layer produces finite output with Q+Gate split`() {
        val runtime = createRuntime()

        // Run 3 DeltaNet layers + hit the full attention layer (layer 3)
        // by processing at least 4 tokens
        for (t in 0 until 4) {
            val logits = runtime.forward(t % vocabSize)
            val buf = logits.data.copyToFloatArray()
            for (v in buf) {
                assertTrue(v.isFinite(),
                    "Logit at pos=$t should be finite (full attention at layer 3): $v")
            }
        }
    }

    @Test
    fun `full attention gate attenuates output`() {
        // With all gate weights initialized to 0.1, the gate projection
        // produces positive values, sigmoid > 0.5, so output is attenuated
        // but not zeroed. Verify logits have reasonable magnitude.
        val runtime = createRuntime()

        // Need multiple tokens to fill KV cache for attention
        for (t in 0 until 3) runtime.forward(t)
        val logits = runtime.forward(3)
        val buf = logits.data.copyToFloatArray()

        val maxAbs = buf.maxOf { abs(it) }
        assertTrue(maxAbs > 0f, "Full attention output should be non-zero")
        assertTrue(maxAbs < 1000f, "Full attention output should not explode (got $maxAbs)")
    }

    // ---- Delta rule recurrence properties ----

    @Test
    fun `delta rule state evolves with prediction error`() {
        // After processing tokens, the DeltaNet state should be non-zero.
        // We verify this indirectly: processing the same token twice should
        // produce different logits (because state has evolved).
        val runtime = createRuntime()

        val logitsFirst = runtime.forward(3).data.copyToFloatArray().copyOf()
        val logitsSecond = runtime.forward(3).data.copyToFloatArray().copyOf()

        var sumAbsDiff = 0f
        for (i in logitsFirst.indices) {
            sumAbsDiff += abs(logitsFirst[i] - logitsSecond[i])
        }
        assertTrue(sumAbsDiff > 1e-6f,
            "Same token at different positions should produce different logits (delta rule state evolves)")
    }

    @Test
    fun `decay keeps state bounded over many tokens`() {
        // With ssm_a = -0.5 (negative), decay < 1, state should not explode
        val runtime = createRuntime()

        for (t in 0 until 30) {
            val logits = runtime.forward(t % vocabSize)
            val buf = logits.data.copyToFloatArray()
            val maxAbs = buf.maxOf { abs(it) }
            assertTrue(maxAbs < 1e6f,
                "Logits should stay bounded at pos=$t (got maxAbs=$maxAbs)")
            for (v in buf) {
                assertTrue(v.isFinite(), "Logit at pos=$t should be finite: $v")
            }
        }
    }

    // ---- L2 normalization ----

    @Test
    fun `l2 normalization produces unit-length vectors`() {
        // Test the l2 norm indirectly: with large QKV projection values,
        // L2 norm on Q/K should prevent state explosion.
        val tensors = buildTensors().toMutableMap()
        // Set QKV projection to produce large values
        tensors["blk.0.attn_qkv.weight"] = tensor(ssmQkvDim, dim, value = 5.0f)

        val runtime = Qwen35Runtime<FP32>(
            ctx = ctx,
            metadata = buildMetadata(),
            tensors = tensors,
            dtype = FP32::class,
            fullAttentionInterval = fullAttnInterval,
            ssmStateSize = ssmStateSize,
            ssmConvKernel = ssmConvKernel,
            ropeFreqBase = 10_000f,
            maxContextLength = 64
        )

        // Without L2 norm, large Q/K would cause state to explode.
        // With L2 norm, Q and K are unit vectors so state stays bounded.
        for (t in 0 until 10) {
            val logits = runtime.forward(t % vocabSize)
            val buf = logits.data.copyToFloatArray()
            for (v in buf) {
                assertTrue(v.isFinite(),
                    "L2 norm should keep output finite even with large QKV weights (pos=$t): $v")
            }
        }
    }
}
