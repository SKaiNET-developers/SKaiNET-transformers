package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.RoPEScaling
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Tests for RoPE's Phase 5b extensions: partialRotaryFactor (rotate only a
 * prefix of head_dim) and proportional scaling. Gemma 4 global-attention
 * layers use partialRotaryFactor=0.25 and PROPORTIONAL scaling (factor=1.0
 * on E2B, so the scaling is a no-op but structurally enabled). Phase 5f.3
 * fixed the proportional formula to match HF reference
 * `_compute_proportional_rope_parameters`.
 */
class RoPEScalingTest {

    private val ctx = DirectCpuExecutionContext()

    @Test
    fun `partialRotaryFactor of 1 leaves no tail untouched — rotates full head`() {
        val rope = RoPE<FP32, Float>(
            headDim = 8,
            maxSeqLen = 4,
            partialRotaryFactor = 1.0f
        )
        assertEquals(8, rope.rotaryDim)
    }

    @Test
    fun `partialRotaryFactor of 0_5 rotates half the head dim`() {
        val rope = RoPE<FP32, Float>(
            headDim = 8,
            maxSeqLen = 4,
            partialRotaryFactor = 0.5f
        )
        assertEquals(4, rope.rotaryDim, "rotaryDim should be headDim × 0.5 = 4")
    }

    @Test
    fun `partialRotaryFactor rounds odd rotaryDim down to even`() {
        // 10 × 0.3 = 3 → round down to 2 (must be pair-able).
        val rope = RoPE<FP32, Float>(
            headDim = 10,
            maxSeqLen = 2,
            partialRotaryFactor = 0.3f
        )
        assertEquals(2, rope.rotaryDim)
    }

    @Test
    fun `rotaryDim below 2 is rejected`() {
        assertFails {
            RoPE<FP32, Float>(headDim = 4, maxSeqLen = 2, partialRotaryFactor = 0.1f)
        }
    }

    @Test
    fun `partial rotation leaves tail values unchanged in INTERLEAVED mode`() {
        // headDim=8, partialRotaryFactor=0.5 → rotate first 4 floats of each head,
        // pass through last 4.
        val rope = RoPE<FP32, Float>(
            headDim = 8,
            maxSeqLen = 4,
            mode = RoPEMode.INTERLEAVED,
            partialRotaryFactor = 0.5f
        )

        val seqLen = 1
        val heads = 1
        // Single head, single position, 8 floats: 0, 1, 2, 3 rotated; 4, 5, 6, 7 pass-through.
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(heads, seqLen, 8),
            FP32::class,
            floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f)
        )

        val out = rope.forward(input, position = 0, ctx)
        val buf = out.data.copyToFloatArray()

        // Tail (indices 4..7) should be untouched.
        assertEquals(4f, buf[4], 1e-6f)
        assertEquals(5f, buf[5], 1e-6f)
        assertEquals(6f, buf[6], 1e-6f)
        assertEquals(7f, buf[7], 1e-6f)

        // At position=0, cos=1 sin=0 so rotation is identity on the rotary portion too.
        assertEquals(0f, buf[0], 1e-6f)
        assertEquals(1f, buf[1], 1e-6f)
        assertEquals(2f, buf[2], 1e-6f)
        assertEquals(3f, buf[3], 1e-6f)
    }

    @Test
    fun `PROPORTIONAL scaling with factor 1 is identical to NONE`() {
        // scalingFactor=1 is a no-op on PROPORTIONAL; tables should match NONE exactly.
        val none = RoPE<FP32, Float>(headDim = 4, maxSeqLen = 8, base = 10000f)
        val proportionalIdentity = RoPE<FP32, Float>(
            headDim = 4, maxSeqLen = 8, base = 10000f,
            scaling = RoPEScaling.PROPORTIONAL, scalingFactor = 1.0f
        )

        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 4),
            FP32::class,
            floatArrayOf(1f, 0f, 0f, 1f)
        )

        val a = none.forward(input, position = 3, ctx).data.copyToFloatArray()
        val b = proportionalIdentity.forward(input, position = 3, ctx).data.copyToFloatArray()

        for (i in a.indices) {
            assertEquals(a[i], b[i], 1e-6f, "element $i differs between NONE and PROPORTIONAL(factor=1)")
        }
    }

    @Test
    fun `PROPORTIONAL scaling with factor greater than 1 shifts table values`() {
        // With factor > 1 the effective base grows, so frequencies shrink and the
        // same position produces a different rotation. Just sanity-check that the
        // output differs from the NONE case — exact values are the scaling's
        // mathematical property, not worth asserting numerically here.
        val none = RoPE<FP32, Float>(headDim = 4, maxSeqLen = 8, base = 10000f)
        val scaled = RoPE<FP32, Float>(
            headDim = 4, maxSeqLen = 8, base = 10000f,
            scaling = RoPEScaling.PROPORTIONAL, scalingFactor = 2.0f
        )

        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 4),
            FP32::class,
            floatArrayOf(1f, 0f, 0f, 1f)
        )

        val a = none.forward(input, position = 3, ctx).data.copyToFloatArray()
        val b = scaled.forward(input, position = 3, ctx).data.copyToFloatArray()

        // At position 3, NONE and PROPORTIONAL(2.0) should differ on at least one element.
        var anyDiff = false
        for (i in a.indices) {
            if (kotlin.math.abs(a[i] - b[i]) > 1e-5f) {
                anyDiff = true
                break
            }
        }
        assertTrue(anyDiff, "PROPORTIONAL scaling with factor=2 must produce different outputs than NONE")
    }

    /**
     * Phase 5f.3 golden test. Verifies the new formula
     *   inv_freq[i] = 1 / base^(2i/headDim)  then  /= factor
     * matches the HF reference `_compute_proportional_rope_parameters`.
     *
     * At `base=10000, headDim=4, factor=2, position=0`, cos=1/sin=0 on
     * everything so the check is trivial at position 0. Use position=1
     * where cos/sin are non-trivial.
     *
     * Hand-derivation:
     *   inv_freq[0] = 1 / 10000^(0/4) = 1, then /2 = 0.5
     *   angle at pos=1: 0.5 → cos(0.5) = 0.8775825, sin(0.5) = 0.4794255
     *   inv_freq[1] = 1 / 10000^(2/4) = 1/100 = 0.01, then /2 = 0.005
     *   angle at pos=1: 0.005 → cos(0.005) ≈ 0.9999875, sin(0.005) ≈ 0.00499998
     *
     * Expect rotation of input (1,0,0,1) at pos=1 under INTERLEAVED pairing
     * to yield approximately:
     *   pair0 = (v0=1, v1=0): (cos*1 - sin*0, sin*1 + cos*0) = (0.8776, 0.4794)
     *   pair1 = (v0=0, v1=1): (cos*0 - sin*1, sin*0 + cos*1) ≈ (-0.005, 0.9999875)
     */
    @Test
    fun `PROPORTIONAL scaling formula matches reference inv_freq table`() {
        val rope = RoPE<FP32, Float>(
            headDim = 4, maxSeqLen = 4, base = 10000f,
            scaling = RoPEScaling.PROPORTIONAL, scalingFactor = 2.0f,
            mode = RoPEMode.INTERLEAVED
        )
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 4), FP32::class, floatArrayOf(1f, 0f, 0f, 1f)
        )
        val out = rope.forward(input, position = 1, ctx).data.copyToFloatArray()
        assertEquals(0.87758f, out[0], 1e-4f, "pair0[0] = cos(0.5)")
        assertEquals(0.47942f, out[1], 1e-4f, "pair0[1] = sin(0.5)")
        assertEquals(-0.005f, out[2], 1e-4f, "pair1[0] = -sin(0.005)")
        assertEquals(0.99999f, out[3], 1e-4f, "pair1[1] = cos(0.005)")
    }

    /**
     * Exponent denominator is `headDim`, not `rotaryDim`. Two configs that
     * have the same rotaryDim but different headDim must produce different
     * cos/sin tables under the new formula. The old NTK formula would have
     * made them identical.
     */
    @Test
    fun `inv_freq exponent denominator is headDim not rotaryDim`() {
        // Both have rotaryDim=4 (so halfRotary=2), but differ in headDim:
        // headDim=8,partial=0.5 vs headDim=4,partial=1.0.
        val big = RoPE<FP32, Float>(
            headDim = 8, maxSeqLen = 4, base = 10000f, partialRotaryFactor = 0.5f,
            mode = RoPEMode.INTERLEAVED
        )
        val small = RoPE<FP32, Float>(
            headDim = 4, maxSeqLen = 4, base = 10000f, partialRotaryFactor = 1.0f,
            mode = RoPEMode.INTERLEAVED
        )

        // At pos=1, inv_freq[1] in big = 1/10000^(2/8) = 1/10000^0.25 ≈ 0.1
        // At pos=1, inv_freq[1] in small = 1/10000^(2/4) = 1/10000^0.5 ≈ 0.01
        // So cos(0.1) ≈ 0.995 vs cos(0.01) ≈ 0.99995 — different.

        // Probe the first 4 (rotated) lanes on each. `big` needs headDim=8,
        // so the last 4 lanes are padding; `small` has headDim=4 already.
        val bigProbe = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 8), FP32::class, floatArrayOf(0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f)
        )
        val smallProbe = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 4), FP32::class, floatArrayOf(0f, 1f, 0f, 1f)
        )
        val bigOut = big.forward(bigProbe, position = 1, ctx).data.copyToFloatArray()
        val smallOut = small.forward(smallProbe, position = 1, ctx).data.copyToFloatArray()

        // Compare the first 4 floats (the rotated portion of each).
        var anyDiff = false
        for (i in 0 until 4) {
            if (kotlin.math.abs(bigOut[i] - smallOut[i]) > 1e-3f) {
                anyDiff = true
                break
            }
        }
        assertTrue(
            anyDiff,
            "headDim=8/partial=0.5 must produce different RoPE table than headDim=4/partial=1 " +
                "(would have been identical under the old NTK formula). " +
                "big[0..3]=${bigOut.toList().subList(0, 4)} small=${smallOut.toList()}"
        )
    }
}
