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
 * prefix of head_dim) and proportional (NTK-aware) scaling. The Gemma 4
 * global-attention layer combines both: partialRotaryFactor=0.5 and
 * PROPORTIONAL scaling with factor=2.0.
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
}
