package sk.ainet.models.gemma

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.test.Test

/**
 * Does SKaiNET's SPLIT_HALF RoPE match llama.cpp's GGML_ROPE_TYPE_NEOX
 * rotation numerically? NEOX pairs x[i] with x[i+d/2], angle = pos*base^(-2i/d):
 *   out[i]     = x[i]*cos - x[i+d/2]*sin
 *   out[i+d/2] = x[i]*sin + x[i+d/2]*cos
 * (gemma3 uses NEOX rope; GGUF stores Q/K in HF layout, no permute for NEOX.)
 */
class RopeNeoxParityTest {
    @Test
    fun splitHalfMatchesNeox() {
        val headDim = 8
        val seq = 3
        val base = 10000.0f
        val ctx = DirectCpuExecutionContext.create()
        val rope = RoPE<FP32, Float>(headDim = headDim, maxSeqLen = 8, base = base, mode = RoPEMode.SPLIT_HALF)

        // input: 1 head, seq positions, headDim — distinct values per slot
        val data = FloatArray(seq * headDim) { (it + 1).toFloat() }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, seq, headDim), FP32::class, data)
        val out = rope.forward(input, position = 0, ctx as ExecutionContext).data.copyToFloatArray()

        val half = headDim / 2
        val invFreq = FloatArray(half) { i -> 1.0f / base.pow(2.0f * i / headDim) }
        var maxErr = 0.0f
        for (s in 0 until seq) {
            for (i in 0 until half) {
                val a = s * invFreq[i]
                val c = cos(a); val sn = sin(a)
                val xi = data[s * headDim + i]
                val xj = data[s * headDim + i + half]
                val expEven = xi * c - xj * sn
                val expOdd = xi * sn + xj * c
                val gotEven = out[s * headDim + i]
                val gotOdd = out[s * headDim + i + half]
                maxErr = maxOf(maxErr, kotlin.math.abs(expEven - gotEven), kotlin.math.abs(expOdd - gotOdd))
                if (s <= 1 && i < 2) println("s=$s i=$i  even exp=$expEven got=$gotEven  odd exp=$expOdd got=$gotOdd")
            }
        }
        println("ROPE_MAXERR $maxErr")
    }
}
