package sk.ainet.models.apertus

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

/**
 * xIELU activation reference implementation.
 *
 * Mutates [buf] in place applying:
 *
 *   alpha_p_eff = softplus(alpha_p)
 *   alpha_n_eff = beta + softplus(alpha_n)
 *
 *   if x > 0:  alpha_p_eff * x*x + beta * x
 *   else:      (expm1(min(x, eps)) - x) * alpha_n_eff + beta * x
 *
 * Apertus models carry per-layer scalar `alpha_p`, `alpha_n`, `beta`, `eps`
 * weights (see [ApertusXIELUParams]). The production network uses an
 * equivalent op emitted by `apertusNetwork()` through SKaiNET's compute
 * graph; this Kotlin reference exists so callers (notably tests) can
 * verify the math without spinning up a runtime, and so future xIELU
 * implementations have a single golden reference to point at.
 *
 * Public for unit testing.
 */
public fun xielu(buf: FloatArray, params: ApertusXIELUParams) {
    val alphaPEff = softplus(params.alphaP)
    val alphaNEff = params.beta + softplus(params.alphaN)

    for (i in buf.indices) {
        val x = buf[i]
        buf[i] = if (x > 0f) {
            alphaPEff * x * x + params.beta * x
        } else {
            val clamped = min(x, params.eps)
            (expm1(clamped) - x) * alphaNEff + params.beta * x
        }
    }
}

/**
 * `softplus(x) = ln(1 + exp(x))`, with the standard large-x asymptotic
 * shortcut to avoid `exp` overflow.
 */
public fun softplus(x: Float): Float {
    return if (x > 20f) x else ln(1f + exp(x))
}

/** `exp(x) - 1` without catastrophic cancellation near zero. */
private fun expm1(x: Float): Float = kotlin.math.expm1(x.toDouble()).toFloat()
