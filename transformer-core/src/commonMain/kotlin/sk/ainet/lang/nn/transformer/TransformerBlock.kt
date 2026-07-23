package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Eager transformer block: a residual-aware container over an ordered list of sub-modules.
 *
 * A block is a flat module list punctuated by [ResidualAdd] markers, e.g.
 * `[norm, attn, residual(), norm, ffn, residual()]`. On [onForward] the input to each residual
 * sub-block is captured at the sub-block's start and handed to that block's [ResidualAdd] so it
 * can add the skip connection — exactly the wiring the network DSL's `residual()` expects.
 *
 * This depends only on `lang-core` + `transformer-core` (no compiler, no io, no platform
 * diagnostics), so it builds on the full KMP matrix **including androidNative** — the on-device
 * (edge NPU / phone) target set. It is the eager core that `HybridTransformerBlock` (llm-core)
 * wraps with its optional JVM/linux compiled fast-path; models that only need eager execution
 * (and export unchanged to StableHLO) should use this directly.
 */
public open class TransformerBlock<T : DType, V>(
    private val modulesList: List<Module<T, V>>,
    override val name: String = "TransformerBlock",
) : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = modulesList

    // For each ResidualAdd at index i, the index at which its residual sub-block began (the module
    // after the previous ResidualAdd, or 0). The block's input is the skip added at that residual.
    private val residualBlockStarts: Map<Int, Int> = buildMap {
        var blockStart = 0
        for (i in modulesList.indices) {
            if (modulesList[i] is ResidualAdd<*, *>) {
                put(i, blockStart)
                blockStart = i + 1
            }
        }
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val outputs = arrayOfNulls<Any>(modulesList.size + 1)
        outputs[0] = input
        var tmp = input
        for (i in modulesList.indices) {
            val module = modulesList[i]
            residualBlockStarts[i]?.let { blockStart ->
                @Suppress("UNCHECKED_CAST")
                (module as ResidualAdd<T, V>).savedInput = outputs[blockStart] as Tensor<T, V>
            }
            tmp = module.forward(tmp, ctx)
            outputs[i + 1] = tmp
        }
        return tmp
    }
}
