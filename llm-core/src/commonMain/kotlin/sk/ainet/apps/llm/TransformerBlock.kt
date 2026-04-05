package sk.ainet.apps.llm

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Sequential module that supports residual (skip) connections.
 *
 * Unlike the generic [MLP][sk.ainet.lang.nn.topology.MLP], this module understands
 * [ResidualAdd] and sets its `savedInput` before executing each residual block.
 *
 * A "residual block" is the group of modules between two [ResidualAdd] boundaries
 * (or between the start and the first [ResidualAdd]). For each [ResidualAdd],
 * `savedInput` is set to the tensor at the start of its block, implementing the
 * standard transformer skip connection: `output = sublayer(x) + x`.
 *
 * Example module list for a transformer layer:
 * ```
 * [rmsNorm, multiHeadAttention, ResidualAdd, rmsNorm, swiGluFFN, ResidualAdd]
 *  ^--- block 1 start --------^              ^--- block 2 start ---^
 * ```
 * - ResidualAdd[0].savedInput = stage input (before rmsNorm)
 * - ResidualAdd[1].savedInput = output of ResidualAdd[0] (after first skip add)
 */
public class TransformerBlock<T : DType, V>(
    private val modulesList: List<Module<T, V>>,
    override val name: String = "TransformerBlock"
) : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = modulesList

    // Precompute residual block boundaries.
    // For each ResidualAdd at index i, the "block start" is the module right after
    // the previous ResidualAdd (or index 0).
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
        // Track the tensor at each module boundary so we can look up
        // the value at any residual block start index.
        val outputs = arrayOfNulls<Any>(modulesList.size + 1)
        outputs[0] = input
        var tmp = input
        for (i in modulesList.indices) {
            val module = modulesList[i]
            val blockStart = residualBlockStarts[i]
            if (blockStart != null) {
                @Suppress("UNCHECKED_CAST")
                (module as ResidualAdd<T, V>).savedInput =
                    outputs[blockStart] as Tensor<T, V>
            }
            tmp = module.forward(tmp, ctx)
            outputs[i + 1] = tmp
        }
        return tmp
    }
}
