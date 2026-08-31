package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Gemma 4 per-block PLE (Per-Layer Embedding) side-channel.
 *
 * Runs at the tail of every decoder layer when `hidden_size_per_layer_input`
 * is set in the config (true for all Gemma 4 variants). Exact forward from
 * `Gemma4TextDecoderLayer.forward` (transformers 5.6.0):
 *
 * ```python
 * if self.hidden_size_per_layer_input:
 *     residual = hidden_states
 *     hidden_states = self.per_layer_input_gate(hidden_states)   # inp_gate   1536 -> 256
 *     hidden_states = self.act_fn(hidden_states)                 # gelu_pytorch_tanh
 *     hidden_states = hidden_states * per_layer_input            # pointwise  [B,S,256]
 *     hidden_states = self.per_layer_projection(hidden_states)   # proj        256 -> 1536
 *     hidden_states = self.post_per_layer_input_norm(hidden_states)  # post_norm RMSNorm
 *     hidden_states = residual + hidden_states
 * ```
 *
 * GGUF tensor names per block:
 * - `blk.N.inp_gate.weight`   → [perLayerDim, hiddenSize] = [256, 1536]
 * - `blk.N.proj.weight`       → [hiddenSize, perLayerDim] = [1536, 256]
 * - `blk.N.post_norm.weight`  → [hiddenSize]              = [1536]
 *
 * [perLayerInput] is set by [GemmaModel.onForward] before each block's
 * forward pass — it holds the `[B, S, perLayerDim]` slice of the pre-computed
 * `per_layer_inputs[:, :, layerIdx, :]` tensor (see [PerLayerEmbedding]).
 *
 * If [perLayerInput] is unset (null), the module is an identity passthrough —
 * matches what happens when a toy-model or other non-Gemma-4 checkpoint runs
 * through a network where PLE was enabled but the outer GemmaModel didn't
 * populate the hook. Should not happen on a real Gemma 4 forward.
 */
@Suppress("UNCHECKED_CAST")
public class PerLayerInputBlockHook<T : DType, V>(
    public val hiddenSize: Int,
    public val perLayerDim: Int,
    /**
     * Phase 5f.6 diagnostic. When `true`, the hook runs the full PLE
     * branch (gate + act + pli-mul + proj + norm) but DISCARDS the
     * result and returns `input` unchanged. Lets us test the hypothesis
     * that the `residual + x` add is the wrong op on single-stream
     * Gemma 4 (HF Gemma 3n adds PLE to inactive AltUp streams `[1:]`,
     * not the main stream — stripping AltUp may mean PLE is vestigial).
     * See `memory/ple_bug_hypothesis.md`.
     */
    public val sideChannelOnly: Boolean = false,
    override val name: String = "PerLayerInputBlockHook"
) : Module<T, V>(), ModuleParameters<T, V> {

    /**
     * Pre-set by the outer [GemmaModel] before each block runs. Shape
     * `[batch, seqLen, perLayerDim]` — this layer's slice of the
     * top-level `per_layer_inputs` tensor. Consumed (zeroed after read)
     * on each forward so cross-forward leaks of stale state are obvious.
     */
    public var perLayerInput: Tensor<T, V>? = null

    private fun voidWeight(shape: Shape): VoidOpsTensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        Any::class as KClass<T>
    )

    public val postNorm: RMSNormalization<T, V> =
        RMSNormalization(intArrayOf(hiddenSize), name = "$name.post_norm")

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter(
            "$name.inp_gate.weight",
            voidWeight(Shape(perLayerDim, hiddenSize))
        ),
        ModuleParameter.WeightParameter(
            "$name.proj.weight",
            voidWeight(Shape(hiddenSize, perLayerDim))
        )
    )

    override val modules: List<Module<T, V>> = listOf(postNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val pli = perLayerInput
            ?: return input  // Pass-through when the outer wrapper didn't set it.
        perLayerInput = null
        val ops = ctx.ops
        val residual = input
        val wInpGate = params[0].value
        val wProj = params[1].value
        val prof = sk.ainet.lang.nn.transformer.PhaseProfile
        var x = prof.time("ple.gate") { linearProject(ops, input, wInpGate) }   // [B,S,1536] → [B,S,256]
        x = prof.time("ple.act") { ops.gelu(x) }                                // gelu_pytorch_tanh
        x = prof.time("ple.mul") { ops.multiply(x, pli) }                       // pointwise with per_layer_input
        x = prof.time("ple.proj") { linearProject(ops, x, wProj) }              // [B,S,256] → [B,S,1536]
        x = postNorm.forward(x, ctx)                                            // RMSNorm at hidden_size
        if (sideChannelOnly) return residual           // Phase 5f.6 diagnostic toggle.
        return prof.time("ple.residual") { ops.add(residual, x) }
    }
}
