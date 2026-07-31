package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Conv1d
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine **v2** streaming audio FRONTEND in the SKaiNET NN DSL — the last vendor-ONNX graph, ported so
 * the whole v2 pipeline is self-compiled. Reproduces `frontend.onnx` (validated bit-exact by
 * `validate_moonshine_v2_frontend.py`, cos 1.0):
 *
 *   audio `[1, samples]`  (samples must be a multiple of [FRAME])
 *     → frame into contiguous [FRAME]-sample frames → `[n, FRAME]`   (200 Hz)
 *     → per-frame CMVN (eps 1e-6): (x − mean) / √(var + eps)
 *     → asinh(exp(log_k) · x) compression         (asinh(y) = ln(y + √(y²+1)))
 *     → filterbank matmul `[FRAME, DIM]`           → `[n, DIM]`
 *     → SiLU
 *     → causal conv1d(DIM→MID, k5, s2)  (left-pad 4) → SiLU     (100 Hz)
 *     → causal conv1d(MID→DIM, k5, s2)  (left-pad 4)            (50 Hz)
 *     → transpose → features `[1, frames, DIM]`   (the v2 encoder's input)
 *
 * Weights (baked by `bake_moonshine_v2_frontend.py`, weight-norm resolved to plain convs):
 *   `fe_filterbank.weight [FRAME,DIM]`, `fe_conv1.weight [MID,DIM,5]`+`.bias`,
 *   `fe_conv2.weight [DIM,MID,5]`+`.bias`, scalar `fe_log_k`.
 * The two convs are causal (left-pad k−1 = 4), matching the streaming ONNX's zeroed conv-state buffers.
 */
public class MoonshineV2Frontend<T : DType, V>(
    private val dtype: KClass<T>,
) : Module<T, V>(), ModuleParameters<T, V> {

    override val name: String = "moonshine_v2_frontend"

    private fun void(shape: Shape): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            @Suppress("UNCHECKED_CAST")
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )

    // Own params (filterbank + log_k); the two convs are sub-modules (their params bake by name).
    // Read via `.value` in forward — baking replaces the parameter's value, not the initial void tensor.
    private val fbParam = ModuleParameter.WeightParameter<T, V>("fe_filterbank.weight", void(Shape(FRAME, DIM)), false)
    private val logKParam = ModuleParameter.WeightParameter<T, V>("fe_log_k", void(Shape(1)), false)
    override val params: List<ModuleParameter<T, V>> = listOf(fbParam, logKParam)

    private val conv1 = Conv1d<T, V>(
        inChannels = DIM, outChannels = MID, kernelSize = KERNEL, stride = STRIDE, bias = true, name = "fe_conv1",
        initWeights = void(Shape(MID, DIM, KERNEL)), initBias = void(Shape(MID)),
    )
    private val conv2 = Conv1d<T, V>(
        inChannels = MID, outChannels = DIM, kernelSize = KERNEL, stride = STRIDE, bias = true, name = "fe_conv2",
        initWeights = void(Shape(DIM, MID, KERNEL)), initBias = void(Shape(DIM)),
    )
    override val modules: List<Module<T, V>> = listOf(conv1, conv2)

    /** Raw audio `[1, samples]` → v2 encoder features `[1, frames, DIM]`. */
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val audio = input.bind(ctx)
        val samples = audio.shape[audio.rank - 1]
        val n = samples / FRAME

        // frame → [n, FRAME]; per-frame CMVN over the FRAME axis.
        val fr = ops.reshape(audio, Shape(n, FRAME))
        val mean = ops.unsqueeze(ops.mean(fr, dim = 1), 1)                          // [n,1]
        val centered = ops.subtract(fr, mean)                                       // [n,FRAME]
        val variance = ops.unsqueeze(ops.mean(ops.multiply(centered, centered), dim = 1), 1)
        val normed = ops.divide(centered, ops.sqrt(ops.addScalar(variance, EPS)))    // [n,FRAME]

        // asinh(exp(log_k) · normed) = ln(y + √(y²+1)), y = exp(log_k)·normed
        val y = ops.multiply(normed, ops.exp(logKParam.value))                       // [n,FRAME] (broadcast [1])
        val asinh = ops.log(ops.add(y, ops.sqrt(ops.addScalar(ops.multiply(y, y), 1.0f))))

        // filterbank + SiLU → [n, DIM]; to conv layout [1, DIM, n].
        var x = ops.silu(ops.matmul(asinh, fbParam.value))                           // [n, DIM]
        x = ops.unsqueeze(ops.transpose(x), 0)                                       // [n,DIM]→[DIM,n]→[1,DIM,n]

        // causal conv1 (left-pad 4) → SiLU; causal conv2 (left-pad 4).
        x = ops.silu(conv1.forward(ops.concat(listOf(zeros(ctx, DIM), x), dim = 2), ctx))
        x = conv2.forward(ops.concat(listOf(zeros(ctx, MID), x), dim = 2), ctx)      // [1, DIM, frames]

        // [1, DIM, frames] → [1, frames, DIM] (rank-3 transpose reverses all dims → squeeze/2D-swap/unsqueeze).
        return ops.unsqueeze(ops.transpose(ops.squeeze(x, 0)), 0)
    }

    // A causal left-pad of [PAD] zero frames as a constant `[1, channels, PAD]`.
    private fun zeros(ctx: ExecutionContext, channels: Int): Tensor<T, V> =
        ctx.fromFloatArray(Shape(1, channels, PAD), dtype, FloatArray(channels * PAD))

    private companion object {
        const val FRAME = 80
        const val DIM = 320
        const val MID = 640
        const val KERNEL = 5
        const val STRIDE = 2
        const val PAD = KERNEL - 1   // causal left-pad
        const val EPS = 1e-6f
    }
}

/** Build the Moonshine v2 audio frontend in the NN DSL. Input `[1, samples]` (multiple of 80) → `[1, frames, 320]`. */
public fun <T : DType, V> moonshineV2Frontend(dtype: KClass<T>): MoonshineV2Frontend<T, V> =
    MoonshineV2Frontend(dtype)
