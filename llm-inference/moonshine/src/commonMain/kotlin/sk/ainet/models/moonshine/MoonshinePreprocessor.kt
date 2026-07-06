package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Conv1d
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.GroupNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.tensor.tanh
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine-tiny audio FRONTEND (preprocessor) in the SKaiNET NN DSL — the last piece needed for a
 * solely-SKaiNET pipeline (was a vendor ONNX graph). Verified against `enc_frontend.onnx`:
 *
 *   input `[1, samples]`
 *     → unsqueeze → `[1, 1, samples]`
 *     → conv1 (1→dim,   k=127, s=64, no bias) → tanh
 *     → groupnorm (1 group, affine, eps=1e-5)
 *     → conv2 (dim→2·dim, k=7, s=3) → gelu
 *     → conv3 (2·dim→dim, k=3, s=2) → gelu
 *     → transpose → `[1, frames, dim]`  (the encoder's input; 80000 samples → 207 frames)
 *
 * (The ONNX `Reshape+InstanceNormalization+Reshape+Mul+Add` is exactly `nn.GroupNorm(1, dim)` +
 * affine; the `Div/Erf/Add/Mul/Mul` blocks are erf-GELU — the DSL GELU is the tanh approximation,
 * a ~1e-3 difference the downstream encoder tolerates.)
 *
 * Weights (from the HF encoder checkpoint): `encoder.conv{1,2,3}.weight`, `conv{2,3}.bias`,
 * `encoder.groupnorm.weight/bias`. Build with the model dtype; bake before tracing to export.
 */
public class MoonshinePreprocessor<T : DType, V>(
    cfg: MoonshineConfig,
    dtype: KClass<T>,
    // When true, emit the conv output `[1, dim, frames]` WITHOUT the final transpose — the layout the
    // board pipeline feeds to the (board-layout) encoder, so this is a drop-in for the vendor
    // `preprocessor_cpu.vmfb`. Default false = `[1, frames, dim]` (matches `enc_frontend.onnx`).
    private val boardLayout: Boolean = false,
) : Module<T, V>() {

    override val name: String = "moonshine_preprocessor"
    private val dim = cfg.dim

    private fun void(shape: Shape, dtype: KClass<T>): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            @Suppress("UNCHECKED_CAST")
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )

    private val conv1 = Conv1d<T, V>(
        inChannels = 1, outChannels = dim, kernelSize = cfg.conv1Kernel, stride = cfg.conv1Stride,
        bias = false, name = "conv1", initWeights = void(Shape(dim, 1, cfg.conv1Kernel), dtype),
    )
    private val groupnorm = GroupNormalization<T, V>(
        numGroups = 1, numChannels = dim, eps = cfg.layerNormEps.toDouble(), affine = true,
        name = "groupnorm", dtype = dtype,
    )
    private val conv2 = Conv1d<T, V>(
        inChannels = dim, outChannels = cfg.conv2Out, kernelSize = cfg.conv2Kernel, stride = cfg.conv2Stride,
        bias = true, name = "conv2",
        initWeights = void(Shape(cfg.conv2Out, dim, cfg.conv2Kernel), dtype), initBias = void(Shape(cfg.conv2Out), dtype),
    )
    private val conv3 = Conv1d<T, V>(
        inChannels = cfg.conv2Out, outChannels = dim, kernelSize = cfg.conv3Kernel, stride = cfg.conv3Stride,
        bias = true, name = "conv3",
        initWeights = void(Shape(dim, cfg.conv2Out, cfg.conv3Kernel), dtype), initBias = void(Shape(dim), dtype),
    )

    override val modules: List<Module<T, V>> = listOf(conv1, groupnorm, conv2, conv3)

    private val eps = cfg.layerNormEps

    /** Raw audio `[1, samples]` → encoder features `[1, frames, dim]`. */
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val x0 = input.bind(ctx)
        var x = ops.unsqueeze(x0, 1)                 // [1, 1, samples]
        x = conv1.forward(x, ctx).tanh()             // [1, dim, L1]
        x = groupNorm1(x, ctx)                       // [1, dim, L1]
        x = conv2.forward(x, ctx).gelu()             // [1, 2·dim, L2]
        x = conv3.forward(x, ctx).gelu()             // [1, dim, frames]
        if (boardLayout) return x                    // board feeds [1, dim, frames] to the encoder
        // [1, dim, frames] → [1, frames, dim] via a 2D swap (rank-3 ops.transpose reverses all dims).
        return ops.unsqueeze(ops.transpose(ops.squeeze(x, 0)), 0)
    }

    // GroupNorm(1 group): normalise over (channels, length) jointly, then per-channel affine. The core
    // GroupNormalization.forward is a stub (reshapeForGroups NotImplemented), so compute it with ops;
    // gamma/beta stay in the [groupnorm] module so they bake by name.
    private fun groupNorm1(x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val c = x.shape[1]
        val l = x.shape[2]
        val flat = ops.reshape(x, Shape(1, c * l))
        val mean = ops.unsqueeze(ops.mean(flat, dim = 1), 1)                       // [1,1]
        val centered = ops.subtract(flat, mean)
        val variance = ops.unsqueeze(ops.mean(ops.multiply(centered, centered), dim = 1), 1)
        val normed = ops.reshape(ops.divide(centered, ops.sqrt(ops.addScalar(variance, eps))), Shape(1, c, l))
        val gamma = ops.reshape(groupnorm.params[0].value, Shape(1, c, 1))
        val beta = ops.reshape(groupnorm.params[1].value, Shape(1, c, 1))
        return ops.add(ops.multiply(normed, gamma), beta)
    }
}

/** Build the Moonshine-tiny audio frontend in the NN DSL. */
public fun <T : DType, V> moonshinePreprocessor(
    cfg: MoonshineConfig,
    dtype: KClass<T>,
    boardLayout: Boolean = false,
): MoonshinePreprocessor<T, V> = MoonshinePreprocessor(cfg, dtype, boardLayout)
