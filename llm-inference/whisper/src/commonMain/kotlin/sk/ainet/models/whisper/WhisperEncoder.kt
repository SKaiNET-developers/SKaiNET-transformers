package sk.ainet.models.whisper

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Conv1d
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Whisper ENCODER in the SKaiNET NN DSL, built at a configurable (short) audio
 * context — the "4-second window" of the Android pipeline (`audioCtx=200`,
 * mel `[1, 80, 400]` → features `[1, 200, 384]`).
 *
 *   conv1 (nMels→dim, k=3, s=1, p=1) → GELU
 *   conv2 (dim→dim,  k=3, s=2, p=1) → GELU        (NCW, stride-2 halves frames)
 *   → permute `[1, audioCtx, dim]` → + positional embedding
 *   → [encoderLayers] pre-norm blocks (biased attention except k_proj, GELU MLP)
 *   → ln_post
 *
 * The positional embedding is a PARAMETER (`enc_pos.weight` `[audioCtx, dim]`):
 * the weight baker feeds the checkpoint's `encoder.embed_positions.weight`
 * sliced to `[0:audioCtx]` — Whisper's encoder positions are a deterministic
 * sinusoid, so the slice IS the short-context table (the load-bearing trick
 * validated by the ONNX pipeline; a diagnostic assert compares it against
 * freshly computed sinusoids at bake time).
 */
public class WhisperEncoderLayer<T : DType, V>(
    cfg: WhisperConfig,
    layer: Int,
    dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "enc.$layer"

    private val attnNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(cfg.dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "enc.$layer.attn_norm",
        dtype = dtype,
    )
    public val attn: WhisperAttentionProjections<T, V> =
        WhisperAttentionProjections("enc.$layer.attn", cfg.dim, cfg.nHeads, cfg.headDim, dtype)
    private val mlpNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(cfg.dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "enc.$layer.mlp_norm",
        dtype = dtype,
    )
    private val fc1 = sk.ainet.lang.nn.transformer.VoidDense<T, V>("enc.$layer.fc1", cfg.ffnDim, cfg.dim, dtype, addBias = true)
    private val fc2 = sk.ainet.lang.nn.transformer.VoidDense<T, V>("enc.$layer.fc2", cfg.dim, cfg.ffnDim, dtype, addBias = true)

    override val modules: List<Module<T, V>> = listOf(attnNorm, attn, mlpNorm, fc1, fc2)

    private val seq = cfg.audioCtx

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val x = input.bind(ctx)
        val sn = attnNorm.forward(x, ctx)
        val k = linearProject(sn, attn.kProj, ctx)
        val v = linearProjectBias(attn.vProj, sn, ctx)
        val afterAttn = ops.add(x, attn.attend(sn, k, v, seq, seq, mask = null, ctx = ctx))
        val h = ops.gelu(linearProjectBias(fc1, mlpNorm.forward(afterAttn, ctx), ctx))
        return ops.add(afterAttn, linearProjectBias(fc2, h, ctx))
    }

    private fun linearProject(
        x: Tensor<T, V>,
        dense: sk.ainet.lang.nn.transformer.VoidDense<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> = sk.ainet.lang.nn.transformer.linearProject(ctx.ops, x, dense.params[0].value)
}

public class WhisperEncoderModel<T : DType, V>(
    private val cfg: WhisperConfig,
    private val dtype: KClass<T>,
) : Module<T, V>(), sk.ainet.lang.nn.topology.ModuleParameters<T, V> {

    override val name: String = "whisper_encoder"

    private fun void(vararg dims: Int): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = Shape(*dims)
            @Suppress("UNCHECKED_CAST")
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )

    private val conv1 = Conv1d<T, V>(
        inChannels = cfg.nMels, outChannels = cfg.dim, kernelSize = 3, stride = 1, padding = 1,
        bias = true, name = "enc_conv1",
        initWeights = void(cfg.dim, cfg.nMels, 3), initBias = void(cfg.dim),
    )
    private val conv2 = Conv1d<T, V>(
        inChannels = cfg.dim, outChannels = cfg.dim, kernelSize = 3, stride = 2, padding = 1,
        bias = true, name = "enc_conv2",
        initWeights = void(cfg.dim, cfg.dim, 3), initBias = void(cfg.dim),
    )
    private val layers: List<WhisperEncoderLayer<T, V>> =
        (0 until cfg.encoderLayers).map { WhisperEncoderLayer(cfg, it, dtype) }
    private val lnPost = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(cfg.dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "enc_ln_post",
        dtype = dtype,
    )

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("enc_pos.weight", void(cfg.audioCtx, cfg.dim)),
    )

    override val modules: List<Module<T, V>> = listOf(conv1, conv2) + layers + listOf(lnPost)

    /** mel `[1, nMels, melFrames]` → features `[1, audioCtx, dim]`. */
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        var x = ops.gelu(conv1.forward(input.bind(ctx), ctx))
        x = ops.gelu(conv2.forward(x, ctx))                  // [1, dim, audioCtx]
        x = ops.permute(x, intArrayOf(0, 2, 1))              // [1, audioCtx, dim]
        val pos = ops.reshape(params[0].value, Shape(1, cfg.audioCtx, cfg.dim))
        x = ops.add(x, pos)
        for (layer in layers) x = layer.forward(x, ctx)
        return lnPost.forward(x, ctx)
    }
}
