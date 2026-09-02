package sk.ainet.models.gemma3n

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/*
 * DSL modules for the Gemma 3n-specific machinery (the #377 DSL migration): AltUp, Laurel,
 * activation-sparsity FFN and the per-layer-input application. All math goes through
 * `ctx.ops` so the modules are traceable for the StableHLO → IREE export path. The
 * reference implementation is HF `transformers` `modeling_gemma3n.py` (verified against
 * the installed 5.x source); working tensors are rank-2 `[seq, hidden]`.
 */

@Suppress("UNCHECKED_CAST")
internal fun <T : DType, V> voidParam(name: String, shape: Shape, dtype: KClass<T>?): ModuleParameter<T, V> =
    ModuleParameter.WeightParameter(
        name,
        VoidOpsTensor(
            object : TensorData<T, V> {
                override val shape: Shape = shape
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>,
        ),
    )

/**
 * Per-layer AltUp (Alternating Updates) block — HF `Gemma3nTextAltUp`.
 *
 * Maintains `numInputs` parallel hidden streams; only the active one runs the expensive
 * transformer sub-layers, the rest are predicted/corrected via a learned router:
 *
 * ```
 * modalities(x) = tanh( modality_router( router_norm(x) * 1/hidden ) )
 * predict:  coefs = prediction_coefs(modalities)            # [S, n²]
 *           pred_i = h_i + Σ_j coefs[:, i·n+j] ⊙ h_j
 * correct:  coefs = correction_coefs(modalities(activated)) + 1
 *           innovation = activated − pred_active
 *           corr_i = pred_i + coefs[:, i] ⊙ innovation
 * scale_corrected_output(x) = x * correct_output_scale      # [hidden]
 * ```
 */
public class Gemma3nAltUpBlock<T : DType, V>(
    private val hiddenSize: Int,
    private val numInputs: Int,
    private val activeIdx: Int,
    rmsEps: Float,
    private val dtype: KClass<T>? = null,
    override val name: String = "altup",
) : Module<T, V>(), ModuleParameters<T, V> {

    /** `router_norm` — scale-full RMSNorm over hidden, plain (non-unit-offset) weight. */
    public val routerNorm: RMSNormalization<T, V> = RMSNormalization(
        intArrayOf(hiddenSize), rmsEps.toDouble(), unitOffset = false, name = "$name.altup_router_norm", dtype = dtype,
    )

    override val params: List<ModuleParameter<T, V>> = listOf(
        voidParam("$name.altup_router.weight", Shape(numInputs, hiddenSize), dtype),
        voidParam("$name.altup_predict_coef.weight", Shape(numInputs * numInputs, numInputs), dtype),
        voidParam("$name.altup_correct_coef.weight", Shape(numInputs, numInputs), dtype),
        voidParam("$name.altup_correct_scale.weight", Shape(hiddenSize), dtype),
    )

    override val modules: List<Module<T, V>> = listOf(routerNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = input

    private fun modalities(x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val normed = routerNorm.forward(x, ctx)
        val scaled = ops.mulScalar(normed, 1.0f / hiddenSize)
        return ops.tanh(linearProject(ops, scaled, params[0].value))    // [S, n]
    }

    /** One learned scalar column `[.., 1]` broadcast-multiplied over `[.., H]` —
     *  rank-general (the trunk runs rank-2 `[S, H]` eagerly, rank-3 `[B, S, H]` under
     *  the export trace). */
    private fun scaleBy(coefs: Tensor<T, V>, col: Int, x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        ctx.ops.multiply(x, ctx.ops.narrow(coefs, dim = coefs.rank - 1, start = col, length = 1))

    public fun predict(streams: List<Tensor<T, V>>, ctx: ExecutionContext): List<Tensor<T, V>> {
        val ops = ctx.ops
        val m = modalities(streams[activeIdx], ctx)
        val coefs = linearProject(ops, m, params[1].value)              // [S, n²]
        return List(numInputs) { i ->
            var pred = streams[i]
            for (j in 0 until numInputs) {
                pred = ops.add(pred, scaleBy(coefs, i * numInputs + j, streams[j], ctx))
            }
            pred
        }
    }

    public fun correct(
        predictions: List<Tensor<T, V>>,
        activated: Tensor<T, V>,
        ctx: ExecutionContext,
    ): List<Tensor<T, V>> {
        val ops = ctx.ops
        val m = modalities(activated, ctx)
        val coefs = ops.addScalar(linearProject(ops, m, params[2].value), 1.0f)   // [S, n]
        val innovation = ops.subtract(activated, predictions[activeIdx])
        return List(numInputs) { i ->
            ops.add(predictions[i], scaleBy(coefs, i, innovation, ctx))
        }
    }

    /** `x * correct_output_scale` (element-wise over hidden). */
    public fun scaleCorrectedOutput(x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        ctx.ops.multiply(x, params[3].value)
}

/**
 * Model-level AltUp stream projections — HF `altup_projections` / `altup_unembed_projections`.
 * The GGUF stores each set as ONE 3D tensor (`altup_proj.weight`, `altup_unembd_proj.weight`,
 * logical `[numInputs-1, hidden, hidden]`); slices are narrowed out at forward time.
 *
 * Both directions renormalize the projected stream to the active stream's per-token RMS
 * magnitude (HF: `target_magnitude / max(rms(proj), 1e-5)`).
 */
public class Gemma3nAltUpGlobals<T : DType, V>(
    private val hiddenSize: Int,
    private val numInputs: Int,
    private val dtype: KClass<T>? = null,
    override val name: String = "altup_globals",
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = listOf(
        voidParam("$name.altup_proj.weight", Shape(numInputs - 1, hiddenSize, hiddenSize), dtype),
        voidParam("$name.altup_unembd_proj.weight", Shape(numInputs - 1, hiddenSize, hiddenSize), dtype),
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = input

    /** Per-token RMS magnitude `[S, 1]`: `sqrt(mean(x², dim=-1))`. */
    private fun magnitude(x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val meanSq = ops.mean(ops.multiply(x, x), dim = -1)             // [S]
        return ops.unsqueeze(ops.sqrt(meanSq), dim = -1)                // [S, 1]
    }

    private fun sliceOf(param: ModuleParameter<T, V>, k: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        // The GGUF stores the stack as ne=[hidden, hidden, numExtra] (ggml: ne2 slowest), and
        // the engine surfaces the raw buffer under that ne-ordered shape — so the slice index
        // is SLOWEST in memory. Reinterpret row-major as [numExtra, hidden, hidden] first,
        // then narrow the leading dim; each slice's buffer is the converter's [out, in]
        // row-major matrix.
        val stacked = ops.reshape(param.value, Shape(numInputs - 1, hiddenSize, hiddenSize))
        val sliced = ops.narrow(stacked, dim = 0, start = k, length = 1)
        return ops.reshape(sliced, Shape(hiddenSize, hiddenSize))
    }

    private fun projectRenormed(
        x0mag: Tensor<T, V>,
        stream: Tensor<T, V>,
        param: ModuleParameter<T, V>,
        k: Int,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        val proj = linearProject(ops, stream, sliceOf(param, k, ctx))
        val newMagSq = ops.mean(ops.multiply(proj, proj), dim = -1)      // [S]
        val newMag = ops.unsqueeze(ops.sqrt(ops.clamp(newMagSq, 1e-5f, Float.MAX_VALUE)), dim = -1)
        return ops.multiply(proj, ops.divide(x0mag, newMag))
    }

    /** HF stream init: `[h0] + [renorm(altup_projections[k](h0))]`. */
    public fun initStreams(h0: Tensor<T, V>, ctx: ExecutionContext): List<Tensor<T, V>> {
        val mag = magnitude(h0, ctx)
        return listOf(h0) + List(numInputs - 1) { k -> projectRenormed(mag, h0, params[0], k, ctx) }
    }

    /** HF finalize: mean of `[h0] + [renorm(altup_unembed_projections[k](h_k+1))]`. */
    public fun mergeStreams(streams: List<Tensor<T, V>>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val mag = magnitude(streams[0], ctx)
        var acc = streams[0]
        for (k in 0 until numInputs - 1) {
            acc = ops.add(acc, projectRenormed(mag, streams[k + 1], params[1], k, ctx))
        }
        return ops.mulScalar(acc, 1.0f / numInputs)
    }
}

/**
 * Laurel (Learned Augmented Residual Layer) — HF `Gemma3nTextLaurelBlock`:
 * `x + post_laurel_norm(linear_right(linear_left(x)))`.
 */
public class Gemma3nLaurelBlock<T : DType, V>(
    hiddenSize: Int,
    laurelRank: Int,
    rmsEps: Float,
    dtype: KClass<T>? = null,
    override val name: String = "laurel",
) : Module<T, V>() {

    public val linearLeft: VoidDense<T, V> = VoidDense("$name.laurel_l", laurelRank, hiddenSize, dtype)
    public val linearRight: VoidDense<T, V> = VoidDense("$name.laurel_r", hiddenSize, laurelRank, dtype)
    public val postNorm: RMSNormalization<T, V> = RMSNormalization(
        intArrayOf(hiddenSize), rmsEps.toDouble(), unitOffset = false, name = "$name.laurel_post_norm", dtype = dtype,
    )

    override val modules: List<Module<T, V>> = listOf(linearLeft, linearRight, postNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val low = linearLeft.forward(input, ctx)
        val back = linearRight.forward(low, ctx)
        return ctx.ops.add(input, postNorm.forward(back, ctx))
    }
}

/**
 * Gemma 3n FFN — gelu-gated (`down(gelu(gate(x)) * up(x))`) with optional Gaussian-top-k
 * activation sparsity on the gate projection (HF `Gemma3nTextMLP._gaussian_topk`):
 *
 * ```
 * cutoff = mean(gate, -1) + std_pop(gate, -1) * stdMultiplier
 * gate   = relu(gate - cutoff)
 * ```
 *
 * `stdMultiplier` comes precomputed per layer from the GGUF (`activation_sparsity_scale`,
 * `Φ⁻¹(0.95) ≈ 1.6449` on sparse layers, `-inf` on the rest — non-finite disables the
 * whole branch at build time). Std is population (unbiased=False): `sqrt(E[x²] − E[x]²)`.
 */
public class Gemma3nSparseGeGluFFN<T : DType, V>(
    hiddenSize: Int,
    ffnDim: Int,
    private val stdMultiplier: Float,
    dtype: KClass<T>? = null,
    override val name: String = "ffn",
) : Module<T, V>() {

    // Param names follow the llama/HF convention the engine resolver maps to
    // `blk.N.ffn_{gate,up,down}.weight`.
    public val gate: VoidDense<T, V> = VoidDense("$name.gate_proj", ffnDim, hiddenSize, dtype)
    public val up: VoidDense<T, V> = VoidDense("$name.up_proj", ffnDim, hiddenSize, dtype)
    public val down: VoidDense<T, V> = VoidDense("$name.down_proj", hiddenSize, ffnDim, dtype)

    public val sparsityEnabled: Boolean = stdMultiplier.isFinite() && stdMultiplier > 0f

    override val modules: List<Module<T, V>> = listOf(gate, up, down)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        var g = gate.forward(input, ctx)
        if (sparsityEnabled) {
            val mean = ops.mean(g, dim = -1)                               // [S]
            val meanSq = ops.mean(ops.multiply(g, g), dim = -1)            // [S]
            val varPop = ops.subtract(meanSq, ops.multiply(mean, mean))
            val std = ops.sqrt(ops.clamp(varPop, 0f, Float.MAX_VALUE))
            val cutoff = ops.unsqueeze(
                ops.add(mean, ops.mulScalar(std, stdMultiplier)), dim = -1,
            )                                                              // [S, 1]
            g = ops.relu(ops.subtract(g, cutoff))
        }
        val activated = ops.gelu(g)
        val upOut = up.forward(input, ctx)
        return down.forward(ops.multiply(activated, upOut), ctx)
    }
}

/**
 * Per-layer-input application — the tail of HF `Gemma3nTextDecoderLayer.forward`.
 * Takes the (scaled) corrected active stream and this layer's `per_layer_input` slice,
 * returns the DELTA that gets added to the non-active streams:
 * `post_norm( proj( gelu(inp_gate(x)) ⊙ per_layer_input ) )`.
 *
 * The gemma-4 lane's `PerLayerInputBlockHook` applies the same transform but adds it to
 * the main residual (gemma-4 has no AltUp streams); gemma3n adds it to streams `1..n-1`,
 * so this module returns the delta and `Gemma3nModel` does the stream adds.
 */
public class Gemma3nPerLayerApply<T : DType, V>(
    hiddenSize: Int,
    perLayerDim: Int,
    rmsEps: Float,
    dtype: KClass<T>? = null,
    override val name: String = "per_layer_input",
) : Module<T, V>() {

    public val inpGate: VoidDense<T, V> = VoidDense("$name.inp_gate", perLayerDim, hiddenSize, dtype)
    public val proj: VoidDense<T, V> = VoidDense("$name.proj", hiddenSize, perLayerDim, dtype)
    public val postNorm: RMSNormalization<T, V> = RMSNormalization(
        intArrayOf(hiddenSize), rmsEps.toDouble(), unitOffset = false, name = "$name.post_norm", dtype = dtype,
    )

    override val modules: List<Module<T, V>> = listOf(inpGate, proj, postNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = input

    public fun computeDelta(
        activeCorrected: Tensor<T, V>,
        perLayerInput: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        val gated = ops.gelu(inpGate.forward(activeCorrected, ctx))
        val mixed = ops.multiply(gated, perLayerInput)
        return postNorm.forward(proj.forward(mixed, ctx), ctx)
    }
}
