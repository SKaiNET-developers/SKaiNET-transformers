package sk.ainet.models.gemma

import kotlin.math.sqrt
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
 * Top-level Gemma 4 Per-Layer Embedding module.
 *
 * Computes the `per_layer_inputs` tensor of shape
 * `[batch, seqLen, numLayers, perLayerDim]` that feeds each decoder layer's
 * [PerLayerInputBlockHook]. Exact math from
 * `Gemma4TextModel.get_per_layer_inputs` / `project_per_layer_inputs` in
 * transformers 5.6.0:
 *
 * ```python
 * # (1) token-identity component — separate embedding table, scaled by sqrt(ple_dim).
 * raw = embed_tokens_per_layer(input_ids) * sqrt(perLayerDim)
 * raw = raw.reshape(B, S, numLayers, perLayerDim)
 *
 * # (2) context-aware component — linear projection of main embedding output.
 * proj = per_layer_model_projection(inputs_embeds) * (1 / sqrt(hiddenSize))
 * proj = proj.reshape(B, S, numLayers, perLayerDim)
 * proj = per_layer_projection_norm(proj)
 *
 * # (3) combine with 1/sqrt(2) scale.
 * per_layer_inputs = (proj + raw) * (1 / sqrt(2))
 * ```
 *
 * GGUF tensor names (all three are top-level, not per-block):
 * - `per_layer_token_embd.weight`  → `embed_tokens_per_layer`
 *   shape in HF: `[vocabSize, numLayers * perLayerDim]`
 *   GGUF dtype: typically Q6_K for E2B — loader dequants to FP32 on load.
 * - `per_layer_model_proj.weight`  → `per_layer_model_projection`
 *   shape in HF: `[hiddenSize, numLayers * perLayerDim]` (Linear `[out, in]` convention).
 *   GGUF dtype: BF16 for E2B.
 * - `per_layer_proj_norm.weight`   → `per_layer_projection_norm`
 *   shape: `[perLayerDim]`, F32.
 */
@Suppress("UNCHECKED_CAST")
public class PerLayerEmbedding<T : DType, V>(
    public val vocabSize: Int,
    public val hiddenSize: Int,
    public val numLayers: Int,
    public val perLayerDim: Int,
    public val rmsEps: Float = 1e-6f,
    override val name: String = "per_layer"
) : Module<T, V>(), ModuleParameters<T, V> {

    private val perLayerTotal: Int = numLayers * perLayerDim
    private val inputScale: Float = 1.0f / sqrt(2.0f)
    private val projScale: Float = 1.0f / sqrt(hiddenSize.toFloat())
    private val embedScale: Float = sqrt(perLayerDim.toFloat())

    private fun voidWeight(shape: Shape): VoidOpsTensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        Any::class as KClass<T>
    )

    public val projectionNorm: RMSNormalization<T, V> =
        RMSNormalization(intArrayOf(perLayerDim), rmsEps.toDouble(), name = "$name.projection_norm")

    override val params: List<ModuleParameter<T, V>> = listOf(
        // Per-layer token embedding: [vocabSize, perLayerTotal]. Big table —
        // for real Gemma 4 E2B this is 262 144 × 8 960 = ~2.3B entries,
        // ~9 GB FP32 after dequant. Memory impact flagged in the loader's
        // optional-tensor path; Phase 5f.5c will switch to a lazy-row gather.
        ModuleParameter.WeightParameter(
            "$name.embed_tokens.weight",
            voidWeight(Shape(vocabSize, perLayerTotal))
        ),
        // Context projection: [perLayerTotal, hiddenSize] in [out, in]
        // convention. 8 960 × 1 536 ≈ 13.8M FP32 = 55 MB. Negligible.
        ModuleParameter.WeightParameter(
            "$name.model_proj.weight",
            voidWeight(Shape(perLayerTotal, hiddenSize))
        )
    )

    // Read the runtime-populated values via params indexing — the void
    // placeholders above are replaced by WeightMapper during load.
    private val embedTokensWeight: Tensor<T, V> get() = params[0].value
    private val modelProjWeight: Tensor<T, V> get() = params[1].value

    override val modules: List<Module<T, V>> = listOf(projectionNorm)

    /**
     * Module forward is not meaningful for this module — it's called
     * imperatively from [GemmaModel.onForward] via [compute]. Return the
     * input unchanged so it fits the `Module<T, V>` contract for
     * weight-mapper traversal.
     */
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = input

    /**
     * Compute `per_layer_inputs[batch, seq, layer, perLayerDim]` given the
     * raw token IDs (for the token-identity embedding lookup) and the main
     * embedding output `inputsEmbeds` (for the context-aware projection).
     *
     * @param tokenIds shape `[batch, seq]` with dtype Int32. If the runtime
     *   feeds a flat `[seq]` tensor (single-batch), upstream code should
     *   unsqueeze to 2D before calling.
     * @param inputsEmbeds shape `[batch, seq, hiddenSize]`.
     * @return `per_layer_inputs` tensor shape `[batch, seq, numLayers, perLayerDim]`.
     */
    public fun compute(
        tokenIds: Tensor<*, *>,
        inputsEmbeds: Tensor<T, V>,
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val ops = ctx.ops

        // (1) Token-identity: gather rows of embedTokensWeight by tokenIds.
        // We do the gather via a manual buffer build; `ops.gather` would also
        // work but needs Int32 handling that varies by backend. Since the
        // activation dtype is FP32 and the weight is FP32 after dequant, this
        // is a plain float copy.
        val idsShape = tokenIds.shape
        require(idsShape.rank == 2) {
            "$name.compute: tokenIds must be [batch, seq], got shape=${tokenIds.shape}"
        }
        val batch = idsShape[0]
        val seq = idsShape[1]
        val idsFlat = intArrayOf(*IntArray(batch * seq) { idx ->
            // tokens arrive as Int32 — read elementwise.
            @Suppress("UNCHECKED_CAST")
            (tokenIds.data as TensorData<*, Any>)
                .get(idx / seq, idx % seq)
                .let { (it as? Int) ?: (it as Number).toInt() }
        })
        val rawBuf = FloatArray(batch * seq * perLayerTotal)
        val weightData = embedTokensWeight.data
        if (weightData is RowDequantSource) {
            // Source kept in its packed/encoded form on load — Q-bytes for
            // the GGUF path, BF16 mmap for the SafeTensors path. We dequant
            // only the rows we need (one per token per decode step), even
            // though the full logical tensor is much larger than 2 GB.
            for (i in 0 until batch * seq) {
                val tokenId = idsFlat[i]
                require(tokenId in 0 until vocabSize) {
                    "$name.compute: token id $tokenId out of range [0, $vocabSize)"
                }
                val row = weightData.dequantRow(tokenId)
                row.copyInto(rawBuf, i * perLayerTotal)
            }
        } else {
            // FP32 path — used by toy-fixture tests and any non-quant source.
            val weightBuf = weightData.copyToFloatArray()
            for (i in 0 until batch * seq) {
                val tokenId = idsFlat[i]
                require(tokenId in 0 until vocabSize) {
                    "$name.compute: token id $tokenId out of range [0, $vocabSize)"
                }
                val src = tokenId * perLayerTotal
                val dst = i * perLayerTotal
                weightBuf.copyInto(rawBuf, dst, src, src + perLayerTotal)
            }
        }
        // Apply scaled-word-embedding scale factor: raw *= sqrt(perLayerDim).
        for (i in rawBuf.indices) rawBuf[i] *= embedScale

        val rawReshaped = ctx.fromFloatArray<T, V>(
            Shape(batch, seq, numLayers, perLayerDim),
            dtype,
            rawBuf
        )

        // (2) Context-aware component: linearProject(inputsEmbeds, modelProjWeight)
        // then reshape to [B, S, numLayers, perLayerDim], scale, and norm.
        // inputsEmbeds shape: [B, S, hiddenSize] — flatten to [B*S, hiddenSize]
        // for the matmul.
        val flatEmbeds = ops.reshape(inputsEmbeds, Shape(batch * seq, hiddenSize))
        var proj = linearProject(ops, flatEmbeds, modelProjWeight)  // [B*S, perLayerTotal]
        proj = ops.reshape(proj, Shape(batch, seq, numLayers, perLayerDim))

        // Scale by 1/sqrt(hidden_size) via a scalar multiply. Build a [1]
        // scalar tensor so broadcasting gives a uniform scale.
        val scaleTensor = ctx.fromFloatArray<T, V>(
            Shape(1), dtype, floatArrayOf(projScale)
        )
        proj = ops.multiply(proj, scaleTensor)
        proj = projectionNorm.forward(proj, ctx)

        // (3) Combine: (proj + raw) * (1 / sqrt(2)).
        val combineScale = ctx.fromFloatArray<T, V>(
            Shape(1), dtype, floatArrayOf(inputScale)
        )
        val combined = ops.multiply(ops.add(proj, rawReshaped), combineScale)
        return combined
    }
}
