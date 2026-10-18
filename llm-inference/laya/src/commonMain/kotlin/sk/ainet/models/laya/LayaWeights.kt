package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.nn.quant.PackedRowDequantTensorData
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Maps module parameters of [LayaNetwork] to checkpoint tensor names after [LayaWeightPreparation]
 * has split the fused tensors and renamed the projections to the engine's `q_proj` / `o_proj` /
 * `gate_proj` vocabulary. The module path is `root/segment/.../module`; the parameter name already
 * carries the module name (`attn.q_proj.weight`, `norm1.bias`, `linear1.weight`), so the tensor
 * name is the root, the `layers.N` / `embeddings` segments, and the parameter name.
 */
public class LayaSafeTensorsNameResolver : WeightNameResolver {
    override fun resolve(modulePath: String, paramName: String): String? {
        val segments = modulePath.split('/')
        val root = segments.firstOrNull() ?: return null
        if (root !in ROOTS) return null
        val keep = segments.drop(1).dropLast(1).filter { it == "embeddings" || it.startsWith("layers.") }
        return (listOf(root) + keep + paramName).joinToString(".")
    }

    private companion object {
        val ROOTS = setOf("encoder", "head", "scorer")
    }
}

/** The tensors a Laya checkpoint holds beyond the module parameters. */
public class LayaExtraTensors<T : DType, V>(
    /** `type_emb.weight`, `[3, hidden]`. */
    public val typeEmbedding: Tensor<T, V>,
    /** `temperature`, `[3]`; kept for parity reporting, decoding reads `rl_agent_config.json`. */
    public val temperature: Tensor<T, V>?,
)

public class LayaPreparedWeights<T : DType, V>(
    public val encoder: List<WeightTensor<T, V>>,
    public val head: List<WeightTensor<T, V>>,
    public val scorer: List<WeightTensor<T, V>>,
    public val extras: LayaExtraTensors<T, V>,
    /** Tensors the adapter does not consume: `act_head.*` (REQ-EXE-007). */
    public val ignored: List<String>,
)

/**
 * Turns the raw checkpoint tensors into the shapes the module tree declares (REQ-EXE-002/003):
 * splits `Wqkv` and `in_proj_weight` / `in_proj_bias` into thirds, splits the fused `Wi` into the
 * activated half (`gate_proj`) and the multiplier half (`up_proj`), renames `Wo` / `out_proj` to
 * `o_proj` / `down_proj`, and synthesises the zero shift of the encoder's bias-free LayerNorms.
 * Unknown tensors fail loading; `act_head.*` is ignored by design.
 */
public object LayaWeightPreparation {

    public fun <T : DType, V> prepare(
        raw: List<WeightTensor<T, V>>,
        encoderConfig: LayaEncoderConfig,
        agentConfig: LayaAgentConfig,
        ctx: ExecutionContext,
        dtype: KClass<T>,
    ): LayaPreparedWeights<T, V> {
        val byName = raw.associateBy { it.name }
        fun need(name: String): WeightTensor<T, V> = byName[name]
            ?: throw UnsupportedLayaCheckpointException("checkpoint has no tensor '$name'")

        val d = encoderConfig.hiddenSize
        val encoder = ArrayList<WeightTensor<T, V>>()
        val head = ArrayList<WeightTensor<T, V>>()
        val scorer = ArrayList<WeightTensor<T, V>>()
        val consumed = HashSet<String>()
        fun take(name: String): WeightTensor<T, V> = need(name).also { consumed += name }
        fun zeros(name: String, size: Int): WeightTensor<T, V> =
            WeightTensor(name, listOf(size), ctx.fromFloatArray(Shape(size), dtype, FloatArray(size)))

        // --- encoder ---------------------------------------------------------------------------
        // A packed token table (SKEEP-007 Tier 2 lane) is read row by row by the embedding layer
        // through RowDequantSource; dense tables pass through unchanged.
        encoder += take("encoder.embeddings.tok_embeddings.weight").let { t ->
            val wrapped = PackedRowDequantTensorData.wrapIfRowDequantable(t.tensor.data as TensorData<FP32, Float>)
            if (wrapped === t.tensor.data) t else WeightTensor(t.name, t.shape, ctx.fromData(wrapped as TensorData<T, V>, dtype))
        }
        encoder += take("encoder.embeddings.norm.weight")
        encoder += byName["encoder.embeddings.norm.bias"]?.also { consumed += it.name } ?: zeros("encoder.embeddings.norm.bias", d)
        for (i in 0 until encoderConfig.numHiddenLayers) {
            val p = "encoder.layers.$i"
            if (i > 0) {
                encoder += take("$p.attn_norm.weight")
                encoder += byName["$p.attn_norm.bias"]?.also { consumed += it.name } ?: zeros("$p.attn_norm.bias", d)
            }
            val qkv = take("$p.attn.Wqkv.weight")
            expectShape(qkv, 3 * d, d)
            val (q, k, v) = splitRows(qkv, 3, ctx, dtype)
            encoder += WeightTensor("$p.attn.q_proj.weight", listOf(d, d), q)
            encoder += WeightTensor("$p.attn.k_proj.weight", listOf(d, d), k)
            encoder += WeightTensor("$p.attn.v_proj.weight", listOf(d, d), v)
            encoder += take("$p.attn.Wo.weight").renamed("$p.attn.o_proj.weight")
            encoder += take("$p.mlp_norm.weight")
            encoder += byName["$p.mlp_norm.bias"]?.also { consumed += it.name } ?: zeros("$p.mlp_norm.bias", d)
            val wi = take("$p.mlp.Wi.weight")
            expectShape(wi, 2 * encoderConfig.intermediateSize, d)
            val (activated, multiplier) = splitRows(wi, 2, ctx, dtype)
            encoder += WeightTensor("$p.mlp.gate_proj.weight", listOf(encoderConfig.intermediateSize, d), activated)
            encoder += WeightTensor("$p.mlp.up_proj.weight", listOf(encoderConfig.intermediateSize, d), multiplier)
            encoder += take("$p.mlp.Wo.weight").renamed("$p.mlp.down_proj.weight")
        }
        encoder += take("encoder.final_norm.weight")
        encoder += byName["encoder.final_norm.bias"]?.also { consumed += it.name } ?: zeros("encoder.final_norm.bias", d)

        // --- head ------------------------------------------------------------------------------
        for (i in 0 until agentConfig.headLayers) {
            val p = "head.layers.$i"
            for (n in listOf("norm1", "norm2")) { head += take("$p.$n.weight"); head += take("$p.$n.bias") }
            val w = take("$p.self_attn.in_proj_weight")
            expectShape(w, 3 * d, d)
            val (q, k, v) = splitRows(w, 3, ctx, dtype)
            head += WeightTensor("$p.self_attn.q_proj.weight", listOf(d, d), q)
            head += WeightTensor("$p.self_attn.k_proj.weight", listOf(d, d), k)
            head += WeightTensor("$p.self_attn.v_proj.weight", listOf(d, d), v)
            val b = take("$p.self_attn.in_proj_bias")
            val (qb, kb, vb) = splitRows(b, 3, ctx, dtype)
            head += WeightTensor("$p.self_attn.q_proj.bias", listOf(d), qb)
            head += WeightTensor("$p.self_attn.k_proj.bias", listOf(d), kb)
            head += WeightTensor("$p.self_attn.v_proj.bias", listOf(d), vb)
            head += take("$p.self_attn.out_proj.weight").renamed("$p.self_attn.o_proj.weight")
            head += take("$p.self_attn.out_proj.bias").renamed("$p.self_attn.o_proj.bias")
            for (n in listOf("linear1", "linear2")) { head += take("$p.$n.weight"); head += take("$p.$n.bias") }
        }

        // --- scorer and extras -----------------------------------------------------------------
        for (n in listOf("scorer.0.weight", "scorer.0.bias", "scorer.1.weight", "scorer.1.bias", "scorer.3.weight", "scorer.3.bias")) scorer += take(n)
        val typeEmb = take("type_emb.weight")
        expectShape(typeEmb, 3, d)
        val temperature = byName["temperature"]?.also { consumed += it.name }

        val leftovers = raw.map { it.name }.filter { it !in consumed }
        val unexpected = leftovers.filter { !it.startsWith("act_head.") }
        if (unexpected.isNotEmpty()) {
            throw UnsupportedLayaCheckpointException("checkpoint carries tensors this adapter does not know: ${unexpected.take(10)}")
        }
        return LayaPreparedWeights(encoder, head, scorer, LayaExtraTensors(typeEmb.tensor, temperature?.tensor), leftovers)
    }

    private fun <T : DType, V> WeightTensor<T, V>.renamed(name: String): WeightTensor<T, V> = WeightTensor(name, shape, tensor)

    private fun expectShape(t: WeightTensor<*, *>, vararg dims: Int) {
        if (t.shape != dims.toList()) {
            throw UnsupportedLayaCheckpointException("tensor '${t.name}' has shape ${t.shape}, expected ${dims.toList()}")
        }
    }

    /**
     * Splits a `[rows, …]` tensor into [parts] equal row blocks, copying each into its own tensor.
     *
     * A packed weight stays packed (SKEEP-007 Tier 2): in canonical row-major block storage a row
     * block is a contiguous byte range (`rowsPerPart × blocksPerRow × bytesPerBlock`), so the fused
     * `Wqkv` / `Wi` / `in_proj` matrices split without an FP32 transient and the parts are served by
     * the packed kernels like every other projection. Dense weights take the float path as before.
     */
    internal fun <T : DType, V> splitRows(source: WeightTensor<T, V>, parts: Int, ctx: ExecutionContext, dtype: KClass<T>): List<Tensor<T, V>> {
        val rows = source.shape[0]
        require(rows % parts == 0) { "cannot split ${source.name} with $rows rows into $parts parts" }
        val rowsPerPart = rows / parts
        val inner = source.shape.drop(1).fold(1) { a, b -> a * b }
        splitPackedRows(source, parts, rowsPerPart, inner, ctx, dtype)?.let { return it }
        val block = rowsPerPart * inner
        val data = source.tensor.data.copyToFloatArray()
        return List(parts) { part ->
            val out = data.copyOfRange(part * block, (part + 1) * block)
            val shape = if (inner == 1) Shape(rowsPerPart) else Shape(rowsPerPart, *source.shape.drop(1).toIntArray())
            ctx.fromFloatArray(shape, dtype, out)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> splitPackedRows(
        source: WeightTensor<T, V>, parts: Int, rowsPerPart: Int, inner: Int, ctx: ExecutionContext, dtype: KClass<T>,
    ): List<Tensor<T, V>>? {
        val packed = source.tensor.data as? PackedBlockStorage ?: return null
        if (source.shape.size != 2 || packed.blockOrder != BlockOrder.ROW_MAJOR) return null
        if (inner % packed.blockSize != 0) return null
        val blocksPerRow = inner / packed.blockSize
        val totalBlocks = source.shape[0] * blocksPerRow
        if (packed.blockCount != totalBlocks) return null
        val bytesPerBlock = packed.packedData.size / totalBlocks
        val partBytes = rowsPerPart * blocksPerRow * bytesPerBlock
        val shape = Shape(rowsPerPart, inner)
        return List(parts) { part ->
            val bytes = packed.packedData.copyOfRange(part * partBytes, (part + 1) * partBytes)
            val data: TensorData<*, *> = when (packed.encoding) {
                TensorEncoding.Q8_0 -> Q8_0BlockTensorData(shape, bytes)
                TensorEncoding.Q4_0 -> Q4_0BlockTensorData(shape, bytes)
                else -> return null
            }
            ctx.fromData(data as TensorData<T, V>, dtype)
        }
    }
}
