package sk.ainet.models.bert

import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.types.FP32

/**
 * Resolves network DSL module paths to llama.cpp GGUF-BERT tensor names, so
 * [createBertEncoderRuntime] loads a GGUF checkpoint the same way it loads SafeTensors:
 *
 * ```kotlin
 * val loader = StreamingGgufParametersLoader(
 *     sourceProvider = { source },
 *     weightFormFor = { name -> BertGguf.weightFormFor(name) },
 * )
 * val tensors = BertNetworkLoader.loadWeightTensors(listOf(loader), ctx, FP32::class)
 * val runtime = createBertEncoderRuntime<FP32>(config, tensors, ctx, BertGgufNameResolver.forTensors(tensors))
 * ```
 *
 * GGUF-BERT naming is not uniform across converters. Two conversions of
 * `sentence-transformers/all-MiniLM-L6-v2` agree on every name except the three LayerNorms:
 *
 * | DSL parameter | GGUF name | alternative |
 * |---|---|---|
 * | word embeddings | `token_embd.weight` | |
 * | position embeddings | `position_embd.weight` | |
 * | token type embeddings | `token_types.weight` | |
 * | embeddings LayerNorm | `token_embd_norm.{weight,bias}` | `output_norm.{weight,bias}` |
 * | layer N query / key / value | `blk.N.attn_{q,k,v}.{weight,bias}` | |
 * | layer N attention output | `blk.N.attn_output.{weight,bias}` | |
 * | layer N attention LayerNorm | `blk.N.attn_output_norm.{weight,bias}` | `blk.N.attn_norm.{weight,bias}` |
 * | layer N intermediate | `blk.N.ffn_up.{weight,bias}` | |
 * | layer N output | `blk.N.ffn_down.{weight,bias}` | |
 * | layer N output LayerNorm | `blk.N.layer_output_norm.{weight,bias}` | `blk.N.ffn_norm.{weight,bias}` |
 *
 * [tensorNames] decides between a name and its alternative, so both conventions resolve
 * without the caller knowing which converter wrote the file.
 *
 * Returned names carry the `bert.` prefix that [BertNetworkLoader.normalizeTensorNames] adds to
 * every loaded tensor (GGUF names never start with `bert.`, so normalization always applies).
 *
 * @param tensorNames names of the loaded tensors, as [BertNetworkLoader.loadWeightTensors]
 *   reports them (with the `bert.` prefix)
 */
public class BertGgufNameResolver(private val tensorNames: Set<String>) : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val block = LAYER.find(modulePath)?.let { "blk.${it.groupValues[1]}" }
        val suffix = if (paramName.endsWith(".weight")) "weight" else "bias"

        return when {
            modulePath.endsWith("/word_embeddings") -> pick("token_embd.weight")
            paramName == "position_embeddings.weight" -> pick("position_embd.weight")
            paramName == "token_type_embeddings.weight" -> pick("token_types.weight")
            modulePath.endsWith("/LayerNorm") -> pick("token_embd_norm.$suffix", "output_norm.$suffix")

            block == null -> null

            paramName.contains("q_proj.") -> pick("$block.attn_q.$suffix")
            paramName.contains("k_proj.") -> pick("$block.attn_k.$suffix")
            paramName.contains("v_proj.") -> pick("$block.attn_v.$suffix")
            paramName.contains("o_proj.") -> pick("$block.attn_output.$suffix")

            modulePath.endsWith("/attn_ln") -> pick("$block.attn_output_norm.$suffix", "$block.attn_norm.$suffix")
            modulePath.endsWith("/intermediate") -> pick("$block.ffn_up.$suffix")
            // "/output_ln" is checked before "/output": the second is a suffix of neither, but
            // keeping the more specific path first makes the order irrelevant to future edits.
            modulePath.endsWith("/output_ln") -> pick("$block.layer_output_norm.$suffix", "$block.ffn_norm.$suffix")
            modulePath.endsWith("/output") -> pick("$block.ffn_down.$suffix")

            else -> null
        }
    }

    /** The first candidate the checkpoint carries, else the first candidate (reported as missing). */
    private fun pick(vararg names: String): String {
        val prefixed = names.map { "bert.$it" }
        return prefixed.firstOrNull { it in tensorNames } ?: prefixed.first()
    }

    public companion object {
        private val LAYER = Regex("""encoder\.layer\.(\d+)\.(?:attn|ffn)""")

        /** A resolver for the tensors [BertNetworkLoader.loadWeightTensors] returned. */
        public fun forTensors(tensors: List<WeightTensor<*, *>>): BertGgufNameResolver =
            BertGgufNameResolver(tensors.mapTo(HashSet()) { it.name })
    }
}

/** Loading choices for a GGUF-BERT checkpoint. */
public object BertGguf {

    private val AS_STORED = WeightForm(shape = WeightShapeOrientation.OUT_IN)
    private val AS_FP32 = WeightForm(
        encoding = EncodingRequest.DequantizeTo(FP32),
        shape = WeightShapeOrientation.OUT_IN,
    )

    /** True for the three embedding tables: `token_embd`, `position_embd`, `token_types`. */
    public fun isEmbeddingTable(tensorName: String): Boolean =
        tensorName.contains("embd") || tensorName.contains("token_types")

    /**
     * The form a GGUF-BERT tensor should load in — pass it as `weightFormFor` to
     * `StreamingGgufParametersLoader`.
     *
     * Every 2-D weight is relabelled `[out, in]` ([WeightShapeOrientation.OUT_IN]): GGUF lists
     * dimensions in `ne` order, the BERT network expects the HuggingFace order, and the bytes
     * already match it. With [keepQuantizedPacked] the encoder's quantized matmul weights stay in
     * their packed blocks; without it every weight is dequantized to FP32. Embedding tables are
     * always dequantized: the position and token-type tables are read element-wise.
     */
    public fun weightFormFor(tensorName: String, keepQuantizedPacked: Boolean = true): WeightForm =
        if (keepQuantizedPacked && !isEmbeddingTable(tensorName)) AS_STORED else AS_FP32
}
