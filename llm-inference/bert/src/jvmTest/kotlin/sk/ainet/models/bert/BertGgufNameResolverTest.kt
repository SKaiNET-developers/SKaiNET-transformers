package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * A checkpoint that carries llama.cpp GGUF-BERT tensor names must load into the same network
 * as the HuggingFace-named one. The synthetic weights are renamed, so equal embeddings prove
 * every DSL parameter resolved to the right tensor (same-shaped Q/K/V included).
 */
class BertGgufNameResolverTest {

    private val ctx = DirectCpuExecutionContext()
    private val cfg = BertSyntheticFixtures.tinyConfig()
    private val tokens = intArrayOf(1, 5, 3, 7, 2)

    /** The two norm-layer conventions seen in published all-MiniLM-L6-v2 conversions. */
    private enum class Norms(val embeddings: String, val attention: String, val output: String) {
        LONG("token_embd_norm", "attn_output_norm", "layer_output_norm"),
        SHORT("output_norm", "attn_norm", "ffn_norm"),
    }

    private fun ggufName(hfName: String, norms: Norms): String {
        val name = hfName.removePrefix("bert.")
        val suffix = name.substringAfterLast('.')
        if (!name.startsWith("encoder.layer.")) {
            return when {
                name == "embeddings.word_embeddings.weight" -> "token_embd.weight"
                name == "embeddings.position_embeddings.weight" -> "position_embd.weight"
                name == "embeddings.token_type_embeddings.weight" -> "token_types.weight"
                name.startsWith("embeddings.LayerNorm.") -> "${norms.embeddings}.$suffix"
                else -> error("unexpected tensor $hfName")
            }
        }
        val rest = name.removePrefix("encoder.layer.")
        val block = "blk.${rest.substringBefore('.')}"
        val part = rest.substringAfter('.').substringBeforeLast('.')
        return when (part) {
            "attention.self.query" -> "$block.attn_q.$suffix"
            "attention.self.key" -> "$block.attn_k.$suffix"
            "attention.self.value" -> "$block.attn_v.$suffix"
            "attention.output.dense" -> "$block.attn_output.$suffix"
            "attention.output.LayerNorm" -> "$block.${norms.attention}.$suffix"
            "intermediate.dense" -> "$block.ffn_up.$suffix"
            "output.dense" -> "$block.ffn_down.$suffix"
            "output.LayerNorm" -> "$block.${norms.output}.$suffix"
            else -> error("unexpected tensor $hfName")
        }
    }

    private fun ggufNamed(norms: Norms): List<WeightTensor<FP32, Float>> =
        BertNetworkLoader.normalizeTensorNames(
            BertSyntheticFixtures.weightTensors(ctx, cfg).map { it.copy(name = ggufName(it.name, norms)) }
        )

    private fun reference(): FloatArray =
        createBertEncoderRuntime<FP32>(cfg, BertSyntheticFixtures.weightTensors(ctx, cfg), ctx).encode(tokens)

    private fun assertLoadsLikeHuggingFace(norms: Norms) {
        val tensors = ggufNamed(norms)
        val runtime = createBertEncoderRuntime<FP32>(cfg, tensors, ctx, BertGgufNameResolver.forTensors(tensors))
        assertContentEquals(reference(), runtime.encode(tokens))
    }

    @Test
    fun `long norm names load like the HuggingFace checkpoint`() = assertLoadsLikeHuggingFace(Norms.LONG)

    @Test
    fun `short norm names load like the HuggingFace checkpoint`() = assertLoadsLikeHuggingFace(Norms.SHORT)

    @Test
    fun `a missing tensor is reported, not silently skipped`() {
        val tensors = ggufNamed(Norms.LONG).filterNot { it.name == "bert.blk.1.ffn_up.weight" }
        assertFailsWith<IllegalArgumentException> {
            createBertEncoderRuntime<FP32>(cfg, tensors, ctx, BertGgufNameResolver.forTensors(tensors))
        }
    }

    @Test
    fun `weight forms relabel every tensor and keep only encoder weights packed`() {
        val encoder = BertGguf.weightFormFor("blk.0.attn_q.weight")
        assertEquals(WeightShapeOrientation.OUT_IN, encoder.shape)
        assertEquals(EncodingRequest.KeepAsStored, encoder.encoding)

        for (table in listOf("token_embd.weight", "position_embd.weight", "token_types.weight")) {
            val form = BertGguf.weightFormFor(table)
            assertEquals(WeightShapeOrientation.OUT_IN, form.shape, table)
            assertEquals(EncodingRequest.DequantizeTo(FP32), form.encoding, table)
        }

        val expanded = BertGguf.weightFormFor("blk.0.attn_q.weight", keepQuantizedPacked = false)
        assertEquals(EncodingRequest.DequantizeTo(FP32), expanded.encoding)
    }
}
