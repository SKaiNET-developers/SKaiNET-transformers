package sk.ainet.models.apertus

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path as IoPath
import kotlinx.io.files.SystemFileSystem
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32

/**
 * End-to-end fixture for the engine-backed [ApertusSafeTensorsLoader] (SKaiNET#1246): a synthetic
 * 2-shard HF checkpoint in a temp dir — BF16 projections, F32 norms (one stored `[1, dim]` to
 * exercise the family-side shape normalization), scalar xIELU parameters, a hand-written
 * `model.safetensors.index.json`, and the minimal `config.json` the loader reads.
 *
 * Covers what the collapse onto `ShardedSafeTensorsParametersLoader` must preserve: every
 * GGUF-named slot is populated after renaming, values round-trip exactly (bf16-representable
 * fixture), tied embeddings, the xIELU scalars arrive as floats, an unmapped INT64 decoy is
 * neither delivered nor allowed to trip the engine's fail-fast dtype pre-scan, and the dtype
 * policy reaches the engine (`Require(BF16)` keeps native storage).
 */
class ApertusSafeTensorsLoaderFixtureTest {

    private val ctx = DirectCpuExecutionContext()

    private val hidden = 8
    private val headDim = 4
    private val heads = 2
    private val kvHeads = 1
    private val ffn = 16
    private val vocab = 16

    @Test
    fun `every mapped slot is present, norms are normalized and values round-trip`() {
        withFixture { indexPath ->
            val weights = runBlocking { ApertusSafeTensorsLoader(indexPath).loadToMap(ctx, FP32::class) }
            val t = weights.tensors

            val expectedSlots = listOf(
                ApertusTensorNames.TOKEN_EMBEDDINGS, ApertusTensorNames.OUTPUT_NORM, ApertusTensorNames.OUTPUT_WEIGHT,
                ApertusTensorNames.attnNorm(0), ApertusTensorNames.attnQ(0), ApertusTensorNames.attnK(0),
                ApertusTensorNames.attnV(0), ApertusTensorNames.attnOut(0), ApertusTensorNames.attnQNorm(0),
                ApertusTensorNames.attnKNorm(0), ApertusTensorNames.ffnNorm(0), ApertusTensorNames.ffnUp(0),
                ApertusTensorNames.ffnDown(0),
            )
            for (slot in expectedSlots) {
                assertNotNull(t[slot], "slot $slot must be populated after HF -> GGUF renaming; got ${t.keys}")
            }
            assertEquals(expectedSlots.toSet(), t.keys, "no extra or missing slots")

            // Tied embeddings (no lm_head.weight in the shards, tie_word_embeddings=true).
            assertTrue(t[ApertusTensorNames.OUTPUT_WEIGHT] === t[ApertusTensorNames.TOKEN_EMBEDDINGS])

            // Values and shapes across both shards.
            assertEquals(listOf(vocab, hidden), t[ApertusTensorNames.TOKEN_EMBEDDINGS]!!.shape.dimensions.toList())
            assertEquals(values(vocab * hidden, EMBED_BASE).toList(), floats(t[ApertusTensorNames.TOKEN_EMBEDDINGS]!!).toList())
            assertEquals(values(heads * headDim * hidden, Q_BASE).toList(), floats(t[ApertusTensorNames.attnQ(0)]!!).toList())

            // The attention norm was stored [1, hidden]; the runtime slot must be [hidden] with the same values.
            assertEquals(listOf(hidden), t[ApertusTensorNames.attnNorm(0)]!!.shape.dimensions.toList())
            assertEquals(values(hidden, LN_BASE).toList(), floats(t[ApertusTensorNames.attnNorm(0)]!!).toList())

            // xIELU scalars arrive as plain floats (shape [] and [1] both accepted).
            val xielu = weights.xieluParams[0]!!
            assertEquals(0.8f, xielu.alphaP)
            assertEquals(0.5f, xielu.alphaN)
            assertEquals(0.75f, xielu.beta)
            assertEquals(0.125f, xielu.eps)

            // Default policy (Any) widens BF16 to dense FP32.
            assertTrue(t[ApertusTensorNames.attnQ(0)]!!.data is FloatArrayTensorData<*>, "Any policy widens BF16")

            // The decoy never reached the family.
            assertFalse(t.keys.any { "rotary" in it })
        }
    }

    @Test
    fun `Require BF16 keeps native storage for the projections and leaves F32 norms dense`() {
        withFixture { indexPath ->
            val weights = runBlocking {
                ApertusSafeTensorsLoader(indexPath, dtypePolicy = DTypePolicy.Require(BF16)).loadToMap(ctx, FP32::class)
            }
            val q = weights.tensors[ApertusTensorNames.attnQ(0)]!!
            assertTrue(q.data is Bf16DenseTensorData, "Require(BF16) must keep the projection bf16-native, got ${q.data::class.simpleName}")
            assertEquals(values(heads * headDim * hidden, Q_BASE).toList(), floats(q).toList())
            val norm = weights.tensors[ApertusTensorNames.ffnNorm(0)]!!
            assertTrue(norm.data is FloatArrayTensorData<*>, "F32 norms stay dense FP32")
        }
    }

    // --- fixture ---------------------------------------------------------------

    private fun withFixture(block: (indexPath: String) -> Unit) {
        val dir = Files.createTempDirectory("apertus-st-fixture")
        try {
            val shard1 = "model-00001-of-00002.safetensors"
            val shard2 = "model-00002-of-00002.safetensors"
            val names1 = mutableListOf<String>()
            val names2 = mutableListOf<String>()

            writeShard(dir.resolve(shard1)) {
                bf16("model.embed_tokens.weight", listOf(vocab, hidden), EMBED_BASE, names1)
                f32("model.norm.weight", listOf(hidden), NORM_BASE, names1)
            }
            writeShard(dir.resolve(shard2)) {
                val l = "model.layers.0"
                // Stored [1, hidden] on purpose: the family normalizes it to [hidden].
                f32("$l.attention_layernorm.weight", listOf(1, hidden), LN_BASE, names2)
                bf16("$l.self_attn.q_proj.weight", listOf(heads * headDim, hidden), Q_BASE, names2)
                bf16("$l.self_attn.k_proj.weight", listOf(kvHeads * headDim, hidden), 0.125f, names2)
                bf16("$l.self_attn.v_proj.weight", listOf(kvHeads * headDim, hidden), 0.25f, names2)
                bf16("$l.self_attn.o_proj.weight", listOf(hidden, heads * headDim), 0.375f, names2)
                f32("$l.self_attn.q_norm.weight", listOf(headDim), 4.5f, names2)
                f32("$l.self_attn.k_norm.weight", listOf(headDim), 5.0f, names2)
                f32("$l.feedforward_layernorm.weight", listOf(hidden), 2.5f, names2)
                bf16("$l.mlp.up_proj.weight", listOf(ffn, hidden), 0.625f, names2)
                bf16("$l.mlp.down_proj.weight", listOf(hidden, ffn), 0.75f, names2)
                // xIELU scalars: two stored as rank-0, two as [1].
                tensorF32("$l.mlp.act_fn.alpha_p", emptyList<Long>(), floatArrayOf(0.8f)); names2 += "$l.mlp.act_fn.alpha_p"
                tensorF32("$l.mlp.act_fn.alpha_n", emptyList<Long>(), floatArrayOf(0.5f)); names2 += "$l.mlp.act_fn.alpha_n"
                tensorBF16("$l.mlp.act_fn.beta", listOf(1L), floatArrayOf(0.75f)); names2 += "$l.mlp.act_fn.beta"
                tensorF32("$l.mlp.act_fn.eps", listOf(1L), floatArrayOf(0.125f)); names2 += "$l.mlp.act_fn.eps"
                // Unmapped decoy: an INT64 tensor the family never asks for. If the allowlist
                // did not exempt it, the engine's fail-fast pre-scan would reject the whole load.
                tensorI64("$l.self_attn.rotary_emb.inv_freq_ids", listOf(4L), LongArray(4) { it.toLong() })
                names2 += "$l.self_attn.rotary_emb.inv_freq_ids"
            }

            val weightMap = (names1.map { "\"$it\":\"$shard1\"" } + names2.map { "\"$it\":\"$shard2\"" })
                .joinToString(",")
            val totalSize = Files.size(dir.resolve(shard1)) + Files.size(dir.resolve(shard2))
            Files.writeString(
                dir.resolve("model.safetensors.index.json"),
                """{"metadata":{"total_size":$totalSize},"weight_map":{$weightMap}}"""
            )
            Files.writeString(
                dir.resolve("config.json"),
                """
                {
                  "model_type": "apertus",
                  "hidden_size": $hidden,
                  "num_hidden_layers": 1,
                  "num_attention_heads": $heads,
                  "num_key_value_heads": $kvHeads,
                  "head_dim": $headDim,
                  "intermediate_size": $ffn,
                  "vocab_size": $vocab,
                  "tie_word_embeddings": true
                }
                """.trimIndent()
            )
            block(dir.resolve("model.safetensors.index.json").toString())
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    private fun writeShard(path: Path, block: SafeTensorsWriter.() -> Unit) {
        SystemFileSystem.sink(IoPath(path.toString())).buffered().use { sink ->
            SafeTensorsWriter.write(sink, block)
        }
    }

    private fun SafeTensorsWriter.bf16(name: String, shape: List<Int>, base: Float, names: MutableList<String>) {
        tensorBF16(name, shape.map { it.toLong() }, values(shape.fold(1) { a, b -> a * b }, base)); names += name
    }

    private fun SafeTensorsWriter.f32(name: String, shape: List<Int>, base: Float, names: MutableList<String>) {
        tensorF32(name, shape.map { it.toLong() }, values(shape.fold(1) { a, b -> a * b }, base)); names += name
    }

    /** bf16-exact values: base + i/8 stays within 7 mantissa bits for these tiny extents. */
    private fun values(n: Int, base: Float) = FloatArray(n) { base + (it % 16) * 0.125f }

    private fun floats(t: sk.ainet.lang.tensor.Tensor<FP32, Float>): FloatArray = t.data.copyToFloatArray()

    private companion object {
        const val EMBED_BASE = 1.0f
        const val NORM_BASE = 2.0f
        const val Q_BASE = -1.0f
        const val LN_BASE = 0.5f
    }
}
