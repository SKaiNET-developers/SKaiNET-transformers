package sk.ainet.models.gemma3n

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
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
 * End-to-end fixture for the engine-backed [Gemma3nSafeTensorsWeightLoader] (SKaiNET#1246): a
 * synthetic 2-shard HF checkpoint in a temp dir — BF16 projections, F32 norms, an E2B-style
 * layer (no AltUp, no Laurel) plus the PLE globals, a hand-written
 * `model.safetensors.index.json`, and the minimal `config.json` the loader reads.
 *
 * Covers what the collapse onto `ShardedSafeTensorsParametersLoader` must preserve: every
 * present GGUF-named slot is populated after renaming and absent optional (E4B) slots stay
 * absent, values round-trip exactly (bf16-representable fixture), tied output/embedding, an
 * unmapped INT64 decoy is neither delivered nor allowed to trip the engine's fail-fast dtype
 * pre-scan, and the dtype policy reaches the engine (`Require(BF16)` keeps native storage).
 * The >2 GiB PLE size guard is not exercised — a fixture that size is not a unit test.
 */
class Gemma3nSafeTensorsWeightLoaderFixtureTest {

    private val ctx = DirectCpuExecutionContext()

    private val hidden = 8
    private val headDim = 4
    private val heads = 2
    private val kvHeads = 1
    private val ffn = 16
    private val vocab = 16
    private val pleDim = 4

    @Test
    fun `present slots are mapped, absent E4B slots stay absent, values round-trip`() {
        withFixture { indexPath ->
            val weights = runBlocking { Gemma3nSafeTensorsWeightLoader(indexPath).loadToMap(ctx, FP32::class) }
            val t = weights.tensors

            val expectedSlots = listOf(
                Gemma3nTensorNames.TOKEN_EMBEDDINGS, Gemma3nTensorNames.OUTPUT_NORM, Gemma3nTensorNames.OUTPUT_WEIGHT,
                Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD, Gemma3nTensorNames.PER_LAYER_MODEL_PROJ,
                Gemma3nTensorNames.PER_LAYER_PROJ_NORM,
                Gemma3nTensorNames.inputLayernorm(0), Gemma3nTensorNames.attnQ(0), Gemma3nTensorNames.attnK(0),
                Gemma3nTensorNames.attnV(0), Gemma3nTensorNames.attnOut(0), Gemma3nTensorNames.postAttentionLayernorm(0),
                Gemma3nTensorNames.ffnGate(0), Gemma3nTensorNames.ffnUp(0), Gemma3nTensorNames.ffnDown(0),
                Gemma3nTensorNames.perLayerInput(0), Gemma3nTensorNames.attnQNorm(0), Gemma3nTensorNames.attnKNorm(0),
                Gemma3nTensorNames.postAttentionNorm(0), Gemma3nTensorNames.postFfwNorm(0), Gemma3nTensorNames.postNorm(0),
                Gemma3nTensorNames.inputGate(0), Gemma3nTensorNames.proj(0),
            )
            for (slot in expectedSlots) {
                assertNotNull(t[slot], "slot $slot must be populated after HF -> GGUF renaming; got ${t.keys}")
            }
            assertEquals(expectedSlots.toSet(), t.keys, "no extra or missing slots")

            // Optional E4B tensors were not in the shards and must not be synthesized.
            assertNull(t[Gemma3nTensorNames.altupRouter(0)])
            assertNull(t[Gemma3nTensorNames.laurelL(0)])
            assertNull(t[Gemma3nTensorNames.ALTUP_PROJ])

            // Weight tying: the output weight IS the token embedding.
            assertTrue(t[Gemma3nTensorNames.OUTPUT_WEIGHT] === t[Gemma3nTensorNames.TOKEN_EMBEDDINGS])

            assertEquals(listOf(vocab, hidden), t[Gemma3nTensorNames.TOKEN_EMBEDDINGS]!!.shape.dimensions.toList())
            assertEquals(values(vocab * hidden, EMBED_BASE).toList(), floats(t[Gemma3nTensorNames.TOKEN_EMBEDDINGS]!!).toList())
            assertEquals(values(hidden, NORM_BASE).toList(), floats(t[Gemma3nTensorNames.OUTPUT_NORM]!!).toList())
            assertEquals(listOf(heads * headDim, hidden), t[Gemma3nTensorNames.attnQ(0)]!!.shape.dimensions.toList())
            assertEquals(values(heads * headDim * hidden, Q_BASE).toList(), floats(t[Gemma3nTensorNames.attnQ(0)]!!).toList())
            assertEquals(listOf(vocab, pleDim), t[Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD]!!.shape.dimensions.toList())

            assertTrue(t[Gemma3nTensorNames.attnQ(0)]!!.data is FloatArrayTensorData<*>, "Any policy widens BF16")
            assertFalse(t.keys.any { "vision" in it })
        }
    }

    @Test
    fun `Require BF16 keeps native storage for the projections and leaves F32 norms dense`() {
        withFixture { indexPath ->
            val weights = runBlocking {
                Gemma3nSafeTensorsWeightLoader(indexPath, dtypePolicy = DTypePolicy.Require(BF16)).loadToMap(ctx, FP32::class)
            }
            val q = weights.tensors[Gemma3nTensorNames.attnQ(0)]!!
            assertTrue(q.data is Bf16DenseTensorData, "Require(BF16) must keep the projection bf16-native, got ${q.data::class.simpleName}")
            assertEquals(values(heads * headDim * hidden, Q_BASE).toList(), floats(q).toList())
            val norm = weights.tensors[Gemma3nTensorNames.inputLayernorm(0)]!!
            assertTrue(norm.data is FloatArrayTensorData<*>, "F32 norms stay dense FP32")
        }
    }

    // --- fixture ---------------------------------------------------------------

    private fun withFixture(block: (indexPath: String) -> Unit) {
        val dir = Files.createTempDirectory("gemma3n-st-fixture")
        try {
            val shard1 = "model-00001-of-00002.safetensors"
            val shard2 = "model-00002-of-00002.safetensors"
            val names1 = mutableListOf<String>()
            val names2 = mutableListOf<String>()

            writeShard(dir.resolve(shard1)) {
                bf16("model.language_model.embed_tokens.weight", listOf(vocab, hidden), EMBED_BASE, names1)
                f32("model.language_model.norm.weight", listOf(hidden), NORM_BASE, names1)
                bf16("model.language_model.per_layer_token_embd.weight", listOf(vocab, pleDim), 0.5f, names1)
                bf16("model.language_model.per_layer_model_proj.weight", listOf(pleDim, hidden), 0.25f, names1)
                f32("model.language_model.per_layer_proj_norm.weight", listOf(pleDim), 1.5f, names1)
            }
            writeShard(dir.resolve(shard2)) {
                val l = "model.language_model.layers.0"
                f32("$l.input_layernorm.weight", listOf(hidden), LN_BASE, names2)
                bf16("$l.self_attn.q_proj.weight", listOf(heads * headDim, hidden), Q_BASE, names2)
                bf16("$l.self_attn.k_proj.weight", listOf(kvHeads * headDim, hidden), 0.125f, names2)
                bf16("$l.self_attn.v_proj.weight", listOf(kvHeads * headDim, hidden), 0.25f, names2)
                bf16("$l.self_attn.o_proj.weight", listOf(hidden, heads * headDim), 0.375f, names2)
                f32("$l.post_attention_layernorm.weight", listOf(hidden), 2.0f, names2)
                bf16("$l.mlp.gate_proj.weight", listOf(ffn, hidden), 0.5f, names2)
                bf16("$l.mlp.up_proj.weight", listOf(ffn, hidden), 0.625f, names2)
                bf16("$l.mlp.down_proj.weight", listOf(hidden, ffn), 0.75f, names2)
                bf16("$l.per_layer_projection.weight", listOf(hidden, pleDim), 1.0f, names2)
                f32("$l.self_attn.q_norm.weight", listOf(headDim), 4.5f, names2)
                f32("$l.self_attn.k_norm.weight", listOf(headDim), 5.0f, names2)
                f32("$l.post_attention_norm.weight", listOf(hidden), 2.5f, names2)
                f32("$l.post_ffw_norm.weight", listOf(hidden), 3.0f, names2)
                f32("$l.post_norm.weight", listOf(hidden), 3.5f, names2)
                bf16("$l.inp_gate.weight", listOf(pleDim, hidden), 0.875f, names2)
                bf16("$l.proj.weight", listOf(hidden, pleDim), 1.25f, names2)
                // Unmapped decoy: an INT64 tensor the family never asks for.
                tensorI64("model.vision_tower.patch_embedding.position_ids", listOf(4L), LongArray(4) { it.toLong() })
                names2 += "model.vision_tower.patch_embedding.position_ids"
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
                  "model_type": "gemma3n",
                  "architectures": ["Gemma3nForConditionalGeneration"],
                  "text_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": $hidden,
                    "num_attention_heads": $heads,
                    "num_key_value_heads": $kvHeads,
                    "head_dim": $headDim,
                    "vocab_size": $vocab,
                    "intermediate_size": $ffn,
                    "hidden_size_per_layer_input": $pleDim,
                    "num_kv_shared_layers": 0,
                    "layer_types": ["full_attention"]
                  }
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
