package sk.ainet.models.gemma

import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.io.safetensors.readTextFile
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration tests for Gemma 4 SafeTensors loading.
 * Requires Gemma 4 model files in models-gemma4/ directory.
 */
class Gemma4SafeTensorsIntegrationTest {

    private val modelsDir = "models-gemma4"
    private val configPath = "$modelsDir/config.json"
    private val indexPath = "$modelsDir/model.safetensors.index.json"

    private fun skipIfModelNotPresent() {
        assumeTrue(
            "Skipping test - Gemma 4 model files not present in $modelsDir",
            File(configPath).exists()
        )
    }

    @Tag("integration")
    @Test
    fun `test Gemma4 config json parsing`() {
        skipIfModelNotPresent()

        val configJson = readTextFile(configPath)
        assertNotNull(configJson, "Failed to read config.json")

        val metadata = Gemma4ConfigParser.parseFromJson(configJson)

        assertEquals("gemma4", metadata.architecture)
        assertTrue(metadata.blockCount > 0, "blockCount should be positive")
        assertTrue(metadata.embeddingLength > 0, "embeddingLength should be positive")
        assertTrue(metadata.vocabSize > 0, "vocabSize should be positive")
        assertTrue(metadata.headCount > 0, "headCount should be positive")
        assertTrue(metadata.kvHeadCount > 0, "kvHeadCount should be positive")
        assertTrue(metadata.headDim > 0, "headDim should be positive")
        assertTrue(metadata.globalHeadDim > 0, "globalHeadDim should be positive")
        assertTrue(metadata.slidingWindow > 0, "slidingWindow should be positive")
        assertTrue(metadata.kvSharedLayers > 0, "kvSharedLayers should be positive")
        assertTrue(metadata.layerTypes.isNotEmpty(), "layerTypes should not be empty")
        assertEquals("proportional", metadata.ropeParametersFull.ropeType)

        println("Gemma 4 config parsing test PASSED")
        println("  Architecture: ${metadata.architecture}")
        println("  Layers: ${metadata.blockCount}")
        println("  Hidden size: ${metadata.embeddingLength}")
        println("  Global head dim: ${metadata.globalHeadDim}")
        println("  Vocab size: ${metadata.vocabSize}")
        println("  RoPE type (full): ${metadata.ropeParametersFull.ropeType}")
    }

    @Tag("integration")
    @Test
    fun `test Gemma4 safetensors index loading`() = runBlocking {
        skipIfModelNotPresent()
        assumeTrue(
            "Skipping - safetensors index not present",
            File(indexPath).exists()
        )

        val reader = StreamingShardedSafeTensorsReader.openFromIndex(indexPath)

        try {
            assertTrue(reader.tensors.isNotEmpty(), "No tensors loaded")
            assertTrue(reader.isComplete, "Model should be complete")

            val expectedTensors = listOf(
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
                "model.language_model.layers.0.input_layernorm.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "model.language_model.layers.0.self_attn.v_proj.weight",
                "model.language_model.layers.0.self_attn.o_proj.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.up_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight"
            )

            for (name in expectedTensors) {
                val tensor = reader.tensors.find { it.name == name }
                assertNotNull(tensor, "Missing expected tensor: $name")
            }

            println("Gemma 4 SafeTensors index loading test PASSED")
            println("  Total tensors: ${reader.tensors.size}")
            println("  Shards loaded: ${reader.loadedShards.size}")
        } finally {
            reader.close()
        }
    }

    @Test
    fun `test Gemma4 tensor name mapping`() {
        assertEquals("token_embd.weight", Gemma4TensorNames.TOKEN_EMBEDDINGS)
        assertEquals("output_norm.weight", Gemma4TensorNames.OUTPUT_NORM)
        assertEquals("output.weight", Gemma4TensorNames.OUTPUT_WEIGHT)

        assertEquals("blk.0.attn_norm.weight", Gemma4TensorNames.inputLayernorm(0))
        assertEquals("blk.5.attn_q.weight", Gemma4TensorNames.attnQ(5))
        assertEquals("blk.10.ffn_gate.weight", Gemma4TensorNames.ffnGate(10))
        assertEquals("blk.3.attn_q_norm.weight", Gemma4TensorNames.attnQNorm(3))
    }

    @Test
    fun `test Gemma4Config fromMetadata creates valid config`() {
        val metadata = Gemma4ModelMetadata(
            architecture = "gemma4",
            embeddingLength = 2304,
            contextLength = 131072,
            blockCount = 34,
            headCount = 8,
            kvHeadCount = 4,
            intermediateSize = 9216,
            headDim = 256,
            globalHeadDim = 256,
            vocabSize = 262144,
            slidingWindow = 512,
            kvSharedLayers = 20,
            layerTypes = listOf("sliding_attention", "full_attention"),
            ropeParametersFull = Gemma4RopeConfig(
                base = 1000000f,
                ropeType = "proportional",
                factor = 2.0f,
                partialRotaryFactor = 0.5f
            ),
            ropeParametersSliding = Gemma4RopeConfig(base = 10000f),
            maxPositionEmbeddings = 131072
        )

        val config = Gemma4Config.fromMetadata(metadata)

        assertEquals(2304, config.hiddenSize)
        assertEquals(34, config.numLayers)
        assertEquals(256, config.globalHeadDim)
        assertEquals("proportional", config.ropeType)
        assertEquals(0.5f, config.partialRotaryFactor)
        assertEquals(15, config.effectiveCacheLayers)
    }
}
