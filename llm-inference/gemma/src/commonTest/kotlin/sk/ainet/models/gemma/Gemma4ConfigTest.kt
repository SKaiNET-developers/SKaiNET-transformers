package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class Gemma4ConfigTest {

    @Test
    fun `E2B default has correct dimensions`() {
        val config = Gemma4Config.E2B_DEFAULT
        assertEquals(2304, config.hiddenSize)
        assertEquals(34, config.numLayers)
        assertEquals(8, config.numAttentionHeads)
        assertEquals(4, config.numKvHeads)
        assertEquals(256, config.headDim)
        assertEquals(256, config.globalHeadDim)
        assertEquals(9216, config.intermediateSize)
        assertEquals(512, config.slidingWindow)
        assertEquals(131072, config.maxPositionEmbeddings)
    }

    @Test
    fun `E2B default has correct KV sharing`() {
        val config = Gemma4Config.E2B_DEFAULT
        assertEquals(20, config.kvSharedLayers)
        // Layer 13 is the boundary (34 - 20 = 14)
        assertFalse(config.isKvShared(13))
        assertTrue(config.isKvShared(14))
        assertTrue(config.isKvShared(33))
        assertEquals(15, config.effectiveCacheLayers)
    }

    @Test
    fun `E2B default last layer is always global`() {
        val config = Gemma4Config.E2B_DEFAULT
        assertEquals(LayerType.GLOBAL, config.getLayerType(33))
        assertTrue(config.isGlobalLayer(33))
    }

    @Test
    fun `E2B default has correct layer type pattern`() {
        val config = Gemma4Config.E2B_DEFAULT
        // Pattern: 5 sliding + 1 full repeating
        for (i in 0 until 5) {
            assertEquals(LayerType.SLIDING, config.getLayerType(i), "Layer $i should be sliding")
        }
        assertEquals(LayerType.GLOBAL, config.getLayerType(5), "Layer 5 should be global")
        // Next group
        for (i in 6 until 11) {
            assertEquals(LayerType.SLIDING, config.getLayerType(i), "Layer $i should be sliding")
        }
        assertEquals(LayerType.GLOBAL, config.getLayerType(11), "Layer 11 should be global")
    }

    @Test
    fun `getHeadDim returns correct dim per layer type`() {
        val config = Gemma4Config(
            headDim = 128,
            globalHeadDim = 256,
            layerTypes = listOf("sliding_attention", "full_attention")
        )
        assertEquals(128, config.getHeadDim(0))
        assertEquals(256, config.getHeadDim(1))
    }

    @Test
    fun `getRotaryDim applies partial rotary factor for global layers`() {
        val config = Gemma4Config(
            headDim = 256,
            globalHeadDim = 256,
            partialRotaryFactor = 0.5f,
            layerTypes = listOf("sliding_attention", "full_attention")
        )
        assertEquals(256, config.getRotaryDim(0)) // sliding: full rotation
        assertEquals(128, config.getRotaryDim(1)) // global: partial (256 * 0.5)
    }

    @Test
    fun `getRopeBase returns different bases per layer type`() {
        val config = Gemma4Config(
            ropeBaseLocal = 10000f,
            ropeBaseGlobal = 1000000f,
            layerTypes = listOf("sliding_attention", "full_attention")
        )
        assertEquals(10000f, config.getRopeBase(0))
        assertEquals(1000000f, config.getRopeBase(1))
    }

    @Test
    fun `GQA dimensions are correct`() {
        val config = Gemma4Config.E2B_DEFAULT
        assertEquals(2, config.numHeadsPerKv)
        assertEquals(2048, config.queryDim) // 8 * 256
        assertEquals(1024, config.kvDim)    // 4 * 256
    }

    @Test
    fun `fromMetadata maps all fields correctly`() {
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
                originalMaxPositionEmbeddings = 8192,
                partialRotaryFactor = 0.5f
            ),
            ropeParametersSliding = Gemma4RopeConfig(
                base = 10000f,
                ropeType = "default"
            ),
            maxPositionEmbeddings = 131072
        )

        val config = Gemma4Config.fromMetadata(metadata)
        assertEquals(2304, config.hiddenSize)
        assertEquals(34, config.numLayers)
        assertEquals(8, config.numAttentionHeads)
        assertEquals(4, config.numKvHeads)
        assertEquals(256, config.headDim)
        assertEquals(256, config.globalHeadDim)
        assertEquals(9216, config.intermediateSize)
        assertEquals(512, config.slidingWindow)
        assertEquals(10000f, config.ropeBaseLocal)
        assertEquals(1000000f, config.ropeBaseGlobal)
        assertEquals("proportional", config.ropeType)
        assertEquals(2.0f, config.ropeFactor)
        assertEquals(0.5f, config.partialRotaryFactor)
        assertEquals(8192, config.originalMaxPositionEmbeddings)
        assertEquals(20, config.kvSharedLayers)
        assertEquals(131072, config.maxPositionEmbeddings)
    }

    @Test
    fun `config parser parses E2B config json`() {
        val json = """
        {
            "model_type": "gemma4",
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "text_config": {
                "num_hidden_layers": 34,
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "global_head_dim": 256,
                "vocab_size": 262144,
                "intermediate_size": 9216,
                "max_position_embeddings": 131072,
                "sliding_window": 512,
                "num_kv_shared_layers": 20,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "rope_parameters": {
                    "full": {
                        "rope_type": "proportional",
                        "factor": 2.0,
                        "original_max_position_embeddings": 8192,
                        "partial_rotary_factor": 0.5
                    },
                    "sliding": {
                        "rope_type": "default",
                        "original_max_position_embeddings": 8192
                    }
                }
            }
        }
        """.trimIndent()

        val metadata = Gemma4ConfigParser.parseFromJson(json)
        assertEquals("gemma4", metadata.architecture)
        assertEquals(34, metadata.blockCount)
        assertEquals(2304, metadata.embeddingLength)
        assertEquals(8, metadata.headCount)
        assertEquals(4, metadata.kvHeadCount)
        assertEquals(256, metadata.headDim)
        assertEquals(256, metadata.globalHeadDim)
        assertEquals(262144, metadata.vocabSize)
        assertEquals(9216, metadata.intermediateSize)
        assertEquals(131072, metadata.contextLength)
        assertEquals(512, metadata.slidingWindow)
        assertEquals(20, metadata.kvSharedLayers)
        assertEquals(2, metadata.bosTokenId)
        assertEquals(1, metadata.eosTokenId)
        assertEquals(0, metadata.padTokenId)

        // Verify layer types
        assertEquals(6, metadata.layerTypes.size)
        assertEquals("sliding_attention", metadata.layerTypes[0])
        assertEquals("full_attention", metadata.layerTypes[5])

        // Verify RoPE parameters
        assertEquals("proportional", metadata.ropeParametersFull.ropeType)
        assertEquals(2.0f, metadata.ropeParametersFull.factor)
        assertEquals(8192, metadata.ropeParametersFull.originalMaxPositionEmbeddings)
        assertEquals(0.5f, metadata.ropeParametersFull.partialRotaryFactor)
        assertEquals("default", metadata.ropeParametersSliding.ropeType)
    }

    @Test
    fun `config parser defaults optional fields`() {
        val json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "num_hidden_layers": 34,
                "hidden_size": 2304,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 262144
            }
        }
        """.trimIndent()

        val metadata = Gemma4ConfigParser.parseFromJson(json)
        assertEquals(256, metadata.globalHeadDim)
        assertEquals(512, metadata.slidingWindow)
        assertEquals(20, metadata.kvSharedLayers)
        assertEquals(131072, metadata.contextLength)
        assertEquals(2, metadata.bosTokenId)
    }

    @Test
    fun `getCacheLayerIndex maps shared layers correctly`() {
        val config = Gemma4Config(numLayers = 34, kvSharedLayers = 20)
        // Non-shared layers map to themselves
        assertEquals(0, config.getCacheLayerIndex(0))
        assertEquals(13, config.getCacheLayerIndex(13))
        // Shared layers (14+) all map to slot 14
        assertEquals(14, config.getCacheLayerIndex(14))
        assertEquals(14, config.getCacheLayerIndex(20))
        assertEquals(14, config.getCacheLayerIndex(33))
    }

    @Test
    fun `E4B default has correct dimensions`() {
        val config = Gemma4Config.E4B_DEFAULT
        assertEquals(2560, config.hiddenSize)
        assertEquals(42, config.numLayers)
        assertEquals(10, config.numAttentionHeads)
        assertEquals(5, config.numKvHeads)
        assertEquals(10240, config.intermediateSize)
    }
}
