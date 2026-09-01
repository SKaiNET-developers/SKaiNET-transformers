package sk.ainet.models.gemma3n

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class Gemma3nConfigTest {

    @Test
    fun `E2B default has no AltUp`() {
        val config = Gemma3nConfig.E2B_DEFAULT
        assertEquals(1, config.numAltupInputs)
        assertFalse(config.hasAltUp)
        assertEquals(0f, config.getActivationSparsity(0))
    }

    @Test
    fun `E4B default has AltUp with 4 inputs`() {
        val config = Gemma3nConfig.E4B_DEFAULT
        assertEquals(4, config.numAltupInputs)
        assertTrue(config.hasAltUp)
        assertEquals(0, config.altupActiveIdx)
    }

    @Test
    fun `E4B default has correct sparsity pattern`() {
        val config = Gemma3nConfig.E4B_DEFAULT
        // First 10 layers: 95% sparsity
        for (i in 0 until 10) {
            assertEquals(0.95f, config.getActivationSparsity(i), "Layer $i")
        }
        // Remaining 25 layers: no sparsity
        for (i in 10 until 35) {
            assertEquals(0f, config.getActivationSparsity(i), "Layer $i")
        }
    }

    @Test
    fun `E4B default has 16384 intermediate size`() {
        val config = Gemma3nConfig.E4B_DEFAULT
        assertEquals(16384, config.getIntermediateSize(0))
        assertEquals(16384, config.getIntermediateSize(34))
    }

    @Test
    fun `getActivationSparsity returns 0 for out of range`() {
        val config = Gemma3nConfig.E4B_DEFAULT
        assertEquals(0f, config.getActivationSparsity(100))
    }

    @Test
    fun `fromMetadata maps AltUp fields`() {
        val metadata = Gemma3nModelMetadata(
            architecture = "gemma3n",
            embeddingLength = 2048,
            perLayerEmbeddingLength = 256,
            contextLength = 8192,
            blockCount = 35,
            headCount = 8,
            kvHeadCount = 2,
            feedForwardLengths = List(35) { 16384 },
            headDim = 256,
            vocabSize = 262400,
            slidingWindow = 512,
            ropeBaseLocal = 10000f,
            ropeBaseGlobal = 1000000f,
            kvSharedLayers = 15,
            layerPattern = listOf("sliding", "sliding", "sliding", "sliding", "full"),
            numAltupInputs = 4,
            altupActiveIdx = 0,
            activationSparsityPattern = List(10) { 0.95f } + List(25) { 0.0f }
        )

        val config = Gemma3nConfig.fromMetadata(metadata)
        assertEquals(4, config.numAltupInputs)
        assertEquals(0, config.altupActiveIdx)
        assertTrue(config.hasAltUp)
        assertEquals(0.95f, config.getActivationSparsity(0))
        assertEquals(0f, config.getActivationSparsity(10))
    }

    @Test
    fun `config parser parses E4B fields from JSON`() {
        val json = """
        {
            "model_type": "gemma3n",
            "text_config": {
                "num_hidden_layers": 35,
                "hidden_size": 2048,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 262400,
                "intermediate_size": 16384,
                "altup_num_inputs": 4,
                "altup_active_idx": 0,
                "activation_sparsity_pattern": [0.95, 0.95, 0.95, 0.0, 0.0]
            }
        }
        """.trimIndent()

        val metadata = Gemma3nConfigParser.parseFromJson(json)
        assertEquals(4, metadata.numAltupInputs)
        assertEquals(0, metadata.altupActiveIdx)
        assertEquals(5, metadata.activationSparsityPattern.size)
        assertEquals(0.95f, metadata.activationSparsityPattern[0])
        assertEquals(0f, metadata.activationSparsityPattern[3])
    }

    @Test
    fun `config parser defaults AltUp fields for E2B`() {
        val json = """
        {
            "model_type": "gemma3n",
            "text_config": {
                "num_hidden_layers": 30,
                "hidden_size": 2048,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 262400,
                "intermediate_size": 8192
            }
        }
        """.trimIndent()

        val metadata = Gemma3nConfigParser.parseFromJson(json)
        assertEquals(1, metadata.numAltupInputs)
        assertEquals(0, metadata.altupActiveIdx)
        assertTrue(metadata.activationSparsityPattern.isEmpty())
    }
}
