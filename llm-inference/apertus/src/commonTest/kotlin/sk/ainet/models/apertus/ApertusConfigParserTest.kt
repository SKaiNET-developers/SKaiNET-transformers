package sk.ainet.models.apertus

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class ApertusConfigParserTest {

    @Test
    fun testParseApertusConfig() {
        val json = """
        {
            "architectures": ["ApertusForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "xielu",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "model_type": "apertus",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "qk_norm": "per_head",
            "rope_theta": 12000000.0,
            "tie_word_embeddings": false,
            "vocab_size": 131072
        }
        """.trimIndent()

        val metadata = ApertusConfigParser.parse(json)

        assertEquals("apertus", metadata.architecture)
        assertEquals(4096, metadata.embeddingLength)
        assertEquals(8192, metadata.contextLength)
        assertEquals(32, metadata.blockCount)
        assertEquals(32, metadata.headCount)
        assertEquals(8, metadata.kvHeadCount)
        assertEquals(14336, metadata.feedForwardLength)
        assertEquals(128, metadata.ropeDimensionCount) // 4096 / 32
        assertEquals(131072, metadata.vocabSize)
        assertEquals(12000000f, metadata.ropeTheta)
        assertTrue(metadata.qkNorm)
        assertEquals("xielu", metadata.hiddenAct)
    }

    @Test
    fun testParseWithDefaults() {
        val json = """
        {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 8192,
            "vocab_size": 131072
        }
        """.trimIndent()

        val metadata = ApertusConfigParser.parse(json)

        assertEquals("apertus", metadata.architecture)
        assertEquals(2048, metadata.contextLength)
        assertEquals(128, metadata.ropeDimensionCount) // 2048 / 16
        assertEquals(12000000f, metadata.ropeTheta) // default
        assertTrue(metadata.qkNorm) // default
        assertEquals("xielu", metadata.hiddenAct) // default
    }

    @Test
    fun testParseWithTiedEmbeddings() {
        val json = """
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 131072,
            "tie_word_embeddings": true
        }
        """.trimIndent()

        val metadata = ApertusConfigParser.parse(json)
        assertTrue(metadata.tiedEmbeddings)
    }

    @Test
    fun testIsTiedEmbeddings() {
        val jsonTied = """{"tie_word_embeddings": true, "hidden_size": 1}"""
        assertTrue(ApertusConfigParser.isTiedEmbeddings(jsonTied))
    }

    @Test
    fun testMissingRequiredField() {
        val json = """
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32
        }
        """.trimIndent()

        assertFailsWith<IllegalStateException> {
            ApertusConfigParser.parse(json)
        }
    }

    @Test
    fun testParseWithExplicitHeadDim() {
        val json = """
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 131072,
            "head_dim": 64
        }
        """.trimIndent()

        val metadata = ApertusConfigParser.parse(json)
        assertEquals(64, metadata.ropeDimensionCount)
    }
}
