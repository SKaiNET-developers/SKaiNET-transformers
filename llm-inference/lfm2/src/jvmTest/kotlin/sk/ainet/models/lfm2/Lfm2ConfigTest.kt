package sk.ainet.models.lfm2

import kotlinx.serialization.json.*
import kotlin.test.*

internal fun tinyConfigJson(): JsonObject = Json.parseToJsonElement("""
    {
      "architectures":["Lfm2ForCausalLM"], "model_type":"lfm2", "hidden_size":4,
      "num_attention_heads":2, "num_key_value_heads":1, "vocab_size":8,
      "num_hidden_layers":2, "layer_types":["conv","full_attention"],
      "conv_L_cache":3, "conv_bias":false, "block_use_swiglu":true,
      "block_ff_dim":8, "block_auto_adjust_ff_dim":false, "block_multiple_of":4,
      "norm_eps":0.00001, "rope_parameters":{"rope_type":"default","rope_theta":1000000.0},
      "max_position_embeddings":32, "tie_embedding":true, "bos_token_id":1, "eos_token_id":7
    }
""").jsonObject

class Lfm2ConfigTest {
    private fun modify(name: String, value: JsonElement): String =
        JsonObject(tinyConfigJson() + (name to value)).toString()

    @Test
    fun createsRuntimeDimensionsAndStopTokensWithoutAnEvaluationFixture() {
        val config = Lfm2Config.parse(tinyConfigJson().toString())
        assertEquals(4, config.hiddenSize)
        assertEquals(8, config.effectiveIntermediateSize)
        assertEquals(setOf(7), config.eosTokenIds)
        assertEquals(1, config.bosTokenId)
        assertEquals(32, config.toDimensions().contextLimit)
        assertEquals(16, config.toDimensions(16).contextLimit)
        assertFailsWith<IllegalArgumentException> { config.toDimensions(33) }
        assertFailsWith<IllegalArgumentException> { config.toDimensions(0) }
    }

    @Test
    fun ffnAdjustmentPreservesReferencePrecedenceAndNullBranch() {
        val config = tinyConfigJson().toMutableMap().apply {
            put("block_auto_adjust_ff_dim", JsonPrimitive(true))
            put("block_ff_dim", JsonPrimitive(6656))
            put("intermediate_size", JsonPrimitive(128))
            put("block_multiple_of", JsonPrimitive(256))
        }
        assertEquals(4608, Lfm2Config.parse(JsonObject(config).toString()).effectiveIntermediateSize)
        config["block_ffn_dim_multiplier"] = JsonNull
        assertEquals(4437, Lfm2Config.parse(JsonObject(config).toString()).effectiveIntermediateSize)
        config["block_ffn_dim_multiplier"] = JsonPrimitive(1e100)
        assertFailsWith<IllegalArgumentException> { Lfm2Config.parse(JsonObject(config).toString()) }
    }

    @Test
    fun rejectsUnsupportedArchitectureSemanticsBeforeLoadingWeights() {
        val unsupported = mapOf(
            "model_type" to JsonPrimitive("lfm2_moe"),
            "conv_bias" to JsonPrimitive(true),
            "attention_bias" to JsonPrimitive(true),
            "block_use_swiglu" to JsonPrimitive(false),
            "use_pos_enc" to JsonPrimitive(false),
            "tie_embedding" to JsonPrimitive(false),
            "block_dim" to JsonPrimitive(8),
            "head_dim" to JsonPrimitive(8),
            "num_hidden_layers" to JsonPrimitive(3),
            "num_key_value_heads" to JsonPrimitive(3),
            "block_norm_eps" to JsonPrimitive(0.01),
            "norm_eps" to JsonPrimitive(-1),
            "rope_scaling" to buildJsonObject { put("factor", 2) },
            "rope_parameters" to buildJsonObject { put("rope_type", "linear"); put("rope_theta", 10000) },
            "layer_types" to buildJsonArray { add("moe"); add("full_attention") },
        )
        for ((key, value) in unsupported) assertFailsWith<IllegalArgumentException>(key) { Lfm2Config.parse(modify(key, value)) }
    }

    @Test
    fun supportsDeclaredEosSetAndRejectsOutOfVocabularyIds() {
        assertEquals(setOf(6, 7), Lfm2Config.parse(modify("eos_token_id", buildJsonArray { add(6); add(7); add(7) })).eosTokenIds)
        assertFailsWith<IllegalArgumentException> { Lfm2Config.parse(modify("bos_token_id", JsonPrimitive(8))) }
        assertFailsWith<IllegalArgumentException> { Lfm2Config.parse(modify("eos_token_id", JsonPrimitive(-1))) }
        assertFailsWith<IllegalArgumentException> { Lfm2Config.parse(modify("tie_word_embeddings", JsonPrimitive(false))) }
    }
}
