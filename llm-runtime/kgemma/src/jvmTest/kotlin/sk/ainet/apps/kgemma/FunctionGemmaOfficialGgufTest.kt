package sk.ainet.apps.kgemma

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.io.JvmRandomAccessSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Env-gated tests against the OFFICIAL google/functiongemma-270m-it GGUF
 * (e.g. unsloth/functiongemma-270m-it-GGUF, BF16 recommended at this size):
 *
 *   FUNCTIONGEMMA_GGUF=/path/to/functiongemma-270m-it-BF16.gguf \
 *     ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:jvmTest \
 *     --tests "*FunctionGemmaOfficialGgufTest*"
 *
 * Two layers:
 *  - PIN: the checkpoint's own vocab + embedded `tokenizer.chat_template` must
 *    contain exactly the control tokens `FunctionGemmaOfficialChatTemplate`
 *    emits — the rendering is never allowed to drift from the release.
 *  - E2E: declare one tool, ask a matching question, greedy-decode, assert one
 *    parsed ToolCall and a natural stop.
 */
class FunctionGemmaOfficialGgufTest {

    private val ggufPath: String? = System.getenv("FUNCTIONGEMMA_GGUF")

    private fun requireGguf(): String? {
        if (ggufPath.isNullOrBlank()) {
            println("SKIP: FUNCTIONGEMMA_GGUF not set")
            return null
        }
        return ggufPath
    }

    @Test
    fun pin_template_tokens_against_embedded_chat_template_and_vocab() {
        val gguf = requireGguf() ?: return
        val fields = JvmRandomAccessSource.open(gguf).use { source ->
            sk.ainet.io.gguf.StreamingGGUFReader.open(source).fields
        }
        val embedded = fields["tokenizer.chat_template"] as? String
        val tokenizer = JvmRandomAccessSource.open(gguf).use { src ->
            GGUFTokenizer.fromRandomAccessSource(src)
        }

        val required = listOf(
            "<start_of_turn>", "<end_of_turn>",
            "<start_function_declaration>", "<end_function_declaration>",
            "<start_function_call>", "<end_function_call>",
            "<start_function_response>", "<end_function_response>",
            "<escape>",
        )
        for (tok in required) {
            assertTrue(
                tokenizer.tokenId(tok) != null,
                "vocab must carry $tok as a single token (official FunctionGemma release)",
            )
            // Presence is not enough — encode must emit the single id (token_type
            // CONTROL/USER_DEFINED drives the atomic-specials branch). If this
            // fails, the model never actually sees the declaration blocks.
            val ids = tokenizer.encode(tok)
            assertTrue(
                ids.size == 1 && ids[0] == tokenizer.tokenId(tok),
                "$tok must encode atomically to its vocab id ${tokenizer.tokenId(tok)}, got ${ids.toList()}",
            )
        }
        if (embedded != null) {
            for (tok in required) {
                assertTrue(
                    embedded.contains(tok),
                    "embedded chat_template must reference $tok — template drift? Embedded:\n$embedded",
                )
            }
            assertTrue(
                embedded.contains("declaration:") && embedded.contains("call:"),
                "compact declaration/call prefixes expected in embedded template:\n$embedded",
            )
        } else {
            println("WARN: GGUF carries no tokenizer.chat_template — vocab pin only")
        }
    }

    @Test
    fun pin_prompt_tokenization_against_llamacpp_reference() {
        val gguf = requireGguf() ?: return
        val tokenizer = JvmRandomAccessSource.open(gguf).use { src ->
            GGUFTokenizer.fromRandomAccessSource(src)
        }
        // The exact prompt FunctionGemmaOfficialChatTemplate renders for the
        // e2e get_weather case, tokenized by llama.cpp b10621 (/tokenize,
        // add_special+parse_special) against this same BF16 GGUF. 85 ids,
        // starting <bos>=2, <start_of_turn>=105.
        val prompt = "<start_of_turn>developer\n" +
            "You are a model that can do function calling with the following functions" +
            "<start_function_declaration>declaration:get_weather{description:<escape>Get current weather for a location<escape>," +
            "parameters:{properties:{location:{description:<escape>City name<escape>,type:<escape>STRING<escape>}}," +
            "required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n" +
            "<start_of_turn>user\nWhat's the weather in Paris?<end_of_turn>\n<start_of_turn>model\n"
        val reference = intArrayOf(
            2, 105, 55060, 107, 3048, 659, 496, 2028, 600, 740, 776, 1292, 11687, 607, 506, 2269,
            5151, 46, 163688, 236787, 828, 236779, 19323, 236782, 7777, 236787, 52, 3407, 1873,
            7606, 573, 496, 4563, 52, 236764, 19031, 29616, 15921, 29616, 7125, 29616, 7777,
            236787, 52, 17698, 1463, 52, 236764, 2084, 236787, 52, 35410, 52, 5237, 15979, 24845,
            52, 7125, 52, 1604, 2084, 236787, 52, 60688, 52, 1807, 47, 106, 107, 105, 2364, 107,
            3689, 236789, 236751, 506, 7606, 528, 9079, 236881, 106, 107, 105, 4368, 107,
        )
        val ours = intArrayOf(tokenizer.bosTokenId) + tokenizer.encode(prompt)
        val firstDiff = (0 until minOf(ours.size, reference.size)).firstOrNull { ours[it] != reference[it] }
        assertTrue(
            ours.contentEquals(reference),
            "tokenization diverges from llama.cpp reference: ours n=${ours.size} ref n=${reference.size}, " +
                "first diff at ${firstDiff ?: "length"}: " +
                "ours=${firstDiff?.let { ours.getOrNull(it) }} (${firstDiff?.let { runCatching { tokenizer.decode(ours[it]) }.getOrNull() }}) " +
                "ref=${firstDiff?.let { reference.getOrNull(it) }} (${firstDiff?.let { runCatching { tokenizer.decode(reference[it]) }.getOrNull() }}); " +
                "ours=${ours.toList()}",
        )
    }

    @Test
    fun e2e_declares_tool_and_gets_one_parsed_call() {
        val gguf = requireGguf() ?: return
        val fg = FunctionGemma.fromGguf(gguf, style = FunctionGemma.Style.OFFICIAL)
        val weatherTool = ToolDefinition(
            name = "get_weather",
            description = "Get current weather for a location",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("location") {
                        put("type", "string")
                        put("description", "City name")
                    }
                }
            },
        )
        val turn = fg.callWithTools(
            userText = "What's the weather in Paris?",
            tools = listOf(weatherTool),
            maxTokens = 64,
            temperature = 0f,
        )
        println("OFFICIAL text='${turn.text}' calls=${turn.calls}")
        assertTrue(turn.calls.isNotEmpty(), "expected a parsed tool call, got text='${turn.text}'")
        val call = turn.calls.first()
        assertEquals("get_weather", call.tool)
        assertTrue(
            (call.args["location"] ?: "").contains("Paris", ignoreCase = true),
            "expected location≈Paris, got ${call.args}",
        )
    }
}
