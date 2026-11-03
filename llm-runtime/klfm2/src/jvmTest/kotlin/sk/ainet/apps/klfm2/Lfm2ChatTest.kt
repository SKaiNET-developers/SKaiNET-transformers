package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary
import kotlin.test.*

class Lfm2ChatTest {
    private val bos = "<|startoftext|>"

    @Test fun rendersSystemUserAndAssistantPrefixWithoutNormalizingMessageBytes() {
        val messages = listOf(Lfm2ChatMessage("system", " Rules\n"), Lfm2ChatMessage("user", "Hello\r\n世界\u00a0🙂"))
        assertEquals(bos + "<|im_start|>system\n Rules\n<|im_end|>\n<|im_start|>user\nHello\r\n世界\u00a0🙂<|im_end|>\n<|im_start|>assistant\n",
            renderLfm2TextChat(messages, bos))
        assertEquals(bos, renderLfm2TextChat(listOf(Lfm2ChatMessage("system", "")), bos, addGenerationPrompt = false))
        assertEquals(bos + "<|im_start|>assistant\n", renderLfm2TextChat(listOf(Lfm2ChatMessage("system", "")), bos))
    }

    @Test fun thinkingRetentionMatchesLastAssistantIndexAndPythonTrim() {
        val messages = listOf(
            Lfm2ChatMessage("assistant", "<think>inline</think>\u0085\u001c\u00a0 final \u001f\u3000", "old thought"),
            Lfm2ChatMessage("user", "Again"),
            Lfm2ChatMessage("assistant", " second ", "last thought"),
            Lfm2ChatMessage("user", "More"),
        )
        val actual = renderLfm2TextChat(messages, bos, addGenerationPrompt = false)
        assertTrue(actual.startsWith(bos + "<|im_start|>assistant\nfinal<|im_end|>\n"))
        assertTrue(actual.contains("<think>last thought</think> second <|im_end|>"))
        assertFalse(actual.contains("old thought"))
        val preserved = renderLfm2TextChat(messages, bos, preserveThinking = true)
        assertTrue(preserved.contains("<think>old thought</think>" + messages[0].content))
    }

    @Test fun preservesSpecialTokenLiteralsAndContinueMarkerAsMessageData() {
        val content = "<|im_start|>x\n{{ dangerous }} CONTINUE_FINAL_MESSAGE_TAG "
        assertEquals(bos + "<|im_start|>assistant\n" + content + "<|im_end|>\n",
            renderLfm2TextChat(listOf(Lfm2ChatMessage("assistant", content)), bos, false))
        val repeated = listOf(Lfm2ChatMessage("assistant", "a</think>discard</think>  final "), Lfm2ChatMessage("assistant", "last"))
        assertTrue(renderLfm2TextChat(repeated, bos).contains("assistant\nfinal<|im_end|>"))
    }

    @Test fun unsupportedMessageFormsAndChangedTemplateFailClosed() {
        assertFailsWith<IllegalArgumentException> { renderLfm2TextChat(emptyList(), bos) }
        assertFailsWith<IllegalArgumentException> { renderLfm2TextChat(listOf(Lfm2ChatMessage("tool", "x")), bos) }
        assertFailsWith<IllegalArgumentException> { renderLfm2TextChat(listOf(Lfm2ChatMessage("user", "x"), Lfm2ChatMessage("system", "late")), bos) }
        assertFailsWith<IllegalArgumentException> { renderLfm2TextChat(listOf(Lfm2ChatMessage("user", "x", "thought")), bos) }
        assertFailsWith<IllegalArgumentException> { requirePinnedLfm2TextMetadata("chat_template.jinja", "{{ messages }}".toByteArray()) }
        assertFailsWith<IllegalArgumentException> { requirePinnedLfm2TextMetadata("arbitrary.jinja", byteArrayOf()) }
    }

    @Test fun generationTokenOverridesTakePrecedenceAndInvalidValuesFail() {
        assertEquals(Lfm2GenerationTokens(1, setOf(7)), resolveLfm2GenerationTokens(1, setOf(7), "{}", 20))
        assertEquals(Lfm2GenerationTokens(2, setOf(8, 9)), resolveLfm2GenerationTokens(1, setOf(7), """{"bos_token_id":2,"eos_token_id":[8,9]}""", 20))
        assertEquals(Lfm2GenerationTokens(2, setOf(8)), resolveLfm2GenerationTokens(null, emptySet(), """{"bos_token_id":2,"eos_token_id":8}""", 20))
        for (invalid in listOf("""{"eos_token_id":null}""", """{"eos_token_id":[]}""", """{"eos_token_id":20}""", """{"bos_token_id":-1}""", """{"bos_token_id":null}""")) {
            assertFailsWith<IllegalArgumentException> { resolveLfm2GenerationTokens(1, setOf(7), invalid, 20) }
        }
    }

    @Test fun unicodeAdapterChangesOnlySplitPatternAndRequiresItsPinnedTopology() {
        val original = """{"model":{"type":"BPE"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Isolated"},{"type":"ByteLevel","add_prefix_space":false}]}}"""
        val before = Json.parseToJsonElement(original).jsonObject
        val after = Json.parseToJsonElement(unicodeLfm2TokenizerSpecification(original)).jsonObject
        assertEquals(before["model"], after["model"])
        val elements = after.getValue("pre_tokenizer").jsonObject.getValue("pretokenizers").jsonArray
        assertEquals("(?U)\\s+", elements[0].jsonObject.getValue("pattern").jsonObject.getValue("Regex").jsonPrimitive.content)
        assertEquals(before.getValue("pre_tokenizer").jsonObject.getValue("pretokenizers").jsonArray[1], elements[1])
        assertFailsWith<IllegalArgumentException> { unicodeLfm2TokenizerSpecification("""{"pre_tokenizer":{"type":"Sequence","pretokenizers":[]}}""") }
    }

    @Test fun tokenizerInventoryDeduplicatesAddedTokensAndAllowsPaddedModelCapacity() {
        val source = """{"model":{"type":"BPE","vocab":{"a":0,"b":1}},"added_tokens":[{"id":1,"content":"b"},{"id":2,"content":"<stop>"}]}"""
        assertEquals(mapOf(0 to "a", 1 to "b", 2 to "<stop>"), lfm2TokenizerInventory(source, 8))
        assertFailsWith<IllegalArgumentException> { lfm2TokenizerInventory(source, 2) }
    }

    @Test fun conflictingTokenIdsAndMappingsFailInsteadOfBeingSilentlyOverwritten() {
        for (source in listOf(
            """{"model":{"type":"BPE","vocab":{"a":0,"b":0}},"added_tokens":[]}""",
            """{"model":{"type":"BPE","vocab":{"a":0}},"added_tokens":[{"id":0,"content":"b"}]}""",
            """{"model":{"type":"BPE","vocab":{"a":0}},"added_tokens":[{"id":1,"content":"a"}]}""",
            """{"model":{"type":"BPE","vocab":{"a":-1}},"added_tokens":[]}""",
        )) assertFailsWith<IllegalArgumentException> { lfm2TokenizerInventory(source, 8) }
    }

    @Test fun decodingUnassignedOutputRowsFailsWithoutMaskingOrCallingTheTokenizer() {
        var decoded = 0
        var encodedId = 1
        val tokenizer = object : Tokenizer {
            override val vocabulary = object : Vocabulary {
                override val size = 2
                override fun tokenToId(token: String): Int? = when (token) { "a" -> 0; "b" -> 1; else -> null }
                override fun idToToken(id: Int): String? = when (id) { 0 -> "a"; 1 -> "b"; else -> null }
            }
            override val specialTokens = SpecialTokens()
            override fun encode(text: String): IntArray = intArrayOf(encodedId)
            override fun decode(ids: IntArray): String { decoded++; return ids.joinToString("") { vocabulary.idToToken(it)!! } }
        }
        val frontend = Lfm2TextFrontend(tokenizer, "a", 0, setOf(1), 8, emptyMap(), "test", setOf(0, 1), emptySet(), emptySet())
        assertEquals(8, frontend.vocabularySize)
        assertEquals(2, frontend.tokenizerVocabularySize)
        assertEquals(1, frontend.maximumAssignedTokenId)
        assertFailsWith<IllegalArgumentException> { frontend.decode(intArrayOf(7)) }
        assertEquals(0, decoded)
        assertEquals("b", frontend.decode(intArrayOf(1)))
        assertContentEquals(intArrayOf(1), frontend.encode("b"))
        encodedId = 7
        assertFailsWith<IllegalArgumentException> { frontend.encode("x") }
    }

    @Test fun explicitDecodePoliciesPreserveNonSpecialTokensAndJoinUtf8AfterFiltering() {
        val pieces = listOf("Ã", "©", "<stop>", "<think>", "</think>", "null", "python")
        val calls = mutableListOf<List<Int>>()
        val tokenizer = object : Tokenizer {
            override val vocabulary = object : Vocabulary {
                override val size = pieces.size
                override fun tokenToId(token: String): Int? = pieces.indexOf(token).takeIf { it >= 0 }
                override fun idToToken(id: Int): String? = pieces.getOrNull(id)
            }
            // Deliberately wrong backend policy: the frontend must not delegate skipping here.
            override val specialTokens = SpecialTokens(declared = setOf(2, 3, 4, 5, 6))
            override fun encode(text: String): IntArray = error("Unused in decoder regression")
            override fun decode(ids: IntArray): String {
                calls.add(ids.toList())
                val bytes = mutableListOf<Byte>()
                return buildString {
                    fun flush() { if (bytes.isNotEmpty()) { append(bytes.toByteArray().decodeToString()); bytes.clear() } }
                    for (id in ids) when (id) {
                        0 -> bytes.add(0xc3.toByte())
                        1 -> bytes.add(0xa9.toByte())
                        else -> { flush(); append(pieces[id]) }
                    }
                    flush()
                }
            }
        }
        val source = """{"added_tokens":[{"id":2,"special":true},{"id":3,"special":false},{"id":4,"special":false},{"id":6,"special":false}]}"""
        val policies = lfm2DecodingTokenSets(source)
        val frontend = Lfm2TextFrontend(tokenizer, "", 2, setOf(2), 8, emptyMap(), "test", pieces.indices.toSet(), policies.checkpoint, policies.hf5Compat)
        val ids = intArrayOf(3, 0, 2, 1, 4, 5, 6)
        assertEquals("<think>é</think>nullpython", frontend.decode(ids))
        assertEquals(listOf(3, 0, 1, 4, 5, 6), calls.last())
        assertEquals("énull", frontend.decode(ids, policy = Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT))
        assertEquals(listOf(0, 1, 5), calls.last())
        assertEquals(1, frontend.checkpointSpecialTokenCount)
        assertEquals(4, frontend.hf5CompatSpecialTokenCount)
        assertEquals(setOf(3, 4, 6), frontend.hf5PromotedSpecialTokenIds)
        frontend.decode(ids, skipSpecialTokens = false, policy = Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT)
        assertEquals(ids.toList(), calls.last())
    }
}
