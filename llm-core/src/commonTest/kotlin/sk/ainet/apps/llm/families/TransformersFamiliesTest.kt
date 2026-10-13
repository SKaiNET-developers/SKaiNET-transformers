package sk.ainet.apps.llm.families

import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.model.InMemoryCheckpointView
import sk.ainet.model.ModelFormat
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Descriptor coverage gate (SKEEP-006 acceptance): architectures disjoint, every shipped family identified. */
class TransformersFamiliesTest {

    @Test
    fun `every architecture string is claimed by exactly one family`() {
        val seen = HashMap<String, String>()
        for (f in TransformersFamilies.all) for (a in f.architectures) {
            val prev = seen.put(a.lowercase(), f.family)
            assertTrue(prev == null, "architecture '$a' claimed by both $prev and ${f.family}")
        }
    }

    @Test
    fun `gguf architectures of the shipped runtimes resolve to their family`() {
        val cases = mapOf(
            "llama" to "llama", "mistral" to "llama", "qwen2" to "qwen", "qwen3" to "qwen", "gemma3" to "gemma",
            "gemma4" to "gemma", "gemma3n" to "gemma", "apertus" to "apertus", "bitnet" to "bitnet", "bitnet-25" to "bitnet",
            "bert" to "bert", "voxtral" to "voxtral", "whisper" to "whisper", "moonshine" to "moonshine", "t5" to "t5",
        )
        for ((arch, family) in cases) {
            val id = TransformersFamilies.registry.identify(InMemoryCheckpointView(ModelFormat.GGUF, mapOf("general.architecture" to arch)))
            assertEquals(family, id.family, arch)
            assertNotNull(TransformersFamilies.forFamily(family), family)
        }
    }

    @Test
    fun `engine name maps survive the family override`() {
        val id = TransformersFamilies.registry.identify(InMemoryCheckpointView(ModelFormat.GGUF, mapOf("general.architecture" to "qwen3")))
        assertNotNull(TransformersFamilies.registry.nameMapFor(id))
    }

    @Test
    @Suppress("DEPRECATION")
    fun `deprecated enum view agrees with the registry`() {
        for (f in TransformersFamilies.all) for (a in f.architectures) {
            val enum = sk.ainet.apps.llm.ModelRegistry.detect(a)
            if (enum != sk.ainet.apps.llm.ModelFamily.UNKNOWN) assertEquals(f.family, enum.id, a)
        }
    }

    @Test
    fun `family decoration adds the end-of-turn ids to the stop set`() {
        val json = """
            {"added_tokens":[{"id":2,"content":"<|endoftext|>","special":true},{"id":3,"content":"<|im_end|>","special":true}],
             "model":{"type":"BPE","vocab":{"a":0,"b":1,"<|endoftext|>":2,"<|im_end|>":3,"ab":4},"merges":["a b"]}}
        """.trimIndent()
        val view = InMemoryCheckpointView.ofText(ModelFormat.SAFETENSORS, mapOf("model_type" to "qwen2"), mapOf("tokenizer.json" to json))
        val text = TextFrontEnds.resolver(TransformersFamilies.registry).resolveText(view)
        assertEquals(setOf(2, 3), text.tokenizer.specialTokens.stopIds)
        assertEquals(listOf(4, 3), text.tokenizer.encode("ab<|im_end|>").toList())
    }
}
