package sk.ainet.apps.kgemma

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.Gemma4ChatTemplate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Golden-token parity gate for Gemma 4 E2B on a **high-confidence** reference
 * position — the kind the maturity gate requires.
 *
 * The `"Hi"` diagnostic prompt is a poor parity probe: llama.cpp's own
 * distribution there is nearly flat (top-5 inside ~1.3 logprob), so the argmax
 * flips on ordinary Q4_K numeric noise. This prompt is the opposite: the
 * reference picks `The` at logprob -0.01 (≈99%), 5.4 nats clear of the
 * runner-up, and completes "The translation of "hello world" to German is
 * **Hallo Welt**."
 *
 * Reference captured with llama.cpp b10621 against the same
 * `gemma-4-E2B-it-Q4_K_M.gguf` (sha256 740185b2…), `/tokenize` with
 * `parse_special`, then `/completion` with those ids at temperature 0.
 *
 * Gated on `GEMMA4_E2B_GGUF_PATH`.
 */
class Gemma4ChatGoldenTokenTest {

    private companion object {
        /** llama.cpp tokenization of the rendered single-user-turn chat prompt. */
        val REFERENCE_PROMPT_IDS = intArrayOf(
            2, 105, 2364, 107, 40414, 756, 23391, 1902, 236789, 531, 9115, 106, 107, 105, 4368, 107,
        )
        /** llama.cpp's greedy first prediction: id 818 = "The", logprob -0.01. */
        const val REFERENCE_FIRST_TOKEN = 818
    }

    /**
     * Diagnostic: does the logit scale collapse with prompt LENGTH or with the
     * presence of CONTROL tokens? Prints max|logit| and the argmax after
     * prefills of increasing length, for an all-ordinary-token prompt and for
     * the reference chat prompt. Run with GEMMA4_E2B_GGUF_PATH set.
     */
    @Test
    fun probe_logit_scale_by_prefill_length() {
        val path = System.getenv("GEMMA4_E2B_GGUF_PATH")?.trim().orEmpty()
        if (path.isEmpty() || System.getenv("GEMMA4_PROBE") != "1") {
            println("[skip] set GEMMA4_E2B_GGUF_PATH and GEMMA4_PROBE=1"); return
        }
        val tokenizer = JvmRandomAccessSource.open(path).use { GGUFTokenizer.fromRandomAccessSource(it) }
        val memSeg = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSeg)
        try {
            val runtime = runBlocking {
                Gemma4Ingestion<FP32>(ctx = ctx, dtype = FP32::class, config = Gemma4LoadConfig())
                    .loadDslRuntimeStreaming { JvmRandomAccessSource.open(path) }
            }
            // Ordinary prose, no control tokens: BOS + encoded sentence.
            val prose = intArrayOf(tokenizer.bosTokenId) +
                tokenizer.encode("The quick brown fox jumps over the lazy dog and then keeps running through the field")
            fun report(label: String, ids: IntArray, upTo: Int) {
                runtime.reset()
                var logits = runtime.forward(ids[0])
                for (i in 1 until upTo) logits = runtime.forward(ids[i])
                val buf = logits.data.copyToFloatArray()
                var best = 0
                for (i in buf.indices) if (buf[i] > buf[best]) best = i
                println(
                    "%-10s n=%2d maxLogit=%+8.3f argmax=%6d '%s'".format(
                        label, upTo, buf[best], best,
                        runCatching { tokenizer.decode(best) }.getOrElse { "?" }.replace("\n", "\\n"),
                    )
                )
            }
            for (n in intArrayOf(2, 4, 8, 12, 16)) {
                if (n <= prose.size) report("prose", prose, n)
            }
            for (n in intArrayOf(2, 4, 8, 12, 16)) {
                report("chat", REFERENCE_PROMPT_IDS, n)
            }
        } finally {
            memSeg.close()
        }
    }

    /**
     * A/B: is the chat-prompt divergence caused by the keep-packed +
     * row-dequant `token_embd` path, or does it reproduce with a fully
     * dequantized dense FP32 load too? Prints the greedy argmax after the
     * reference prefill for BOTH weight forms.
     */
    @Test
    fun probe_packed_vs_dense_token_embd() {
        val path = System.getenv("GEMMA4_E2B_GGUF_PATH")?.trim().orEmpty()
        if (path.isEmpty() || System.getenv("GEMMA4_PROBE") != "1") {
            println("[skip] set GEMMA4_E2B_GGUF_PATH and GEMMA4_PROBE=1"); return
        }
        val tokenizer = JvmRandomAccessSource.open(path).use { GGUFTokenizer.fromRandomAccessSource(it) }
        for ((label, cfg) in listOf(
            "packed(default)" to Gemma4LoadConfig(),
            "dense(DEQUANT_ALL)" to Gemma4LoadConfig(weightForm = sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL),
        )) {
            val memSeg = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSeg)
            try {
                val runtime = runBlocking {
                    Gemma4Ingestion<FP32>(ctx = ctx, dtype = FP32::class, config = cfg)
                        .loadDslRuntimeStreaming { JvmRandomAccessSource.open(path) }
                }
                var logits = runtime.forward(REFERENCE_PROMPT_IDS[0])
                for (i in 1 until REFERENCE_PROMPT_IDS.size) logits = runtime.forward(REFERENCE_PROMPT_IDS[i])
                val buf = logits.data.copyToFloatArray()
                var best = 0
                for (i in buf.indices) if (buf[i] > buf[best]) best = i
                println(
                    "AB %-20s argmax=%6d '%s' maxLogit=%+.3f logit[818]=%+.3f".format(
                        label, best,
                        runCatching { tokenizer.decode(best) }.getOrElse { "?" }.replace("\n", "\\n"),
                        buf[best], buf.getOrElse(818) { Float.NaN },
                    )
                )
            } finally {
                memSeg.close()
            }
        }
        println("AB reference: argmax=818 'The' (llama.cpp, logprob -0.01)")
    }

    @Test
    fun chat_prompt_tokenizes_and_predicts_like_llamacpp() {
        val path = System.getenv("GEMMA4_E2B_GGUF_PATH")?.trim().orEmpty()
        if (path.isEmpty()) {
            println("[skip] GEMMA4_E2B_GGUF_PATH not set."); return
        }

        val tokenizer = JvmRandomAccessSource.open(path).use { GGUFTokenizer.fromRandomAccessSource(it) }

        // 1) Tokenizer parity: our template's rendering must produce llama.cpp's ids.
        val rendered = Gemma4ChatTemplate().apply(
            messages = listOf(ChatMessage(ChatRole.USER, "Translate 'hello world' to German")),
            addGenerationPrompt = true,
        )
        val encoded = tokenizer.encode(rendered)
        val ours = if (encoded.isEmpty() || encoded[0] != tokenizer.bosTokenId) {
            intArrayOf(tokenizer.bosTokenId) + encoded
        } else encoded
        val firstDiff = (0 until minOf(ours.size, REFERENCE_PROMPT_IDS.size))
            .firstOrNull { ours[it] != REFERENCE_PROMPT_IDS[it] }
        assertTrue(
            ours.contentEquals(REFERENCE_PROMPT_IDS),
            "prompt tokenization diverges from llama.cpp: ours(n=${ours.size})=${ours.toList()} " +
                "ref(n=${REFERENCE_PROMPT_IDS.size})=${REFERENCE_PROMPT_IDS.toList()} firstDiff=$firstDiff",
        )

        // 2) Forward-pass parity: greedy argmax after the same prefill.
        val memSeg = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSeg)
        try {
            val runtime = runBlocking {
                Gemma4Ingestion<FP32>(ctx = ctx, dtype = FP32::class, config = Gemma4LoadConfig())
                    .loadDslRuntimeStreaming { JvmRandomAccessSource.open(path) }
            }
            var logits = runtime.forward(REFERENCE_PROMPT_IDS[0])
            for (i in 1 until REFERENCE_PROMPT_IDS.size) logits = runtime.forward(REFERENCE_PROMPT_IDS[i])
            val buf = logits.data.copyToFloatArray()
            val top5 = buf.toList().mapIndexed { i, v -> i to v }.sortedByDescending { it.second }.take(5)
            println("top-5: " + top5.joinToString { (id, s) ->
                "%d(%s)=%.3f".format(id, runCatching { tokenizer.decode(id) }.getOrElse { "?" }, s)
            })
            println("logit[$REFERENCE_FIRST_TOKEN]=${buf.getOrNull(REFERENCE_FIRST_TOKEN)}")
            assertEquals(
                REFERENCE_FIRST_TOKEN, top5.first().first,
                "greedy first token must match llama.cpp (818='The', ref logprob -0.01); " +
                    "got ${top5.first().first} ('${runCatching { tokenizer.decode(top5.first().first) }.getOrElse { "?" }}')",
            )
        } finally {
            memSeg.close()
        }
    }
}
