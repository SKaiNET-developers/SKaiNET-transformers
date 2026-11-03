package sk.ainet.apps.klfm2

import sk.ainet.models.lfm2.HybridLfm2Model
import sk.ainet.models.lfm2.Lfm2Session

/** Greedy generation only. The prompt and requested decode budget must fit the model context. */
data class Lfm2GenerationOptions(val maxNewTokens: Int = 32, val preserveThinking: Boolean = false)

/**
 * One emitted token and the decoded completion so far. [text] is cumulative, not an
 * append-only delta: byte-level tokens can revise a previously incomplete UTF-8 suffix.
 * Stop tokens are included in token IDs and events. Decoding skips tokenizer special tokens.
 */
data class Lfm2GeneratedToken(val tokenId: Int, val index: Int, val text: String, val isStopToken: Boolean)

data class Lfm2TextResult(
    val promptTokenIds: IntArray,
    val generatedTokenIds: IntArray,
    val text: String,
    val stopReason: String,
    /** Tokens actually forwarded into model state; the last emitted token need not be consumed. */
    val stateInputTokens: Int,
)

internal interface GenerationTextCodec {
    val stopTokenIds: Set<Int>
    fun encodeMessages(messages: List<Lfm2ChatMessage>, preserveThinking: Boolean): IntArray
    fun decode(tokenIds: IntArray): String
}

internal interface GenerationTokenSession : AutoCloseable {
    val position: Int
    fun forwardBatched(tokenIds: IntArray): FloatArray
    fun forward(tokenId: Int): FloatArray
}

/**
 * Thread-confined synchronous text generation with a fresh hybrid-state session per call.
 * Cancellation is cooperative at forward boundaries; it cannot interrupt a running kernel.
 * Callback/model/tokenizer exceptions propagate after session cleanup. Callbacks receive
 * cumulative decoded completion text and token IDs, never an assumed UTF-8 token fragment.
 * Decoding defaults to checkpoint-declared special tokens. Pinned HF 5 compatibility
 * is available as an explicit policy for reference-runtime comparisons.
 */
class Lfm2TextGenerator internal constructor(
    private val vocabularySize: Int,
    private val contextLimit: Int,
    private val codec: GenerationTextCodec,
    private val openSession: () -> GenerationTokenSession,
) {
    constructor(model: HybridLfm2Model, frontend: Lfm2TextFrontend,
        decodingPolicy: Lfm2DecodingPolicy = Lfm2DecodingPolicy.CHECKPOINT,
    ) : this(
        model.dimensions.vocabulary,
        model.dimensions.contextLimit,
        object : GenerationTextCodec {
            override val stopTokenIds: Set<Int> get() = frontend.stopTokenIds
            override fun encodeMessages(messages: List<Lfm2ChatMessage>, preserveThinking: Boolean): IntArray =
                frontend.encodeMessages(messages, addGenerationPrompt = true, preserveThinking = preserveThinking)
            override fun decode(tokenIds: IntArray): String = frontend.decode(tokenIds, skipSpecialTokens = true, policy = decodingPolicy)
        },
        {
            val session = Lfm2Session(model)
            object : GenerationTokenSession {
                override val position: Int get() = session.position
                override fun forwardBatched(tokenIds: IntArray): FloatArray =
                    session.forwardBatched(tokenIds).data.copyToFloatArray()
                override fun forward(tokenId: Int): FloatArray = session.forward(tokenId).data.copyToFloatArray()
                override fun close() = session.close()
            }
        },
    ) {
        require(frontend.vocabularySize == model.dimensions.vocabulary) { "Tokenizer and model vocabulary sizes differ" }
    }

    init {
        require(vocabularySize > 0 && contextLimit > 0)
    }

    fun generate(
        messages: List<Lfm2ChatMessage>,
        options: Lfm2GenerationOptions = Lfm2GenerationOptions(),
        isCancelled: () -> Boolean = { false },
        onToken: (Lfm2GeneratedToken) -> Unit = {},
    ): Lfm2TextResult {
        require(options.maxNewTokens in 1..contextLimit) { "maxNewTokens must be positive and within the model context limit" }
        val stops = codec.stopTokenIds.toSet()
        require(stops.isNotEmpty() && stops.all { it in 0 until vocabularySize }) { "Generation requires valid stop token IDs" }
        val prompt = codec.encodeMessages(messages.toList(), options.preserveThinking).copyOf()
        require(prompt.isNotEmpty() && prompt.all { it in 0 until vocabularySize }) { "Prompt token IDs must be nonempty and inside the vocabulary" }
        require(prompt.size.toLong() + options.maxNewTokens - 1 <= contextLimit) { "Prompt and requested decode budget exceed the model context limit" }
        val generated = mutableListOf<Int>()
        var text = ""
        fun result(reason: String, position: Int) =
            Lfm2TextResult(prompt.copyOf(), generated.toIntArray(), text, reason, position)
        if (isCancelled()) return result("cancelled", 0)

        return openSession().use { session ->
            check(session.position == 0) { "Each generation request requires a fresh token session" }
            if (isCancelled()) return@use result("cancelled", 0)
            var logits = session.forwardBatched(prompt)
            while (true) {
                // Also observes cancellation requested while prefill or a decode kernel was running.
                if (isCancelled()) return@use result("cancelled", session.position)
                require(logits.size == vocabularySize && logits.all { it.isFinite() }) { "Generation received malformed or non-finite logits" }
                var token = 0
                for (index in 1 until logits.size) if (logits[index] > logits[token]) token = index
                generated.add(token)
                text = codec.decode(generated.toIntArray())
                val isStop = token in stops
                onToken(Lfm2GeneratedToken(token, generated.lastIndex, text, isStop))
                // A token already emitted as EOS or at the length limit completes that request.
                if (isStop) return@use result("eos", session.position)
                if (generated.size == options.maxNewTokens) return@use result("length", session.position)
                if (isCancelled()) return@use result("cancelled", session.position)
                logits = session.forward(token)
            }
            @Suppress("UNREACHABLE_CODE")
            error("Unreachable generation loop")
        }
    }
}
