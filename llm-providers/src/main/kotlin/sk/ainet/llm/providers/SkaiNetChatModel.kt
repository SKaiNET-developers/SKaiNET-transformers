package sk.ainet.llm.providers

import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.sampleFromTensor
import sk.ainet.lang.types.DType
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.ChatResponse
import sk.ainet.llm.api.ChatResponseChunk
import sk.ainet.llm.api.FinishReason
import sk.ainet.llm.api.Generation
import sk.ainet.llm.api.Message
import sk.ainet.llm.api.StreamingChatModel
import sk.ainet.llm.api.Usage

/**
 * Adapts an `InferenceRuntime` + `Tokenizer` + `ChatTemplate` to the neutral
 * [StreamingChatModel] SPI.
 *
 * **Threading:** A single instance is **not** thread-safe — KV-cache state is
 * mutated across forward passes. Use one instance per concurrent request, or
 * wrap many instances in a pool decorator.
 *
 * **Sampling knobs:** Only [ChatOptions.temperature], [ChatOptions.maxTokens],
 * and [ChatOptions.stopSequences] are honored today. `topK` / `topP` /
 * `frequencyPenalty` are accepted in the request shape (for API parity with
 * Spring AI / OpenAI) but ignored — the underlying runtime currently does
 * temperature-only categorical sampling.
 */
public class SkaiNetChatModel<T : DType>(
    private val runtime: InferenceRuntime<T>,
    private val tokenizer: Tokenizer,
    private val chatTemplate: ChatTemplate,
    public override val defaultOptions: ChatOptions = ChatOptions.DEFAULTS,
    private val bosTokenId: Int = tokenizer.bosTokenId,
    private val eosTokenId: Int = tokenizer.eosTokenId,
    private val random: Random = Random.Default,
    private val modelId: String? = null,
) : StreamingChatModel {

    override fun call(request: ChatRequest): ChatResponse {
        val opts = merged(request.options)
        val promptTokens = renderAndTokenize(request)
        val sb = StringBuilder()
        var produced = 0
        var finish = FinishReason.LENGTH
        runGenerationLoop(
            promptTokens = promptTokens,
            maxTokens = opts.maxTokens ?: 512,
            temperature = opts.temperature ?: 0f,
        ) { tokenId ->
            if (tokenId == eosTokenId) {
                finish = FinishReason.STOP
                false
            } else {
                sb.append(tokenizer.decode(tokenId))
                produced++
                val stopAt = findStopSequenceStart(sb, opts.stopSequences)
                if (stopAt != null) {
                    sb.setLength(stopAt)
                    finish = FinishReason.STOP
                    false
                } else {
                    true
                }
            }
        }
        val text = sb.toString()
        val toolCalls = if (chatTemplate.containsToolCall(text))
            chatTemplate.parseToolCalls(text).map { it.toApi() }
        else emptyList()
        if (toolCalls.isNotEmpty()) finish = FinishReason.TOOL_CALL

        val message = Message.assistant(text, toolCalls)
        return ChatResponse(
            generations = listOf(Generation(message, finish)),
            usage = Usage(promptTokens = promptTokens.size, completionTokens = produced),
            modelId = modelId,
        )
    }

    override fun stream(request: ChatRequest): Flow<ChatResponseChunk> = channelFlow {
        withContext(Dispatchers.Default) {
            val opts = merged(request.options)
            val promptTokens = renderAndTokenize(request)
            val sb = StringBuilder()
            var produced = 0
            var finish = FinishReason.LENGTH

            runGenerationLoop(
                promptTokens = promptTokens,
                maxTokens = opts.maxTokens ?: 512,
                temperature = opts.temperature ?: 0f,
            ) { tokenId ->
                if (!isActive) return@runGenerationLoop false
                if (tokenId == eosTokenId) {
                    finish = FinishReason.STOP
                    return@runGenerationLoop false
                }
                val piece = tokenizer.decode(tokenId)
                val before = sb.length
                sb.append(piece)
                produced++

                val stopAt = findStopSequenceStart(sb, opts.stopSequences)
                if (stopAt != null) {
                    finish = FinishReason.STOP
                    val keep = (stopAt - before).coerceAtLeast(0)
                    if (keep > 0) {
                        trySend(ChatResponseChunk(delta = piece.substring(0, keep)))
                    }
                    sb.setLength(stopAt)
                    return@runGenerationLoop false
                }
                val sendResult = trySend(ChatResponseChunk(delta = piece))
                !sendResult.isClosed
            }

            val text = sb.toString()
            val toolCalls = if (chatTemplate.containsToolCall(text))
                chatTemplate.parseToolCalls(text).map { it.toApi() }
            else emptyList()
            if (toolCalls.isNotEmpty()) finish = FinishReason.TOOL_CALL

            send(
                ChatResponseChunk(
                    delta = "",
                    toolCallDelta = toolCalls,
                    finishReason = finish,
                    usage = Usage(promptTokens = promptTokens.size, completionTokens = produced),
                )
            )
        }
    }.buffer(Channel.UNLIMITED)

    override fun close() {
        runtime.reset()
    }

    private fun merged(opts: ChatOptions?): ChatOptions =
        opts?.let {
            ChatOptions(
                model = it.model ?: defaultOptions.model,
                temperature = it.temperature ?: defaultOptions.temperature,
                topK = it.topK ?: defaultOptions.topK,
                topP = it.topP ?: defaultOptions.topP,
                maxTokens = it.maxTokens ?: defaultOptions.maxTokens,
                stopSequences = if (it.stopSequences.isNotEmpty()) it.stopSequences else defaultOptions.stopSequences,
                seed = it.seed ?: defaultOptions.seed,
            )
        } ?: defaultOptions

    private fun renderAndTokenize(request: ChatRequest): IntArray {
        val agentMessages = request.messages.map { it.toAgent() }
        val agentTools = request.tools.map { it.toAgent() }
        val rendered = chatTemplate.apply(agentMessages, agentTools, addGenerationPrompt = true)
        return tokenizer.encode(rendered)
    }

    /**
     * Runs the autoregressive loop synchronously on the calling thread.
     *
     * Mirrors the prefill/sample structure of [sk.ainet.apps.llm.generate] but exposes
     * a per-token `Boolean` callback so callers can stop early on EOS or stop sequences.
     */
    private fun runGenerationLoop(
        promptTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        onSampled: (Int) -> Boolean,
    ) {
        require(maxTokens > 0) { "maxTokens must be > 0" }
        runtime.reset()

        val full = if (promptTokens.isEmpty() || promptTokens[0] != bosTokenId) {
            intArrayOf(bosTokenId) + promptTokens
        } else {
            promptTokens
        }

        var token = full[0]
        var pos = 0
        var produced = 0
        while (produced < maxTokens) {
            val logits = runtime.forward(token)
            val next = if (pos + 1 < full.size) {
                full[pos + 1]
            } else {
                sampleFromTensor(logits, temperature, random)
            }
            if (pos + 1 >= full.size) {
                if (!onSampled(next)) return
                produced++
            }
            token = next
            pos++
        }
    }
}
