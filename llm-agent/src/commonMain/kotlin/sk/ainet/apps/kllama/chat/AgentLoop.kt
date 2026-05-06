package sk.ainet.apps.kllama.chat

import kotlin.random.Random
import sk.ainet.apps.kllama.agent.GenerateResult
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.lang.types.DType

/**
 * Configuration for the agent loop.
 *
 * @param maxToolRounds Maximum number of tool-call rounds before forcing a final response.
 * @param maxTokensPerRound Maximum tokens to generate in each round.
 * @param temperature Sampling temperature.
 * @param random Random generator for sampling.
 */
public data class AgentConfig @kotlin.jvm.JvmOverloads constructor(
    val maxToolRounds: Int = 5,
    val maxTokensPerRound: Int = 512,
    val temperature: Float = 0.7f,
    val random: Random = Random.Default
)

/**
 * Callback interface for observing the agent loop.
 */
public interface AgentListener {
    /** Called when the model produces text (may be called multiple times per round). */
    public fun onToken(token: String) {}

    /**
     * Called after each prompt token is fed through the model during prefill,
     * with `(done, total)` where `done` is 1-based and `total` is the prompt
     * length in tokens. Prefill is autoregressive (one `forward` per token),
     * so on a CPU-only runtime there is a long silence between the start of
     * a round and the first [onToken]; this callback lets a UI surface that.
     */
    public fun onPrefillProgress(done: Int, total: Int) {}

    /** Called when the model's full response for a round is available. */
    public fun onAssistantMessage(text: String) {}

    /**
     * Called once for each thinking block (e.g. Gemma 4 `<|think>...<think|>`)
     * emitted by the model this round. Thinking blocks are stripped before
     * being persisted to the conversation so they do not leak into the next
     * prompt; this callback is the only way to observe them.
     */
    public fun onThinking(text: String) {}

    /** Called when tool calls are detected in the model's response. */
    public fun onToolCalls(calls: List<ToolCall>) {}

    /** Called when a tool returns a result. */
    public fun onToolResult(call: ToolCall, result: String) {}

    /**
     * Called when a parsed tool call fails JSON-Schema validation against its
     * [ToolDefinition]. The loop treats the call as failed, feeds the reason
     * back to the model as the tool result, and does NOT invoke the tool.
     */
    public fun onToolCallValidationFailed(call: ToolCall, reason: String) {}

    /** Called when the agent loop finishes. */
    public fun onComplete(finalResponse: String) {}
}

/**
 * Orchestrates the generate -> parse -> execute -> re-prompt cycle.
 *
 * The agent loop:
 * 1. Formats the conversation using the [template]
 * 2. Generates tokens until EOS or max tokens
 * 3. Parses the output for tool calls
 * 4. If tool calls are found, executes them and appends results to the conversation
 * 5. Repeats until no tool calls are found or [AgentConfig.maxToolRounds] is reached
 *
 * @param T The DType of the model weights.
 * @param runtime The inference runtime for token generation.
 * @param template The chat template for formatting prompts.
 * @param toolRegistry Registry of available tools.
 * @param eosTokenId The EOS token ID used to detect generation completion.
 * @param config Agent configuration.
 * @param decode Function to decode a token ID to a string fragment.
 */
public class AgentLoop<T : DType>(
    private val runtime: InferenceRuntime<T>,
    private val template: ChatTemplate,
    private val toolRegistry: ToolRegistry,
    private val eosTokenId: Int,
    private val config: AgentConfig = AgentConfig(),
    private val decode: (Int) -> String
) {

    /**
     * Run the agent loop with the given conversation history.
     *
     * @param messages Initial conversation messages (typically system + user).
     * @param listener Optional listener for observing the loop.
     * @return The final assistant response text.
     */
    public fun run(
        messages: MutableList<ChatMessage>,
        listener: AgentListener? = null
    ): String {
        val tools = toolRegistry.definitions()
        val toolsByName = tools.associateBy { it.name }
        var lastResponse = ""
        var lastVisibleResponse = ""

        for (round in 0 until config.maxToolRounds) {
            // Reset runtime state for each round (re-process full context)
            runtime.reset()

            // Format the full conversation into a prompt
            val prompt = template.apply(messages, tools, addGenerationPrompt = true)
            val promptTokens = encodePrompt(prompt)

            // Generate until EOS or max tokens
            val result: GenerateResult = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = config.maxTokensPerRound,
                eosTokenId = eosTokenId,
                temperature = config.temperature,
                random = config.random,
                onToken = { tokenId -> listener?.onToken(decode(tokenId)) },
                decode = decode,
                onPrefill = { done, total -> listener?.onPrefillProgress(done, total) }
            )

            lastResponse = result.text
            listener?.onAssistantMessage(lastResponse)

            // Extract thinking blocks first so the listener sees them, and
            // strip them out of the text we persist to the conversation.
            template.parseThinkingBlocks(lastResponse).forEach { block ->
                listener?.onThinking(block)
            }
            lastVisibleResponse = template.stripThinking(lastResponse)

            // Parse tool calls from the RAW text so a model can emit thinking
            // and tool_call in the same response.
            val toolCalls = template.parseToolCalls(lastResponse)

            if (toolCalls.isEmpty()) {
                // No tool calls — this is the final response
                messages.add(ChatMessage(role = ChatRole.ASSISTANT, content = lastVisibleResponse))
                listener?.onComplete(lastVisibleResponse)
                return lastVisibleResponse
            }

            // Tool calls found — execute them
            listener?.onToolCalls(toolCalls)
            messages.add(
                ChatMessage(
                    role = ChatRole.ASSISTANT,
                    content = lastVisibleResponse,
                    toolCalls = toolCalls
                )
            )

            for (call in toolCalls) {
                val toolResult = executeWithValidation(call, toolsByName, listener)
                listener?.onToolResult(call, toolResult)
                messages.add(
                    ChatMessage(
                        role = ChatRole.TOOL,
                        content = toolResult,
                        toolCallId = call.id
                    )
                )
            }
        }

        // Max rounds exhausted — return last response
        listener?.onComplete(lastVisibleResponse)
        return lastVisibleResponse
    }

    /**
     * Validate [call] against the schema declared by its [ToolDefinition]. On
     * failure, notify the listener and return the validation error as the
     * tool result so the model sees the problem on the next round instead of
     * crashing inside the tool. Unknown tools bypass validation and reach
     * [ToolRegistry.execute], which handles them explicitly.
     */
    private fun executeWithValidation(
        call: ToolCall,
        toolsByName: Map<String, ToolDefinition>,
        listener: AgentListener?
    ): String {
        val def = toolsByName[call.name]
        if (def != null) {
            val result = ToolCallValidator.validate(call, def)
            if (result is ToolCallValidationResult.Invalid) {
                listener?.onToolCallValidationFailed(call, result.reason)
                return "validation error: ${result.reason}"
            }
        }
        return toolRegistry.execute(call)
    }

    /**
     * Simple prompt-to-tokens encoding.
     * Encodes each character individually — the runtime's tokenizer should be used
     * for real encoding. This is a fallback for when only a decode function is available.
     */
    private fun encodePrompt(prompt: String): IntArray {
        // This is handled at the CLI level where we have access to the tokenizer.
        // The AgentLoop receives pre-encoded tokens in practice.
        // For now, we store the prompt and let the CLI layer handle encoding.
        throw UnsupportedOperationException(
            "AgentLoop.encodePrompt should not be called directly. " +
                "Use AgentLoop.run() with pre-encoded prompts via runWithEncoder()."
        )
    }

    /**
     * Run the agent loop with an explicit tokenizer encode function.
     *
     * @param messages Conversation history.
     * @param encode Function to encode a string to token IDs.
     * @param listener Optional listener.
     * @return The final assistant response text.
     */
    public fun runWithEncoder(
        messages: MutableList<ChatMessage>,
        encode: (String) -> IntArray,
        listener: AgentListener? = null
    ): String {
        val tools = toolRegistry.definitions()
        val toolsByName = tools.associateBy { it.name }
        var lastResponse = ""
        var lastVisibleResponse = ""

        for (round in 0 until config.maxToolRounds) {
            runtime.reset()

            val prompt = template.apply(messages, tools, addGenerationPrompt = true)
            val promptTokens = encode(prompt)

            val result: GenerateResult = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = config.maxTokensPerRound,
                eosTokenId = eosTokenId,
                temperature = config.temperature,
                random = config.random,
                onToken = { tokenId -> listener?.onToken(decode(tokenId)) },
                decode = decode,
                onPrefill = { done, total -> listener?.onPrefillProgress(done, total) }
            )

            lastResponse = result.text
            listener?.onAssistantMessage(lastResponse)

            template.parseThinkingBlocks(lastResponse).forEach { block ->
                listener?.onThinking(block)
            }
            lastVisibleResponse = template.stripThinking(lastResponse)

            val toolCalls = template.parseToolCalls(lastResponse)

            if (toolCalls.isEmpty()) {
                messages.add(ChatMessage(role = ChatRole.ASSISTANT, content = lastVisibleResponse))
                listener?.onComplete(lastVisibleResponse)
                return lastVisibleResponse
            }

            listener?.onToolCalls(toolCalls)
            messages.add(
                ChatMessage(
                    role = ChatRole.ASSISTANT,
                    content = lastVisibleResponse,
                    toolCalls = toolCalls
                )
            )

            for (call in toolCalls) {
                val toolResult = executeWithValidation(call, toolsByName, listener)
                listener?.onToolResult(call, toolResult)
                messages.add(
                    ChatMessage(
                        role = ChatRole.TOOL,
                        content = toolResult,
                        toolCallId = call.id
                    )
                )
            }
        }

        listener?.onComplete(lastVisibleResponse)
        return lastVisibleResponse
    }
}
