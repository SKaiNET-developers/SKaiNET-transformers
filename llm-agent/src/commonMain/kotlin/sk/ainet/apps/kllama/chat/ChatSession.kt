package sk.ainet.apps.kllama.chat

import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.lang.types.DType

/**
 * Bundles an [InferenceRuntime] with a [Tokenizer] and [ModelMetadata] to provide
 * chat, agent, and tool-calling capabilities for any model.
 *
 * This decouples tool calling from any specific runner — any runner that can
 * produce an [InferenceRuntime] and a [Tokenizer] can create a [ChatSession]
 * and get chat/agent/demo modes for free.
 *
 * Usage:
 * ```kotlin
 * val session = ChatSession(runtime, tokenizer, metadata)
 * session.chat(maxTokens = 512, temperature = 0.7f)       // interactive chat
 * session.agent(maxTokens = 512, temperature = 0.7f)      // interactive agent with tools
 * session.demo(maxTokens = 256, temperature = 0.7f)       // interactive tool calling demo
 * session.demoSingleShot("What is 2+2?", maxTokens = 256) // non-interactive single prompt
 * ```
 */
public class ChatSession<T : DType>(
    public val runtime: InferenceRuntime<T>,
    public val tokenizer: Tokenizer,
    public val metadata: ModelMetadata = ModelMetadata(),
    templateName: String? = null,
    public val defaultSystemPrompt: String = DEFAULT_SYSTEM_PROMPT
) {
    private val provider: ToolCallingSupport = ToolCallingSupportResolver.resolveOrFallback(metadata, templateName)
    private val template: ChatTemplate = provider.createChatTemplate()

    /**
     * Run a single agent round with the given prompt and tools.
     * Returns the final response text. Non-interactive — suitable for smoke tests.
     *
     * @param prompt The user prompt.
     * @param tools Tools to register. If empty, uses default calculator + list_files.
     * @param maxTokens Maximum tokens per generation round.
     * @param temperature Sampling temperature.
     * @param systemPrompt System prompt for this turn; falls back to [defaultSystemPrompt].
     * @param listener Optional listener for observing the agent loop.
     * @return The final assistant response text.
     */
    public fun runSingleTurn(
        prompt: String,
        tools: List<Tool> = emptyList(),
        maxTokens: Int = 256,
        temperature: Float = 0.7f,
        systemPrompt: String? = null,
        listener: AgentListener? = null
    ): String {
        val registry = ToolRegistry()
        tools.forEach { registry.register(it) }

        val agentLoop = AgentLoop(
            runtime = runtime,
            template = template,
            toolRegistry = registry,
            eosTokenId = tokenizer.eosTokenId,
            config = AgentConfig(
                maxToolRounds = 5,
                maxTokensPerRound = maxTokens,
                temperature = temperature
            ),
            decode = { tokenId -> tokenizer.decode(tokenId) }
        )

        val effectiveSystemPrompt = systemPrompt ?: defaultSystemPrompt
        val messages = mutableListOf(
            ChatMessage(role = ChatRole.SYSTEM, content = effectiveSystemPrompt),
            ChatMessage(role = ChatRole.USER, content = prompt)
        )

        return agentLoop.runWithEncoder(
            messages = messages,
            encode = { text -> tokenizer.encode(text) },
            listener = listener
        )
    }

    /**
     * Create an [AgentLoop] configured for this session.
     */
    public fun createAgentLoop(
        toolRegistry: ToolRegistry,
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ): AgentLoop<T> {
        return AgentLoop(
            runtime = runtime,
            template = template,
            toolRegistry = toolRegistry,
            eosTokenId = tokenizer.eosTokenId,
            config = AgentConfig(
                maxToolRounds = 5,
                maxTokensPerRound = maxTokens,
                temperature = temperature
            ),
            decode = { tokenId -> tokenizer.decode(tokenId) }
        )
    }

    /** The resolved chat template for this session. */
    public val chatTemplate: ChatTemplate get() = template

    /** The resolved tool calling provider family name. */
    public val providerFamily: String get() = provider.family

    /** Encode text to token IDs using this session's tokenizer. */
    public fun encode(text: String): IntArray = tokenizer.encode(text)

    /** Decode a token ID to text using this session's tokenizer. */
    public fun decode(tokenId: Int): String = tokenizer.decode(tokenId)

    public companion object {
        public const val DEFAULT_SYSTEM_PROMPT: String =
            "You are a helpful assistant with access to tools."
    }
}
