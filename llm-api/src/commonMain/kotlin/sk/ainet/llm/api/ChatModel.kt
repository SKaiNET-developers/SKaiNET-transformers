package sk.ainet.llm.api

import kotlin.AutoCloseable
import kotlinx.coroutines.flow.Flow

/**
 * Provider-neutral chat completion SPI.
 *
 * Mirrors the shape of Spring AI's `ChatModel` so the design is familiar to JVM
 * developers, but has zero Spring or Reactor dependencies — only `kotlin-stdlib`
 * and `kotlinx-coroutines`. Framework adapters (Spring AI, LangChain4j, ...) live
 * in companion repositories and translate to this interface at the seam.
 *
 * Implementations are typically **not** thread-safe — they hold mutable
 * KV-cache state across forward passes. Callers requiring concurrent invocation
 * should use a pool decorator.
 */
public interface ChatModel : AutoCloseable {

    /** Synchronous chat completion. */
    public fun call(request: ChatRequest): ChatResponse

    /** Default options used when [ChatRequest.options] is `null`. */
    public val defaultOptions: ChatOptions

    /** Free any native or runtime resources. Default no-op. */
    override fun close(): Unit = Unit
}

/**
 * Token-by-token streaming variant.
 *
 * Implementations emit a [ChatResponseChunk] per generated token (or per fixed
 * batch of tokens). The terminal chunk carries [ChatResponseChunk.finishReason]
 * and final [ChatResponseChunk.usage]. Cancellation of the [Flow] aborts generation.
 */
public interface StreamingChatModel : ChatModel {
    public fun stream(request: ChatRequest): Flow<ChatResponseChunk>
}
