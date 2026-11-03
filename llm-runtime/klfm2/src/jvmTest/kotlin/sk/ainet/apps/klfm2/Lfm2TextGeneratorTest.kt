package sk.ainet.apps.klfm2

import kotlin.test.*

class Lfm2TextGeneratorTest {
    private val messages = listOf(Lfm2ChatMessage("user", "Hello"))

    private class Codec(
        var prompt: IntArray = intArrayOf(1, 2),
        override val stopTokenIds: Set<Int> = setOf(7),
        val decoder: (IntArray) -> String = { ids -> ids.filter { it != 7 }.joinToString(",") },
    ) : GenerationTextCodec {
        var encoded = 0
        var preserveThinking: Boolean? = null
        val decoded = mutableListOf<List<Int>>()
        override fun encodeMessages(messages: List<Lfm2ChatMessage>, preserveThinking: Boolean): IntArray {
            encoded++
            this.preserveThinking = preserveThinking
            return prompt
        }
        override fun decode(tokenIds: IntArray): String {
            decoded.add(tokenIds.toList())
            return decoder(tokenIds)
        }
    }

    private class Session(val rows: List<FloatArray>, val afterForward: () -> Unit = {}) : GenerationTokenSession {
        override var position = 0
        var closed = false
        val forwarded = mutableListOf<List<Int>>()
        override fun forwardBatched(tokenIds: IntArray): FloatArray {
            check(!closed)
            forwarded.add(tokenIds.toList())
            position += tokenIds.size
            afterForward()
            return rows[forwarded.lastIndex].copyOf()
        }
        override fun forward(tokenId: Int): FloatArray = forwardBatched(intArrayOf(tokenId))
        override fun close() { closed = true; position = 0 }
    }

    private fun row(token: Int): FloatArray = FloatArray(8) { if (it == token) 1f else -1f }

    private class Factory(private val rows: List<FloatArray>, private val afterForward: () -> Unit = {}) {
        val sessions = mutableListOf<Session>()
        fun open(): GenerationTokenSession = Session(rows, afterForward).also(sessions::add)
    }

    @Test fun greedyUsesComputedTokensStopsAtEosAndStartsFreshForEveryRequest() {
        val codec = Codec()
        val factory = Factory(listOf(row(4), row(5), row(7)))
        val generator = Lfm2TextGenerator(8, 10, codec, factory::open)
        repeat(2) {
            val events = mutableListOf<Lfm2GeneratedToken>()
            val result = generator.generate(messages, Lfm2GenerationOptions(5, preserveThinking = true), onToken = events::add)
            assertContentEquals(intArrayOf(1, 2), result.promptTokenIds)
            assertContentEquals(intArrayOf(4, 5, 7), result.generatedTokenIds)
            assertEquals("4,5", result.text)
            assertEquals("eos", result.stopReason)
            assertEquals(4, result.stateInputTokens)
            assertEquals(listOf(4, 5, 7), events.map { e -> e.tokenId })
            assertEquals(listOf(false, false, true), events.map { e -> e.isStopToken })
            assertEquals(listOf(0, 1, 2), events.map { e -> e.index })
        }
        assertEquals(true, codec.preserveThinking)
        assertEquals(2, factory.sessions.size)
        factory.sessions.forEach {
            assertTrue(it.closed)
            assertEquals(0, it.position)
            assertEquals(listOf(listOf(1, 2), listOf(4), listOf(5)), it.forwarded)
        }
    }

    @Test fun lengthBoundAndTiesUseLowestIdWithoutFeedingTheLastEmittedToken() {
        val ties = row(3).also { it[2] = it[3] }
        val factory = Factory(listOf(ties, row(4)))
        val result = Lfm2TextGenerator(8, 3, Codec(), factory::open).generate(messages, Lfm2GenerationOptions(2))
        assertContentEquals(intArrayOf(2, 4), result.generatedTokenIds)
        assertEquals("length", result.stopReason)
        assertEquals(3, result.stateInputTokens)
        assertEquals(listOf(listOf(1, 2), listOf(2)), factory.sessions.single().forwarded)
        assertTrue(factory.sessions.single().closed)
    }

    @Test fun streamDecodesTheWholePrefixInsteadOfConcatenatingTokenPieces() {
        val codec = Codec(decoder = { ids -> if (ids.size == 1) "\uFFFD" else "é" })
        val factory = Factory(listOf(row(3), row(4)))
        val events = mutableListOf<Lfm2GeneratedToken>()
        val result = Lfm2TextGenerator(8, 5, codec, factory::open)
            .generate(messages, Lfm2GenerationOptions(2), onToken = events::add)
        assertEquals(listOf("\uFFFD", "é"), events.map { it.text })
        assertEquals("é", result.text)
        assertEquals(listOf(listOf(3), listOf(3, 4)), codec.decoded)
    }

    @Test fun preCancelledRequestDoesNotOpenOrForwardASession() {
        val factory = Factory(listOf(row(3)))
        val result = Lfm2TextGenerator(8, 5, Codec(), factory::open)
            .generate(messages, Lfm2GenerationOptions(2), isCancelled = { true })
        assertEquals("cancelled", result.stopReason)
        assertContentEquals(intArrayOf(), result.generatedTokenIds)
        assertEquals("", result.text)
        assertEquals(0, result.stateInputTokens)
        assertTrue(factory.sessions.isEmpty())
    }

    @Test fun cancellationDuringPrefillClosesStateWithoutEmittingAToken() {
        var cancelled = false
        val factory = Factory(listOf(row(3))) { cancelled = true }
        val events = mutableListOf<Lfm2GeneratedToken>()
        val result = Lfm2TextGenerator(8, 5, Codec(), factory::open)
            .generate(messages, Lfm2GenerationOptions(2), isCancelled = { cancelled }, onToken = events::add)
        assertEquals("cancelled", result.stopReason)
        assertEquals(2, result.stateInputTokens)
        assertTrue(events.isEmpty())
        assertTrue(factory.sessions.single().closed)
    }

    @Test fun cancellationFromTokenCallbackStopsBeforeTheNextDecode() {
        var cancelled = false
        val factory = Factory(listOf(row(3)))
        val result = Lfm2TextGenerator(8, 5, Codec(), factory::open).generate(messages,
            Lfm2GenerationOptions(3), isCancelled = { cancelled }, onToken = { cancelled = true })
        assertEquals("cancelled", result.stopReason)
        assertContentEquals(intArrayOf(3), result.generatedTokenIds)
        assertEquals(2, result.stateInputTokens)
        assertEquals(1, factory.sessions.single().forwarded.size)
        assertTrue(factory.sessions.single().closed)
    }

    @Test fun cancellationDuringDecodeKeepsOnlyPreviouslyEmittedTokens() {
        var calls = 0
        var cancelled = false
        val factory = Factory(listOf(row(3), row(4))) { calls++; if (calls == 2) cancelled = true }
        val events = mutableListOf<Lfm2GeneratedToken>()
        val result = Lfm2TextGenerator(8, 5, Codec(), factory::open).generate(messages,
            Lfm2GenerationOptions(3), isCancelled = { cancelled }, onToken = events::add)
        assertEquals("cancelled", result.stopReason)
        assertContentEquals(intArrayOf(3), result.generatedTokenIds)
        assertEquals(3, result.stateInputTokens)
        assertEquals(1, events.size)
        assertTrue(factory.sessions.single().closed)
    }

    @Test fun callbackExceptionsCloseTheRequestAndAllowAnIndependentRetry() {
        val factory = Factory(listOf(row(3), row(7)))
        val generator = Lfm2TextGenerator(8, 5, Codec(), factory::open)
        val failure = IllegalStateException("consumer failed")
        assertSame(failure, assertFailsWith<IllegalStateException> {
            generator.generate(messages, Lfm2GenerationOptions(2), onToken = { throw failure })
        })
        assertTrue(factory.sessions.single().closed)
        val retried = generator.generate(messages, Lfm2GenerationOptions(2))
        assertContentEquals(intArrayOf(3, 7), retried.generatedTokenIds)
        assertEquals("eos", retried.stopReason)
        assertTrue(factory.sessions.all { it.closed })
    }

    @Test fun invalidOptionsPromptAndStopsFailBeforeOpeningModelState() {
        val codec = Codec()
        val factory = Factory(listOf(row(3)))
        val generator = Lfm2TextGenerator(8, 5, codec, factory::open)
        assertFailsWith<IllegalArgumentException> { generator.generate(messages, Lfm2GenerationOptions(0)) }
        assertFailsWith<IllegalArgumentException> { generator.generate(messages, Lfm2GenerationOptions(6)) }
        assertEquals(0, codec.encoded)
        assertFailsWith<IllegalArgumentException> { generator.generate(messages, Lfm2GenerationOptions(5)) }
        codec.prompt = intArrayOf(8)
        assertFailsWith<IllegalArgumentException> { generator.generate(messages, Lfm2GenerationOptions(1)) }
        codec.prompt = intArrayOf()
        assertFailsWith<IllegalArgumentException> { generator.generate(messages, Lfm2GenerationOptions(1)) }
        assertFailsWith<IllegalArgumentException> {
            Lfm2TextGenerator(8, 5, Codec(stopTokenIds = emptySet()), factory::open).generate(messages, Lfm2GenerationOptions(1))
        }
        assertFailsWith<IllegalArgumentException> {
            Lfm2TextGenerator(8, 5, Codec(stopTokenIds = setOf(8)), factory::open).generate(messages, Lfm2GenerationOptions(1))
        }
        assertTrue(factory.sessions.isEmpty())
    }

    @Test fun malformedLogitsAndDecodeFailuresCloseTheSession() {
        for (bad in listOf(floatArrayOf(1f), row(3).also { it[0] = Float.NaN }, row(3).also { it[2] = Float.POSITIVE_INFINITY })) {
            val factory = Factory(listOf(bad))
            assertFailsWith<IllegalArgumentException> {
                Lfm2TextGenerator(8, 5, Codec(), factory::open).generate(messages, Lfm2GenerationOptions(1))
            }
            assertTrue(factory.sessions.single().closed)
        }
        val factory = Factory(listOf(row(3)))
        assertFailsWith<IllegalStateException> {
            Lfm2TextGenerator(8, 5, Codec(decoder = { error("decode failed") }), factory::open)
                .generate(messages, Lfm2GenerationOptions(1))
        }
        assertTrue(factory.sessions.single().closed)
    }
}
