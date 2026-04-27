package sk.ainet.llm.providers

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class StopSequencesTest {

    @Test fun `returns null when no stop sequences configured`() {
        assertNull(findStopSequenceStart("anything goes here", emptyList()))
    }

    @Test fun `returns null when stop sequence is absent`() {
        assertNull(findStopSequenceStart("hello world", listOf("###")))
    }

    @Test fun `returns index of single stop`() {
        assertEquals(5, findStopSequenceStart("hello###world", listOf("###")))
    }

    @Test fun `returns earliest of multiple stops`() {
        // "STOP" at index 6, "###" at index 11
        assertEquals(6, findStopSequenceStart("hello STOP and ###", listOf("###", "STOP")))
    }

    @Test fun `ignores empty stop sequences`() {
        assertEquals(5, findStopSequenceStart("hello###", listOf("", "###")))
    }
}
