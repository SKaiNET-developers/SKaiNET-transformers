package sk.ainet.models.lfm2

import kotlin.test.*

class ShortConvTest {
    private val fixture = readFixture(javaClass.getResource("/short-conv.json")!!.readText())

    private fun model(kernel: Int = fixture.config.kernelSize): ShortConv = ShortConv(
        fixture.config.hiddenSize, kernel, fixture.weights.inProjection,
        fixture.weights.conv.map { it.takeLast(kernel) }, fixture.weights.outProjection,
    )

    @Test
    fun matchesIndependentOutputsAndPersistentState() {
        val result = evaluateFixture(fixture)
        assertEquals(8, result.cases)
        assertEquals(17, result.chunks)
        assertTrue(result.comparedValues > 300)
    }

    @Test
    fun rejectedInputDoesNotAdvanceState() {
        val model = model()
        val prefix = fixture.cases.first().chunks.first().input.take(2)
        model.forward(prefix)
        val before = model.state()
        assertFailsWith<IllegalArgumentException> { model.forward(listOf(listOf(1f))) }
        assertFailsWith<IllegalArgumentException> { model.forward(emptyList()) }
        assertFailsWith<IllegalArgumentException> { model.forward(listOf(List(4) { Float.NaN })) }
        assertEquals(before, model.state())
        assertEquals(2, model.position)
    }

    @Test
    fun interleavedInstancesRemainIndependentAndResetReplays() {
        val input = fixture.cases.first().chunks.first().input
        val first = model()
        val second = model()
        val expected = second.forward(input)
        first.forward(input.take(2))
        second.forward(input.takeLast(1))
        val actual = first.forward(input.drop(2))
        for (i in actual.indices) for (j in actual[i].indices) {
            assertTrue(kotlin.math.abs(actual[i][j] - expected[i + 2][j]) < 1e-5)
        }
        first.reset()
        assertEquals(0, first.position)
        assertTrue(first.state().flatten().all { it == 0f })
        assertEquals(expected, first.forward(input))
    }

    @Test
    fun kernelOneWorksWithoutHistoryPadding() {
        val input = fixture.cases.first().chunks.first().input
        val whole = model(kernel = 1).forward(input)
        val streamed = model(kernel = 1)
        val parts = input.flatMap { streamed.forward(listOf(it)) }
        for (i in whole.indices) for (j in whole[i].indices) {
            assertTrue(kotlin.math.abs(whole[i][j] - parts[i][j]) < 1e-5)
        }
    }

    @Test
    fun incorrectReferenceAndMalformedWeightsAreRejected() {
        val case = fixture.cases.first()
        val chunk = case.chunks.first()
        val wrong = chunk.expectedOutput.map { row -> row.map { it + 1f } }
        assertFailsWith<IllegalArgumentException> {
            evaluateFixture(fixture.copy(cases = listOf(case.copy(chunks = listOf(chunk.copy(expectedOutput = wrong))))))
        }
        assertFailsWith<IllegalArgumentException> {
            ShortConv(4, 3, fixture.weights.inProjection.dropLast(1), fixture.weights.conv, fixture.weights.outProjection)
        }
    }
}

