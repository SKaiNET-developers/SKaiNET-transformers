package sk.ainet.performance

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.performance.jvm.JvmBenchmarkOrchestrator

class JvmBenchmarkOrchestratorTest {
    @Test
    fun `smoke scenario runs end to end`() = runBlocking {
        val orchestrator = JvmBenchmarkOrchestrator()

        val result = orchestrator.run(BenchmarkRunRequest(scenarioId = "jvm-smoke"))

        assertEquals("jvm-smoke", result.scenarioId)
        assertEquals("jvm", result.target)
        assertTrue(result.cases.isNotEmpty())
        assertEquals(BenchmarkCaseStatus.SUCCESS, result.cases.single().status)
        assertTrue(result.cases.single().metrics.isNotEmpty())
    }
}
