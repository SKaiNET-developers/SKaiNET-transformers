package sk.ainet.performance

import kotlin.io.path.createTempFile
import kotlin.io.path.deleteIfExists
import kotlin.io.path.writeText
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.performance.jvm.JvmBenchmarkOrchestrator
import sk.ainet.performance.jvm.JvmModelResolver

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

    @Test
    fun `list scenarios includes llama throughput`() {
        val orchestrator = JvmBenchmarkOrchestrator()

        val scenarioIds = orchestrator.listScenarios().map { it.scenario.id }

        assertTrue("llama-runtime-throughput" in scenarioIds)
        assertTrue("jvm-smoke" in scenarioIds)
    }

    @Test
    fun `resolver prefers cli over system property over env`() {
        val cliFile = createTempFile(prefix = "cli-model", suffix = ".gguf")
        val sysFile = createTempFile(prefix = "sys-model", suffix = ".gguf")
        val envFile = createTempFile(prefix = "env-model", suffix = ".gguf")
        try {
            cliFile.writeText("cli")
            sysFile.writeText("sys")
            envFile.writeText("env")

            val resolver = JvmModelResolver(
                systemPropertyLookup = { if (it == "skainet.model.path") sysFile.toString() else null },
                environmentLookup = { if (it == "SKAINET_MODEL_PATH") envFile.toString() else null },
            )

            val cliResolved = resolver.resolve(cliFile.toString())
            val sysResolved = resolver.resolve(null)
            val envOnlyResolver = JvmModelResolver(
                systemPropertyLookup = { null },
                environmentLookup = { if (it == "SKAINET_MODEL_PATH") envFile.toString() else null },
            )
            val envResolved = envOnlyResolver.resolve(null)

            assertEquals(cliFile.toString(), cliResolved.path.toString())
            assertEquals("cli", cliResolved.source)
            assertEquals(sysFile.toString(), sysResolved.path.toString())
            assertEquals("system-property:skainet.model.path", sysResolved.source)
            assertEquals(envFile.toString(), envResolved.path.toString())
            assertEquals("env:SKAINET_MODEL_PATH", envResolved.source)
        } finally {
            cliFile.deleteIfExists()
            sysFile.deleteIfExists()
            envFile.deleteIfExists()
        }
    }
}
