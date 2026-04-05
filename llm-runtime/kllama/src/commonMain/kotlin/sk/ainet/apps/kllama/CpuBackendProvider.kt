package sk.ainet.apps.kllama

import sk.ainet.apps.llm.backend.BackendProvider
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext

/**
 * CPU backend provider — always available on all platforms.
 */
public class CpuBackendProvider : BackendProvider {
    override val name: String = "cpu"
    override val displayName: String = "CPU (SIMD)"
    override val priority: Int = 0
    override fun isAvailable(): Boolean = true
    override fun createContext(): ExecutionContext = DirectCpuExecutionContext()
}
