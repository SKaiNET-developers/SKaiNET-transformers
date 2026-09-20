package sk.ainet.apps.kllama

import sk.ainet.apps.llm.backend.BackendCapabilities
import sk.ainet.apps.llm.backend.BackendOptions
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
    override val capabilities: BackendCapabilities =
        BackendCapabilities(supportedDTypes = setOf("FP32"), supportsCompile = false, usesNpu = false)
    override fun isAvailable(): Boolean = true
    override fun createContext(options: BackendOptions): ExecutionContext {
        installPlatformNativeKernels()
        return DirectCpuExecutionContext()
    }
}
