package sk.ainet.apps.kgemma

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.lang.memory.ExperimentalMemoryApi
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Guards the assumption this module now relies on: **kgemma performs no kernel bootstrap of its
 * own**, because `KernelDispatch` populates itself on first use (engine-side `ensureInstalled()`
 * plus the `ViewKernelPack` ServiceLoader SPI).
 *
 * Two ways that assumption can silently break, both of which land us back on the decoding
 * reference kernel — correct output, ~1000x slower, no error:
 *
 *  1. the engine dependency is downgraded to a version predating the self-heal, or
 *  2. `skainet-backend-native-cpu` drops off kgemma's **runtime** classpath, so the FFM row-major
 *     pack is no longer discoverable. Note it is a `ServiceLoader` dependency now, not a compile
 *     one — nothing in this module references it by symbol, which is exactly why it is easy to
 *     "clean up" by mistake.
 */
@OptIn(ExperimentalMemoryApi::class)
class KernelBootstrapOrderingTest {

    @Test
    fun dispatch_self_heals_without_any_bootstrap_from_this_module() {
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()

        // No install call anywhere — the engine must handle it.
        KernelDispatch.ensureInstalled()

        val names = KernelDispatch.kernels().map { it.name }
        println("BOOT self-heal n=${names.size} providers=${KernelRegistry.availableNames()} kernels=${names.sorted()}")

        assertTrue(
            names.any { it.endsWith("-fp32") },
            "providers must be discovered before KernelPacks.install(), else only the reference " +
                "kernel lands; got $names",
        )
        assertTrue(
            names.any { it.startsWith("ffm-rowmajor-Q4_K") },
            "the FFM row-major pack must be ServiceLoader-discoverable — check that " +
                "skainet-backend-native-cpu is still on the runtime classpath; got $names",
        )
        assertTrue(
            KernelDispatch.mappedServableEncodings().isNotEmpty(),
            "K-quant weights should be servable zero-copy from a mapping",
        )
    }
}
