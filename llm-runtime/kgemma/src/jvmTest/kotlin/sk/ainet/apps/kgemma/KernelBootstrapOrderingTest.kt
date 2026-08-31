package sk.ainet.apps.kgemma

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.exec.kernel.FfmRowMajorKernelPack
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Model-free probe for the kernel-bootstrap ORDERING trap.
 *
 * `KernelPacks.install()` defaults its provider to `KernelRegistry.bestAvailable()`,
 * which is `null` while the provider registry is still empty — and the registry is
 * only populated lazily, by `DefaultCpuOpsJvm.ensureKernelProviders()`, the first
 * time an ops instance touches a kernel. Every current bootstrap
 * (kgemma's `KgemmaKernels`, kllama's CLI, `KLlamaJava`, `skainet-cli`) runs at
 * process start, i.e. BEFORE any ops instance exists.
 *
 * Prints what actually lands in `KernelDispatch` for both orderings so the
 * difference is visible rather than assumed.
 */
class KernelBootstrapOrderingTest {

    private fun snapshot(label: String) {
        val names = KernelDispatch.kernels().map { it.name }.sorted()
        println("BOOT %-28s n=%2d kernels=%s".format(label, names.size, names))
    }

    @Test
    fun install_before_vs_after_provider_discovery() {
        // (1) install() with an empty provider registry — the naive bootstrap.
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()
        KernelPacks.install()
        FfmRowMajorKernelPack.install()
        snapshot("cold (no provider discovery)")
        val cold = KernelDispatch.kernels().map { it.name }.toSet()

        // (2) The same calls, after provider discovery has run.
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()
        KernelServiceLoader.installAll()
        println("BOOT providers after installAll: ${KernelRegistry.availableNames()}")
        KernelPacks.install()
        FfmRowMajorKernelPack.install()
        snapshot("warm (installAll first)")
        val warm = KernelDispatch.kernels().map { it.name }.toSet()

        // The gap this test exists to pin: provider-derived tiers (dense FP32
        // views + the input-block-major packed kernels) only appear in (2).
        assertTrue(
            warm.size > cold.size,
            "expected provider discovery to add kernels; cold=$cold warm=$warm",
        )
        assertTrue(
            warm.any { it.endsWith("-fp32") } && cold.none { it.endsWith("-fp32") },
            "dense-FP32 view kernels should be the ones gained; cold=$cold warm=$warm",
        )
    }

    /** [KgemmaKernels.ensureInstalled] must land the full warm set, not the cold one. */
    @Test
    fun kgemma_bootstrap_installs_the_full_set() {
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()
        KgemmaKernels.ensureInstalled()
        snapshot("KgemmaKernels.ensureInstalled")
        val names = KernelDispatch.kernels().map { it.name }
        assertTrue(
            names.any { it.endsWith("-fp32") },
            "bootstrap must discover providers before KernelPacks.install(); got $names",
        )
        assertTrue(
            names.any { it.startsWith("ffm-rowmajor-Q4_K") },
            "bootstrap must install the row-major pack that serves GGUF K-quants; got $names",
        )
    }
}
