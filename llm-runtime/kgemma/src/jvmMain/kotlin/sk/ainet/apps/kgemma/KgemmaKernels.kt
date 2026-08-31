package sk.ainet.apps.kgemma

import java.util.concurrent.atomic.AtomicBoolean
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.exec.kernel.FfmRowMajorKernelPack
import sk.ainet.lang.memory.ExperimentalMemoryApi

/**
 * Installs the 0.51 view-keyed kernel tiers exactly once per process (mirrors
 * `KLlamaJava.ensureKernelPacksInstalled` / `kllama-cli`, #354). Every kgemma
 * entry point — [Gemma4ChatModel], the kgemma CLI, [FunctionGemma] — must call
 * this before the first forward pass: without it, MAPPED/keep-packed weights
 * fall to KernelDispatch's decoding reference kernel, which is correct but
 * measured at hours-scale on real checkpoints rather than sub-second.
 *
 * **Ordering matters.** `KernelPacks.install()` defaults its provider to
 * `KernelRegistry.bestAvailable()`, which is null while the provider registry is
 * empty — and the registry is only filled lazily, by
 * `DefaultCpuOpsJvm.ensureKernelProviders()`, the first time an ops instance
 * touches a kernel. A bootstrap that runs at process start (all of them do)
 * therefore installs only the reference kernel from that call and silently skips
 * the dense-FP32 view kernels and the input-block-major packed tier: 8 of 17
 * kernels land instead of all 17 (pinned by `KernelBootstrapOrderingTest`). The
 * GGUF decode path happens to survive on the row-major FFM pack alone, which
 * installs unconditionally — but nothing else does. Discovering providers first
 * closes that gap.
 */
public object KgemmaKernels {

    private val installed = AtomicBoolean(false)

    @OptIn(ExperimentalMemoryApi::class)
    public fun ensureInstalled() {
        if (!installed.compareAndSet(false, true)) return
        if (KernelRegistry.providers().isEmpty()) KernelServiceLoader.installAll()
        KernelPacks.install()
        FfmRowMajorKernelPack.install()
    }
}
