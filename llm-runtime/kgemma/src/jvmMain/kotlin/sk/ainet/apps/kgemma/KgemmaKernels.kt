package sk.ainet.apps.kgemma

import java.util.concurrent.atomic.AtomicBoolean
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.exec.kernel.FfmRowMajorKernelPack
import sk.ainet.lang.memory.ExperimentalMemoryApi

/**
 * Installs the 0.51 view-keyed kernel tiers exactly once per process (mirrors
 * `KLlamaJava.ensureKernelPacksInstalled` / `kllama-cli`, #354). Every kgemma
 * entry point — [Gemma4ChatModel], the kgemma CLI, [FunctionGemma] — must call
 * this before the first forward pass: without it, MAPPED/keep-packed weights
 * fall to KernelDispatch's decoding reference kernel, which is correct but
 * measured at hours-scale on real checkpoints rather than sub-second.
 */
public object KgemmaKernels {

    private val installed = AtomicBoolean(false)

    @OptIn(ExperimentalMemoryApi::class)
    public fun ensureInstalled() {
        if (!installed.compareAndSet(false, true)) return
        KernelPacks.install()
        FfmRowMajorKernelPack.install()
    }
}
