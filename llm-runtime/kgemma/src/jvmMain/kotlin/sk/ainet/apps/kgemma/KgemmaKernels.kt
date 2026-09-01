package sk.ainet.apps.kgemma

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.lang.memory.ExperimentalMemoryApi

/**
 * Kernel bootstrap facade for the kgemma entry points ([GemmaChatModel], the kgemma CLI,
 * [FunctionGemma]).
 *
 * Since engine 0.52.0 the dispatcher self-heals: `KernelDispatch.ensureInstalled()` discovers
 * providers first, then every `ViewKernelPack` on the classpath (row-major Q-series AND the
 * ternary/BitNet packs, SKaiNET#1240) — the ordering trap this object used to work around
 * (providers-before-packs, pinned by `KernelBootstrapOrderingTest`) is handled inside the
 * engine. The facade stays so call sites keep one explicit bootstrap point that runs before
 * the loaders' mapped-staging decisions.
 *
 * All-or-nothing note (the transformers#370 trap): explicit kernel registration suppresses
 * auto-install — so this must remain the ONLY bootstrap; adding a manual `install()` call
 * beside it would silently disable discovery for everything else.
 */
public object KgemmaKernels {

    @OptIn(ExperimentalMemoryApi::class)
    public fun ensureInstalled() {
        KernelDispatch.ensureInstalled()
    }
}
