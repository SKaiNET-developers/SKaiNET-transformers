package sk.ainet.apps.llm.diag

import sk.ainet.lang.tensor.Tensor

/**
 * No-op fallbacks for every non-JVM target (JS, wasm, native, Android-via-
 * registryBasedMain). Diagnostic dumps are JVM-only — there's no JFR or
 * comparable profiling story on these platforms, so the env gates always
 * read false and the stat dumps emit nothing.
 */
public actual fun envFlag(name: String): Boolean = false

public actual fun dumpStats(label: String, tensor: Tensor<*, *>) {
    // No-op on non-JVM. Dumps are debug aids for JVM profiling and aren't
    // worth maintaining a JS/wasm-compatible printf path.
}
