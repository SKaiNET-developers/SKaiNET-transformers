package sk.ainet.apps.llm.diag

import sk.ainet.lang.tensor.Tensor

/**
 * Multiplatform shim for the per-block / per-MHA debug dumps that
 * `HybridTransformerBlock` and `MultiHeadAttention` emit when their
 * env-var gates are set. The dumps are JVM-only (they reference
 * `System.getenv`, `MemorySegmentTensorData`, and `String.format`,
 * none of which are available on JS/wasm/native), so this file
 * declares the platform abstractions and the actuals live in
 * `jvmMain` (real impl) and `registryBasedMain` (no-op fallback for
 * every non-JVM target).
 */

/**
 * Read an env-var as a boolean flag. JVM: `System.getenv(name) == "1"`.
 * Other targets: always returns false (env vars don't exist or aren't
 * a useful concept in JS/wasm/native browser/node hosts).
 */
public expect fun envFlag(name: String): Boolean

/**
 * Print one line of stats about [tensor] (min/max/mean/rms/argmax) and a
 * fingerprint of the last sequence position. JVM does the full formatted
 * dump; other targets emit nothing — diagnostics are JVM-only.
 */
public expect fun dumpStats(label: String, tensor: Tensor<*, *>)
