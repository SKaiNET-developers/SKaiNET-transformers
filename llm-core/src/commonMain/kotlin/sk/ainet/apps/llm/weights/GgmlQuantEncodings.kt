package sk.ainet.apps.llm.weights

import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Map a GGUF [GGMLQuantizationType] to the engine [TensorEncoding] of the
 * formats with a packed CPU matmul kernel (+ lazy `ops.transpose` support),
 * or `null` for types that must be dequantized.
 *
 * This is the ggml-keyed front door to the shared
 * [sk.ainet.lang.nn.quant.BlockQuantPacking] packer (#184 hoist 2): model
 * modules call `qt.toBlockEncoding()?.let { BlockQuantPacking.pack(bytes, it, shape) }`
 * and keep only weight *selection* and naming for themselves.
 */
public fun GGMLQuantizationType.toBlockEncoding(): TensorEncoding? = when (this) {
    GGMLQuantizationType.Q4_K -> TensorEncoding.Q4_K
    GGMLQuantizationType.Q5_K -> TensorEncoding.Q5_K
    GGMLQuantizationType.Q6_K -> TensorEncoding.Q6_K
    GGMLQuantizationType.Q8_0 -> TensorEncoding.Q8_0
    GGMLQuantizationType.Q4_0 -> TensorEncoding.Q4_0
    GGMLQuantizationType.Q5_0 -> TensorEncoding.Q5_0
    GGMLQuantizationType.Q5_1 -> TensorEncoding.Q5_1
    else -> null
}

/**
 * Runtime kernel gate for the packed-quant conversion path (#170): `true` iff
 * some registered, available [sk.ainet.backend.api.kernel.KernelProvider]
 * carries an `FP32 activations x packed-<this>` matmul kernel, so a weight
 * packed by [sk.ainet.lang.nn.quant.BlockQuantPacking] will actually dispatch
 * to a packed kernel instead of the generic elementwise matmul (which reads
 * block-major bytes with row-major strides and would be numerically wrong
 * after the lazy packed transpose).
 *
 * Converters gate NEW packed formats on this check and keep the FP32 dequant
 * fallback for `false` — availability-based, not engine-version-based: under
 * engine 0.39.0 the scalar/Panama Q5_0/Q5_1 kernels already report here, and
 * the native FFM/Kotlin-Native/JNI tiers added by SKaiNET#951 (0.40.0) light
 * up through the same query without any transformers-side change.
 *
 * On the JVM this installs `ServiceLoader`-discovered providers first
 * (idempotent), mirroring the engine's own lazy `ensureKernelProviders`, so
 * the answer is correct even before the first matmul runs. On registry-based
 * targets (Kotlin/Native, Android, JS/WASM) providers are registered by the
 * platform backend factories; converters run with a live [sk.ainet.context.ExecutionContext],
 * which implies that registration has happened.
 */
public fun GGMLQuantizationType.hasPackedMatmulKernel(): Boolean {
    val encoding = toBlockEncoding() ?: return false
    ensurePlatformKernelProviders()
    return KernelRegistry.providers().any {
        it.isAvailable() && it.supports("matmul", listOf("Float32", encoding.name))
    }
}

/**
 * Populate [KernelRegistry] with the platform's discoverable providers before
 * a capability query. JVM: `KernelServiceLoader.installAll()` (idempotent).
 * Registry-based targets: no-op — providers are registered explicitly by the
 * backend factories (see `BackendRegistry.registryBased.kt`).
 */
internal expect fun ensurePlatformKernelProviders()
