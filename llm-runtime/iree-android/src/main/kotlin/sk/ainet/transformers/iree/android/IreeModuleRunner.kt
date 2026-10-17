package sk.ainet.transformers.iree.android

/** A dense float32 tensor, row-major: what [IreeModuleRunner] takes and returns. */
public class FloatTensor(public val data: FloatArray, public val shape: IntArray) {
    init {
        require(shape.isNotEmpty() && shape.size <= MAX_RANK) { "rank must be 1..$MAX_RANK, got ${shape.size}" }
        require(shape.all { it > 0 }) { "dimensions must be positive: ${shape.toList()}" }
        val n = shape.fold(1L) { a, d -> a * d }
        require(n == data.size.toLong()) { "shape ${shape.toList()} needs $n values, got ${data.size}" }
    }

    override fun toString(): String = "FloatTensor(${shape.joinToString("x")})"

    internal companion object {
        const val MAX_RANK = 8

        /** The native side's `[rank, d0, d1, …]` shape buffer → the result tensor. */
        fun fromNative(data: FloatArray, shapeBuffer: IntArray): FloatTensor {
            val rank = shapeBuffer[0]
            require(rank in 1..MAX_RANK) { "native output rank $rank" }
            return FloatTensor(data, shapeBuffer.copyOfRange(1, 1 + rank))
        }
    }
}

/**
 * JNI wrapper for `libskainet_iree_module.so` — a generic single-shot runner over the IREE C runtime: one
 * compiled module, one entry function taking one float32 tensor and returning one float32 tensor (static
 * shapes). For batch cartridges — a detector, a classifier, an embedding graph — that are a plain tensor function.
 *
 * This class knows nothing about any specific model: [vmfbPath], [irpaPath] and [functionName] are all
 * caller-supplied. With an [irpaPath], weights are EXTERNAL, bound at create time from the `.irpa` (scope `model`)
 * via the IREE `io_parameters` VM module; with `null`, the vmfb carries them.
 *
 * Not thread-safe: one [run] at a time per instance. The package/class name is the JNI symbol contract with the
 * `.so` — do not move/rename.
 *
 * @param functionName the exported function, bare (`yolov8_detect`) or module-qualified (`module.yolov8_detect`).
 * @param taskTopologyGroupCount local-task worker groups (see [IreeTaskTopology]); `null` lets IREE decide.
 */
public class IreeModuleRunner(
    vmfbPath: String,
    irpaPath: String?,
    functionName: String,
    device: String = DEFAULT_DEVICE,
    taskTopologyGroupCount: Int? = null,
) : AutoCloseable {

    public val functionName: String = qualify(functionName)
    private var handle: Long = 0

    init {
        require(taskTopologyGroupCount == null || taskTopologyGroupCount > 0) {
            "taskTopologyGroupCount must be >= 1, got $taskTopologyGroupCount"
        }
        loadNativeLibrary()
        handle = nativeCreate(device, vmfbPath, irpaPath.orEmpty(), this.functionName, taskTopologyGroupCount ?: 0)
        check(handle != 0L) {
            "IREE module runner create failed (device='$device', function='${this.functionName}', vmfb='$vmfbPath', " +
                "irpa=${irpaPath ?: "none"}); the cause is in logcat under tag skainet_iree."
        }
    }

    /** Invokes the function on [input]; throws with the IREE status logged under `skainet_iree` on failure. */
    public fun run(input: FloatTensor): FloatTensor {
        check(handle != 0L) { "runner is closed" }
        val shapeBuffer = IntArray(1 + FloatTensor.MAX_RANK)
        val out = nativeRun(handle, input.data, input.shape, shapeBuffer)
            ?: throw IllegalStateException(
                "IREE module run failed (function='$functionName', input $input); the cause is in logcat under tag skainet_iree.",
            )
        return FloatTensor.fromNative(out, shapeBuffer)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0
        }
    }

    private external fun nativeCreate(device: String, vmfbPath: String, irpaPath: String, functionName: String, taskTopologyGroupCount: Int): Long
    private external fun nativeRun(handle: Long, input: FloatArray, inputShape: IntArray, outputShape: IntArray): FloatArray?
    private external fun nativeDestroy(handle: Long)

    public companion object {
        /** CPU driver. */
        public const val DEFAULT_DEVICE: String = "local-task"

        /** GPU driver (Vulkan); needs a `vulkan-spirv`-compiled vmfb. The `.so` ships with it compiled in. */
        public const val VULKAN_DEVICE: String = "vulkan"

        /** The runtime resolves functions by module-qualified name; a bare export name gets `module.`. */
        internal fun qualify(name: String): String {
            require(name.isNotBlank()) { "functionName must not be blank" }
            return if ('.' in name) name else "module.$name"
        }

        @Volatile private var loaded: Boolean = false
        private fun loadNativeLibrary() {
            if (loaded) return
            synchronized(this) {
                if (loaded) return
                System.loadLibrary("skainet_iree_module")
                loaded = true
            }
        }
    }
}
