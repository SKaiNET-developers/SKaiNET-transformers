package sk.ainet.transformers.iree.android

/**
 * JNI wrapper for `libskainet_iree_redecode.so` — a generic fixed-seq redecode cartridge
 * over the real IREE C runtime API. Drives ANY DSL-compiled vmfb that follows the redecode
 * graph contract (see [IreeRedecodeDecoder]): [step] takes the full padded `[seq]` token
 * buffer and returns the `[seq]` predicted-next-token ids for every position (causal
 * masking makes padding safe) — the caller reads the one position it needs and grows the
 * buffer, matching `gemma-iree`'s `GemmaDecoder` re-decode loop.
 *
 * This class knows nothing about any specific model: [vmfbPath], [irpaPath], and
 * [functionName] are all caller-supplied. Weights are EXTERNAL — bound at session-create
 * time from the `.irpa` via the IREE `io_parameters` VM module, not baked into the vmfb.
 *
 * The package/class name is the JNI symbol contract with the `.so` — do not move/rename.
 *
 * @param taskTopologyGroupCount local-task worker groups for the device this session creates
 *   (SKaiNET SKEEP-005 phase 2: the vmfb carries no core count; this is the run-time knob, the
 *   same `--task_topology_group_count` `iree-run-module` takes). `null` leaves IREE's own
 *   topology detection in charge. Map an engine schedule with [IreeTaskTopology.groupCountFor];
 *   read the `SKAINET_TASK_GROUPS` environment knob with [IreeTaskTopology.fromEnv].
 */
public class IreeRedecodeSession(
    vmfbPath: String,
    irpaPath: String,
    private val functionName: String,
    device: String = DEFAULT_DEVICE,
    taskTopologyGroupCount: Int? = null,
) : AutoCloseable {

    private var handle: Long = 0

    init {
        loadNativeLibrary()
        // The runtime resolves functions by module-qualified name (`module.gemma`); a bare name from the
        // export contract (`FunctionGemmaContract.FN_REDECODE` = "gemma") creates fine and then fails every
        // step inside the JNI without a message (#404). Qualify it here so both spellings work.
        val qualifiedName = if ('.' in functionName) functionName else "module.$functionName"
        handle = if (taskTopologyGroupCount == null) {
            nativeCreate(device, vmfbPath, irpaPath, qualifiedName)
        } else {
            require(taskTopologyGroupCount > 0) { "taskTopologyGroupCount must be >= 1, got $taskTopologyGroupCount" }
            try {
                nativeCreateWithTopology(device, vmfbPath, irpaPath, qualifiedName, taskTopologyGroupCount)
            } catch (e: UnsatisfiedLinkError) {
                // A `.so` built before the knob existed: refuse rather than silently run on IREE's default topology.
                throw IllegalStateException(
                    "libskainet_iree_redecode.so predates the task-topology knob (nativeCreateWithTopology missing); " +
                        "rebuild it with native/build-iree-redecode.sh or pass taskTopologyGroupCount = null.", e,
                )
            }
        }
        require(handle != 0L) {
            "IREE redecode session native create failed (device='$device', function='$functionName', " +
                "taskGroups=${taskTopologyGroupCount ?: "auto"}); check libskainet_iree_redecode.so, the vmfb, and the irpa."
        }
    }

    /** One redecode step: [tokenIds] must be exactly the vmfb's fixed `seq` length. */
    public fun step(tokenIds: IntArray): IntArray? {
        if (handle == 0L) return null
        val out = nativeStep(handle, tokenIds)
        if (out == null) {
            // The native side drops its iree_status_t (#404); make the failure visible at least.
            android.util.Log.e("IreeRedecodeSession", "nativeStep returned null (function='$functionName', seq=${tokenIds.size}): input buffer, call, or output mapping failed in the IREE runtime")
        }
        return out
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0
        }
    }

    private external fun nativeCreate(device: String, vmfbPath: String, irpaPath: String, functionName: String): Long
    private external fun nativeCreateWithTopology(device: String, vmfbPath: String, irpaPath: String, functionName: String, taskTopologyGroupCount: Int): Long
    private external fun nativeStep(handle: Long, tokenIds: IntArray): IntArray?
    private external fun nativeDestroy(handle: Long)

    public companion object {
        /** CPU driver — the primary, numerically-verified device string. */
        public const val DEFAULT_DEVICE: String = "local-task"

        /** GPU driver (Vulkan) — requires a `.so` built with `--vulkan` and a
         *  `vulkan-spirv`-compiled vmfb; see the module README. */
        public const val VULKAN_DEVICE: String = "vulkan"

        @Volatile private var loaded: Boolean = false
        private fun loadNativeLibrary() {
            if (loaded) return
            synchronized(this) {
                if (loaded) return
                System.loadLibrary("skainet_iree_redecode")
                loaded = true
            }
        }
    }
}
