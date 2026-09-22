package sk.ainet.transformers.iree.android

/**
 * Streaming Moonshine v2 speech-to-text over the IREE runtime (`libskainet_moonshine_stream.so`): the
 * five graphs of `MoonshineV2ExportCli` — frontend, encoder, adapter, masked prefill, dynamic with-past
 * step — plus the shared decoder parameter archive, driven as a real streaming loop on the device.
 *
 * [feedPcm] takes mono 16 kHz float PCM as it arrives; the native side windows it (64-frame windows,
 * hop 44 frames = 0.88 s), runs frontend / encoder / adapter, appends finalized memory, decodes
 * incrementally and returns the updated cumulative partial transcript whenever it changed. [finish]
 * flushes the tail, runs one exact full re-decode over the final memory, returns the final transcript
 * and resets for the next utterance. [reset] aborts the current utterance and keeps the engine.
 *
 * One instance = one utterance at a time; callers serialize access. Per-hop stage timings are logged
 * under the `moonshine-timing` tag.
 *
 * The geometry (stream width, layers, heads) is compiled into the library — the default build is the
 * `tiny` class (320 / 6 / 8), which the English and German tiny streaming checkpoints share. Per-language
 * attention bands and vocabulary size come from the graphs and `vocab.bin`, not from the library.
 * Every native failure surfaces as an [IllegalStateException].
 * The package/class name is the JNI symbol contract with the library — do not move/rename.
 */
public class IreeMoonshineStream(
    /** `"vulkan"` (GPU — the real-time path on Mali) or `"local-task"` (CPU). */
    device: String,
    /** The eight files of a materialized Moonshine v2 streaming model, by absolute path. */
    files: Files,
) : AutoCloseable {
    /** Absolute paths of the model files the runtime opens. */
    public data class Files(
        val frontendVmfb: String,
        val encoderVmfb: String,
        val adapterVmfb: String,
        val prefillVmfb: String,
        val stepVmfb: String,
        val paramsIrpa: String,
        val vocabBin: String,
        val decEmbedBin: String,
    )

    private var handle: Long = 0

    init {
        loadNativeLibrary()
        handle = nativeCreate(
            device, files.frontendVmfb, files.encoderVmfb, files.adapterVmfb, files.prefillVmfb, files.stepVmfb,
            files.paramsIrpa, files.vocabBin, files.decEmbedBin,
        )
        check(handle != 0L) { "IreeMoonshineStream: native create failed (device='$device') — check the model files and, for 'vulkan', that the device has a Vulkan driver" }
    }

    /** Feed mono 16 kHz float PCM in [-1, 1]; returns the updated cumulative partial transcript, or null when unchanged. */
    public fun feedPcm(samples: FloatArray): String? =
        if (handle == 0L || samples.isEmpty()) null else nativeFeedPcm(handle, samples)

    /** End of utterance: the exact final transcript (or null); the stream is reset afterwards. */
    public fun finish(): String? = if (handle == 0L) null else nativeFinish(handle)

    /** Abort the current utterance, keep the engine. */
    public fun reset() { if (handle != 0L) nativeReset(handle) }

    override fun close() { if (handle != 0L) { nativeDestroy(handle); handle = 0 } }

    private external fun nativeCreate(
        device: String, fe: String, enc: String, adp: String, prefill: String, step: String,
        params: String, vocab: String, embed: String,
    ): Long
    private external fun nativeFeedPcm(handle: Long, pcm: FloatArray): String?
    private external fun nativeFinish(handle: Long): String?
    private external fun nativeReset(handle: Long)
    private external fun nativeDestroy(handle: Long)

    public companion object {
        public const val VULKAN_DEVICE: String = "vulkan"
        public const val CPU_DEVICE: String = "local-task"

        /** The eight file names a materialized model directory holds, in the order [Files] takes them. */
        public val FILE_NAMES: List<String> = listOf(
            "frontend.vmfb", "encoder.vmfb", "adapter.vmfb", "prefill.vmfb", "step.vmfb", "params.irpa", "vocab.bin", "dec_embed.bin",
        )

        /** [Files] for a directory that holds the eight files under their standard names. */
        public fun filesIn(dir: java.io.File): Files {
            val p = FILE_NAMES.map { java.io.File(dir, it).absolutePath }
            return Files(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        }

        @Volatile private var loaded = false
        private fun loadNativeLibrary() {
            if (loaded) return
            synchronized(this) { if (!loaded) { System.loadLibrary("skainet_moonshine_stream"); loaded = true } }
        }
    }
}
