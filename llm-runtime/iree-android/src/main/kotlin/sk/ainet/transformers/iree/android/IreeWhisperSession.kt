package sk.ainet.transformers.iree.android

/**
 * Whisper speech-to-text over the IREE runtime (`libskainet_whisper_session.so`): the three graphs of
 * `WhisperExportCli` (`:llm-inference:whisper:exportWhisper`, whisper contract version 2) — encoder, masked
 * prefill, KV step — plus the parameter archive prefill and step share.
 *
 * [decode] takes one log-mel window (`[nMels, melFrames]` row-major, Whisper's frontend: e.g. SKaiNET-audio's
 * `MelSpectrogram` with `MelConfig.whisper(maxFrames = melFrames)`) and returns the greedy token ids after the
 * prompt, end-of-text excluded. Detokenization is the caller's: the ids index the checkpoint's `tokenizer.json`.
 *
 * Nothing about the checkpoint is compiled into the library: the [Contract] (from the export's
 * `manifest.json`) carries geometry, function names, parameter scope and the prompt, so one library serves
 * tiny, base, small and the English-only variants.
 *
 * The encoder and the decoder graphs can run on different devices: `IreeWhisperSession("vulkan", files, contract,
 * decoderDevice = "local-task")` puts the encoder — one large batch over the whole window, where a GPU pays — on
 * the GPU and prefill and step — chains of small matrix-vector products — on the CPU; the encoder features cross
 * once through host memory. The vmfbs must be compiled for the device each one runs on.
 *
 * One instance = one decode at a time; callers serialize
 * access. Every native failure surfaces as an [IllegalStateException].
 * The package/class name is the JNI symbol contract with the library — do not move/rename.
 */
public class IreeWhisperSession(
    /** `"vulkan"` (GPU) or `"local-task"` (CPU): the encoder's device, and the decoder's unless [decoderDevice] says otherwise. */
    device: String,
    /** The four model files, by absolute path. */
    files: Files,
    /** The graph contract the files were exported with. */
    public val contract: Contract,
    /** The device prefill and step run on. */
    decoderDevice: String = device,
) : AutoCloseable {
    /** Absolute paths of the model files the runtime opens. */
    public data class Files(
        val encoderVmfb: String,
        val prefillVmfb: String,
        val stepVmfb: String,
        val paramsIrpa: String,
    )

    /**
     * What the runtime needs from the export's `manifest.json` (whisper contract version 2): the three function
     * names, the parameter scope prefill and step reference, the geometry, end-of-text and the prompt
     * (`<sot> <lang> <transcribe> <notimestamps>`, or `<sot> <notimestamps>` for an English-only checkpoint).
     */
    public data class Contract(
        val encoderFunction: String,
        val prefillFunction: String,
        val stepFunction: String,
        val parameterScope: String,
        val nLayers: Int,
        val dModel: Int,
        val audioCtx: Int,
        val nMels: Int,
        val melFrames: Int,
        val maxPositions: Int,
        val vocab: Int,
        val eot: Int,
        val promptIds: IntArray,
    ) {
        init {
            require(promptIds.isNotEmpty() && promptIds.size < maxPositions) { "prompt of ${promptIds.size} tokens does not fit $maxPositions positions" }
        }

        internal fun geometry(): IntArray = intArrayOf(nLayers, dModel, audioCtx, nMels, melFrames, maxPositions, vocab)

        public companion object {
            /** The contract version this class reads. */
            public const val VERSION: Int = 2

            /**
             * Minimal parser for `manifest.json` (no JSON dependency). `nMels` is not in the manifest; it is the
             * checkpoint's (80 for tiny … small, 128 for large-v3) — pass it when it differs.
             */
            public fun fromManifest(json: String, nMels: Int = 80): Contract {
                fun int(key: String): Int = Regex("\"$key\"\\s*:\\s*(-?\\d+)").find(json)?.groupValues?.get(1)?.toInt()
                    ?: throw IllegalArgumentException("whisper manifest: no integer '$key'")
                fun str(key: String): String = Regex("\"$key\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1)
                    ?: throw IllegalArgumentException("whisper manifest: no string '$key'")
                val version = int("contractVersion")
                require(version == VERSION) { "whisper contract version $version; this runtime reads version $VERSION" }
                val prompt = Regex("\"promptIds\"\\s*:\\s*\\[([^\\]]*)]").find(json)?.groupValues?.get(1)
                    ?: throw IllegalArgumentException("whisper manifest: no 'promptIds'")
                return Contract(
                    encoderFunction = str("encoder"), prefillFunction = str("prefill"), stepFunction = str("step"),
                    parameterScope = str("parameterScope"),
                    nLayers = int("nLayers"), dModel = int("dModel"), audioCtx = int("audioCtx"), nMels = nMels,
                    melFrames = int("melFrames"), maxPositions = int("maxPositions"), vocab = int("vocab"), eot = int("eot"),
                    promptIds = prompt.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray(),
                )
            }
        }
    }

    private var handle: Long = 0

    init {
        loadNativeLibrary()
        handle = nativeCreate(
            device, decoderDevice, files.encoderVmfb, files.prefillVmfb, files.stepVmfb, files.paramsIrpa, contract.parameterScope,
            contract.encoderFunction, contract.prefillFunction, contract.stepFunction, contract.geometry(),
        )
    }

    /**
     * Greedy decode of one log-mel window. Returns the generated ids, end-of-text excluded.
     *
     * @param mel `[nMels, melFrames]` row-major.
     * @param maxNewTokens upper bound on generated tokens; 0 = until end-of-text or a full KV cache.
     * @param repeatNgram stop as soon as the last [repeatNgram] tokens already occurred earlier and drop them —
     *   greedy decoding over a short window can transcribe the utterance and then start it again instead of
     *   ending. 0 disables the guard.
     */
    public fun decode(mel: FloatArray, maxNewTokens: Int = 0, repeatNgram: Int = DEFAULT_REPEAT_NGRAM, promptIds: IntArray = contract.promptIds): IntArray {
        check(handle != 0L) { "IreeWhisperSession is closed" }
        require(mel.size == contract.nMels * contract.melFrames) { "mel has ${mel.size} values, the encoder takes ${contract.nMels} x ${contract.melFrames}" }
        return nativeDecode(handle, mel, promptIds, contract.eot, maxNewTokens, repeatNgram)
    }

    /**
     * What the last [decode] cost: `encoderMs`, `prefillMs`, `stepMs`, `totalMs`, `steps`, `tokens`, and how it
     * ended (`endedOnEot`, `stoppedOnRepeat`; both 0 = token budget or full KV cache). Empty before the first decode.
     */
    public fun stats(): Map<String, String> {
        if (handle == 0L) return emptyMap()
        val raw = nativeStats(handle) ?: return emptyMap()
        return raw.split(',').mapNotNull { pair ->
            val k = pair.substringBefore('=', "")
            if (k.isEmpty()) null else k to pair.substringAfter('=')
        }.toMap()
    }

    override fun close() { if (handle != 0L) { nativeDestroy(handle); handle = 0 } }

    private external fun nativeCreate(
        encoderDevice: String, decoderDevice: String, enc: String, prefill: String, step: String, params: String, scope: String,
        encFn: String, prefillFn: String, stepFn: String, geometry: IntArray,
    ): Long
    private external fun nativeDecode(handle: Long, mel: FloatArray, promptIds: IntArray, eot: Int, maxNewTokens: Int, repeatNgram: Int): IntArray
    private external fun nativeStats(handle: Long): String?
    private external fun nativeDestroy(handle: Long)

    public companion object {
        public const val VULKAN_DEVICE: String = "vulkan"
        public const val CPU_DEVICE: String = "local-task"
        public const val DEFAULT_REPEAT_NGRAM: Int = 3

        /** The four file names of a compiled model directory, in the order [Files] takes them. */
        public val FILE_NAMES: List<String> = listOf("encoder.vmfb", "prefill.vmfb", "step.vmfb", "params.irpa")

        /** [Files] for a directory that holds the four files under their standard names. */
        public fun filesIn(dir: java.io.File): Files {
            val p = FILE_NAMES.map { java.io.File(dir, it).absolutePath }
            return Files(p[0], p[1], p[2], p[3])
        }

        @Volatile private var loaded = false
        private fun loadNativeLibrary() {
            if (loaded) return
            synchronized(this) { if (!loaded) { System.loadLibrary("skainet_whisper_session"); loaded = true } }
        }
    }
}
