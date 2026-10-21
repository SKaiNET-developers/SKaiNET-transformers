package sk.ainet.models.whisper

import java.io.File

/**
 * Test-side support for the whisper export (the export itself is [WhisperExportHarness] in jvmMain):
 * the env-gated checkpoint, the golden greedy sequence, `.npy` I/O for `iree-run-module`, the recorded
 * resources and the dockerized IREE toolchain used by [WhisperExportTest] (G2a) and
 * [WhisperVmfbParityTest] (G2b).
 *
 * The goldens were recorded from `primeline/whisper-tiny-german-1224` (tiny multilingual) at audioCtx=200,
 * maxPositions=48 with the German transcribe prompt — the configuration [export] uses.
 */
object WhisperExportTestSupport {

    val cfg = WhisperConfig() // tiny multilingual @ audioCtx=200
    const val MAX_P = 48
    const val SCOPE = "model"
    const val IMAGE = "skainet/iree-compiler:3.11.0"

    /** Golden greedy sequence recorded from the validated ONNX pipeline (G1). */
    val GOLDEN_TOKENS = intArrayOf(37, 1248, 2804, 5553, 635, 20314, 13)

    fun snapshotDir(): File? =
        System.getenv("WHISPER_SAFETENSORS")?.let { File(it) }
            ?.takeIf { File(it, "model.safetensors").exists() }

    fun exportOutDir(): File =
        File(System.getenv("WHISPER_EXPORT_OUT") ?: "build/whisper-export").absoluteFile

    /** The export of the env-gated checkpoint in the goldens' configuration. */
    fun export(outDir: File): WhisperExportHarness.ExportResult {
        val snap = requireNotNull(snapshotDir()) { "WHISPER_SAFETENSORS not set / invalid" }
        val harness = WhisperExportHarness(cfg, WhisperExportOptions(audioCtx = cfg.audioCtx, maxPositions = MAX_P, language = "de", scope = SCOPE)) { println(it) }
        return harness.export(File(snap, "model.safetensors"), outDir)
    }

    fun manifestJson(): String =
        WhisperExportHarness(cfg, WhisperExportOptions(audioCtx = cfg.audioCtx, maxPositions = MAX_P, language = "de", scope = SCOPE)).manifestJson()

    // ------------------------------------------------------------------ npy I/O

    /** Write a little-endian `.npy` (v1.0). [descr] = `<f4` or `<i4`. */
    private fun writeNpy(file: File, descr: String, shape: IntArray, bytes: ByteArray) {
        val shapeStr = shape.joinToString(", ") + (if (shape.size == 1) "," else "")
        var header = "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr), }"
        val pad = 64 - ((10 + header.length + 1) % 64)
        header += " ".repeat(if (pad == 64) 0 else pad) + "\n"
        val hlen = header.length
        file.outputStream().buffered().use { out ->
            out.write(
                byteArrayOf(
                    0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(),
                    'P'.code.toByte(), 'Y'.code.toByte(), 1, 0,
                    (hlen and 0xff).toByte(), ((hlen shr 8) and 0xff).toByte(),
                )
            )
            out.write(header.toByteArray(Charsets.US_ASCII))
            out.write(bytes)
        }
    }

    fun writeNpyFloat(file: File, data: FloatArray, shape: IntArray) {
        val b = ByteArray(data.size * 4)
        for (i in data.indices) {
            val v = data[i].toRawBits()
            b[i * 4] = v.toByte(); b[i * 4 + 1] = (v shr 8).toByte()
            b[i * 4 + 2] = (v shr 16).toByte(); b[i * 4 + 3] = (v shr 24).toByte()
        }
        writeNpy(file, "<f4", shape, b)
    }

    fun writeNpyInt(file: File, data: IntArray, shape: IntArray) {
        val b = ByteArray(data.size * 4)
        for (i in data.indices) {
            val v = data[i]
            b[i * 4] = v.toByte(); b[i * 4 + 1] = (v shr 8).toByte()
            b[i * 4 + 2] = (v shr 16).toByte(); b[i * 4 + 3] = (v shr 24).toByte()
        }
        writeNpy(file, "<i4", shape, b)
    }

    /** Read a contiguous little-endian float32 `.npy` (what `--output=@f.npy` writes). */
    fun readNpyFloat(file: File): FloatArray {
        val b = file.readBytes()
        require(b.size > 10 && b[0] == 0x93.toByte()) { "not a .npy: ${file.name}" }
        val headerLen = (b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
        val dataStart = 10 + headerLen
        val n = (b.size - dataStart) / 4
        return FloatArray(n) { i ->
            val o = dataStart + i * 4
            val bits = (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8) or
                ((b[o + 2].toInt() and 0xff) shl 16) or ((b[o + 3].toInt() and 0xff) shl 24)
            Float.fromBits(bits)
        }
    }

    fun resourceFloats(name: String): FloatArray {
        val bytes = checkNotNull(javaClass.getResourceAsStream("/whisper-de/$name")) {
            "missing resource $name"
        }.readBytes()
        return FloatArray(bytes.size / 4) { i ->
            var bits = 0
            for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
            Float.fromBits(bits)
        }
    }

    fun argmax(a: FloatArray, from: Int = 0, len: Int = a.size - from): Int {
        var best = 0
        for (i in 0 until len) if (a[from + i] > a[from + best]) best = i
        return best
    }

    // ------------------------------------------------------------------ docker

    /** Runs the pinned IREE toolchain image with [workDir] mounted at /work. */
    class Toolchain(private val workDir: File, private val image: String = IMAGE) {

        fun available(): Boolean = try {
            ProcessBuilder("docker", "image", "inspect", image)
                .redirectErrorStream(true).start()
                .also { it.inputStream.readBytes() }
                .waitFor() == 0
        } catch (_: Exception) {
            false
        }

        /** Run one toolchain subcommand; returns exit code + combined output. */
        fun run(vararg args: String): Pair<Int, String> {
            val cmd = listOf(
                "docker", "run", "--rm",
                "-v", "${workDir.absolutePath}:/work", "-w", "/work", image,
            ) + args
            val p = ProcessBuilder(cmd).redirectErrorStream(true).start()
            val out = p.inputStream.bufferedReader().readText()
            return p.waitFor() to out
        }

        fun compileCpu(mlir: String, vmfb: String): Pair<Int, String> = run(
            "compile", "--",
            "--iree-input-type=stablehlo",
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            mlir, "-o", vmfb,
        )
    }
}
