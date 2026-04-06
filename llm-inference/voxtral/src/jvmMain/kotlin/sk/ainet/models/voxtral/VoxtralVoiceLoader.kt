package sk.ainet.models.voxtral

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipInputStream
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.name

/**
 * Loads Voxtral voice embeddings from PyTorch `.pt` files.
 *
 * PyTorch `.pt` files are ZIP archives containing pickled metadata and raw tensor data.
 * Voice embeddings are BF16 tensors of shape [N, 3072] stored at a path like
 * `archive/data/0` or `data/0` inside the ZIP.
 *
 * This loader extracts the raw tensor data and converts BF16 → FP32.
 */
public object VoxtralVoiceLoader {

    private const val DEFAULT_DIM = 3072

    /**
     * Load a voice embedding from a .pt file.
     *
     * @param path Path to the .pt file
     * @param dim Expected embedding dimension (default: 3072)
     * @return [VoxtralVoice] with the loaded embeddings
     */
    public fun load(path: Path, dim: Int = DEFAULT_DIM): VoxtralVoice {
        val voiceName = path.name.removeSuffix(".pt")
        val bytes = readTensorDataFromPt(path)
        val floats = dequantBF16(bytes)

        // Infer number of frames from total floats and dim
        require(floats.size % dim == 0) {
            "Tensor data size ${floats.size} is not divisible by dim $dim"
        }
        val numFrames = floats.size / dim

        return VoxtralVoice(
            name = voiceName,
            embeddings = floats,
            numFrames = numFrames,
            dim = dim
        )
    }

    /**
     * Load a voice by name from a model directory.
     * Looks for `<voiceName>.pt` in the directory.
     *
     * @param modelDir Path to the model directory (e.g. Voxtral-4B-TTS-2603/)
     * @param voiceName Voice name (e.g. "casual_male")
     * @return [VoxtralVoice] or null if the file doesn't exist
     */
    public fun loadFromDir(modelDir: Path, voiceName: String, dim: Int = DEFAULT_DIM): VoxtralVoice? {
        val ptFile = modelDir.resolve(VoxtralVoices.filename(voiceName))
        if (!ptFile.exists()) return null
        return load(ptFile, dim)
    }

    /**
     * List available voice .pt files in a model directory.
     */
    public fun listAvailable(modelDir: Path): List<String> {
        if (!modelDir.isDirectory()) return emptyList()
        return Files.list(modelDir).use { stream ->
            stream
                .filter { it.name.endsWith(".pt") }
                .map { it.name.removeSuffix(".pt") }
                .filter { it in VoxtralVoices.PRESETS }
                .toList()
                .sorted()
        }
    }

    /**
     * Extract raw tensor data bytes from a PyTorch .pt ZIP archive.
     *
     * PyTorch saves tensors in a ZIP with structure:
     * - `archive/data.pkl` — pickle metadata (we skip this)
     * - `archive/data/0` — raw tensor bytes (this is what we need)
     *
     * Some files use `data/0` instead of `archive/data/0`.
     */
    private fun readTensorDataFromPt(path: Path): ByteArray {
        val candidates = listOf("archive/data/0", "data/0", "voice_embed/data/0")
        var tensorData: ByteArray? = null

        ZipInputStream(Files.newInputStream(path)).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                val name = entry.name
                if (candidates.any { name.endsWith(it) } ||
                    (name.contains("/data/") && name.endsWith("/0"))) {
                    val baos = ByteArrayOutputStream()
                    val buf = ByteArray(8192)
                    var len: Int
                    while (zis.read(buf).also { len = it } != -1) {
                        baos.write(buf, 0, len)
                    }
                    tensorData = baos.toByteArray()
                    break
                }
                entry = zis.nextEntry
            }
        }

        return tensorData ?: error("No tensor data found in ${path}. Expected archive/data/0 or data/0 inside the ZIP.")
    }

    /**
     * Convert BF16 byte array to FP32 float array.
     * BF16 is just the upper 16 bits of FP32.
     */
    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val numFloats = bytes.size / 2
        val out = FloatArray(numFloats)
        for (i in 0 until numFloats) {
            val offset = i * 2
            val lo = bytes[offset].toInt() and 0xFF
            val hi = bytes[offset + 1].toInt() and 0xFF
            // BF16 → FP32: shift left by 16 bits
            val bits = (hi shl 24) or (lo shl 16)
            out[i] = Float.fromBits(bits)
        }
        return out
    }
}
