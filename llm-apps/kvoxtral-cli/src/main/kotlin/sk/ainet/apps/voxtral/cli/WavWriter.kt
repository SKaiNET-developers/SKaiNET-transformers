package sk.ainet.apps.voxtral.cli

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path

/**
 * Writes PCM audio data as a WAV file.
 *
 * Supports 16-bit signed PCM (the most common WAV format).
 * Audio data is provided as normalized floats in [-1.0, 1.0].
 */
object WavWriter {

    /**
     * Write a WAV file from float audio samples.
     *
     * @param path Output file path
     * @param samples Audio samples normalized to [-1.0, 1.0]
     * @param sampleRate Sample rate in Hz (default: 24000 for Voxtral)
     * @param channels Number of audio channels (default: 1 for mono)
     * @param bitsPerSample Bits per sample (default: 16)
     */
    fun write(
        path: Path,
        samples: FloatArray,
        sampleRate: Int = 24000,
        channels: Int = 1,
        bitsPerSample: Int = 16
    ) {
        Files.newOutputStream(path).use { out ->
            write(out, samples, sampleRate, channels, bitsPerSample)
        }
    }

    /**
     * Write WAV data to an output stream.
     */
    fun write(
        out: OutputStream,
        samples: FloatArray,
        sampleRate: Int = 24000,
        channels: Int = 1,
        bitsPerSample: Int = 16
    ) {
        val bytesPerSample = bitsPerSample / 8
        val blockAlign = channels * bytesPerSample
        val byteRate = sampleRate * blockAlign
        val dataSize = samples.size * bytesPerSample
        val fileSize = 36 + dataSize

        val buf = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        buf.put('R'.code.toByte())
        buf.put('I'.code.toByte())
        buf.put('F'.code.toByte())
        buf.put('F'.code.toByte())
        buf.putInt(fileSize)
        buf.put('W'.code.toByte())
        buf.put('A'.code.toByte())
        buf.put('V'.code.toByte())
        buf.put('E'.code.toByte())

        // fmt sub-chunk
        buf.put('f'.code.toByte())
        buf.put('m'.code.toByte())
        buf.put('t'.code.toByte())
        buf.put(' '.code.toByte())
        buf.putInt(16)              // sub-chunk size (PCM = 16)
        buf.putShort(1)             // audio format (1 = PCM)
        buf.putShort(channels.toShort())
        buf.putInt(sampleRate)
        buf.putInt(byteRate)
        buf.putShort(blockAlign.toShort())
        buf.putShort(bitsPerSample.toShort())

        // data sub-chunk
        buf.put('d'.code.toByte())
        buf.put('a'.code.toByte())
        buf.put('t'.code.toByte())
        buf.put('a'.code.toByte())
        buf.putInt(dataSize)

        out.write(buf.array())

        // Write PCM samples (float → 16-bit signed)
        val sampleBuf = ByteBuffer.allocate(bytesPerSample).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            val clamped = sample.coerceIn(-1.0f, 1.0f)
            val pcm = (clamped * 32767.0f).toInt().toShort()
            sampleBuf.clear()
            sampleBuf.putShort(pcm)
            out.write(sampleBuf.array())
        }
    }
}
