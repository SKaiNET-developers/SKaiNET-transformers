package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Loads Voxtral weights from SafeTensors format, capturing ALL tensor types:
 * backbone, acoustic model, and codec.
 *
 * Unlike [DecoderSafeTensorsLoader] which only maps backbone tensors,
 * this loader uses [VoxtralHfTensorNameMapper] to capture the full model
 * including acoustic model and codec weights.
 *
 * Backbone + acoustic tensors are stored in [DecoderGgufWeights.tensors] with
 * canonical names. Codec tensors are returned separately via [loadAll].
 */
public class VoxtralSafeTensorsLoader<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val metadata: GgufDecoderMetadata,
    private val tiedEmbeddings: Boolean = true
) {

    /**
     * Result of loading all Voxtral tensors.
     */
    public data class VoxtralWeights<T : DType>(
        /** Backbone weights (LLaMA-compatible, for DSL weight mapping). */
        val backbone: DecoderGgufWeights<T, Float>,
        /** All tensors by canonical name (backbone + acoustic + codec). */
        val allTensors: Map<String, Tensor<T, Float>>
    )

    /**
     * Load all tensors from SafeTensors file.
     */
    public fun loadAll(randomAccessProvider: () -> RandomAccessSource): VoxtralWeights<T> {
        val allTensors = mutableMapOf<String, Tensor<T, Float>>()

        StreamingSafeTensorsReader.open(randomAccessProvider()).use { reader ->
            val tensorInfoMap = reader.tensors.associateBy { it.name }

            for (info in reader.tensors) {
                if (info.name.endsWith(".qb")) continue

                val canonicalName = VoxtralHfTensorNameMapper.toCanonical(info.name)
                if (canonicalName == null) {
                    println("  SKIPPED (unmapped): ${info.name} (${info.dtype} ${info.shape})")
                    continue
                }

                val tensor = when (info.dataType) {
                    DataType.BFLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = dequantBF16(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.FLOAT16 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = dequantF16(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.FLOAT32 -> {
                        val bytes = reader.loadTensorData(info)
                        val floats = bytesToFloatArray(bytes)
                        val targetShape = normalizeNormShape(info.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    DataType.QUANT4 -> {
                        val qbName = info.name + ".qb"
                        val qbInfo = tensorInfoMap[qbName]
                        val rawQ4 = reader.loadTensorData(info)
                        val qbBytes = qbInfo?.let { reader.loadTensorData(it) }
                        val targetShape = Shape(*info.shape.map { it.toInt() }.toIntArray())
                        val floats = dequantQ4(rawQ4, qbBytes, info.shape, qbInfo?.shape)
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromFloatArray<T, Float>(targetShape, dtype, floats) as Tensor<T, Float>
                    }
                    else -> {
                        println("WARNING: Skipping tensor '${info.name}' with unsupported dtype ${info.dtype}")
                        continue
                    }
                }

                allTensors[canonicalName] = tensor
                println("  Loaded: ${info.name} (${info.dtype} ${info.shape}) → $canonicalName ${tensor.shape}")
            }
        }

        // Handle tied embeddings
        if (tiedEmbeddings && !allTensors.containsKey(DecoderTensorNames.OUTPUT_WEIGHT)) {
            val embedding = allTensors[DecoderTensorNames.TOKEN_EMBEDDINGS]
            if (embedding != null) {
                allTensors[DecoderTensorNames.OUTPUT_WEIGHT] = embedding
                println("  Tied: ${DecoderTensorNames.OUTPUT_WEIGHT} → ${DecoderTensorNames.TOKEN_EMBEDDINGS}")
            }
        }

        // Split into backbone weights (for DecoderGgufWeights compat) and full map.
        // Exclude non-backbone tensors: acoustic, codec, and audio codebook embeddings.
        val backboneTensors = allTensors.filterKeys { name ->
            !name.startsWith("acoustic.") &&
                !name.startsWith("codec.") &&
                !name.startsWith("audio_codebook_embeddings.")
        }

        val backbone = DecoderGgufWeights<T, Float>(
            metadata = metadata,
            tensors = backboneTensors
        )

        return VoxtralWeights(backbone = backbone, allTensors = allTensors)
    }

    // ========== Dequantization helpers (same as DecoderSafeTensorsLoader) ==========

    private fun normalizeNormShape(shape: List<Long>): Shape {
        return if (shape.size == 2 && shape[0] == 1L) {
            Shape(shape[1].toInt())
        } else {
            Shape(*shape.map { it.toInt() }.toIntArray())
        }
    }

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val bf16Low = bytes[offset].toInt() and 0xFF
            val bf16High = bytes[offset + 1].toInt() and 0xFF
            val bits = (bf16High shl 24) or (bf16Low shl 16)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    private fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    private fun dequantQ4(
        q4Bytes: ByteArray,
        qbBytes: ByteArray?,
        q4Shape: List<Long>,
        qbShape: List<Long>?
    ): FloatArray {
        val rows = q4Shape[0].toInt()
        val cols = q4Shape[1].toInt()
        val totalElements = rows * cols
        val out = FloatArray(totalElements)

        if (qbBytes == null) {
            for (i in 0 until totalElements) {
                val byteIdx = i / 2
                val nibble = if (i % 2 == 0) (q4Bytes[byteIdx].toInt() and 0x0F)
                else (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                out[i] = (if (nibble >= 8) nibble - 16 else nibble).toFloat()
            }
            return out
        }

        val scales = bytesToFloatArray(qbBytes)
        val numGroups = if (qbShape != null && qbShape.size == 2) qbShape[1].toInt() else scales.size / rows
        val groupSize = if (numGroups > 0) cols / numGroups else 32

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val flatIdx = row * cols + col
                val byteIdx = flatIdx / 2
                val nibble = if (flatIdx % 2 == 0) (q4Bytes[byteIdx].toInt() and 0x0F)
                else (q4Bytes[byteIdx].toInt() shr 4) and 0x0F
                val q = if (nibble >= 8) nibble - 16 else nibble
                val groupIdx = col / groupSize
                val scaleIdx = row * numGroups + groupIdx
                val scale = if (scaleIdx < scales.size) scales[scaleIdx] else 1.0f
                out[flatIdx] = q.toFloat() * scale
            }
        }
        return out
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            val bits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }
}
