package sk.ainet.models.lfm2

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

data class Lfm2WeightRecord(
    val tensorName: String,
    val shape: List<Int>,
    val sourceDtype: String,
    val fp32Sha256: String,
    val fp32Bytes: Long,
)

data class LoadedLfm2Model(
    val model: HybridLfm2Model,
    val weights: Map<String, Tensor<FP32, Float>>,
    val records: List<Lfm2WeightRecord>,
    val checkpointSha256: String,
)

/** Load a single dense SafeTensors checkpoint without reference fixtures or checkpoint-specific names. */
fun loadLfm2Model(
    config: Lfm2Config,
    checkpoint: File,
    contextLimit: Int = minOf(2048, config.maximumPositions),
    expectedCheckpointSha256: String? = null,
    ctx: DirectCpuExecutionContext = DirectCpuExecutionContext.create(),
    rmsReduction: RmsReductionPolicy = RmsReductionPolicy.BALANCED_FP32,
): LoadedLfm2Model {
    val dimensions = config.toDimensions(contextLimit)
    val expected = HybridLfm2Model.expectedWeights(dimensions)
    require(checkpoint.isFile && checkpoint.canRead()) { "Checkpoint must be a readable single SafeTensors file" }
    if (expectedCheckpointSha256 != null) require(expectedCheckpointSha256.matches(Regex("[0-9a-fA-F]{64}"))) { "Invalid expected checkpoint SHA-256" }
    val checkpointHash = checkpoint.inputStream().buffered().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().hex()
    }
    require(expectedCheckpointSha256 == null || checkpointHash.equals(expectedCheckpointSha256, ignoreCase = true)) { "Checkpoint SHA-256 mismatch" }
    val sourceDtypes = StreamingSafeTensorsReader.open(JvmRandomAccessSource.open(checkpoint.absolutePath)).use { reader ->
        val tensors = reader.tensors
        require(tensors.size == expected.size && tensors.map { it.name }.toSet() == expected.keys) { "Checkpoint tensor inventory differs from configured LFM2 architecture" }
        val types = linkedMapOf<String, String>()
        var nextOffset = 0L
        for (info in tensors.sortedBy { it.dataOffsetStart }) {
            val shape = expected.getValue(info.name)
            require(info.shape == shape.dimensions.map(Int::toLong)) { "Checkpoint shape mismatch: ${info.name}" }
            require(info.dtype in setOf("BF16", "F16", "F32")) { "Unsupported checkpoint dtype ${info.dtype}: ${info.name}" }
            val itemBytes = if (info.dtype == "F32") 4L else 2L
            val elements = shape.dimensions.fold(1L) { total, size -> Math.multiplyExact(total, size.toLong()) }
            require(elements <= Int.MAX_VALUE / 4) { "Tensor exceeds the current CPU loader allocation limit: ${info.name}" }
            val byteCount = Math.multiplyExact(elements, itemBytes)
            require(info.dataOffsetStart == nextOffset && info.dataOffsetEnd == Math.addExact(nextOffset, byteCount)) { "Invalid or overlapping tensor extent: ${info.name}" }
            require(info.sizeInBytes == byteCount && info.absoluteDataOffset == Math.addExact(reader.dataOffset, nextOffset)) { "Invalid tensor byte count: ${info.name}" }
            nextOffset = info.dataOffsetEnd
            types[info.name] = info.dtype
        }
        require(Math.addExact(reader.dataOffset, nextOffset) == checkpoint.length()) { "Checkpoint payload is truncated or has unclaimed bytes" }
        types
    }

    val weights = linkedMapOf<String, Tensor<FP32, Float>>()
    runBlocking {
        SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(checkpoint.absolutePath) })
            .load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                require(name in expected && weights.put(name, tensor) == null) { "Unexpected or repeated loaded tensor: $name" }
            }
    }
    require(weights.keys == expected.keys) { "Loader returned an incomplete LFM2 checkpoint" }
    val records = expected.map { (name, shape) ->
        val tensor = weights.getValue(name)
        require(tensor.shape == shape) { "Loaded tensor shape mismatch: $name" }
        val values = tensor.data.copyToFloatArray()
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteBuffer.allocate(256 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) {
            require(value.isFinite()) { "Non-finite checkpoint weight: $name" }
            if (buffer.remaining() < 4) { digest.update(buffer.array(), 0, buffer.position()); buffer.clear() }
            buffer.putFloat(value)
        }
        digest.update(buffer.array(), 0, buffer.position())
        Lfm2WeightRecord(name, shape.dimensions.toList(), sourceDtypes.getValue(name), digest.digest().hex(), values.size * 4L)
    }
    // The model uses the same embed_tokens tensor for lookup and output projection.
    return LoadedLfm2Model(HybridLfm2Model(dimensions, ctx, weights, rmsReduction), weights.toMap(), records, checkpointHash)
}

private fun ByteArray.hex(): String = joinToString("") { "%02x".format(it) }
