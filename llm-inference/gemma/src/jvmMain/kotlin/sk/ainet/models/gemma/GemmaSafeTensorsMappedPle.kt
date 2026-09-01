package sk.ainet.models.gemma

import java.lang.foreign.Arena
import java.nio.channels.FileChannel
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import sk.ainet.context.ExecutionContext
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32

/**
 * JVM-only post-processor for [GemmaWeights] that injects the
 * `embed_tokens_per_layer` table via a memory-mapped read when the source
 * tensor is too large for the eager `ByteArray`-based loader path.
 *
 * `GemmaSafeTensorsLoader` skips this tensor when its raw size
 * exceeds the JVM `ByteArray` limit (~2 GB; the BF16 table on Gemma 4 E2B
 * is 4.7 GB). Calling this after the load mmap's the tensor's byte
 * region, wraps it in a [SafeTensorsPerLayerTokenEmbedTensorData] for
 * lazy per-row dequantisation, and inserts it under the GGUF-style key
 * the rest of the runtime expects, which flips PLE auto-detection on in
 * `GemmaNetworkLoader`.
 *
 * Lifetime: the returned tensor's storage is owned by [arena]; close the
 * arena (typically when the chat model is closed) to release the mmap'd
 * region. Don't close it earlier — `PerLayerEmbedding.compute` reads
 * rows on every decode step.
 */
public object GemmaSafeTensorsMappedPle {

    private const val HF_EMBED_TOKENS_PER_LAYER = "model.language_model.embed_tokens_per_layer.weight"

    /**
     * If the embed-tokens-per-layer tensor is absent from [weights]
     * (typically because the eager loader skipped it for size), mmap it
     * from the SafeTensors checkpoint at [indexPath] and return an
     * updated [GemmaWeights] with the tensor injected.
     *
     * Returns [weights] unchanged when:
     *  - the tensor is already loaded (small fixture, F32 source, etc.), or
     *  - the checkpoint doesn't contain the tensor at all (unimodal Gemma 4
     *    variants without PLE).
     *
     * @param indexPath path to `model.safetensors.index.json` (or a
     *   directory / single shard the original loader was given).
     * @param ctx execution context used to wrap the new [TensorData].
     * @param arena arena that owns the mmap'd region's lifetime.
     */
    public fun injectIfMissing(
        weights: GemmaWeights<FP32, Float>,
        indexPath: String,
        ctx: ExecutionContext,
        arena: Arena,
    ): GemmaWeights<FP32, Float> {
        if (weights.tensors.containsKey(GemmaTensorNames.PER_LAYER_TOKEN_EMBD)) return weights

        val reader = runCatching {
            kotlinx.coroutines.runBlocking {
                StreamingShardedSafeTensorsReader.openFromIndex(indexPath)
            }
        }.getOrNull() ?: return weights

        return reader.use { r ->
            // Resolve the tensor's metadata + file location via the upstream
            // sharded `loadTensorStorageMapped` (added in skainet-io-safetensors
            // 0.22.1). The returned `TensorStorage`'s `BufferHandle.FileBacked`
            // tells us the path / offset / size — we then run the JVM-specific
            // `FileChannel.map(..., Arena)` to materialise a `MemorySegment` of
            // the right byte range. (`BufferAccessor` upstream returns
            // ByteArrays, which are 2 GB-capped; we need a long-indexed
            // segment for the 4.7 GB PLE table on Gemma 4 E2B.)
            val info = r.tensors.firstOrNull { it.name == HF_EMBED_TOKENS_PER_LAYER }
                ?: return@use weights
            val storage = r.loadTensorStorageMapped(info)
            val handle = storage.buffer
            require(handle is BufferHandle.FileBacked) {
                "Expected file-backed handle from loadTensorStorageMapped, got ${handle::class}"
            }

            val segment = FileChannel.open(Paths.get(handle.path), StandardOpenOption.READ).use { fc ->
                fc.map(
                    FileChannel.MapMode.READ_ONLY,
                    handle.fileOffset,
                    handle.sizeInBytes,
                    arena,
                )
            }

            val logicalShape = if (storage.shape.rank == 2) storage.shape
                else flattenTrailingDims(storage.shape)
            val data = SafeTensorsPerLayerTokenEmbedTensorData(
                logicalShape = logicalShape,
                segment = segment,
                sourceDtype = info.dtype,
            )
            val tensor = ctx.fromData(data, FP32::class)
            weights.copy(
                tensors = weights.tensors + (GemmaTensorNames.PER_LAYER_TOKEN_EMBD to tensor),
            )
        }
    }

    /**
     * Collapse a rank-3+ shape into a 2-D `[rows, cols]` shape by
     * multiplying out every dimension after the first. Used in case a
     * future Gemma-4 checkpoint stores the table as
     * `[vocab, num_layers, per_layer_dim]` instead of the flat
     * `[vocab, per_layer_total]` we observe today on E2B.
     */
    private fun flattenTrailingDims(shape: Shape): Shape {
        require(shape.rank >= 2) { "Cannot flatten rank-${shape.rank} shape: $shape" }
        val rows = shape[0]
        var cols = 1
        for (i in 1 until shape.rank) cols *= shape[i]
        return Shape(rows, cols)
    }
}
