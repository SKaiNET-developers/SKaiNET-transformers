package sk.ainet.lang.nn.quant

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Shared GGUF-block → engine-`*BlockTensorData` packing (#184 hoist 2).
 *
 * Every model that keeps quantized matmul weights packed used to carry its own
 * copy of the same two steps (gemma `GemmaQuantLayout.packGemmaKQuant`, llama
 * `LlamaQuantLayout.packLlamaKQuant`, apertus' JVM converter):
 *
 * 1. (pre-transposed path only) re-layout the checkpoint's row-major block
 *    order to the input-block-major order the `matmulQ*` kernels index
 *    (`(blockIdx * outDim + r)`), and
 * 2. wrap the bytes in the engine block tensor-data type for the format.
 *
 * This object is that logic, once, keyed by the engine's [TensorEncoding]
 * (a `skainet-lang-core` type — so this file needs no GGUF dependency; model
 * modules map their `GGMLQuantizationType` to an encoding and keep only weight
 * *selection* and naming). Supported: the seven formats with first-class CPU
 * matmul kernels — Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 / Q5_0 / Q5_1.
 *
 * ## Byte-order contract (engine ≥ 0.40.1)
 *
 * The engine's `ops.transpose` on packed data performs a **physical**
 * canonical→input-block-major block-grid permutation (SKaiNET#968 fix) and
 * therefore **requires its input bytes in the checkpoint's canonical row-major
 * block order**. Consequently:
 *
 * - [pack] stores the checkpoint bytes **verbatim** (canonical, `[out, in]`)
 *   and relies on the engine transpose inside `linearProject`'s classic path
 *   to produce kernel order — an O(bytes) copy per forward.
 * - [packPreTransposed] performs the relayout **once at load time** and marks
 *   the result [PreTransposedWeight] so `linearProject` never transposes it.
 *   This is the production path.
 *
 * Feeding a [pack] result to a pre-0.40.1 engine (shape-swap-only transpose)
 * re-creates bug SKaiNET#968; this packer requires the 0.40.1 pin.
 */
public object BlockQuantPacking {

    /**
     * Block geometry `(blockElems, bytesPerBlock)` for [encoding], or `null`
     * when the encoding has no packed matmul kernel (callers dequantize).
     */
    public fun blockLayoutFor(encoding: TensorEncoding): Pair<Int, Int>? = when (encoding) {
        TensorEncoding.Q4_K -> TensorEncoding.Q4_K.BLOCK_SIZE to TensorEncoding.Q4_K.BYTES_PER_BLOCK
        TensorEncoding.Q5_K -> TensorEncoding.Q5_K.BLOCK_SIZE to TensorEncoding.Q5_K.BYTES_PER_BLOCK
        TensorEncoding.Q6_K -> TensorEncoding.Q6_K.BLOCK_SIZE to TensorEncoding.Q6_K.BYTES_PER_BLOCK
        TensorEncoding.Q8_0 -> TensorEncoding.Q8_0.BLOCK_SIZE to TensorEncoding.Q8_0.BYTES_PER_BLOCK
        TensorEncoding.Q4_0 -> TensorEncoding.Q4_0.BLOCK_SIZE to TensorEncoding.Q4_0.BYTES_PER_BLOCK
        TensorEncoding.Q5_0 -> TensorEncoding.Q5_0.BLOCK_SIZE to TensorEncoding.Q5_0.BYTES_PER_BLOCK
        TensorEncoding.Q5_1 -> TensorEncoding.Q5_1.BLOCK_SIZE to TensorEncoding.Q5_1.BYTES_PER_BLOCK
        else -> null
    }

    /**
     * Re-layout packed block bytes of a 2-D `[outDim, inDim]` weight from the
     * checkpoint's row-major block order (`(r * blocksPerRow + b) * bytesPerBlock`)
     * to the input-block-major order the `matmulQ*` kernels expect
     * (`(b * outDim + r) * bytesPerBlock`). A block-level 2-D transpose; bytes
     * inside a block are untouched.
     */
    public fun relayoutRowMajorToBlockMajor(
        bytes: ByteArray,
        shape: Shape,
        bytesPerBlock: Int,
        blockSize: Int,
    ): ByteArray {
        require(shape.rank == 2) { "packed matmul weight must be 2D, got rank ${shape.rank}" }
        val outDim = shape[0]
        val inDim = shape[1]
        require(inDim % blockSize == 0) { "packed weight inDim ($inDim) must be a multiple of $blockSize" }
        val blocksPerRow = inDim / blockSize
        val expected = outDim.toLong() * blocksPerRow.toLong() * bytesPerBlock.toLong()
        require(bytes.size.toLong() >= expected) {
            "packed byte buffer ${bytes.size} < expected $expected for [$outDim, $inDim] @ ${bytesPerBlock}B/block"
        }
        val out = ByteArray(bytes.size)
        for (r in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                val srcOff = (r * blocksPerRow + b) * bytesPerBlock
                val dstOff = (b * outDim + r) * bytesPerBlock
                bytes.copyInto(out, dstOff, srcOff, srcOff + bytesPerBlock)
            }
        }
        return out
    }

    /**
     * Validates the same preconditions [relayoutRowMajorToBlockMajor] enforces
     * (rank 2, block-aligned inDim, sufficient bytes) without copying — so
     * [pack]'s no-relayout path fails as loudly as the relayouting path.
     */
    private fun requirePackable(bytes: ByteArray, shape: Shape, bytesPerBlock: Int, blockSize: Int) {
        require(shape.rank == 2) { "packed matmul weight must be 2D, got rank ${shape.rank}" }
        val outDim = shape[0]
        val inDim = shape[1]
        require(inDim % blockSize == 0) { "packed weight inDim ($inDim) must be a multiple of $blockSize" }
        val expected = outDim.toLong() * (inDim / blockSize).toLong() * bytesPerBlock.toLong()
        require(bytes.size.toLong() >= expected) {
            "packed byte buffer ${bytes.size} < expected $expected for [$outDim, $inDim] @ ${bytesPerBlock}B/block"
        }
    }

    /**
     * Pack raw checkpoint `bytes` of logical `[out, in]` [shape] into the
     * heap-packed block tensor data for the format, keeping the checkpoint's
     * **canonical row-major block order verbatim** (no relayout). Returns
     * `null` for encodings without a packed kernel (callers dequantize those
     * to FP32).
     *
     * Canonical order is what the engine's packed `ops.transpose` (≥ 0.40.1)
     * requires: `linearProject`'s classic path transposes the weight every forward,
     * physically permuting the block grid into the kernels' input-block-major
     * order. Prefer [packPreTransposed], which pays that permutation once at
     * load time instead.
     */
    public fun <T : DType> pack(
        bytes: ByteArray,
        encoding: TensorEncoding,
        shape: Shape,
    ): TensorData<T, *>? {
        val (blockElems, bpb) = blockLayoutFor(encoding) ?: return null
        requirePackable(bytes, shape, bpb, blockElems)
        @Suppress("UNCHECKED_CAST")
        return when (encoding) {
            TensorEncoding.Q4_K -> Q4_KBlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q5_K -> Q5_KBlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q6_K -> Q6_KBlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q8_0 -> Q8_0BlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q4_0 -> Q4_0BlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q5_0 -> Q5_0BlockTensorData(shape, bytes) as TensorData<T, *>
            TensorEncoding.Q5_1 -> Q5_1BlockTensorData(shape, bytes) as TensorData<T, *>
            else -> null
        }
    }

    /**
     * Like [pack], but returns the weight *already transposed*: logical shape
     * `[in, out]` over **relaid (input-block-major, kernel-order) bytes**,
     * marked with [PreTransposedWeight] so
     * [sk.ainet.lang.nn.transformer.linearProject] skips `ops.transpose` and
     * feeds the tensor straight to the packed matmul dispatch (#184 hoist 3).
     * This is exactly the tensor data the engine's packed `ops.transpose`
     * (≥ 0.40.1, physical block-grid permutation) would produce from [pack]'s
     * canonical result — computed once at load time instead of every forward.
     *
     * [logicalShape] is still the checkpoint's `[out, in]`; the swap happens
     * here. Returns `null` for encodings without a packed kernel, same as
     * [pack] (callers dequantize those to FP32 and keep the transposing
     * `linearProject` path).
     *
     * The converters' default as of the engine 0.40.0 native-kernel closure
     * train (SKaiNET#951 landed, gate confirmed live): `GemmaQuantLayout
     * .packGemmaKQuant` / `LlamaQuantLayout.packLlamaKQuant` and
     * `GemmaMemSegConverter`'s JVM MemSeg path call this instead of [pack]
     * whenever the kernel-availability gate ([hasPackedMatmulKernel]) has
     * confirmed a packed kernel exists — see #184, #170. [pack] itself is
     * kept reachable (deprecate-don't-delete) as the non-transposed fallback
     * / parity-comparison path, and both `pack*` functions still expose a
     * `preTransposed` parameter to opt back into it.
     */
    public fun <T : DType> packPreTransposed(
        bytes: ByteArray,
        encoding: TensorEncoding,
        logicalShape: Shape,
    ): TensorData<T, *>? {
        val (blockElems, bpb) = blockLayoutFor(encoding) ?: return null
        require(logicalShape.rank == 2) { "packed matmul weight must be 2D, got rank ${logicalShape.rank}" }
        val relaid = relayoutRowMajorToBlockMajor(bytes, logicalShape, bpb, blockElems)
        val transposed = Shape(logicalShape[1], logicalShape[0])
        @Suppress("UNCHECKED_CAST")
        return when (encoding) {
            TensorEncoding.Q4_K -> PreTransposedQ4_K(Q4_KBlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q5_K -> PreTransposedQ5_K(Q5_KBlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q6_K -> PreTransposedQ6_K(Q6_KBlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q8_0 -> PreTransposedQ8_0(Q8_0BlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q4_0 -> PreTransposedQ4_0(Q4_0BlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q5_0 -> PreTransposedQ5_0(Q5_0BlockTensorData(transposed, relaid)) as TensorData<T, *>
            TensorEncoding.Q5_1 -> PreTransposedQ5_1(Q5_1BlockTensorData(transposed, relaid)) as TensorData<T, *>
            else -> null
        }
    }
}
