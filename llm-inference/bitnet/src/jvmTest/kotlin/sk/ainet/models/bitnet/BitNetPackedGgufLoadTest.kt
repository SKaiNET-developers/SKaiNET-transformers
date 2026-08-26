package sk.ainet.models.bitnet

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.NativeTernaryF32GemvKernel
import sk.ainet.exec.kernel.NativeTernaryLmheadKernel
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights

/**
 * transformers#337 end to end: a synthetic BitNet **I2_S** GGUF (BitNet.cpp GROUP_128 flavor,
 * per-tensor trailer scales, llama.cpp KV metadata) loads **packed** — ternary projections as
 * `BitNetB158TensorData`, the lm_head as `BitNetPlanesTensorData` — and produces the same logits
 * as the FP32-widened load of the same file, with and without the vendored NEON kernel packs
 * installed. Correctness never depends on the packs; speed does.
 */
@OptIn(ExperimentalMemoryApi::class)
class BitNetPackedGgufLoadTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 16
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val ffDim = 32
    private val vocabSize = 32
    private val seqLen = 16

    @BeforeTest fun setUp() = KernelDispatch.clearForTesting()
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    // ---- synthetic GGUF ------------------------------------------------------------------

    private class Kv(val key: String, val type: Int, val write: (ByteBuffer) -> Unit)
    private class T(val name: String, val type: Int, val dims: LongArray, val data: ByteArray)

    /** BitNet.cpp `quantize_i2_s` packing (QK=128) + the 32-byte trailer holding the scale ×8. */
    private fun i2sTensor(name: String, out: Int, inDim: Int, scale: Float, seed: Int): T {
        val rng = Random(seed)
        val elements = out * inDim
        check(elements % 128 == 0)
        val payload = ByteArray(elements / 4)
        for (j in 0 until elements) {
            val jb = j % 128
            val idx = (j / 128) * 32 + jb % 32
            payload[idx] = (payload[idx].toInt() or (rng.nextInt(3) shl (6 - 2 * (jb / 32)))).toByte()
        }
        val trailer = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        repeat(8) { trailer.putFloat(scale) }
        // ne order is fastest-varying first: a row-major [out, in] weight declares [in, out].
        return T(name, 36, longArrayOf(inDim.toLong(), out.toLong()), payload + trailer.array())
    }

    private fun f32Tensor(name: String, values: FloatArray, vararg neDims: Long): T {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buf.putFloat(it) }
        return T(name, 0, neDims, buf.array())
    }

    private fun norm(n: Int, seed: Int): FloatArray {
        val rng = Random(seed)
        return FloatArray(n) { 0.8f + rng.nextFloat() * 0.4f }
    }

    private fun writeGguf(tensors: List<T>): File {
        val arch = "bitnet-b1.58"
        fun u32(v: Int): (ByteBuffer) -> Unit = { it.putInt(4); it.putInt(v) }
        fun f32(v: Float): (ByteBuffer) -> Unit = { it.putInt(6); it.putFloat(v) }
        fun str(v: String): (ByteBuffer) -> Unit = { b ->
            b.putInt(8); val e = v.encodeToByteArray(); b.putLong(e.size.toLong()); b.put(e)
        }
        val kvs = listOf(
            Kv("general.architecture", 8, str(arch)),
            Kv("$arch.embedding_length", 4, u32(dim)),
            Kv("$arch.context_length", 4, u32(seqLen)),
            Kv("$arch.block_count", 4, u32(1)),
            Kv("$arch.attention.head_count", 4, u32(nHeads)),
            Kv("$arch.attention.head_count_kv", 4, u32(kvHeads)),
            Kv("$arch.feed_forward_length", 4, u32(ffDim)),
            Kv("$arch.rope.dimension_count", 4, u32(headDim)),
            Kv("$arch.vocab_size", 4, u32(vocabSize)),
            Kv("$arch.attention.layer_norm_rms_epsilon", 6, f32(1e-5f)),
        )

        val head = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747)              // "GGUF"
        head.putInt(3)
        head.putLong(tensors.size.toLong())
        head.putLong(kvs.size.toLong())
        for (kv in kvs) {
            val k = kv.key.encodeToByteArray()
            head.putLong(k.size.toLong()); head.put(k)
            kv.write(head)
        }
        fun padded(size: Int): Int = ((size + 31) / 32) * 32
        var dataOffset = 0L
        for (t in tensors) {
            val n = t.name.encodeToByteArray()
            head.putLong(n.size.toLong()); head.put(n)
            head.putInt(t.dims.size)
            for (d in t.dims) head.putLong(d)
            head.putInt(t.type)
            head.putLong(dataOffset)
            dataOffset += padded(t.data.size).toLong()
        }
        val padding = (32 - (head.position() % 32)) % 32
        repeat(padding) { head.put(0) }

        val file = File.createTempFile("bitnet_i2s_", ".gguf")
        file.deleteOnExit()
        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            for (t in tensors) {
                raf.write(t.data)
                repeat(padded(t.data.size) - t.data.size) { raf.write(0) }
            }
        }
        return file
    }

    private fun buildFile(): File {
        val rng = Random(42)
        val embd = FloatArray(vocabSize * dim) { (rng.nextFloat() - 0.5f) * 0.6f }
        return writeGguf(
            listOf(
                f32Tensor("token_embd.weight", embd, dim.toLong(), vocabSize.toLong()),
                f32Tensor("output_norm.weight", norm(dim, 20), dim.toLong()),
                i2sTensor("output.weight", vocabSize, dim, scale = 0.08f, seed = 11),
                f32Tensor("blk.0.attn_norm.weight", norm(dim, 21), dim.toLong()),
                i2sTensor("blk.0.attn_q.weight", dim, dim, scale = 0.11f, seed = 1),
                i2sTensor("blk.0.attn_k.weight", dim, dim, scale = 0.12f, seed = 2),
                i2sTensor("blk.0.attn_v.weight", dim, dim, scale = 0.13f, seed = 3),
                f32Tensor("blk.0.attn_sub_norm.weight", norm(dim, 22), dim.toLong()),
                i2sTensor("blk.0.attn_output.weight", dim, dim, scale = 0.14f, seed = 4),
                f32Tensor("blk.0.ffn_norm.weight", norm(dim, 23), dim.toLong()),
                i2sTensor("blk.0.ffn_gate.weight", ffDim, dim, scale = 0.15f, seed = 5),
                i2sTensor("blk.0.ffn_up.weight", ffDim, dim, scale = 0.16f, seed = 6),
                f32Tensor("blk.0.ffn_sub_norm.weight", norm(ffDim, 24), ffDim.toLong()),
                i2sTensor("blk.0.ffn_down.weight", dim, ffDim, scale = 0.17f, seed = 7),
            ),
        )
    }

    // ---- load paths ----------------------------------------------------------------------

    private fun loadPacked(file: File): Module<FP32, Float> = runBlocking {
        BitNetPackedGgufLoader.load(ctx, { JvmRandomAccessSource.open(file.path) })
    }

    private fun loadWidened(file: File): Module<FP32, Float> = runBlocking {
        val metadata = sk.ainet.io.gguf.StreamingGGUFReader.open(JvmRandomAccessSource.open(file.path)).use {
            BitNetPackedGgufLoader.metadataFrom(it.fields, it.tensors)
        }
        val tensors = LinkedHashMap<String, Tensor<FP32, Float>>()
        StreamingGgufParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file.path) },
            weightForm = WeightForm(
                encoding = EncodingRequest.DequantizeTo(FP32),
                shape = WeightShapeOrientation.OUT_IN,
            ),
            i2sLayout = I2sGgufLayout.GROUP_128,
        ).load<FP32, Float>(ctx, FP32::class) { name, t -> tensors[name] = t }
        BitNetNetworkLoader.fromWeights(DecoderGgufWeights(metadata, tensors))
    }

    private fun logitsOf(model: Module<FP32, Float>, tokens: IntArray): List<FloatArray> {
        val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        return tokens.map { token ->
            val t = runtime.forward(token)
            FloatArray(vocabSize) { i -> t.data.get(0, i) }
        }
    }

    private fun assertLogitsClose(a: List<FloatArray>, b: List<FloatArray>, what: String) {
        for (step in a.indices) {
            for (i in 0 until vocabSize) {
                val x = a[step][i]; val y = b[step][i]
                assertTrue(x.isFinite() && y.isFinite(), "$what step $step [$i]: $x vs $y")
                assertTrue(
                    abs(x - y) <= 2e-3f * maxOf(1f, abs(y)),
                    "$what step $step logit[$i]: $x vs $y",
                )
            }
        }
    }

    // ---- tests ---------------------------------------------------------------------------

    @Test
    fun packedLoadKeepsTernaryTensorsPackedAndTheLmHeadAsPlanes() {
        val file = buildFile()
        try {
            val model = loadPacked(file)
            fun param(path: String, name: String): Tensor<FP32, Float> {
                var m: Module<FP32, Float> = model
                for (seg in path.split("/")) m = m.modules.first { it.name == seg }
                @Suppress("UNCHECKED_CAST")
                return (m as sk.ainet.lang.nn.topology.ModuleParameters<FP32, Float>)
                    .params.first { it.name.endsWith(name) }.value
            }
            assertIs<BitNetB158TensorData>(
                param("blk.0/attn", "q_proj.weight").data,
                "ternary projection must stay packed (0.25 B/weight)",
            )
            assertIs<BitNetPlanesTensorData>(
                param("output", "weight").data,
                "the lm_head must arrive as BITNET_PLANES",
            )
        } finally {
            file.delete()
        }
    }

    @Test
    fun packedLogitsMatchTheWidenedBaselineWithAndWithoutTheNativePacks() {
        val file = buildFile()
        try {
            val tokens = intArrayOf(1, 7, 3, 12)
            val baseline = logitsOf(loadWidened(file), tokens)

            // reference dispatch (no native packs registered)
            val packedReference = logitsOf(loadPacked(file), tokens)
            assertLogitsClose(packedReference, baseline, "packed(reference) vs widened")

            // with the vendored NeoGPU kernels serving the exact keys
            NativeTernaryF32GemvKernel.install()
            NativeTernaryLmheadKernel.install()
            val packedNative = logitsOf(loadPacked(file), tokens)
            assertLogitsClose(packedNative, baseline, "packed(native) vs widened")
        } finally {
            file.delete()
        }
    }
}
