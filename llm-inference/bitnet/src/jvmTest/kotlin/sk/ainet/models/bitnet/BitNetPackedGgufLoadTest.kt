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

    /** [tied] = a 2B4T-style file: no `output.weight`, the lm_head is tied to `token_embd`. */
    private fun buildFile(tied: Boolean = false): File {
        val rng = Random(42)
        val embd = FloatArray(vocabSize * dim) { (rng.nextFloat() - 0.5f) * 0.6f }
        return writeGguf(
            listOfNotNull(
                f32Tensor("token_embd.weight", embd, dim.toLong(), vocabSize.toLong()),
                f32Tensor("output_norm.weight", norm(dim, 20), dim.toLong()),
                if (tied) null else i2sTensor("output.weight", vocabSize, dim, scale = 0.08f, seed = 11),
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

    private fun loadPacked(file: File, planesLmHead: Boolean = true): Module<FP32, Float> = runBlocking {
        BitNetPackedGgufLoader.load(
            ctx, { JvmRandomAccessSource.open(file.path) }, planesLmHead = planesLmHead,
        )
    }

    private fun param(model: Module<FP32, Float>, path: String, name: String): Tensor<FP32, Float> {
        var m: Module<FP32, Float> = model
        for (seg in path.split("/")) m = m.modules.first { it.name == seg }
        @Suppress("UNCHECKED_CAST")
        return (m as sk.ainet.lang.nn.topology.ModuleParameters<FP32, Float>)
            .params.first { it.name.endsWith(name) }.value
    }

    private fun loadWidened(file: File): Module<FP32, Float> = runBlocking {
        val metadata = sk.ainet.io.gguf.StreamingGGUFReader.open(JvmRandomAccessSource.open(file.path)).use { reader ->
            sk.ainet.models.llama.decoderMetadataFromGguf(reader.fields, reader.tensors)
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

    private fun assertLogitsClose(
        a: List<FloatArray>,
        b: List<FloatArray>,
        what: String,
        relTol: Float = 2e-3f,
    ) {
        for (step in a.indices) {
            for (i in 0 until vocabSize) {
                val x = a[step][i]; val y = b[step][i]
                assertTrue(x.isFinite() && y.isFinite(), "$what step $step [$i]: $x vs $y")
                assertTrue(
                    abs(x - y) <= relTol * maxOf(1f, abs(y)),
                    "$what step $step logit[$i]: $x vs $y",
                )
            }
        }
    }

    private fun argmax(logits: FloatArray): Int {
        var best = 0
        for (i in logits.indices) if (logits[i] > logits[best]) best = i
        return best
    }

    // ---- tests ---------------------------------------------------------------------------

    @Test
    fun packedLoadKeepsTernaryTensorsPackedAndTheLmHeadAsPlanes() {
        val file = buildFile()
        try {
            val model = loadPacked(file)
            assertIs<BitNetB158TensorData>(
                param(model, "blk.0/attn", "q_proj.weight").data,
                "ternary projection must stay packed (0.25 B/weight)",
            )
            assertIs<BitNetPlanesTensorData>(
                param(model, "output", "weight").data,
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

    // ---- tied embeddings (transformers#357) ----------------------------------------------

    @Test
    fun tiedFileServesTheLmHeadAsPlanesFromTheEmbedding() {
        val file = buildFile(tied = true)
        try {
            val model = loadPacked(file)
            assertIs<BitNetPlanesTensorData>(
                param(model, "output", "weight").data,
                "the tied lm_head must be materialized as BITNET_PLANES from token_embd (#357)",
            )
            assertTrue(
                param(model, "token_embd", "weight").data !is BitNetPlanesTensorData,
                "the embedding itself must keep its as-stored form for gathers",
            )
            // Opting out restores the exact dense tied head (the pre-#357 behavior).
            val dense = loadPacked(file, planesLmHead = false)
            assertTrue(
                param(dense, "output", "weight").data !is BitNetPlanesTensorData,
                "planesLmHead = false must keep the dense tied head",
            )
        } finally {
            file.delete()
        }
    }

    @Test
    fun tiedPackedLogitsMatchTheWidenedBaselineWithinThePlanesBound() {
        val file = buildFile(tied = true)
        try {
            val tokens = intArrayOf(1, 7, 3, 12)
            // Widened baseline: dense FP32 everywhere, head served by the exact tied fallback.
            val baseline = logitsOf(loadWidened(file), tokens)

            // The planes tied head is a *bounded* requantization of the (non-ternary) embedding
            // rows — per-weight error ≤ ~0.5·3⁻⁷ of the row scale — not bit-exact like the
            // exactly-ternary output.weight case: assert closeness at that scale plus greedy
            // top-1 stability.
            val packed = logitsOf(loadPacked(file), tokens)
            assertLogitsClose(packed, baseline, "tied packed(reference) vs widened", relTol = 1e-2f)
            for (step in tokens.indices) {
                assertTrue(
                    argmax(packed[step]) == argmax(baseline[step]),
                    "greedy top-1 must survive the planes encoding at step $step",
                )
            }

            // The native lm_head kernel serves the same stored format — tight parity vs reference.
            NativeTernaryF32GemvKernel.install()
            NativeTernaryLmheadKernel.install()
            val native = logitsOf(loadPacked(file), tokens)
            assertLogitsClose(native, packed, "tied packed(native) vs packed(reference)")
        } finally {
            file.delete()
        }
    }
}
