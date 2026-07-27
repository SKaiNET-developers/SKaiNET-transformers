package sk.ainet.models.llama

import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end parity for narrow-float KEEP_NATIVE: a complete tiny LLaMA is loaded twice from the
 * *same* SafeTensors file — once widened to FP32 at load, once kept packed — and both are run
 * through [OptimizedLLMRuntime]. The logits must agree.
 *
 * Why this test exists, when [DecoderSafeTensorsLoaderNarrowFloatTest] already pins the loader:
 * that test proves the bytes survive and decode correctly, which is only half the feature. The
 * other half is what happens once those tensors reach the model — every op that touches a packed
 * weight has to decode it correctly, and since binary16 and bfloat16 are both 2 bytes per element,
 * decoding one as the other does not throw. It returns finite, plausible, wrong logits. Only a
 * numeric comparison against a known-good reference catches that.
 *
 * Both sides of the comparison hold mathematically identical weights: the file is written from
 * values already round-tripped through the codec, so the widened path and the packed path decode
 * the same numbers. [TOLERANCE] covers float accumulation-order differences.
 *
 * **What this does NOT prove.** It does not show that the narrow-float matmul kernel runs. On this
 * chain it does not — see [`narrow weights are materialized to FP32 before matmul`], which pins
 * why. The logits below are bit-identical rather than merely close, because both runs ultimately
 * execute the same FP32 SGEMM; only the moment of decoding differs. [TOLERANCE] stays non-zero so
 * the test keeps passing if a real narrow kernel is ever wired in and shifts accumulation order.
 *
 * The KEEP_NATIVE arm applies to *every* tensor in the file, so this does cover embedding gather
 * and the RMSNorm weight multiply reading through packed storage.
 *
 * [`the parity check would catch a codec mix-up`] is the guard that keeps the tolerance honest.
 *
 * Files are synthesized in-test; no model downloads are involved.
 */
class DecoderNarrowFloatForwardParityTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val seqLen = 32

    /**
     * Absolute logit tolerance. Today both paths converge on the same FP32 SGEMM and agree to the
     * bit, so this is slack for a future narrow kernel whose accumulation order would differ.
     * Sized to swallow that and nothing more — the codec mix-up guard measures the margin actually
     * available and fails if this ever grows large enough to hide a real defect.
     */
    private val TOLERANCE = 1e-4f

    private val metadata = LlamaModelMetadata(
        architecture = "llama",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
    )

    // ---------------------------------------------------------------- codecs

    private fun encodeFp16(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun encodeBf16(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** Round-trip through binary16 so the on-disk file loses nothing further. */
    private fun quantizeFp16(values: FloatArray): FloatArray =
        FloatArray(values.size) { Fp16Codec.decode(Fp16Codec.encode(values[it])) }

    /** Round-trip through bfloat16 — truncate the low 16 mantissa bits. */
    private fun quantizeBf16(values: FloatArray): FloatArray =
        FloatArray(values.size) { Float.fromBits(values[it].toRawBits() and 0xFFFF0000.toInt()) }

    // ------------------------------------------------------------ model data

    /** Deterministic small weights; the same generator [LlamaDslPipelineTest] uses. */
    private fun randn(size: Int, seed: Int): FloatArray {
        val rng = kotlin.random.Random(seed)
        return FloatArray(size) { (rng.nextFloat() - 0.5f) * 0.1f }
    }

    private fun ones(size: Int): FloatArray = FloatArray(size) { 1.0f }

    /** HF-named weights with their SafeTensors shapes, in FP32 before any narrowing. */
    private fun buildHfWeights(): List<Triple<String, List<Int>, FloatArray>> = listOf(
        Triple("model.embed_tokens.weight", listOf(vocabSize, dim), randn(vocabSize * dim, 10)),
        Triple("model.norm.weight", listOf(dim), ones(dim)),
        Triple("lm_head.weight", listOf(vocabSize, dim), randn(vocabSize * dim, 11)),
        Triple("model.layers.0.input_layernorm.weight", listOf(dim), ones(dim)),
        Triple("model.layers.0.self_attn.q_proj.weight", listOf(dim, dim), randn(dim * dim, 1)),
        Triple("model.layers.0.self_attn.k_proj.weight", listOf(dim, dim), randn(dim * dim, 2)),
        Triple("model.layers.0.self_attn.v_proj.weight", listOf(dim, dim), randn(dim * dim, 3)),
        Triple("model.layers.0.self_attn.o_proj.weight", listOf(dim, dim), randn(dim * dim, 4)),
        Triple("model.layers.0.post_attention_layernorm.weight", listOf(dim), ones(dim)),
        Triple("model.layers.0.mlp.gate_proj.weight", listOf(ffDim, dim), randn(ffDim * dim, 5)),
        Triple("model.layers.0.mlp.down_proj.weight", listOf(dim, ffDim), randn(dim * ffDim, 6)),
        Triple("model.layers.0.mlp.up_proj.weight", listOf(ffDim, dim), randn(ffDim * dim, 7)),
    )

    /**
     * Write every weight into one SafeTensors file: 8-byte LE header length, JSON header, data.
     *
     * [declaredDtype] is what the header claims; [encode] is what actually produces the bytes.
     * They are separate parameters on purpose — the codec mix-up guard needs to declare one
     * format while writing the other's bit layout.
     */
    private fun writeModel(
        weights: List<Triple<String, List<Int>, FloatArray>>,
        declaredDtype: String,
        encode: (FloatArray) -> ByteArray,
    ): File {
        val header = StringBuilder("{")
        var offset = 0L
        val payloads = weights.map { (name, shape, values) ->
            val bytes = encode(values)
            if (offset > 0) header.append(",")
            header.append(
                "\"$name\": {\"dtype\": \"$declaredDtype\", \"shape\": [${shape.joinToString(", ")}], " +
                    "\"data_offsets\": [$offset, ${offset + bytes.size}]}",
            )
            offset += bytes.size
            bytes
        }
        header.append("}")
        val headerBytes = header.toString().toByteArray(Charsets.UTF_8)

        val file = Files.createTempFile("decoder_forward_parity", ".safetensors").toFile()
        file.deleteOnExit()
        file.outputStream().use { out ->
            out.write(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(headerBytes.size.toLong()).array(),
            )
            out.write(headerBytes)
            payloads.forEach { out.write(it) }
        }
        return file
    }

    // -------------------------------------------------------------- the runs

    /** Load under [policy] and forward [tokens] in order, returning the logits of the last step. */
    private fun forwardLogits(
        file: File,
        policy: DTypePolicy,
        tokens: IntArray,
        onWeights: (DecoderGgufWeights<FP32, Float>) -> Unit = {},
    ): FloatArray {
        val ctx = DirectCpuExecutionContext()
        val loader = DecoderSafeTensorsLoader(
            ctx = ctx,
            dtype = FP32::class,
            metadata = metadata,
            tiedEmbeddings = false,
            dtypePolicy = policy,
        )
        val provider: () -> RandomAccessSource = { JvmRandomAccessSource.open(file) }
        val weights = loader.loadToMap(provider)
        onWeights(weights)

        val runtime = OptimizedLLMRuntime(
            model = LlamaNetworkLoader.fromWeights(weights),
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
        )
        var logits = FloatArray(0)
        for (token in tokens) {
            logits = runtime.forward(token).data.copyToFloatArray()
        }
        return logits
    }

    /** Multi-token so the comparison runs through the KV cache, not just a single step. */
    private val tokens = intArrayOf(1, 5, 3, 9)

    private fun assertParity(reference: FloatArray, native: FloatArray, label: String) {
        assertEquals(vocabSize, reference.size, "reference logits should be one row of vocab")
        assertEquals(reference.size, native.size, "$label: logit count differs")
        for (i in reference.indices) {
            assertTrue(native[i].isFinite(), "$label: logit[$i] is not finite (${native[i]})")
            assertTrue(
                abs(reference[i] - native[i]) <= TOLERANCE,
                "$label: logit[$i] diverged — widened=${reference[i]} native=${native[i]} " +
                    "delta=${abs(reference[i] - native[i])} tolerance=$TOLERANCE",
            )
        }
    }

    /** Fails the run if KEEP_NATIVE silently didn't engage — otherwise this is FP32 vs FP32. */
    private fun assertActuallyPacked(weights: DecoderGgufWeights<FP32, Float>, label: String) {
        val packed = weights.tensors.values.count { it.data is NarrowFloatTensorData }
        assertEquals(
            weights.tensors.size, packed,
            "$label: expected every tensor to stay packed, only $packed of " +
                "${weights.tensors.size} did — the parity check would be vacuous",
        )
    }

    @Test
    fun `FP16 KEEP_NATIVE forward matches the widened FP32 forward`() {
        val file = writeModel(
            buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeFp16(v)) },
            declaredDtype = "F16",
            encode = ::encodeFp16,
        )

        val reference = forwardLogits(file, DTypePolicy.Any, tokens)
        val native = forwardLogits(file, DTypePolicy.Require(FP16), tokens) {
            assertActuallyPacked(it, "Require(FP16)")
        }

        assertParity(reference, native, "FP16 KEEP_NATIVE")
    }

    @Test
    fun `BF16 KEEP_NATIVE forward matches the widened FP32 forward`() {
        val file = writeModel(
            buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeBf16(v)) },
            declaredDtype = "BF16",
            encode = ::encodeBf16,
        )

        val reference = forwardLogits(file, DTypePolicy.Any, tokens)
        val native = forwardLogits(file, DTypePolicy.Require(BF16), tokens) {
            assertActuallyPacked(it, "Require(BF16)")
        }

        assertParity(reference, native, "BF16 KEEP_NATIVE")
    }

    @Test
    fun `Prefer reaches the same forward result as Require`() {
        // `Prefer` is the policy users are steered toward, because it degrades instead of
        // throwing on a chain that can't keep the format. It must not be a different code path.
        val file = writeModel(
            buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeFp16(v)) },
            declaredDtype = "F16",
            encode = ::encodeFp16,
        )

        val required = forwardLogits(file, DTypePolicy.Require(FP16), tokens)
        val preferred = forwardLogits(file, DTypePolicy.Prefer(FP16), tokens)

        assertEquals(required.size, preferred.size)
        for (i in required.indices) {
            assertEquals(
                required[i].toRawBits(), preferred[i].toRawBits(),
                "Prefer(FP16) and Require(FP16) must be bit-identical at $i",
            )
        }
    }

    @Test
    fun `narrow weights are materialized to FP32 before matmul`() {
        // Documents the gap between "weights stay packed at rest" and "the narrow kernel runs",
        // and explains why the parity tests above come out bit-identical instead of merely close.
        //
        // SafeTensors stores projections as [out, in]. `LlamaRuntime.linearProject` therefore
        // calls `w.t()` before the matmul, and transpose has no narrow-float implementation — it
        // decodes to a plain FP32 dense buffer. Meanwhile `DefaultCpuOpsJvm.chooseQuantizedMatmul`
        // only engages when the weight is already [in, out]. So on this chain the packed data is
        // widened on *every* forward, and the FP16/BF16 SGEMM kernels are never reached.
        //
        // The saving that survives is at-rest memory. The cost is a per-token decode plus a
        // transpose allocation. Anyone wiring up the kernel for real has to remove the `.t()` —
        // when they do, this test should be updated, and the parity tests will start exercising
        // the kernel path they were written for.
        val ctx = DirectCpuExecutionContext()
        val outFeatures = ffDim
        val inFeatures = dim
        val values = quantizeFp16(randn(outFeatures * inFeatures, seed = 5))

        @Suppress("UNCHECKED_CAST")
        val packed = ctx.fromData(
            Fp16DenseTensorData(Shape(outFeatures, inFeatures), encodeFp16(values))
                as TensorData<FP32, Float>,
            FP32::class,
        )
        assertTrue(packed.data is NarrowFloatTensorData, "precondition: weight starts packed")

        val transposed = packed.t()
        assertTrue(
            transposed.data !is NarrowFloatTensorData,
            "transpose is expected to widen today; if this now stays packed, the narrow matmul " +
                "kernel may finally be reachable — revisit the KDoc on this class",
        )

        // ...and the layout the fast path actually requires is the opposite one.
        assertEquals(
            inFeatures, transposed.shape[0],
            "after t() the weight is [in, out] — the orientation chooseQuantizedMatmul wants, " +
                "but by then it is no longer narrow",
        )
    }

    @Test
    fun `the parity check would catch a codec mix-up`() {
        // Keeps [TOLERANCE] honest. Both formats are 2 bytes per element, so if the dispatch
        // ever picked a kernel by byte width instead of by codec, nothing would throw. This
        // measures what such a mix-up costs: write binary16 bit patterns into a file that
        // *declares* BF16, so the loader hands those bytes to the bfloat16 decode.
        val quantized = buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeFp16(v)) }

        val honest = writeModel(quantized, declaredDtype = "F16", encode = ::encodeFp16)
        val mislabelled = writeModel(quantized, declaredDtype = "BF16", encode = ::encodeFp16)

        val correct = forwardLogits(honest, DTypePolicy.Require(FP16), tokens)
        val wrong = forwardLogits(mislabelled, DTypePolicy.Require(BF16), tokens)

        val maxDelta = correct.indices.maxOf { abs(correct[it] - wrong[it]) }
        assertTrue(
            maxDelta > TOLERANCE * 10,
            "reading F16 bytes as BF16 moved the logits by only $maxDelta — that is inside " +
                "10x the parity tolerance ($TOLERANCE), so the parity tests above prove nothing",
        )
    }
}
