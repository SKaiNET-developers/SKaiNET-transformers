package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
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
import kotlin.test.assertSame
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
 * The KEEP_NATIVE arm applies to *every* tensor in the file, so this also covers embedding gather
 * and the RMSNorm weight multiply reading through packed storage.
 *
 * Since engine #888 the matmul weights are relaid input-major at load, so the per-forward
 * transpose is a zero-copy view and the narrow kernel genuinely runs — see
 * [`matmul weights survive the per-forward transpose still packed`] for that property, and
 * [`gathered and rank-1 tensors stay row-major`] for the tensors deliberately left alone.
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
     * Absolute logit tolerance. The two paths now run genuinely different kernels — the narrow
     * SGEMM against the FP32 one — so they no longer agree bit-for-bit; the measured divergence
     * is ~7e-9, a couple of ULPs of accumulation-order difference. This sits three orders of
     * magnitude above that for headroom on other vector widths, and three below the ~1e-2 a codec
     * mix-up produces. The mix-up guard asserts that margin rather than assuming it.
     *
     * Before the input-major relayout landed, these paths *were* bit-identical, because the
     * per-forward transpose widened the weight and both ended up in the same FP32 SGEMM. A return
     * to exact equality here would mean the narrow kernel has stopped running.
     */
    private val TOLERANCE = 1e-5f

    private val metadata = GgufDecoderMetadata(
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

    private val ctx = DirectCpuExecutionContext()

    /** Load the synthesized model under [policy]. */
    private fun loadWeights(file: File, policy: DTypePolicy): DecoderGgufWeights<FP32, Float> {
        val loader = DecoderSafeTensorsLoader(
            ctx = ctx,
            dtype = FP32::class,
            metadata = metadata,
            tiedEmbeddings = false,
            dtypePolicy = policy,
        )
        val provider: () -> RandomAccessSource = { JvmRandomAccessSource.open(file) }
        return loader.loadToMap(provider)
    }

    /** Load under [policy] and forward [tokens] in order, returning the logits of the last step. */
    private fun forwardLogits(
        file: File,
        policy: DTypePolicy,
        tokens: IntArray,
        onWeights: (DecoderGgufWeights<FP32, Float>) -> Unit = {},
    ): FloatArray {
        val weights = loadWeights(file, policy)
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
    fun `matmul weights survive the per-forward transpose still packed`() {
        // The property the whole feature rests on. Weights arrive [out, in]; `Linear.onForward`
        // transposes before every matmul. Before engine #888 that transpose widened the tensor
        // elementwise, so the narrow kernel was unreachable and KEEP_NATIVE was slower than not
        // using it. The loader now relays matmul weights input-major, which makes the transpose a
        // zero-copy view — this asserts the weight is still narrow on the far side of it.
        val file = writeModel(
            buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeFp16(v)) },
            declaredDtype = "F16",
            encode = ::encodeFp16,
        )
        val weights = loadWeights(file, DTypePolicy.Require(FP16))

        val ffnGate = weights.tensors[DecoderTensorNames.ffnGate(0)]
            ?: error("missing ffn_gate")
        assertTrue(
            ffnGate.data is NarrowFloatInputMajorTensorData,
            "a matmul weight must be relaid input-major, got ${ffnGate.data::class.simpleName}",
        )

        val transposed = ffnGate.t()
        assertTrue(
            transposed.data is NarrowFloatTensorData,
            "transpose widened the weight — the narrow kernel is unreachable again",
        )
        assertEquals(
            dim, transposed.shape[0],
            "after t() the weight must be [in, out], the orientation chooseQuantizedMatmul wants",
        )
        assertSame(
            (ffnGate.data as NarrowFloatTensorData).packedData,
            (transposed.data as NarrowFloatTensorData).packedData,
            "the transpose must not copy — a copy per forward is the cost being removed",
        )
    }

    @Test
    fun `gathered and rank-1 tensors stay row-major`() {
        // The counterpart to the test above, and the one that would catch over-applying the
        // relayout. The token embedding is gathered by row, so input-major storage would stride
        // exactly the reads it serves; norms are rank-1 and never transposed at all.
        val file = writeModel(
            buildHfWeights().map { (n, s, v) -> Triple(n, s, quantizeFp16(v)) },
            declaredDtype = "F16",
            encode = ::encodeFp16,
        )
        val weights = loadWeights(file, DTypePolicy.Require(FP16))

        val embedding = weights.tensors[DecoderTensorNames.TOKEN_EMBEDDINGS]
            ?: error("missing token_embd")
        assertTrue(
            embedding.data is NarrowFloatTensorData,
            "the embedding should still be packed — only its layout differs",
        )
        assertTrue(
            embedding.data !is NarrowFloatInputMajorTensorData,
            "the gathered embedding must stay row-major",
        )

        val norm = weights.tensors[DecoderTensorNames.attnNorm(0)] ?: error("missing attn_norm")
        assertEquals(1, norm.shape.rank, "precondition: norms are rank-1")
        assertTrue(
            norm.data !is NarrowFloatInputMajorTensorData,
            "a rank-1 norm must never be relaid — the input-major type rejects rank != 2",
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
