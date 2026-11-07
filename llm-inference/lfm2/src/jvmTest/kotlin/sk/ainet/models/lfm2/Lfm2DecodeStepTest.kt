package sk.ainet.models.lfm2

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.models.lfm2.export.exportLfm2DecodeStep
import sk.ainet.models.lfm2.export.lfm2DecodeStepInputs
import sk.ainet.models.lfm2.export.lfm2FoldableMatmuls
import sk.ainet.models.lfm2.export.lfm2Head
import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.*

class Lfm2DecodeStepTest {
    private val ctx = DirectCpuExecutionContext.create()

    private fun tensor(shape: Shape, values: FloatArray = FloatArray(shape.dimensions.reduce(Int::times))): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    private fun weightsFor(d: ModelDimensions): Map<String, Tensor<FP32, Float>> =
        HybridLfm2Model.expectedWeights(d).mapValues { (name, shape) ->
            val values = FloatArray(shape.dimensions.reduce(Int::times)) { i ->
                if (name.endsWith("norm.weight")) 0.75f + i % 3 * 0.1f
                else (((i * 7 + name.hashCode().mod(17)) % 19) - 9) * 0.025f
            }
            tensor(shape, values)
        }

    /** Host-side controls of one step at [position], as [Lfm2CachedTensorNetworkTest] builds them. */
    private fun controls(d: ModelDimensions, capacity: Int, position: Int): List<Tensor<FP32, Float>> {
        val cosines = FloatArray(d.headDim)
        val sines = FloatArray(d.headDim)
        for (i in 0 until d.headDim / 2) {
            val angle = position * d.theta.pow(-2.0 * i / d.headDim)
            for (lane in intArrayOf(i, i + d.headDim / 2)) {
                cosines[lane] = cos(angle).toFloat(); sines[lane] = sin(angle).toFloat()
            }
        }
        return listOf(
            tensor(Shape(1, d.headDim), cosines), tensor(Shape(1, d.headDim), sines),
            tensor(Shape(1, capacity + 1), FloatArray(capacity + 1) { if (it < position || it == capacity) 0f else -Float.MAX_VALUE }),
            tensor(Shape(1, capacity, 1), FloatArray(capacity) { if (it == position) 1f else 0f }),
        )
    }

    private fun embeddedRow(embedding: Tensor<FP32, Float>, token: Int): Tensor<FP32, Float> =
        ctx.ops.narrow(embedding, dim = 0, start = token, length = 1)

    private fun closeEnough(expected: FloatArray, actual: FloatArray, tolerance: Float = 2e-6f) {
        assertEquals(expected.size, actual.size)
        expected.indices.forEach { i ->
            assertTrue(actual[i].isFinite() && abs(expected[i] - actual[i]) <= tolerance, "Value $i differs: ${expected[i]} versus ${actual[i]}")
        }
    }

    private val small = ModelDimensions(8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6,
        listOf("conv", "full_attention", "conv", "full_attention"), contextLimit = 6)

    @Test
    fun embeddedStepWithTheTiedHeadMatchesTheFullModelAtEveryPositionForEveryLowering() {
        val weights = weightsFor(small)
        val embedding = weights.getValue("model.embed_tokens.weight")
        val model = HybridLfm2Model(small, ctx, weights)
        for (lowering in ShortConvLowering.entries) for (layout in AttentionLayoutPolicy.entries) {
            val step = Lfm2DecodeStep(small, weights, 6, embedding, lowering, layout)
            var states: Map<String, Tensor<FP32, Float>> = step.stateShapes().mapValues { (_, s) -> tensor(s) }
            val state = HybridModelState()
            for ((position, token) in intArrayOf(0, 3, 6, 11, 4, 9).withIndex()) {
                val expected = model.forward(intArrayOf(token), state)
                val (c, s, mask, write) = controls(small, 6, position)
                val actual = step.forward(embeddedRow(embedding, token), c, s, mask, write, states, ctx)
                closeEnough(expected.logits.data.copyToFloatArray(), actual.logits.data.copyToFloatArray())
                states = actual.states
            }
        }
    }

    @Test
    fun multiplySumConvolutionEqualsTheGroupedConvolutionForOneToken() {
        val d = 8
        val k = 3
        fun w(shape: Shape, seed: Int) = tensor(shape, FloatArray(shape.dimensions.reduce(Int::times)) { ((it * seed + 5) % 13 - 6) * 0.05f })
        val inW = w(Shape(3 * d, d), 3)
        val convW = w(Shape(d, 1, k), 5)
        val outW = w(Shape(d, d), 7)
        val x = w(Shape(1, d), 11)
        val history = w(Shape(d, k), 2)
        val grouped = ShortConvTensorOp(d, k, inW, convW, outW, ShortConvLowering.DEPTHWISE).forward(x, history, ctx)
        val summed = ShortConvTensorOp(d, k, inW, convW, outW, ShortConvLowering.MULTIPLY_SUM).forward(x, history, ctx)
        closeEnough(grouped.output.data.copyToFloatArray(), summed.output.data.copyToFloatArray(), 1e-6f)
        assertContentEquals(grouped.nextHistory.data.copyToFloatArray(), summed.nextHistory.data.copyToFloatArray())
    }

    @Test
    fun exportFoldsEveryBlockQuantProjectionAndTheHead() {
        val d = ModelDimensions(32, 40, 4, 2, 8, 64, 3, 1e-5f, 1e6,
            listOf("conv", "full_attention", "conv", "full_attention"), contextLimit = 6)
        val projection = Regex("""model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|conv\.(in|out)_proj|feed_forward\.w[123])\.weight""")
        @Suppress("UNCHECKED_CAST")
        val weights = weightsFor(d).mapValues { (name, t) ->
            if (!projection.matches(name)) t
            else ctx.fromData(Q8_0Quantizer.quantize(t.data.copyToFloatArray(), t.shape) as TensorData<FP32, Float>, FP32::class)
        }
        val head = lfm2Head(weights.getValue("model.embed_tokens.weight"), TensorEncoding.Q4_0, ctx)

        val step = Lfm2DecodeStep(d, weights, 6, head)
        val exported = exportLfm2DecodeStep(step)
        assertEquals(emptyList(), exported.diagnostics.filter { it.startsWith("BLOCK_QUANT_SKIPPED") })
        assertEquals(lfm2FoldableMatmuls(d.layerTypes), exported.foldedBlockQuantMatmuls)
        assertEquals(25, exported.foldedBlockQuantMatmuls)
        assertFalse(exported.mlir.contains("stablehlo.convolution"), "the decode step's convolution is multiply + sum")
        val states = listOf("history_0", "key_1", "value_1", "history_2", "key_3", "value_3")
        assertEquals((listOf("embedded", "cos", "sin", "mask", "write_mask") + states).toSet(), exported.inputs.map { it.role }.toSet())
        assertEquals((listOf("logits") + states).toSet(), exported.outputs.map { it.role }.toSet())
        val shapes = lfm2DecodeStepInputs(step).toMap()
        exported.inputs.forEach { assertEquals(shapes.getValue(it.role).dimensions.toList(), it.shape, it.role) }

        val depthwise = exportLfm2DecodeStep(Lfm2DecodeStep(d, weights, 6, head, ShortConvLowering.DEPTHWISE))
        assertTrue(depthwise.mlir.contains("stablehlo.convolution"))
        assertEquals(25, depthwise.foldedBlockQuantMatmuls)
    }

    /**
     * Gated on `LFM2_MODEL_DIR` (a local LFM2.5-350M with `config.json` and `model.safetensors`):
     * the Q8_0 projections and Q4_0 head of the NPU step against the FP32 model over a short
     * sequence (top-1 and cosine per step), and the export's fold count.
     */
    @Test
    fun realCheckpointQuantizedStepKeepsTheReferenceTopOne() {
        val dir = System.getenv("LFM2_MODEL_DIR")?.let(::File) ?: return
        val config = Lfm2Config.parse(File(dir, "config.json").readText())
        val capacity = 16
        val reference = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity)
        val quantized = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity,
            ctx = ctx, weightFormFor = lfm2ProjectionsAs(TensorEncoding.Q8_0))
        val dims = quantized.model.dimensions
        val embedding = quantized.weights.getValue("model.embed_tokens.weight")
        val step = Lfm2DecodeStep(dims, quantized.weights, capacity, lfm2Head(embedding, TensorEncoding.Q4_0, ctx))

        val tokens = intArrayOf(1, 6, 6423, 708, 1098, 8448, 4924, 523, 2210, 7)
        val state = HybridModelState()
        var states: Map<String, Tensor<FP32, Float>> = step.stateShapes().mapValues { (_, s) -> tensor(s) }
        var agree = 0
        for ((position, token) in tokens.withIndex()) {
            val expected = reference.model.forward(intArrayOf(token), state).logits.data.copyToFloatArray()
            val (c, s, mask, write) = controls(dims, capacity, position)
            val r = step.forward(embeddedRow(embedding, token), c, s, mask, write, states, ctx)
            states = r.states
            val actual = r.logits.data.copyToFloatArray()
            val dot = expected.indices.sumOf { expected[it].toDouble() * actual[it] }
            val cosine = dot / sqrt(expected.sumOf { it.toDouble() * it } * actual.sumOf { it.toDouble() * it })
            val same = expected.indices.maxBy { expected[it] } == actual.indices.maxBy { actual[it] }
            if (same) agree++
            println("position $position token $token: cosine %.5f, top-1 %s".format(cosine, if (same) "same" else "differs"))
            assertTrue(cosine > 0.95, "logits cosine $cosine at position $position")
        }
        assertTrue(agree >= tokens.size - 1, "top-1 agrees at $agree of ${tokens.size} positions")
        val exported = exportLfm2DecodeStep(step)
        assertEquals(lfm2FoldableMatmuls(dims.layerTypes), exported.foldedBlockQuantMatmuls)
        assertEquals(93, exported.foldedBlockQuantMatmuls)
    }

    /**
     * Gated on `LFM2_MODEL_DIR` and `LFM2_PROMPT_IDS` (comma-separated): greedy decoding of the prompt with the FP32
     * model and with the quantized decode step, printed side by side, for checking a device run of the same step.
     */
    @Test
    fun realCheckpointGreedyReference() {
        val dir = System.getenv("LFM2_MODEL_DIR")?.let(::File) ?: return
        val prompt = System.getenv("LFM2_PROMPT_IDS")?.split(',')?.map { it.trim().toInt() }?.toIntArray() ?: return
        val maxNew = 16
        val capacity = 64
        val config = Lfm2Config.parse(File(dir, "config.json").readText())
        val reference = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity)
        val fp32 = reference.model.generateGreedy(prompt, maxNew, setOf(7)).generatedIds.toList()
        val quantized = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity,
            ctx = ctx, weightFormFor = lfm2ProjectionsAs(TensorEncoding.Q8_0))
        val dims = quantized.model.dimensions
        val embedding = quantized.weights.getValue("model.embed_tokens.weight")
        val step = Lfm2DecodeStep(dims, quantized.weights, capacity, lfm2Head(embedding, TensorEncoding.Q4_0, ctx))
        var states: Map<String, Tensor<FP32, Float>> = step.stateShapes().mapValues { (_, s) -> tensor(s) }
        var position = 0
        fun next(token: Int): FloatArray {
            val (c, s, mask, write) = controls(dims, capacity, position++)
            val r = step.forward(embeddedRow(embedding, token), c, s, mask, write, states, ctx)
            states = r.states
            return r.logits.data.copyToFloatArray()
        }
        var logits = FloatArray(0)
        for (t in prompt) logits = next(t)
        val stepIds = ArrayList<Int>()
        while (stepIds.size < maxNew) {
            val t = logits.indices.maxBy { logits[it] }
            stepIds += t
            if (t == 7 || stepIds.size == maxNew) break
            logits = next(t)
        }
        println("greedy fp32 model: ${fp32.joinToString(",")}")
        println("greedy q8_0 step : ${stepIds.joinToString(",")}")
    }

    /**
     * Gated on `LFM2_MODEL_DIR` and `LFM2_GOLDEN_DIR` (an `exportLfm2DecodeStep` output directory):
     * feeds a prompt on the host, then writes the inputs of the next step in the module's argument
     * order as bf16 files plus `args.txt` (`--input=` lines for an IREE runner), and the expected
     * logits (`logits.fp32.f32.bin` from the FP32 model, `logits.step.f32.bin` from this step) under
     * `golden/`, for checking the compiled module on a device.
     */
    @Test
    fun realCheckpointGoldenStepForADevice() {
        val dir = System.getenv("LFM2_MODEL_DIR")?.let(::File) ?: return
        val exportDir = System.getenv("LFM2_GOLDEN_DIR")?.let(::File) ?: return
        val capacity = Regex("\"capacity\"\\s*:\\s*(\\d+)").find(File(exportDir, "export.json").readText())!!.groupValues[1].toInt()
        val order = File(exportDir, "io.txt").readLines().filter { it.startsWith("in ") }.map { it.split(" ")[2] }
        val config = Lfm2Config.parse(File(dir, "config.json").readText())
        val reference = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity)
        val quantized = loadLfm2Model(config, File(dir, "model.safetensors"), contextLimit = capacity,
            ctx = ctx, weightFormFor = lfm2ProjectionsAs(TensorEncoding.Q8_0))
        val dims = quantized.model.dimensions
        val embedding = quantized.weights.getValue("model.embed_tokens.weight")
        val step = Lfm2DecodeStep(dims, quantized.weights, capacity, lfm2Head(embedding, TensorEncoding.Q4_0, ctx))

        val tokens = intArrayOf(1, 6, 6423, 708, 1098, 8448)
        val state = HybridModelState()
        var states: Map<String, Tensor<FP32, Float>> = step.stateShapes().mapValues { (_, s) -> tensor(s) }
        var expected = FloatArray(0)
        for ((position, token) in tokens.withIndex()) {
            expected = reference.model.forward(intArrayOf(token), state).logits.data.copyToFloatArray()
            if (position == tokens.lastIndex) break
            val (c, s, mask, write) = controls(dims, capacity, position)
            states = step.forward(embeddedRow(embedding, token), c, s, mask, write, states, ctx).states
        }
        val position = tokens.lastIndex
        val (c, s, mask, write) = controls(dims, capacity, position)
        // The device mask uses a bf16-safe large negative value instead of -Float.MAX_VALUE.
        val deviceMask = tensor(mask.shape, mask.data.copyToFloatArray().map { if (it < 0f) -1e30f else it }.toFloatArray())
        val inputs = mapOf("embedded" to embeddedRow(embedding, tokens.last()), "cos" to c, "sin" to s, "mask" to deviceMask, "write_mask" to write) + states
        val r = step.forward(inputs.getValue("embedded"), c, s, deviceMask, write, states, ctx)

        val out = File(exportDir, "golden").apply { mkdirs() }
        fun writeBf16(t: Tensor<FP32, Float>, file: File) {
            val bytes = java.nio.ByteBuffer.allocate(t.shape.volume * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (v in t.data.copyToFloatArray()) {
                val bits = java.lang.Float.floatToRawIntBits(v)
                bytes.putShort(((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16).toShort())
            }
            file.writeBytes(bytes.array())
        }
        fun writeF32(values: FloatArray, file: File) {
            val bytes = java.nio.ByteBuffer.allocate(values.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            values.forEach { bytes.putFloat(it) }
            file.writeBytes(bytes.array())
        }
        val args = order.map { role ->
            val t = inputs.getValue(role)
            writeBf16(t, File(out, "$role.bf16.bin"))
            "--input=${t.shape.dimensions.joinToString("x")}xbf16=@$role.bf16.bin"
        }
        File(out, "args.txt").writeText(args.joinToString("\n", postfix = "\n"))
        writeF32(expected, File(out, "logits.fp32.f32.bin"))
        writeF32(r.logits.data.copyToFloatArray(), File(out, "logits.step.f32.bin"))
        val top = expected.indices.maxBy { expected[it] }
        println("golden step at position $position written to $out; FP32 top-1 $top, step top-1 ${r.logits.data.copyToFloatArray().let { l -> l.indices.maxBy { l[it] } }}")
    }
}
