package sk.ainet.models.lfm2

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/** Batch-one FP32 LFM2 short convolution. Projections, gates and Conv1d execute in SKaiNET. */
class ShortConv(
    private val width: Int,
    private val kernel: Int,
    inProjection: List<List<Float>>,
    convolution: List<List<Float>>,
    outProjection: List<List<Float>>,
) {
    private val ctx = DirectCpuExecutionContext.create()
    private val inWeight: Tensor<FP32, Float>
    private val convWeight: Tensor<FP32, Float>
    private val outWeight: Tensor<FP32, Float>
    private val operation: ShortConvTensorOp
    // Detached, chronological last K gated values, matching reference cache representation.
    private var history = FloatArray(width * kernel)
    var position: Int = 0
        private set

    init {
        require(width > 0 && kernel > 0) { "Dimensions must be positive" }
        validate(inProjection, 3 * width, width, "in_proj")
        validate(convolution, width, kernel, "conv")
        validate(outProjection, width, width, "out_proj")
        inWeight = tensor(Shape(3 * width, width), inProjection.flatten().toFloatArray())
        convWeight = tensor(Shape(width, 1, kernel), convolution.flatten().toFloatArray())
        outWeight = tensor(Shape(width, width), outProjection.flatten().toFloatArray())
        operation = ShortConvTensorOp(width, kernel, inWeight, convWeight, outWeight)
    }

    fun reset() {
        history.fill(0f)
        position = 0
    }

    fun state(): List<List<Float>> = List(width) { channel ->
        history.slice(channel * kernel until (channel + 1) * kernel)
    }

    fun provider(): String = ctx.ops.javaClass.name

    fun forward(input: List<List<Float>>): List<List<Float>> {
        require(input.isNotEmpty()) { "Empty chunks are unsupported" }
        validate(input, input.size, width, "input")
        require(position.toLong() + input.size <= Int.MAX_VALUE) { "Position overflow" }
        val length = input.size
        val x = tensor(Shape(length, width), input.flatten().toFloatArray())
        val result = operation.forward(x, tensor(Shape(width, kernel), history), ctx)
        // Host copies are confined to this convenience wrapper, outside the exported function.
        val output = result.output.data.copyToFloatArray()
        val nextHistory = result.nextHistory.data.copyToFloatArray()
        require(output.all { it.isFinite() } && nextHistory.all { it.isFinite() }) { "Non-finite output or convolution state" }
        // Commit state only after a successful forward pass.
        history = nextHistory
        position += length
        return List(length) { row -> output.slice(row * width until (row + 1) * width) }
    }

    private fun tensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    private fun validate(values: List<List<Float>>, rows: Int, columns: Int, name: String) {
        require(values.size == rows && values.all { it.size == columns }) { "$name shape mismatch: expected [$rows,$columns]" }
        require(values.all { row -> row.all { it.isFinite() } }) { "$name contains non-finite values" }
    }
}

