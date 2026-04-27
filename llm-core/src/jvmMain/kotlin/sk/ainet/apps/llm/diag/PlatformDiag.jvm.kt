package sk.ainet.apps.llm.diag

import sk.ainet.lang.tensor.Tensor

public actual fun envFlag(name: String): Boolean = System.getenv(name) == "1"

public actual fun dumpStats(label: String, tensor: Tensor<*, *>) {
    val data = tensor.data
    val arr: FloatArray = when (data) {
        is sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<*> -> data.buffer.copyOf()
        is sk.ainet.lang.tensor.data.MemorySegmentTensorData<*> -> {
            val n = tensor.shape.volume
            val out = FloatArray(n)
            java.lang.foreign.MemorySegment.copy(
                data.segment,
                java.lang.foreign.ValueLayout.JAVA_FLOAT,
                data.segmentByteOffset,
                out, 0, n
            )
            out
        }
        else -> {
            println("$label shape=${tensor.shape.dimensions.toList()} backing=${data::class.simpleName}")
            return
        }
    }
    val rank = tensor.shape.rank
    val featureDim = tensor.shape[rank - 1]
    val lastPosOff = arr.size - featureDim
    var argmaxAbs = 0
    var argmaxAbsVal = 0f
    if (lastPosOff >= 0) {
        for (i in 0 until featureDim) {
            val v = arr[lastPosOff + i]
            if (v.isNaN()) continue
            val av = if (v < 0f) -v else v
            if (av > argmaxAbsVal) { argmaxAbsVal = av; argmaxAbs = i }
        }
    }
    var mn = Float.POSITIVE_INFINITY
    var mx = Float.NEGATIVE_INFINITY
    var sum = 0.0
    var sumSq = 0.0
    for (v in arr) {
        if (!v.isNaN()) {
            if (v < mn) mn = v; if (v > mx) mx = v
            sum += v; sumSq += v.toDouble() * v
        }
    }
    val n = arr.size
    val mean = sum / n
    val rms = kotlin.math.sqrt(sumSq / n)
    println(
        "$label shape=${tensor.shape.dimensions.toList()} min=%+.3f max=%+.3f mean=%+.3f rms=%.3f argmaxAbs=%d v=%+.3f".format(
            mn, mx, mean, rms, argmaxAbs, argmaxAbsVal
        )
    )
    if (lastPosOff >= 0) {
        val fpDims = intArrayOf(0, 12, 16, 100, 438, 660, 809, 1213, 1273, 1295, 1500)
        val fp = fpDims.filter { it < featureDim }.joinToString(" ") { d ->
            "v[$d]=%+.4f".format(arr[lastPosOff + d])
        }
        println("        $fp")
    }
}
