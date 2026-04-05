package sk.ainet.apps.llm

import kotlin.random.Random
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Sample a token ID from a logits tensor.
 *
 * Copies the underlying float buffer before sampling (the original tensor is not mutated).
 */
public fun <T : DType> sampleFromTensor(
    logits: Tensor<T, Float>,
    temperature: Float,
    random: Random = Random.Default
): Int {
    val buf = extractFloatBuffer(logits).copyOf()
    return sampleFromLogits(buf, temperature, random)
}

/**
 * Auto-regressive generation loop on any [InferenceRuntime].
 *
 * Prepends [bosToken] if not already present, feeds prompt tokens
 * through the model (discarding their logits), then samples
 * [steps] new tokens, calling [onToken] for each.
 */
public fun <T : DType> InferenceRuntime<T>.generate(
    prompt: IntArray,
    steps: Int,
    temperature: Float,
    bosToken: Int = 1,
    random: Random = Random.Default,
    onToken: (Int) -> Unit
) {
    require(steps > 0) { "steps must be > 0" }

    val fullPrompt = if (prompt.isNotEmpty() && prompt[0] != bosToken) {
        intArrayOf(bosToken) + prompt
    } else if (prompt.isEmpty()) {
        intArrayOf(bosToken)
    } else {
        prompt
    }

    var token = fullPrompt[0]
    var pos = 0
    var generatedCount = 0
    while (generatedCount < steps) {
        val logits = forward(token)
        val next = if (pos + 1 < fullPrompt.size) {
            fullPrompt[pos + 1]
        } else {
            sampleFromTensor(logits, temperature, random)
        }
        if (pos + 1 >= fullPrompt.size) {
            onToken(next)
            generatedCount++
        }
        token = next
        pos++
    }
}

/** Extract a FloatArray from a tensor, preferring zero-copy when possible. */
internal fun <T : DType> extractFloatBuffer(tensor: Tensor<T, Float>): FloatArray {
    val data = tensor.data
    if (data is FloatArrayTensorData<*>) return data.buffer
    return data.copyToFloatArray()
}
