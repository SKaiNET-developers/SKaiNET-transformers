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
): Int = sk.ainet.lang.nn.transformer.PhaseProfile.time("sample") {
    val buf = extractFloatBuffer(logits).copyOf()
    sampleFromLogits(buf, temperature, random)
}

/**
 * Generation loop on any [InferenceRuntime].
 *
 * Prepends [bosToken] if not already present, ingests the prompt via the
 * configured [prefillStrategy], then samples [steps] new tokens, calling
 * [onToken] for each.
 *
 * The default strategy [PrefillStrategy.Autoregressive] preserves the
 * historical one-token-per-forward semantics. Opt in to
 * [PrefillStrategy.Batched] to get the prefill speedup from
 * [InferenceRuntime.forwardBatched] (typically 3–10× on long prompts).
 *
 * Decode is always autoregressive — each generated token depends on the
 * previous sample, so there is nothing to batch.
 */
public fun <T : DType> InferenceRuntime<T>.generate(
    prompt: IntArray,
    steps: Int,
    temperature: Float,
    bosToken: Int = 1,
    random: Random = Random.Default,
    prefillStrategy: PrefillStrategy = PrefillStrategy.Autoregressive,
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

    when (prefillStrategy) {
        is PrefillStrategy.Autoregressive -> {
            generateAutoregressive(fullPrompt, steps, temperature, random, onToken)
        }
        is PrefillStrategy.Batched -> {
            generateBatched(fullPrompt, steps, temperature, random, prefillStrategy.maxBatch, onToken)
        }
    }
}

private fun <T : DType> InferenceRuntime<T>.generateAutoregressive(
    fullPrompt: IntArray,
    steps: Int,
    temperature: Float,
    random: Random,
    onToken: (Int) -> Unit
) {
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

private fun <T : DType> InferenceRuntime<T>.generateBatched(
    fullPrompt: IntArray,
    steps: Int,
    temperature: Float,
    random: Random,
    maxBatch: Int,
    onToken: (Int) -> Unit
) {
    // Prefill: ingest the prompt in [maxBatch]-sized chunks. forwardBatched
    // returns the logits at the last position of its input, so the last
    // chunk's logits are exactly the post-prompt distribution we need to
    // sample the first generated token from.
    var lastLogits: sk.ainet.lang.tensor.Tensor<T, Float>? = null
    var i = 0
    while (i < fullPrompt.size) {
        val end = if (i + maxBatch < fullPrompt.size) i + maxBatch else fullPrompt.size
        val chunk = fullPrompt.copyOfRange(i, end)
        lastLogits = forwardBatched(chunk)
        i = end
    }
    val postPromptLogits = lastLogits
        ?: error("generate: prompt is empty after BOS prepend — internal invariant broken")

    // Decode: autoregressive sampling. The first sample comes from
    // postPromptLogits; each subsequent step does forward(prev_sample).
    var sample = sampleFromTensor(postPromptLogits, temperature, random)
    onToken(sample)
    var generatedCount = 1
    while (generatedCount < steps) {
        val logits = forward(sample)
        sample = sampleFromTensor(logits, temperature, random)
        onToken(sample)
        generatedCount++
    }
}

/** Extract a FloatArray from a tensor, preferring zero-copy when possible. */
internal fun <T : DType> extractFloatBuffer(tensor: Tensor<T, Float>): FloatArray {
    val data = tensor.data
    if (data is FloatArrayTensorData<*>) return data.buffer
    return data.copyToFloatArray()
}
