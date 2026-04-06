package sk.ainet.models.voxtral

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Voxtral backbone runtime that captures hidden states during generation.
 *
 * Wraps the backbone model (sequential: token_embd → transformer blocks →
 * output_norm → lm_head) and intercepts the output of `output_norm` to
 * provide hidden states for the acoustic model, while still returning
 * logits for autoregressive token generation.
 *
 * The hidden states are accumulated during [generate] and can be retrieved
 * via [lastHiddenStates] for feeding into [VoxtralAcousticRuntime].
 *
 * @param model The backbone Module (from [voxtralBackboneNetwork])
 * @param ctx Execution context
 * @param dtype DType class
 * @param bos Beginning-of-sequence token ID
 */
public class VoxtralBackboneRuntime<T : DType>(
    private val model: Module<T, Float>,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    public val bos: Int = 1,
    private val random: Random = Random.Default
) {
    private val modules = model.modules
    private val allButLast = modules.dropLast(1)  // everything up to and including output_norm
    private val lmHead = modules.last()            // output (lm_head)

    public var position: Int = 0
        private set

    /** Hidden states captured during the last [generate] call. */
    private val hiddenStatesList = mutableListOf<FloatArray>()

    /**
     * Get the accumulated hidden states from the last [generate] call.
     *
     * Returns a tensor of shape [seqLen, dim] where seqLen is the number of
     * tokens generated (prompt + generated). This is the output of the
     * output_norm layer (before the lm_head projection).
     *
     * @return Hidden states tensor, or null if [generate] hasn't been called
     */
    @Suppress("UNCHECKED_CAST")
    public fun lastHiddenStates(): Tensor<T, Float>? {
        if (hiddenStatesList.isEmpty()) return null
        val dim = hiddenStatesList[0].size
        val seqLen = hiddenStatesList.size
        val flat = FloatArray(seqLen * dim)
        for (i in hiddenStatesList.indices) {
            hiddenStatesList[i].copyInto(flat, i * dim)
        }
        return ctx.fromFloatArray<T, Float>(Shape(seqLen, dim), dtype, flat) as Tensor<T, Float>
    }

    /**
     * Single-token forward pass that returns logits and captures hidden state.
     */
    public fun forward(tokenId: Int): Tensor<T, Float> {
        val input = createTokenTensor(tokenId)

        // Forward through all modules except lm_head
        var hidden = input
        for (mod in allButLast) {
            hidden = mod.forward(hidden, ctx)
        }

        // Capture the hidden state (output of output_norm)
        hiddenStatesList.add(hidden.data.copyToFloatArray())

        // Forward through lm_head to get logits
        val logits = lmHead.forward(hidden, ctx)

        position++
        return logits
    }

    /**
     * Reset state (KV caches, position, hidden states).
     */
    public fun reset() {
        resetKVCaches(model)
        position = 0
        hiddenStatesList.clear()
    }

    /**
     * Generate tokens autoregressively while capturing hidden states.
     *
     * After this call, [lastHiddenStates] returns the accumulated hidden states
     * for all positions (prompt + generated).
     *
     * @param prompt Input token IDs
     * @param steps Number of tokens to generate after prompt
     * @param temperature Sampling temperature
     * @param onToken Callback for each generated token
     */
    public fun generate(
        prompt: IntArray,
        steps: Int,
        temperature: Float = 0.7f,
        onToken: (Int) -> Unit
    ) {
        reset()

        // Process prompt (prefill)
        for (i in 0 until prompt.size - 1) {
            forward(prompt[i])
        }

        // Generate from last prompt token
        var nextToken = if (prompt.isNotEmpty()) {
            val logits = forward(prompt.last())
            sample(logits, temperature)
        } else {
            bos
        }

        for (step in 0 until steps) {
            onToken(nextToken)
            val logits = forward(nextToken)
            nextToken = sample(logits, temperature)
        }
    }

    private fun sample(logits: Tensor<T, Float>, temperature: Float): Int {
        val data = logits.data.copyToFloatArray()
        if (temperature <= 0f) {
            // Greedy
            var bestIdx = 0
            var bestVal = data[0]
            for (i in 1 until data.size) {
                if (data[i] > bestVal) { bestVal = data[i]; bestIdx = i }
            }
            return bestIdx
        }

        // Temperature-scaled softmax sampling
        val scaled = FloatArray(data.size)
        var maxVal = data[0]
        for (v in data) if (v > maxVal) maxVal = v

        var sumExp = 0.0
        for (i in data.indices) {
            val e = kotlin.math.exp(((data[i] - maxVal) / temperature).toDouble())
            scaled[i] = e.toFloat()
            sumExp += e
        }

        val r = random.nextDouble() * sumExp
        var cumSum = 0.0
        for (i in scaled.indices) {
            cumSum += scaled[i]
            if (cumSum >= r) return i
        }
        return scaled.size - 1
    }

    @Suppress("UNCHECKED_CAST")
    private fun createTokenTensor(tokenId: Int): Tensor<T, Float> {
        return ctx.fromFloatArray<T, Float>(
            Shape(1), dtype, floatArrayOf(tokenId.toFloat())
        ) as Tensor<T, Float>
    }

    private fun resetKVCaches(module: Module<*, *>) {
        for (child in module.modules) {
            resetKVCaches(child)
        }
        if (module is sk.ainet.lang.nn.transformer.KVCache<*, *>) {
            module.reset()
        }
    }
}
