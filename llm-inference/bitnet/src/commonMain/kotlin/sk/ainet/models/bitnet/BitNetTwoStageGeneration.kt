package sk.ainet.models.bitnet

import kotlin.random.Random
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.sampleFromCandidates
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.types.FP32

/**
 * The `BITNET_PLANES` lm_head weight of [model], or `null` when the model's final module is not
 * a planes-encoded head — the gate a caller checks before reaching for [generateTwoStage].
 *
 * The decoder network template ends in the lm_head projection ([bitnetNetwork] ends in the
 * `"output"` `VoidDense`); its weight is planes-encoded when the model was loaded through
 * [BitNetWeightLoader] with `planesLmHead` on (both the `output.weight` and the tied-2B4T
 * lanes, transformers#337/#357).
 */
@OptIn(ExperimentalMemoryApi::class)
public fun bitnetPlanesHead(model: Module<FP32, Float>): BitNetPlanesTensorData? {
    val head = model.modules.lastOrNull() ?: return null
    @Suppress("UNCHECKED_CAST")
    val params = (head as? ModuleParameters<FP32, Float>)?.params ?: return null
    val weight = params.firstOrNull { it.name.endsWith("weight") }?.value ?: return null
    return weight.data as? BitNetPlanesTensorData
}

/**
 * The two-stage BitNet generation loop (transformers#358): NeoGPU's decode driver expressed
 * over [OptimizedLLMRuntime.forwardHidden] + [BitNetTwoStageDecode] + [sampleFromCandidates].
 *
 * Per step, instead of the full-vocab lm_head matmul inside the module tree:
 *
 * 1. [OptimizedLLMRuntime.forwardHidden] runs the trunk — everything but the head — for the
 *    last-position hidden state (KV cache and position advance exactly as a full forward).
 * 2. [BitNetTwoStageDecode.topK] scans all rows with planes 0–3 (one fused `lmhead_stage1`
 *    call when [native] is given) and exactly rescores the surviving [candidates] rows.
 * 3. [sampleFromCandidates] samples with the standard temperature semantics over that list.
 *
 * Prompt ingestion also runs [OptimizedLLMRuntime.forwardHidden]: prefill logits are discarded
 * anyway, so every prompt position saves its full-vocab projection outright.
 *
 * Exactness: greedy decode (`temperature <= 1e-6`) selects the same token as the full matmul —
 * [BitNetTwoStageDecode.topK]'s bound guarantees the exact top-k (up to the [candidates] cap,
 * NeoGPU's flat-200 heuristic beyond it). At temperature, sampling is restricted to the
 * rescored candidates — the two-stage contract: mass outside the top-[candidates] is treated
 * as zero.
 *
 * Same prompt semantics as the stock `generate` loop: [OptimizedLLMRuntime.bos] is prepended
 * when the prompt does not already start with it, prompt tokens are ingested one per forward,
 * and [onToken] fires only for generated tokens.
 */
@OptIn(ExperimentalMemoryApi::class)
public fun OptimizedLLMRuntime<FP32>.generateTwoStage(
    prompt: IntArray,
    steps: Int,
    temperature: Float,
    head: BitNetPlanesTensorData,
    native: BitNetStage1Kernel? = null,
    candidates: Int = BitNetTwoStageDecode.DEFAULT_CANDIDATES,
    random: Random = Random.Default,
    onToken: (Int) -> Unit,
) {
    require(steps > 0) { "steps must be > 0" }
    val fullPrompt = when {
        prompt.isEmpty() -> intArrayOf(bos)
        prompt[0] != bos -> intArrayOf(bos) + prompt
        else -> prompt
    }

    val k = candidates.coerceIn(1, head.rows)
    var token = fullPrompt[0]
    var pos = 0
    var generated = 0
    while (generated < steps) {
        val next = if (pos + 1 < fullPrompt.size) {
            forwardHidden(token)
            fullPrompt[pos + 1]
        } else {
            // The hidden tensor lives in the per-step forward scope — copy out before the
            // next forwardHidden recycles it, exactly like logits in the stock loop.
            val hidden = forwardHidden(token).data.copyToFloatArray()
            val rescored = BitNetTwoStageDecode.topK(
                weight = head, hidden = hidden, k = k, maxCandidates = k, native = native,
            )
            sampleFromCandidates(rescored, temperature, random)
        }
        if (pos + 1 >= fullPrompt.size) {
            onToken(next)
            generated++
        }
        token = next
        pos++
    }
}
