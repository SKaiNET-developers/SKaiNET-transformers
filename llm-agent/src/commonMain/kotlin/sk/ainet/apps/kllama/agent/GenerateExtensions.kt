@file:Suppress("unused")

package sk.ainet.apps.kllama.agent

import kotlin.random.Random
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.PrefillStrategy
import sk.ainet.apps.llm.generateUntilStop as coreGenerateUntilStop
import sk.ainet.apps.llm.sampleFromLogits as coreSampleFromLogits
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Backward-compatible re-export of stop-token-aware generation.
 *
 * The canonical implementation is now [sk.ainet.apps.llm.generateUntilStop]
 * in `llm-core` (promoted in issue #49 Phase 1 so any runner with an
 * [InferenceRuntime] gets EOS-aware generation without depending on the
 * agent layer). See that function for full parameter documentation.
 */
public fun <T : DType> InferenceRuntime<T>.generateUntilStop(
    prompt: IntArray,
    maxTokens: Int,
    eosTokenId: Int,
    temperature: Float = 0.8f,
    random: Random = Random.Default,
    onToken: ((Int) -> Unit)? = null,
    decode: ((Int) -> String)? = null,
    onPrefill: ((Int, Int) -> Unit)? = null,
    prefillStrategy: PrefillStrategy = PrefillStrategy.Autoregressive
): GenerateResult = coreGenerateUntilStop(
    prompt = prompt,
    maxTokens = maxTokens,
    eosTokenId = eosTokenId,
    temperature = temperature,
    random = random,
    onToken = onToken,
    decode = decode,
    onPrefill = onPrefill,
    prefillStrategy = prefillStrategy
)

/** The multi-stop-token sibling — see [sk.ainet.apps.llm.generateUntilStop]'s `Set` overload. */
public fun <T : DType> InferenceRuntime<T>.generateUntilStop(
    prompt: IntArray,
    maxTokens: Int,
    eosTokenIds: Set<Int>,
    temperature: Float = 0.8f,
    random: Random = Random.Default,
    onToken: ((Int) -> Unit)? = null,
    decode: ((Int) -> String)? = null,
    onPrefill: ((Int, Int) -> Unit)? = null,
    prefillStrategy: PrefillStrategy = PrefillStrategy.Autoregressive
): GenerateResult = coreGenerateUntilStop(
    prompt = prompt,
    maxTokens = maxTokens,
    eosTokenIds = eosTokenIds,
    temperature = temperature,
    random = random,
    onToken = onToken,
    decode = decode,
    onPrefill = onPrefill,
    prefillStrategy = prefillStrategy
)

/**
 * Backward-compatible re-export of tensor-based sampling.
 * The canonical implementation is now [sk.ainet.apps.llm.sampleFromLogits].
 */
public fun <T : DType> sampleFromLogits(
    logits: Tensor<T, Float>,
    temperature: Float,
    random: Random = Random.Default
): Int = coreSampleFromLogits(logits, temperature, random)
