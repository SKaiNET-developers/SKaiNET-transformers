package sk.ainet.models.qwen

import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.llamaNetwork

/**
 * Qwen3 architecture defined via the network DSL.
 *
 * Qwen3 uses the same transformer architecture as LLaMA
 * (GQA + SwiGLU FFN + RoPE + RMSNorm, no attention biases),
 * so this delegates to [llamaNetwork].
 *
 * This function exists as a stable entry point for Qwen consumers
 * and a future extension point if the architecture diverges.
 */
public inline fun <reified T : DType, V> qwenNetwork(
    metadata: LlamaModelMetadata
): Module<T, V> = llamaNetwork<T, V>(metadata)
