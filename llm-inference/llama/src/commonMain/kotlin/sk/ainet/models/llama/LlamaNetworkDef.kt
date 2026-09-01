package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.decoder.decoderTransformerNetwork
import sk.ainet.lang.types.DType

/**
 * Llama architecture defined via the network DSL.
 *
 * Thin caller of the shared [decoderTransformerNetwork] builder with
 * Llama-specific knobs:
 * - RoPE base from `metadata.ropeFreqBase` (typically 10_000)
 * - RMSNorm eps from `metadata.rmsNormEps` (typically 1e-5)
 * - No QK-norm (Llama has no `attn_q_norm` / `attn_k_norm` weights)
 * - SwiGLU FFN
 *
 * Architecture: `Embedding → N × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 * RMSNorm → SwiGLU FFN → Residual) → RMSNorm → Dense`.
 */
public inline fun <reified T : DType, V> llamaNetwork(
    metadata: GgufDecoderMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
): Module<T, V> = decoderTransformerNetwork<T, V>(
    metadata = metadata,
    qkNorm = false,
    maxInferenceLen = maxInferenceLen,
)
