package sk.ainet.lang.nn.dsl.decoder

/**
 * The KV cache a decoder layer is built with (SKEEP-005).
 *
 * - [APPEND] (default): `AppendKVCache` — history grows by concatenation each step; attention
 *   copies the used prefix per layer and token.
 * - [POSITIONAL]: `PositionalKVCache` — a buffer of `maxInferenceLen × nKVHeads × headDim`
 *   floats ×2 per layer allocated up front; attention reads it in place, no per-token copies.
 *   Size `maxInferenceLen` accordingly (Llama-3.2-3B at 4096: ≈ 32 MB per layer).
 */
public enum class DecoderKVCacheKind { APPEND, POSITIONAL }
