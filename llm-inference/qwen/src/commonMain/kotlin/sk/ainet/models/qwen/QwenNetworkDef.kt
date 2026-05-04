package sk.ainet.models.qwen

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.decoder.decoderTransformerNetwork
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Qwen3 architecture defined via the network DSL.
 *
 * Thin caller of the shared [decoderTransformerNetwork] builder with
 * Qwen3-specific knobs:
 * - RoPE base from `metadata.ropeFreqBase` (typically 1_000_000 — read off
 *   the GGUF; falls back to the metadata default of 10_000 if absent).
 * - RMSNorm eps from `metadata.rmsNormEps` (typically 1e-6 for Qwen3).
 * - **QK-Norm enabled** (`attn_q_norm.weight` / `attn_k_norm.weight` per
 *   layer) — the key Qwen-vs-Llama difference at the network level.
 * - SwiGLU FFN — same as Llama.
 *
 * The metadata type stays `LlamaModelMetadata` because Qwen3 GGUFs use the
 * Llama-family tensor naming convention; per-architecture metadata classes
 * are a follow-up rename.
 */
public inline fun <reified T : DType, V> qwenNetwork(
    metadata: LlamaModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
    qkNorm: Boolean = true,
): Module<T, V> = decoderTransformerNetwork<T, V>(
    metadata = metadata,
    qkNorm = qkNorm,
    maxInferenceLen = maxInferenceLen,
)
