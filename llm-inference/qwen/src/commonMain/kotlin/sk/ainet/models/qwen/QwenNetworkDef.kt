package sk.ainet.models.qwen

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.decoder.decoderTransformerNetwork
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Qwen3 architecture defined via the network DSL.
 *
 * Thin caller of the shared [decoderTransformerNetwork] builder with
 * Qwen3-specific knobs:
 * - **RoPE NEOX pairing** ([RoPEMode.SPLIT_HALF]) — Qwen 2/3 GGUFs are
 *   stored with `(buf[i], buf[i + ropeDim/2])` pairing per llama.cpp's
 *   `LLAMA_ROPE_TYPE_NEOX` (mode 2). Picking the LLaMA default of
 *   [RoPEMode.INTERLEAVED] gives correct-by-accident outputs at very
 *   small contexts but compounds wrong rotations across positions —
 *   logits diverged by 5+ on identical FP32 weights vs the legacy
 *   `LlamaRuntime` (which auto-applies NEOX via
 *   `RopeType.forArchitecture("qwen3")`). See #114.
 * - RoPE base from `metadata.ropeFreqBase` (typically 1_000_000 — read off
 *   the GGUF; falls back to the metadata default of 10_000 if absent).
 * - RMSNorm eps from `metadata.rmsNormEps` (typically 1e-6 for Qwen3).
 * - **QK-Norm enabled** (`attn_q_norm.weight` / `attn_k_norm.weight` per
 *   layer) — Qwen3-specific.
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
    attnBias: Boolean = false,
): Module<T, V> = decoderTransformerNetwork<T, V>(
    metadata = metadata,
    qkNorm = qkNorm,
    attnBias = attnBias,
    ropeMode = RoPEMode.SPLIT_HALF,
    maxInferenceLen = maxInferenceLen,
)
