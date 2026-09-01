package sk.ainet.models.bitnet

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.decoder.DecoderFfnKind
import sk.ainet.lang.nn.dsl.decoder.decoderTransformerNetwork
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * BitNet b1.58 (e.g. `microsoft/bitnet-b1.58-2B-4T`) defined via the network DSL.
 *
 * Thin caller of the shared [decoderTransformerNetwork] builder with the BitNet knobs, all
 * verified against NeoGPU's reference driver (`hs_ml_infer.c` — the working 2B4T inference the
 * SKaiNET ternary port tracks, see SKaiNET#1136):
 *
 * - **Squared-ReLU gated FFN with `ffn_sub_norm`** ([DecoderFfnKind.RELU2_SUBLN]):
 *   `down(subNorm(relu(gate(x))² * up(x)))` — not SwiGLU.
 * - **`attn_sub_norm`** ([decoderTransformerNetwork]'s `attnSubNorm`): an RMSNorm on the merged
 *   attention output BEFORE the o_projection.
 * - **RoPE NEOX-style pairing** ([RoPEMode.SPLIT_HALF]): bitnet.cpp places all three BITNET
 *   arches in the `LLAMA_ROPE_TYPE_NEOX` group (pairs offset by `n_rot/2`), and the HF BF16
 *   reference (`microsoft/bitnet-b1.58-2B-4T`) agrees. The previous INTERLEAVED pairing was a
 *   real numerics bug — greedy decode diverged from the reference at generated token 4; with
 *   SPLIT_HALF it matches for 17 tokens and the residual divergence is a reference-side
 *   near-tie (transformers#360's arbitration, 2026-08-31).
 * - RMSNorm eps from metadata (1e-5 for 2B4T).
 *
 * The metadata type stays [LlamaModelMetadata] — BitNet GGUFs use the Llama-family tensor naming
 * convention plus the two sub-norm tensors ([BitNetGGUFNameResolver] handles those).
 *
 * The weights arriving through this def are whatever the loader materialized — dense FP32 for the
 * exact baseline, or packed `BITNET_B1_58` tensors once the packed path is wired (#337): the
 * network definition itself is format-agnostic, per the SKaiNET physiology (kernels are selected
 * by the weight's storage format, never by the model code).
 */
public inline fun <reified T : DType, V> bitnetNetwork(
    metadata: LlamaModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
): Module<T, V> = decoderTransformerNetwork<T, V>(
    metadata = metadata,
    ropeMode = RoPEMode.SPLIT_HALF,
    maxInferenceLen = maxInferenceLen,
    ffnKind = DecoderFfnKind.RELU2_SUBLN,
    attnSubNorm = true,
)
