package sk.ainet.models.moonshine

import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine **v2** DECODER, authored in the SKaiNET NN DSL.
 *
 * CONFIRMED (2026-07-25) against the real **tiny-streaming** `decoder_kv.onnx` + `cross_kv.onnx`: the v2
 * decoder is **architecturally identical to the v1 Moonshine decoder** — each pre-norm layer is
 *   x + SelfAttn(LN(x))                  — causal, **partial-rotary RoPE**, no bias, KV-cached
 *   x + CrossAttn(LN(x), memory K/V)     — attends the adapter memory, no RoPE, non-causal, no bias
 *   x + MLP(LN(x))                       — **gated SiLU** (fused fc1 `[2·ffn,dim]` → silu(gate)*value → fc2)
 * then a final scale-only LayerNorm and the `lm_head` vocab projection (tied to the token embedding). The
 * ONNX confirms this exactly: 6 layers, `Sigmoid`×6 (the SiLU gate), a single shared `Cos`/`Sin` (RoPE),
 * `LayerNormalization`×19 (3/layer + final), and — because v2 factors the cross-attention K/V into the
 * separate `cross_kv` graph — 36 `[320,320]` projections in `decoder_kv` (self q,k,v,o + cross q,o per layer)
 * plus 12 `[320,320]` in `cross_kv` (cross k,v per layer).
 *
 * Only the **dimensions** differ from v1, so this **reuses the proven [MoonshineDecoderModel] /
 * [MoonshineDecoderLayer]** (their `forward` / `forwardPrefill` / `forwardWithPast` are the exact v2 decode
 * contract) rather than duplicating ~200 lines — the v2 config just maps onto the shared [MoonshineConfig].
 * Real tiny dims: `dim=320, decoderLayers=6, nHeads=8, headDim=40, ffn=1280, vocab=32768`, and
 * `rotaryDim=32` (`rotary.inv_freq[16]`) ⇒ `partialRotaryFactor=0.8` (see [MoonshineV2Config]).
 *
 * The v2 **`cross_kv`** graph = the decoder cross-attention K/V projections, which [MoonshineDecoderModel.
 * forwardPrefill] already surfaces as its per-layer `crossK/crossV` outputs; **`decoder_kv`** = its
 * [MoonshineDecoderModel.forwardWithPast] (one token over a growing self-cache + the fixed cross K/V).
 *
 * dtype-portable like the encoder/adapter (`FP32` host, `BF16` Torq). Numeric validation vs the ONNX (RoPE
 * base/factor, tie) is a follow-up — the structure is confirmed here.
 */
public fun <T : DType, V> moonshineV2Decoder(
    cfg: MoonshineV2Config,
    dtype: KClass<T>,
): MoonshineDecoderModel<T, V> = MoonshineDecoderModel(cfg.toDecoderConfig(), dtype)

/**
 * Map the v2 config onto the shared [MoonshineConfig] fields the decoder consumes. Encoder-only fields
 * (conv frontend, `maxFrames`) keep their defaults — the decoder never reads them.
 */
internal fun MoonshineV2Config.toDecoderConfig(): MoonshineConfig = MoonshineConfig(
    dim = dim,
    decoderLayers = decoderLayers,
    nHeads = nHeads,
    headDim = headDim,
    ffnDim = ffnDim,
    vocabSize = vocabSize,
    maxDecodeTokens = maxDecodeTokens,
    ropeBase = ropeBase,
    partialRotaryFactor = partialRotaryFactor,
    layerNormEps = layerNormEps,
)
