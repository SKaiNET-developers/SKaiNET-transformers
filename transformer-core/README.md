# transformer-core

Framework NN primitives — attention, the KV-cache family, embedding, norms, RoPE, SwiGLU/GeGLU FFN,
residual, linear projection — extracted from `llm-core` so they build on the **full Kotlin target matrix
including `androidNativeArm32/Arm64`** (the on-device ARM path). Depends only on `skainet-lang-core`
(which has androidNative); no io/compile/backend deps.

`llm-core` `api`-depends on this module and **re-exports** it, so existing consumers are unaffected.
ARM-native consumers (e.g. `skainet-whisper-kmp`) depend on `transformer-core` directly and reuse
KV-cache/attention instead of reimplementing.

## Why
`llm-core`'s primitives only need `lang-core`, but were trapped there: `llm-core`'s *other* deps
(`io-gguf`, `io-core`, `compile-*`, `backend-cpu`) lack androidNative, so ARM-native consumers couldn't
depend on it. The primitives are **dtype-agnostic** (just call `ops.*`), so this target generalization is
orthogonal to the quant/dtype generalization (issue #178) — they meet cleanly at these primitives.

## What moved (15 files, lang-core-only)
`transformer/*` (KVCache, RoPE, ResidualAdd, MultiHeadAttention, GeGLUFFN, SwiGLUFFN, XIELUActivation,
LayerScalarMul, LinearProjection, VoidDense), `layers/*` (Embedding*), `normalization/RMSNormalization`,
`dsl/TransformerDsl`. **Kept in `llm-core`:** `dsl/decoder/*` (DecoderTransformerNetwork needs
`apps.llm.HybridTransformerBlock`, which is compile-opt-coupled).

One back-reference decoupled: `MultiHeadAttention`'s diagnostic `dumpStats` → a settable `mhaStatSink`
(default no-op) that `HybridTransformerBlock` wires to llm-core's platform `dumpStats` (no behaviour lost).

## Verified
`:transformer-core:` compiles for jvm + androidNativeArm32 + arm64; `:llm-core:jvmTest` green (5/5) via
the re-export.

## Landing (for the maintainer)
Branch `feature/transformer-core` was cut from `release/0.31.0`. To land on `develop` (which has #178's
merged #179/#180):
1. `git fetch origin && git rebase origin/develop` — **no conflicts expected on the moved files**: #178's
   merged work is in the model layer (`GemmaPackedWeights`) + engine (`ops.transpose` Q8_0/Q4_0), not these
   primitives. (Verified against local refs; re-check against fresh `develop`.)
2. Build the full target matrix + `:llm-core:` tests; PR; CI-publish; bump the `skainet`/transformers pins.
3. **Note for future quant work:** the pre-transpose-marker (#178 "Solution C") will land in
   `LinearProjection.kt`, which now lives **here**, not `llm-core`. And `RowDequantSource` + packed-weight
   packing (today in `sk.ainet.models.gemma`) are the next candidates to hoist into a shared `quant` layer
   or this module — that's what makes quant reusable across models *and* whisper.
