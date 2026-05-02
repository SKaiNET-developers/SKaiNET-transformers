# Apertus Support Rollout — COMPLETE

**Status:** complete (3 of 3 PRs merged + deprecated-runtime cleanup).
**Plan PR:** #91. **Implementation PRs:** #92 (routing), #93 (chat-template docs), #94 (tool calling).

## Summary

Apertus (Swiss AI / EPFL multilingual decoder-only transformer) reached production parity with kllama and kgemma over four PRs landed 2026-05-01 and 2026-05-02:

| PR    | Title                                                     |
| ----: | --------------------------------------------------------- |
| #91   | Plan + this tracking doc                                  |
| #92   | `fix(apertus): route through OptimizedLLMRuntime + apertusNetwork()` |
| #93   | `docs(apertus): document chat-template format`            |
| #94   | `feat(apertus): tool calling support`                     |

After this stack:
- `skainet-cli` routes Apertus models through `OptimizedLLMRuntime + apertusNetwork()` (xIELU + QK-Norm + ungated FFN — the previous `LlamaRuntime` fallback silently produced wrong logits).
- `--agent --template=apertus` formats prompts with Apertus's own role tokens (`<|system_start|>`, `<|user_start|>`, `<|assistant_start|>`, `<|tools_prefix|>`, etc.) and parses tool calls back from `<|tools_prefix|>[...]<|tools_suffix|>` JSON arrays.
- `ModelRegistry.APERTUS.supportsToolCalling = true`, `chatTemplateFamily = "apertus"`.
- `KernelRegistry` auto-discovers native FFM kernels for the matmul path via the 0.22.0 native-cpu module.

## What's not in this rollout

- **Optional kapertus-cli rebuild** — was originally listed as PR 4 ("rebuild CLI under `llm-apps/`"). Dropped: the unified `skainet-cli` already covers Apertus end-to-end, model-specific CLIs (kqwen, kapertus, kvoxtral) are being deprecated per commit `81f3506`, and the workspace direction is consolidation rather than per-model binaries. If a downstream consumer needs an Apertus-only fat-jar later, copy the `skainet-cli` shadow setup.
- **Native Apertus kernels** — Apertus shares matmul shapes with Llama; the native FFM kernels from SKaiNET 0.22.0 (Q4_K, FP32) work transparently. No Apertus-specific kernel work needed.
- **TurboQuant KV-cache compression for Apertus** — tracked separately under the TurboQuant workstream.

## Reference docs

- `docs/specs/apertus-chat-template.md` — full spec for the Apertus chat template (PR 2). Source of truth for the `ApertusChatTemplate` implementation.

## Cleanup that landed alongside the rollout (this commit)

The hand-coded `ApertusRuntime.kt` and `ApertusQuantizedRuntime.kt` paths (and their attention backends + smoke tests) were marked `@Deprecated` after PR 1 made `OptimizedLLMRuntime + apertusNetwork()` the canonical path. Removed in this commit alongside the rollout closure:

- `ApertusRuntime.kt` — hand-coded decoder runtime, deprecated.
- `ApertusQuantizedRuntime.kt` — lazy-dequant variant, deprecated.
- `ApertusAttentionBackend.kt` + `ApertusCpuAttentionBackend.kt` — only used by the two deleted runtimes.
- `ApertusRuntimeSmokeTest.kt` + `ApertusQuantizedRuntimeSmokeTest.kt` — exercised the deleted runtimes.

The `xielu()` / `softplus()` activation reference functions previously housed in `ApertusRuntime.kt` were extracted to `ApertusXIELU.kt` so `ApertusXIELUTest` keeps validating the math. The kdoc references in `OptimizedLLMRuntime.kt` and `OutputEquivalenceTest.kt` to "ApertusRuntime" are now stale and worth a follow-up sweep, but they're code comments only and don't break anything.

The remaining apertus library files (`ApertusNetworkDef`, `ApertusNetworkLoader`, `ApertusWeightLoader`, `ApertusSafeTensorsLoader`, `ApertusRuntimeWeights`, `ApertusConfigParser`, `QuantizedTensor`, `ApertusXIELU`, `ApertusIngestion`) cover the whole production path through `apertusNetwork() + OptimizedLLMRuntime`.

## Test footprint (post-cleanup)

- `:llm-inference:apertus:jvmTest` — 12 tests (ConfigParser 6, XIELU 6).
- `:llm-agent:jvmTest --tests '*Apertus*'` — 21 tests (ChatTemplate 10, ParserStrategy 11).
- 33 Apertus-specific tests total, all green.
