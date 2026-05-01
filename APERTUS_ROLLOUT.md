# Apertus Support Rollout

**Status:** plan landed; PRs 1–4 not started.
**Owner:** unassigned.
**Plan PR:** this commit.

## Context

The Apertus model (Swiss AI / EPFL multilingual decoder-only transformer) is **architecturally complete in the transformers library layer** but has three integration gaps that make it semi-broken end-to-end today:

1. **Silent correctness bug in `skainet-cli`.** Lines 168–216 of `llm-apps/skainet-cli/src/main/kotlin/sk/ainet/apps/skainet/cli/Main.kt` route every non-Gemma model — including Apertus — through the deprecated `LlamaRuntime`. Apertus uses **xIELU** activation, **QK-Norm** (RMSNorm on Q and K before RoPE), and **ungated FFN** (no `gate_proj`); none of those branches exist in `LlamaRuntime`. Inference completes, but the logits diverge from what the Apertus checkpoint actually wants. The output is wrong on a level the user can't easily catch unless they compare to a reference.
2. **Tool calling is OFF.** `llm-core/src/commonMain/kotlin/sk/ainet/apps/llm/ModelRegistry.kt:66` lists Apertus as `("apertus", "Apertus", false, "chatml")` — `supportsToolCalling=false`, `chatTemplateFamily="chatml"` (a guess). There is no `ApertusChatTemplate.kt`, no `ApertusToolCallingSupport`, and no entry in `llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/ToolCallingSupportResolver.kt:16`. Apertus models fall back to `GenericToolCallingSupport`, which doesn't know the model's actual prompt format.
3. **`kapertus` runtime is a 1-file stub.** Commit `81f3506` (`chore: remove deprecated-runtime CLIs (kqwen-only / kapertus / kvoxtral)`) deleted the previous `kapertus` CLI's `Main.kt` (~292 lines). What remains in `llm-runtime/kapertus/` is a single `ApertusIngestion.kt` (89 lines) that wraps the weight loaders. There is no kapertus-cli binary equivalent of `kllama-cli`, and no `llm-apps/kapertus-cli/` module.

The architecture / library layer itself is solid:
- `apertusNetwork()` DSL with xIELU, QK-Norm, ungated FFN — `llm-inference/apertus/src/commonMain/kotlin/sk/ainet/models/apertus/ApertusNetworkDef.kt:29`.
- Weight loading from GGUF + SafeTensors + quantized — `ApertusWeightLoader.kt`, `ApertusSafeTensorsLoader.kt`, `ApertusQuantizedRuntime.kt`.
- 4 self-contained `commonTest` smoke tests pass without a model file (`ApertusRuntimeSmokeTest`, `ApertusQuantizedRuntimeSmokeTest`, `ApertusXIELUTest`, `ApertusConfigParserTest`).

**Goal:** lift Apertus to the same level of polish kllama and kgemma have. Track the rollout in this file. The next contributor / session opens this doc, scans the staged-delivery checklist, and picks up where the previous one left off.

---

## Staged delivery

- [ ] **PR 1 — `fix(apertus): route through OptimizedLLMRuntime + apertusNetwork()`** (correctness fix)
- [ ] **PR 2 — `docs(apertus): document chat template format`** (research)
- [ ] **PR 3 — `feat(apertus): tool calling support`** (implementation, depends on PR 2)
- [ ] **PR 4 — `feat(kapertus): rebuild CLI under llm-apps/`** (parity, optional)

Each PR ticks its own checkbox when merged. `Status:` at the top of this doc reflects the most recent merged PR.

---

## PR 1 — fix skainet-cli routing for Apertus

**Why first:** today's `skainet-cli` produces silently-wrong logits for Apertus models. Fix the worst class of bug first; everything else is additive.

**Changes:**
- `llm-apps/skainet-cli/src/main/kotlin/sk/ainet/apps/skainet/cli/Main.kt` lines 168–216: detect `architecture == "apertus"` (or `family == "apertus"`) and branch to `OptimizedLLMRuntime + apertusNetwork()` instead of `LlamaRuntime`. Mirror how Gemma4 is already special-cased in this file.
- Reuse `ApertusNetworkLoader.kt:31` for the module-build path.
- Reuse `apertusNetwork()` from `ApertusNetworkDef.kt:29`.

**Verification:**
- `skainet-cli -m <apertus.gguf> "Hello"` produces coherent output (the divergence is silent today; loading a real checkpoint and seeing meaningful text is the canary).
- xIELU activation actually fires — verify by setting a debug breakpoint or `println` in `ApertusNetworkDef`'s xIELU branch on the first forward pass.

**No new tests required** — the existing `:llm-inference:apertus:commonTest` already exercises `apertusNetwork()` end-to-end at toy scale; the routing fix is a Main.kt change covered by manual run.

---

## PR 2 — document the chat template format

**Why:** before implementing a chat template, we need to know what format Apertus models actually expect. `ModelRegistry.kt:66` lists `chatTemplateFamily="chatml"` but that's a guess — Apertus may use a different format (Alpaca, llama2-style, custom). Without the right template, tool-calling output won't parse correctly even if the rest of PR 3 is right.

**Changes:**
- Inspect a real Apertus GGUF (download from HuggingFace if needed: `swiss-ai/Apertus-1B` or similar). Read the `tokenizer.chat_template` GGUF metadata key.
- Create `docs/explanation/models/apertus-chat-template.md` documenting:
  - The actual chat-template Jinja string from the GGUF, byte-for-byte.
  - Special tokens (`<|im_start|>`-style? `[INST]`-style? Alpaca?).
  - Tool calling format (if the template has any).
  - Whether the template matches an existing family (`chatml`, `llama3`, `gemma`) or needs a new `apertus` strategy.
- Update `ModelRegistry.kt:66` `chatTemplateFamily` if the research shows a different family is correct.

**Verification:** doc exists; rendered template matches the GGUF's `tokenizer.chat_template` byte-for-byte for one canonical message exchange (system + user + assistant + user roles).

---

## PR 3 — tool calling support

**Depends on PR 2** (need the template format documented).

**Changes:**
- New `llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/ApertusChatTemplate.kt` — concrete `ChatTemplate`. Pattern reference: `Llama3ChatTemplate.kt` and `Gemma4ChatTemplate.kt`.
- New `llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/ApertusToolCallingSupport.kt` — analog of `Gemma4ToolCallingSupport`. Bundles the chat template + the tool-call markup parser.
- Register in `ToolCallingSupportResolver.kt:16` — add a branch for `family == "apertus"` returning the new support.
- `ModelRegistry.kt:66`: flip `supportsToolCalling=true`.
- Tests:
  - `ApertusChatTemplateTest.kt` in `llm-agent/src/commonTest/kotlin/sk/ainet/apps/kllama/chat/` — render parity vs the GGUF Jinja template (mirror `Gemma4ChatTemplateHfParityTest` shape).
  - `ApertusToolCallParserStrategyTest.kt` — parse the Apertus tool-call output format.

**Verification:**
- `:llm-agent:commonTest --tests '*Apertus*'` green.
- End-to-end: `skainet-cli -m <apertus.gguf> --agent --template=apertus "What is 17 * 23?"` invokes a calculator tool the same way the kllama TinyLlama tool-calling smoke test does.

---

## PR 4 — rebuild kapertus CLI under llm-apps/ (optional)

**Why:** `kapertus` runtime is a 1-file library facade today. Users wanting a `kapertus-cli` equivalent of `kllama-cli` have nothing. The unified `skainet-cli` works fine after PR 1, so PR 4 is **optional polish** — only worth doing if there's a specific reason to ship a separate Apertus-only CLI binary (smaller fat-JAR, branded distribution, similar parity expectations as `kllama-cli`).

**Changes:**
- New `llm-apps/kapertus-cli/build.gradle.kts` mirroring `kllama-cli/build.gradle.kts`: `kotlin("jvm")` + `shadow` + `application` plugins.
- New `llm-apps/kapertus-cli/src/main/kotlin/sk/ainet/apps/kapertus/cli/Main.kt` — re-implement the deleted (commit `81f3506`) Main.kt, but routed through `OptimizedLLMRuntime + apertusNetwork()`.
- Apply the shadow `mergeServiceFiles()` `doLast` workaround that PR #88 added to `kllama-cli` and `skainet-cli` (the `com.gradleup.shadow:9.4.x` bug — `NativeKernelProviderFactory` gets dropped from the merged services file otherwise).
- Add to `settings.gradle.kts`: `include("llm-apps:kapertus-cli")`.

**Verification:**
- `:llm-apps:kapertus-cli:shadowJar` produces a runnable fat JAR.
- `unzip -p kapertus-all.jar META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` shows all 3 KernelProvider entries (Scalar + PanamaVector + Native).
- `java -jar kapertus-all.jar -m <apertus.gguf> "Hello"` produces coherent output.

---

## Out of scope

- **Native-cpu wiring for Apertus inference.** Works automatically once PR 1 lands: matmul flows through `KernelRegistry.bestAvailable()`; native FFM kernels are auto-discovered via ServiceLoader on the classpath. No Apertus-specific code needed.
- **Q4_K-quantized Apertus checkpoints.** Should work today via the existing Q4_K matmul path; no Apertus-specific code needed.
- **TurboQuant KV-cache compression for Apertus.** Tracked separately under the broader TurboQuant workstream.
- **Removing the deprecated `ApertusRuntime.kt`** (hand-coded path). Leave as `@Deprecated` until consumers have migrated to `OptimizedLLMRuntime + apertusNetwork()`.

---

## Why this lives in transformers, not SKaiNET

The Apertus rollout is entirely transformers-side. Model definition (`apertusNetwork()`), runtime, weight loaders, and tool calling all live under `llm-inference/apertus/`, `llm-runtime/kapertus/`, and `llm-agent/`. The SKaiNET upstream (kernels, tensor ops, ServiceLoader infra) needs no Apertus-specific changes.
