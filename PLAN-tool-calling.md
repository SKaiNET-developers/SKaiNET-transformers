# PLAN — Gemma 4 tool calling: finish partial + missing

Status tracking for wrapping up the Phase 6b tool-calling work. Groups below map to GitHub issues / feature branches. Mark items `[x]` as they land.

## Ground rules

- **Scope split** (per user directive): LLM / tool-calling code stays in `SKaiNET-transformers`. Only compute, ops, and core-framework bugs go upstream to `SKaiNET`. If an upstream change is needed mid-task, fix in `SKaiNET`, `./gradlew publishToMavenLocal` (unsigned — `RELEASE_SIGNING_ENABLED=false` is default), consume from transformers, and reference the upstream issue in the transformers commit footer.
- **GitFlow** (per `GITFLOW.adoc`): feature branches are `feature/ISSUE-<N>-<slug>`, cut from `develop`, merged back via PR with `--no-ff`. Commits use `feat(#N): ...` / `fix(#N): ...` / `docs(#N): ...` format.
- **Issue location**: transformers issues on `SKaiNET-developers/SKaiNET-transformers`. The research repo (`gemma4-research`) is local-only — docs work there is tracked in this plan, no GH issue.

## Groups

### Group 1 — Tool-call robustness · `feature/ISSUE-<N>-tool-call-robustness`
**Issue repo**: SKaiNET-transformers. **Goal**: harden the existing parser + template path.

- [ ] (c) JSON-schema validation: `ToolCallParser` extracts `arguments` but doesn't validate against `ToolDefinition.parameters`. Add schema check + `ToolCallValidationError` surfaced through `AgentLoop`. Test: malformed args rejected with actionable error.
- [ ] (f) Codify tokenizer special tokens: `<|turn>`, `<|tool>`, `<|tool_call>`, `<tool_call|>`, `<|tool_response>` are used by `Gemma4ChatTemplate` but only empirically tested. Add a tokenizer round-trip test proving these encode/decode cleanly against the E2B tokenizer.

### Group 2 — ChatSession polish · `feature/ISSUE-<N>-chat-session-polish`
**Issue repo**: SKaiNET-transformers. **Goal**: make the Gemma 4 DSL path the canonical surface.

- [ ] (d) Parameterize system prompt on `ChatSession` (`ChatSession.kt:67` — currently hard-coded `"You are a helpful assistant with access to tools."`). Constructor arg + sensible default per model family.
- [ ] (e) Remove deprecated `Gemma4Runtime.kt:27-42`. Confirm zero callers (grep) before deletion; update any docs/README references.

### Group 3 — Thinking mode · `feature/ISSUE-<N>-thinking-mode`
**Issue repo**: SKaiNET-transformers. **Goal**: emit/consume `<|think|>...<think|>` blocks.

- [ ] (b) `Gemma4ChatTemplate`: parse `<|think|>` blocks alongside tool calls; buffer separately from user-visible content.
- [ ] `AgentLoop`: expose thinking output via an `AgentListener` callback (no leak into assistant message).
- [ ] Update `GemmaDslToolCallIntegrationTest` (or add a sibling test) to exercise a scripted `<|think|>` → `<|tool_call>` sequence.

### Group 4 — kgemma CLI tools + E2B smoke test · `feature/ISSUE-69-kgemma-tools-e2b` ✅
**Issue**: [#69](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/69). **Goal**: drive tool calling from the CLI against a real checkpoint.

- [x] (a) `kgemma --tools=<names>` flag parses a comma-separated list against a new `DefaultTools` factory (`calculator`, `list_files`). Default (flag absent) keeps prior behavior of calculator-only. Unknown names print an error with the available set. `--tools` without `--agent` is rejected early. `ListFilesTool` promoted from `internal` to `public` in kllama (api baseline refreshed).
- [x] (g) `Gemma4E2BToolCallSmokeTest` in `kgemma/jvmTest` — gated on `GEMMA4_E2B_MODEL_PATH`. Loads the real GGUF via DSL NATIVE_OPTIMIZED, runs a calculator-prompt round, asserts the raw output contains `<|tool_call>` / `<tool_call|>` and that `Gemma4ChatTemplate.parseToolCalls` recovers a `calculator` call. Skips cleanly when the env var is unset so CI stays green. If the test surfaces a grammar mismatch, that's the signal to write `gemma4-research/findings/tool_calling.md` (Group 5) before adjusting the template.

### Group 5 — Research spec doc · no GH issue (local `gemma4-research`)
**Location**: `gemma4-research/findings/tool_calling.md`. Tracked here only.

- [ ] (h) Write `tool_calling.md` capturing: Gemma 4 chat-template grammar (`<|turn>`, `<|tool>`, `<|tool_call>`, `<|think|>`), difference from Gemma 2/3 legacy `functionCall` JSON, constraints observed in practice, links to the transformers implementation.

## Upstream (SKaiNET) — conditional

No upstream work is planned up front. If during any group we hit a compute / op / framework bug in SKaiNET:

1. Open an issue on `SKaiNET-developers/SKaiNET`.
2. `git checkout -b feature/ISSUE-<M>-<slug>` from `develop` in SKaiNET.
3. Fix + `./gradlew publishToMavenLocal` (unsigned).
4. In transformers, temporarily switch the composite-build `includeBuild("../SKaiNET")` off and depend on the published `mavenLocal()` artifact to prove the fix, then restore composite build once upstream PR merges.
5. Transformers commit footer: `Upstream-fix: SKaiNET-developers/SKaiNET#<M>`.

## Execution order

1. Group 2 (smallest, clears deprecated path) →
2. Group 1 (robustness — strengthens foundation before thinking-mode adds complexity) →
3. Group 3 (thinking mode) →
4. Group 4 (CLI + real-E2B — validates the whole stack) →
5. Group 5 (research doc — reflects what shipped).

## Pre-flight

- [ ] **Clean transformers working tree**: `feature/gemma4` currently has 12 dirty files (Phase 5f.6/6b leftovers, untracked `GemmaDslToolCallIntegrationTest.kt`, `GemmaDslPleTest.kt`, etc.). Must be committed, stashed, or triaged before cutting new branches from `develop` to avoid bleeding changes across PRs.
