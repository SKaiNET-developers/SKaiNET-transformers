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

### Group 2 — ChatSession polish · `feature/ISSUE-63-chat-session-polish` ✅
**Issue**: [#63](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/63). **Goal**: make the Gemma 4 DSL path the canonical surface.

- [x] (d) Parameterize system prompt on `ChatSession` — `runSingleTurn(systemPrompt = ...)` + constructor-level `defaultSystemPrompt` default (`ChatSession.DEFAULT_SYSTEM_PROMPT`).
- [x] (e) Remove deprecated `Gemma4Runtime`: deleted `Gemma4Runtime.kt` + `GemmaRuntimeParityTest.kt`; pruned `loadRuntime*` / `buildRuntime` from `Gemma4Ingestion`; removed `--runtime=handcoded` and the `RuntimeKind` enum from `kgemma` CLI.

### Group 3 — Thinking mode · `feature/ISSUE-<N>-thinking-mode`
**Issue repo**: SKaiNET-transformers. **Goal**: emit/consume `<|think|>...<think|>` blocks.

- [ ] (b) `Gemma4ChatTemplate`: parse `<|think|>` blocks alongside tool calls; buffer separately from user-visible content.
- [ ] `AgentLoop`: expose thinking output via an `AgentListener` callback (no leak into assistant message).
- [ ] Update `GemmaDslToolCallIntegrationTest` (or add a sibling test) to exercise a scripted `<|think|>` → `<|tool_call>` sequence.

### Group 4 — kgemma CLI tools + E2B smoke test · `feature/ISSUE-<N>-kgemma-tools-e2b`
**Issue repo**: SKaiNET-transformers. **Goal**: drive tool calling from the CLI against a real checkpoint.

- [ ] (a) `kgemma` `--tools=<name1>,<name2>` flag + default tool registry (calculator, file-read, time — keep set minimal). Wire through `Main.kt` to `ChatSession`.
- [ ] (g) Real-E2B smoke test: load actual Gemma 4 E2B checkpoint (not synthetic weights), run prompt that should elicit a tool call, assert model natively emits `<|tool_call>` format and round-trip completes. Gate test behind env var (checkpoint path) so CI stays green without the weights.

### Group 5 — Research spec doc · no GH issue (local `gemma4-research`) ✅
**Location**: `gemma4-research/findings/tool_calling.md` (committed as root commit `a609e46` of the research repo).

- [x] (h) `tool_calling.md` captures the delimiter-family table (`<|turn>`/`<turn|>`, `<|tool>`/`<tool|>`, `<|tool_call>`/`<tool_call|>`, `<|tool_response>`/`<tool_response|>`, `<|think>`/`<think|>`), differences from Gemma 2/3 (legacy `functionCall` JSON, `<start_of_turn>`/`<end_of_turn>`, non-native system role), implementation pointers into `llm-agent` + the kgemma smoke test, and three open questions for the E2B smoke test to answer (thinking grammar, tokenizer-level specialness, tool-result framing).

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

- [x] **Clean transformers working tree**: Phase 6b follow-up committed as `6c2d729` on `feature/gemma4` bundling the e2e integration test, PLE test, and CLI wiring leftovers.

## GitFlow note

`develop` was 47 commits behind when this plan was drafted, so all sub-feature branches were cut from `feature/gemma4` and PR back into `feature/gemma4` (the integration branch for the Gemma 4 epic). When `feature/gemma4` eventually merges to `develop`, the whole stack ships together.

## Shipped PRs

| # | Group | Status |
|---|-------|--------|
| [#63](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/63) / [PR #64](https://github.com/SKaiNET-developers/SKaiNET-transformers/pull/64) | 2 — ChatSession polish | open |
| [#65](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/65) / [PR #66](https://github.com/SKaiNET-developers/SKaiNET-transformers/pull/66) | 1 — tool-call robustness | open |
| [#67](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/67) / [PR #68](https://github.com/SKaiNET-developers/SKaiNET-transformers/pull/68) | 3 — thinking mode | open |
| [#69](https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/69) / [PR #70](https://github.com/SKaiNET-developers/SKaiNET-transformers/pull/70) | 4 — kgemma tools + E2B smoke | open |
| (none — local) | 5 — research spec | committed in `gemma4-research` as `a609e46` |
