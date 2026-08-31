# Gemma 4 generation is broken: garbage output + ~0.04 tok/s

## RESOLVED

Fixed upstream in the SKaiNET engine repo, PR **#1221** (`fix/gemma4-registries-and-fp32-kernels`,
merged commit `220c77de` — `fix(io,backend): gemma4 tokenizer/arch registry, streaming decodeToken,
observable dispatch fallback`). Verified fixed live in EdgeTranslator after republishing the engine
(`develop` past `220c77de`) and SKaiNET-transformers to mavenLocal as `0.52.0` and rebuilding — Gemma
4 (E2B, Q4_K_M) now translates correctly through the full app path
(`Gemma4ChatModel.fromGguf` → `StreamingChatModel.stream`). Leaving the rest of this file as-is
below as the investigation record (repro, what was ruled out, hypotheses) in case the same
symptom resurfaces on a different Gemma variant/quantization.

## Summary

Gemma 4 (E2B, Q4_K_M GGUF) generation through this stack produces no usable text and runs at
roughly 25 seconds per token. This was found while wiring Gemma into EdgeTranslator's SkaiNet
engine (generalized `(family, tier)` selector, `SkaiNetFamily.GEMMA`) and reproduces two
independent ways:

1. **Through the app** (`Gemma4ChatModel.fromGguf` → `StreamingChatModel.stream(ChatRequest(...))`,
   used by `EdgeTranslator`'s `SkaiNetLlm.jvm.kt`): every `generate()` call completes with an
   **empty string** — the stream emits at least one chunk but its `delta` is empty, and the whole
   call returns `""`.
2. **Through the bare `kgemma` CLI** (`llm-runtime/kgemma/src/jvmMain/kotlin/sk/ainet/apps/kgemma/
   cli/Main.kt`, raw `runtime.generate()` path, no chat template, no `--agent`): produces a
   **repeated literal special-token string**, not real text, and reports a token rate low enough
   to be effectively unusable.

Since the failure reproduces with and without the chat-template/`Gemma4ChatModel` wrapper layer,
the bug is most likely in the underlying Gemma 4 network/tokenizer/kernel path itself
(`Gemma4Ingestion`, `GGUFTokenizer`, or the matmul kernel dispatch for this architecture), not in
`Gemma4ChatModel`'s message formatting.

**Llama 3.2 (1B, Q4_K_M) through the exact same app/session-manager code path works correctly** —
real streaming text, normal speed. This isolates the problem to the Gemma 4 model/engine path
specifically, not the surrounding session/streaming machinery (both families go through the same
`SkaiNetLlm.jvm.kt` → `SkaiNetTranslator` → `AppViewModel` plumbing in EdgeTranslator).

## Repro

Built from a local unsigned `mavenLocal` publish, version `0.51.1-SNAPSHOT`, of:
- SKaiNET engine repo at commit `e7a4d884` (`develop`, includes PR #1215, #1216, #1218 — the
  dense-FP32-over-mapped-storage kernel fix from earlier in this work).
- This repo (`SKaiNET-transformers`), branch `feat/0.51-migration`, at the commit this file is
  added in.

Model file: `unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf`, 3,106,738,272 bytes,
sha256 `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` (verified against HF's
`x-linked-etag` header and a local `shasum -a 256` — the downloaded file is confirmed correct, not
corrupted).

### CLI repro (raw `runtime.generate()`, no chat template)

```
cd SKaiNET-transformers
./gradlew --no-configuration-cache -PskainetMavenLocal :llm-runtime:kgemma:jvmRun \
  --args="/path/to/gemma-4-E2B-it-Q4_K_M.gguf \"Translate 'hello world' to German\" 16 0.0"
```

Observed output:

```
Detected model variant: GEMMA4
Loading Gemma 4 GGUF model from .../gemma-4-E2B-it-Q4_K_M.gguf via gemmaNetwork() + OptimizedLLMRuntime (DSL, engine loader, keep-packed) (streaming)...
Loading embedded GGUF tokenizer...
Tokenizer: SENTENCEPIECE (model=gemma4)
Generating 16 tokens with temperature=0.0...
---
Translate 'hello world' to German<turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|><turn|>
---
tok/s: 0.04089843640165332
```

Every one of the 16 requested tokens decoded to the literal string `<turn|>` — a real GGUF special
token being repeatedly re-selected, not a decode/formatting bug (ruled out: the surrounding prompt
echo and `---` separators render correctly, so the tokenizer's normal-text decode path works fine;
only the *model's own predictions* are degenerate).

`tok/s: 0.04` ⇒ ~24.5 seconds/token. **Caveat, not yet ruled out**: this is an average over only 16
tokens including cold JIT/native-library warmup, so it may not reflect true steady-state
throughput — no per-token timing breakdown was captured. Worth re-measuring with a larger step
count and warmup excluded before treating the absolute number as gospel. Even so, it visibly took
several minutes for 16 tokens, which is unusable regardless of the precise steady-state number.

### App repro (via `Gemma4ChatModel`, real chat template applied)

In EdgeTranslator (`shared/src/jvmMain/kotlin/dev/nucleusframework/offlinetranslator/engine/
SkaiNetLlm.jvm.kt`), the Gemma branch of `generate()`:

```kotlin
val model = chatModel ?: error("SkaiNetLlm.generate called before load()")
val request = ChatRequest(messages = listOf(Message.system(system), Message.user(user)))
val acc = StringBuilder()
model.stream(request).collect { chunk ->
    acc.append(chunk.delta)
    onPartial(acc.toString())
}
acc.toString()
```

Temporary diagnostic logging (since removed) around this call, driven by real UI interaction in the
running app, showed:

```
[diag] translate() called, path=.../gemma-4-E2B-it-Q4_K_M.gguf
[diag] mutex acquired, ensuring loaded
[diag] model loaded, starting generate
[diag] partial received, len=0
[diag] generate returned, len=0
```

i.e. `model.stream(request)` emits at least one chunk (so the stream isn't simply hanging/empty),
but `chunk.delta` is empty on it, and the whole call returns `""`. The delay between "starting
generate" and "generate returned" was long enough to be indistinguishable from a hang in casual
testing — consistent with the ~25s/token rate seen in the CLI repro, if the first (and only)
predicted token was itself something filtered out before reaching `delta` (e.g. an immediate
EOS/stop, or the same kind of special-token degenerate prediction seen in the CLI repro).

For comparison, the same diagnostic logging against **Llama 3.2** (`Llama-3.2-1B-Instruct-Q4_K_M.
gguf`, same session/mutex/streaming machinery, same machine) showed normal incremental growth
(`len=5,6,10,14,18,19,23,25,26...`) at a normal pace — confirming the surrounding app-level
streaming/session code is not at fault.

## What's already ruled out

- **Not a download/corruption issue.** File size and SHA-256 verified exactly against HF's own
  `x-linked-etag` for the blob.
- **Not EdgeTranslator's session/streaming plumbing.** Llama 3.2 works correctly through the
  identical code path (`SkaiNetLlm.jvm.kt` → `SkaiNetTranslator` → `AppViewModel`).
- **Not (solely) a chat-template/message-formatting problem in `Gemma4ChatModel`.** The raw CLI
  path bypasses `Gemma4ChatModel`/`ChatRequest`/chat templates entirely (calls
  `InferenceRuntime.generate()` directly on the raw prompt string) and still produces degenerate
  special-token output, so whatever's wrong is upstream of the chat-template layer.
- **Not obviously an OOM/crash.** No exception was thrown in either repro; both complete cleanly,
  just with wrong/empty output.

## Root cause (per PR #1221's description/commit)

`fix(io,backend): gemma4 tokenizer/arch registry, streaming decodeToken, observable dispatch
fallback` — the fix landed in the engine's tokenizer/architecture registry and the streaming
decode-token path, matching hypothesis (1) below almost exactly (a special-token mapping/registry
gap for the `gemma4` architecture). See that commit's diff in the SKaiNET engine repo for the
precise mechanism if more detail is needed later.

## Not yet investigated (starting points for the next session)

1. **Tokenizer special-token mapping.** `<turn|>` looks like a mangled/wrong rendering of what
   should probably be `<start_of_turn>`/`<end_of_turn>` (Gemma's real chat-turn delimiters,
   confirmed via `tokenizer.ggml.eos_token_id` for other Gemma variants). Worth checking whether
   `GGUFTokenizer`'s SentencePiece vocab for `model=gemma4` has a mismatched/off-by-one token ID
   for these special tokens, or whether the *decode* side is fine but the *model's own logits* are
   what's degenerate (in which case the tokenizer is a red herring and the bug is in weight
   loading/kernel dispatch).
2. **K-quant kernel dispatch for Gemma 4's specific tensor shapes.** Earlier in this work,
   `KernelDispatch.find(KernelKey.matmul(activation, weight))` was confirmed to resolve to
   `ffm-rowmajor-Q4_K` for one specific Gemma4 weight (an attention Q-proj) via a throwaway
   diagnostic — that only proves ONE tensor's dispatch key resolves to the fast kernel, not that
   *every* matmul actually taken during a real generation loop does. Worth re-running that kind of
   diagnostic across all of Gemma4's tensor roles (attention, MLP, embedding, lm_head — Gemma
   architectures notably tie or specially handle the embedding/lm_head weight) to see if some
   subset silently falls through to `ReferenceMatmulKernel`, which would fully explain the
   catastrophic slowness (and could plausibly also explain garbage output if a *different* subset
   falls through to a kernel that decodes correctly but computes wrong values for Gemma4's
   particular block layout).
3. **`Gemma4LoadConfig`/`gemmaNetwork()` architecture parameters.** Confirm the network config
   (head count, head dim, sliding-window pattern, RoPE base, norm epsilon, etc. — Gemma 4 mixes
   local/global attention layers) being built for E2B actually matches what `unsloth/
   gemma-4-E2B-it-GGUF`'s metadata declares. A subtly wrong architecture parameter (e.g. sliding
   window applied to the wrong layers, wrong RoPE scaling) would produce exactly this kind of
   "runs without crashing but predicts garbage" failure.
4. **BOS/EOS/stop-token handling in the raw `generate()` loop.** If the very first predicted token
   in the app's `Gemma4ChatModel.stream()` path is an immediate stop/EOS, `delta` would correctly
   be empty and the stream would correctly end — that's "working as designed" for a *degenerate
   prediction*, but still points back to (1)–(3) as the root cause, not a bug in the stop-handling
   itself.

(Superseded by PR #1221 — kept for reference in case a related symptom shows up again on a
different Gemma variant/quantization that #1221 didn't cover.)

## Environment

- macOS, Apple Silicon (arm64), JDK 21 preview (`--enable-preview --add-modules
  jdk.incubator.vector`), SIMD-accelerated CPU kernel pack confirmed active (`[SKaiNET] Using
  SIMD-accelerated CPU operations (Vector API)` printed at both load and generate time).
- SKaiNET engine `0.51.1-SNAPSHOT` (local mavenLocal, built from `develop` @ `e7a4d884`).
- SKaiNET-transformers `0.51.1-SNAPSHOT` (local mavenLocal, built from `feat/0.51-migration`).

## Unrelated, separately-noted issue found during the same session

A minor race in EdgeTranslator (not this repo) was observed while switching the active SkaiNet
family rapidly: `SkaiNetTranslator.ensureLoaded()` independently re-reads `LlmRuntime.
skainetFamily` via its `family()` closure, while the model *path* was already resolved from the
same field a moment earlier in `AppViewModel.activeModelPath()`. If the family selector changes in
that window, one `load()` call can receive a mismatched `(family, path)` pair (e.g. loading the
Gemma-architecture loader against a Llama GGUF file, or vice versa) — observed twice, both times
self-corrected on the very next `translate()` call. The loader currently accepts a
mismatched-architecture GGUF without validation/erroring, which is itself worth hardening
independently of the race. Tracked for EdgeTranslator, not SKaiNET-transformers — noted here only
because it surfaced during the same debugging session. **Still open** — not addressed by #1221.
