# Qwen3 ground-truth reference

Phase-4 readiness gate for the DSL Qwen path. Tracks #118.

This directory contains a `uv`-managed Python harness that produces
**llama.cpp reference logits** for a small set of fixed prompts on a
real Qwen3 GGUF. The Kotlin smoke test in
`llm-runtime/kllama/src/jvmTest/.../QwenHfReferenceParityTest.kt` (added
in this PR) runs the same GGUF through the SKaiNET-transformers DSL
Qwen path and compares.

If the DSL agrees with `llama.cpp` on greedy top-1 tokens for the first
N steps, Phase 4 (CLI swap) can ship. If not, we have a concrete
correctness failure to root-cause — much more actionable than the
synthetic DSL-vs-legacy parity that #114 spent.

## Why llama.cpp, not HuggingFace `transformers`

HF `transformers` loads from native HF weight format, not GGUF.
Comparing the DSL path (which loads from GGUF) against `transformers`
(which loads from HF format) mixes two error sources — weight
conversion correctness vs. inference-engine correctness.
`llama-cpp-python` loads the **same GGUF bytes** the SKaiNET CLI
consumes; the comparison is apples-to-apples and any divergence is
in the engine, not the data.

`gradienttracer` / `skainet-ground-truth` would have been the obvious
host for this — they already use `uv` + PyTorch + GGUF for SKaiNET
operator-level reference data — but per the SKaiNET upstream scope
boundary (no LLM/transformer code), a Qwen3 forward-pass reference
belongs in this repo, not theirs. The harness follows the same
*pattern* (uv, GGUF outputs, suite-style organisation) without the
cross-repo coupling.

## Layout

| File | Role |
|---|---|
| `pyproject.toml` | `uv` project config; pins `llama-cpp-python`. |
| `prompts.json` | Fixed prompts + comparison parameters (top-K, steps). |
| `generate_reference.py` | Runs llama.cpp, captures top-K logits per step, writes JSON. |
| `results/` | Generated fixtures (gitignored — too large + model-pin-dependent to commit). |

## Regenerating the reference

You need a Qwen3 GGUF locally. The CLI smoke harness defaults to
`~/.lmstudio/models/...`; reuse the same path.

```bash
cd tests/ground-truth/qwen3
uv sync
uv run python generate_reference.py \
    --model "$HOME/.lmstudio/models/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf" \
    --out results/qwen3-1.7B-q8_0.json
```

The Kotlin parity test reads `results/qwen3-1.7B-q8_0.json` (or
whatever path you configure via the `QWEN3_REFERENCE_FIXTURE`
environment variable) and skips gracefully if the fixture is missing —
so the test is safe to commit even before any maintainer has
regenerated the reference.

## When to regenerate

- The pinned `llama-cpp-python` version changes meaningfully.
- `prompts.json` changes.
- A different reference Qwen3 GGUF (different size / quant) is desired.
- llama.cpp ships a non-trivial Qwen3 inference fix.

The fixture file itself is gitignored. Phase-4 readiness is then
something each maintainer can verify locally:

1. Generate the reference once.
2. Run the Kotlin parity test.
3. Inspect divergence — top-1 agreement on the first 8 greedy steps is
   the pass criterion.

## Pass / fail criterion

For each prompt in `prompts.json`:
- DSL top-1 token at each of the first `steps` greedy steps must equal
  llama.cpp's top-1 at the same step. **Strict equality** — no
  numerical tolerance, because top-1 is a discrete decision.
- Optional: top-K logit KL-divergence below threshold. Not asserted by
  default; useful as diagnostic when top-1 disagrees.

If any prompt fails, the parity test prints the divergent step plus
the top-K from both engines for that step, then fails. That output is
the input to a #114-style root-cause investigation — but this time
against a real-correctness reference, not a synthetic one.
