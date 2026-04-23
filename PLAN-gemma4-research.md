# Plan: research the unwired Gemma 4 E2B features before implementing

## Context

Phase 5f.1 (QK-Norm) and 5f.2 (sandwich norms) landed. The DSL path now
emits `"Hi relieved"` on real Gemma 4 E2B Q4_K_M instead of `"Hi ??"`, so
those two features were clearly load-bearing. Remaining Phase 5f features
are architecturally unclear — the tensor names give shape and location but
**not semantics**. Guessing wrong will move the output further from E2B, not
closer. This plan researches each before wiring.

Unknowns, in rough order of probable impact (worst-to-best):

1. `blk.N.layer_output_scale.weight [1]` — a scalar per block. Almost
   certainly multiplies *something* but where?
2. `blk.N.post_norm.weight [1536]` — a 5th norm per block on top of
   `attn_norm` + `post_attention_norm` + `ffn_norm` + `post_ffw_norm`.
   Position in the topology unknown.
3. PLE triad: `per_layer_token_embd` [8960, 262144] Q6_K,
   `per_layer_model_proj` [1536, 8960] BF16, `per_layer_proj_norm` [256] F32.
   Feeds each block somehow — needs a new forward-pass primitive.
4. `blk.N.inp_gate.weight [1536, 256]` + `blk.N.proj.weight [256, 1536]`.
   Shapes suggest a 256-dim side channel. Could be PLE's block-level hook.
5. p-RoPE formula validation — the DSL uses `RoPEScaling.PROPORTIONAL` with
   `base × factor^(rotaryDim/(rotaryDim-2))`. Need to confirm this matches
   `Gemma4AttentionBackend.applyRopeGqa` AND the reference HF implementation.

## Approach

Use `uv` to spin a Python env with authoritative reference implementations,
then for each unknown feature produce a specification document that the
next Phase 5f.N implementation PR can consume directly.

### Why `uv`

- Fast, reproducible venv creation (no system Python pollution).
- Lockfile-based, so the research is repeatable later if we need to
  re-check a claim.
- Mechanical: `uv run python -c "..."` or `uv run script.py` — no manual
  `source .venv/bin/activate` dance.

The research is READ-ONLY (inspect models, print tensors, match shapes);
nothing gets written into any SKaiNET repo until each feature ships as its
own implementation PR.

## Research environment setup

One-time setup, done in a fresh directory outside either repo to keep the
research scratchpad out of version control:

```
mkdir -p /home/miso/projects/SK-tran-0.19/gemma4-research
cd /home/miso/projects/SK-tran-0.19/gemma4-research
uv init --python 3.11  # Gemma 4 transformers needs >=3.10; pin for reproducibility
uv add transformers torch "accelerate>=0.33" gguf "numpy<2" sentencepiece
# Optional, only if we want to cross-check against llama.cpp's own decoder:
# uv add llama-cpp-python
```

Gemma 4 E2B-it is gated on HuggingFace — if the `AutoModel.from_pretrained`
call fails with a 401, we fall back to loading from the local GGUF via
`gguf.GGUFReader` (shape + raw tensor bytes only, no forward pass from
that source) and compare structures against the published Gemma 4 HF
`modeling_gemma.py` source fetched as a standalone file.

## Per-feature research tasks

Each task produces a single markdown file in
`/home/miso/projects/SK-tran-0.19/gemma4-research/findings/` summarising
what the feature does, where in the forward pass it applies, and any edge
cases. These are the INPUTS to the matching Phase 5f.N implementation PRs.

### R1 — `layer_output_scale` [1 per block]

**Question.** What does `layer_output_scale[k]` multiply, and when?

**Commands.**
```
uv run python - <<'PY'
from transformers import AutoConfig, AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="float32")
# Locate the parameter that corresponds to layer_output_scale.
for n, p in m.named_parameters():
    if "layer_output_scale" in n or "output_scale" in n or n.endswith(".scale"):
        print(n, tuple(p.shape))
# Then print the source of the module that owns it:
import inspect
for mod_name, mod in m.named_modules():
    if any("layer_output_scale" in n for n, _ in mod.named_parameters(recurse=False)):
        print(mod_name, type(mod).__name__)
        print(inspect.getsourcefile(type(mod)))
PY
```

**Expected output.** File path to `modeling_gemma4.py` + owning module
name. Then `grep -n layer_output_scale <path>` to find the forward-pass
use. Record: which tensor it multiplies, where in the residual chain,
whether it's applied pre- or post-sandwich-norms.

**Deliverable.** `findings/layer_output_scale.md` — one paragraph on
semantics, exact pseudocode for where it fits in our
`HybridTransformerBlock.directForward` ordering.

### R2 — `post_norm` [1536 per block]

**Question.** In our current stage order
`attn_norm → MHA → post_attention_norm → residual → ffn_norm → FFN →
post_ffw_norm → residual`, where does the fifth norm (`post_norm`) fit?

**Commands.**
```
uv run python - <<'PY'
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="float32")
block = m.language_model.model.layers[0]
import inspect
print(inspect.getsource(type(block).forward))
PY
```

**Expected output.** The Gemma 4 decoder-layer forward pass with every
norm call site visible.

**Deliverable.** `findings/post_norm.md` — state which of: (a) a final
per-block norm wrapping the full block output after both residuals,
(b) a norm inside the PLE path (see R3/R4), or (c) something else.

### R3 — PLE (Per-Layer Embedding)

**Question.** How are `per_layer_token_embd` [8960, 262144],
`per_layer_model_proj` [1536, 8960], and `per_layer_proj_norm` [256]
wired into each transformer block?

**Commands.**
```
uv run python - <<'PY'
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="float32")
import inspect
top = m.language_model.model
print(inspect.getsource(type(top).forward))
PY
```

And trace which parameters are touched on a forward pass of a single
token:

```
uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="float32")
m.eval()
x = tok("Hi", return_tensors="pt").input_ids
hooks = {}
def hook(n):
    def _h(mod, inp, out):
        shp = tuple(out.shape) if hasattr(out, "shape") else None
        hooks[n] = shp
    return _h
for n, mod in m.named_modules():
    if "per_layer" in n:
        mod.register_forward_hook(hook(n))
with torch.no_grad():
    m(x)
for k, v in hooks.items():
    print(k, v)
PY
```

**Deliverable.** `findings/ple.md` — sequence diagram (text) of where
PLE injects into the decoder, with exact tensor shapes at each step.
Note whether PLE is per-layer additive or gating.

### R4 — `inp_gate` [1536, 256] + `proj` [256, 1536]

**Question.** These two per-block matrices look like a 1536 → 256 → 1536
projection loop. The 256 dim matches `per_layer_proj_norm`'s shape,
suggesting inp_gate and proj are the PLE's block-level hook.

**Commands.** Same forward-hook trick as R3, filter for `inp_gate` /
`proj` modules. Also inspect `DecoderLayer.__init__` to see how these
modules are connected to PLE projections.

**Deliverable.** `findings/ple_block_hook.md` — exact math of the 1536 →
256 → 1536 path and how it combines with the main residual stream.

### R5 — p-RoPE formula validation

**Question.** Does our `RoPE.kt`'s proportional scaling match the
reference? Current formula (paraphrased):
`effective_base = base × factor^(rotaryDim / (rotaryDim - 2))`.

**Commands.**
```
uv run python - <<'PY'
from transformers import AutoModelForCausalLM
import inspect
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="float32")
# Find the RotaryEmbedding module on a global layer.
rope_modules = [(n, mod) for n, mod in m.named_modules() if "rotary" in type(mod).__name__.lower() or "rope" in n.lower()]
for n, mod in rope_modules[:3]:
    print(n, type(mod).__name__, inspect.getsourcefile(type(mod)))
    print(inspect.getsource(type(mod).forward))
PY
```

Then dump the `inv_freq` tensor from the global-attention RoPE module and
compare numerically to what our `RoPE.kt` produces for the same
`(base, factor, rotaryDim, partialRotaryFactor)` configuration.

**Deliverable.** `findings/prope_formula.md` — either "our formula is
correct, locked in with a reference `inv_freq` numerical table" or "our
formula is wrong, here is the correct one plus a failing-test specimen".

### R6 — Cross-check: end-to-end prompt comparison

Once R1–R5 are known, the research dir should also contain a reference
log of Gemma 4 E2B-it's exact token output on a short prompt, so each
implementation PR has a concrete target.

```
uv run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
m = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it",
    torch_dtype="float32", device_map="cpu")
m.eval()
x = tok("Hi", return_tensors="pt").input_ids
with torch.no_grad():
    out = m.generate(x, max_new_tokens=10, do_sample=False, temperature=0.0)
print(tok.decode(out[0]))
PY
```

**Deliverable.** `findings/reference_outputs.md` — exact decoded tokens
for a handful of short prompts at `temperature=0.0`. Our DSL-path
`kgemma --runtime=dsl` should converge on these as each feature lands.

## Non-goals for this plan

- Do NOT start implementing any Phase 5f.N feature from the research
  directly. The research deliverables are specs; each spec becomes a
  separate implementation PR reviewed on its own merits.
- Do NOT depend on network access to HuggingFace in SKaiNET tests. The
  research happens in an external scratch dir; only the findings docs
  travel back into the repo (committed to `docs/` or referenced from
  implementation PRs).
- Do NOT set up `uv` inside either SKaiNET repo's build. The Kotlin
  build stays pure Gradle — Python is a research tool, not a runtime
  dependency.

## Success criteria

Plan is complete when:

1. Each of `findings/layer_output_scale.md`, `post_norm.md`, `ple.md`,
   `ple_block_hook.md`, `prope_formula.md`, `reference_outputs.md`
   exists and states the semantics with enough precision to drive the
   corresponding implementation PR.
2. The `uv` lockfile is committed to the research dir (outside SKaiNET)
   so the research environment is reproducible.
3. The remaining Phase 5f features in `PLAN-unified-pipeline.md` each
   have a link to the matching findings doc.

## Out of scope

- HuggingFace access if the Gemma 4 model is gated and we don't have a
  token. In that case, fall back to the public reference
  implementation file from the `transformers` github (which is not
  gated) plus local GGUF inspection via `gguf.GGUFReader`. The model
  weights are *not* strictly required to read the forward-pass source.
- Porting any Python code into Kotlin. This plan is research-only.
- Modifying SKaiNET tests or runtimes — those changes wait for their
  own implementation PRs.
