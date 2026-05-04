"""
Generate ground-truth Qwen3 reference fixtures via llama.cpp.

Why llama.cpp and not HF transformers
-------------------------------------
HF transformers loads from native HuggingFace weight format, not GGUF.
Comparing the DSL path (which loads from GGUF) against HF-on-original-
weights mixes two error sources — weight conversion vs. inference-engine
correctness. Using `llama-cpp-python` to load the SAME GGUF that the
SKaiNET-transformers CLI uses keeps the comparison apples-to-apples:
same bytes in, two different inference engines, compare outputs.

Usage
-----
    cd tests/ground-truth/qwen3
    uv sync
    uv run python generate_reference.py \\
        --model ~/.lmstudio/models/Qwen3-1.7B-Q8_0.gguf \\
        --out results/qwen3-1.7B-q8_0.json

Re-run whenever:
    - the prompts.json set changes
    - the target reference model file changes
    - llama-cpp-python pin changes meaningfully

The generated JSON is the input to QwenHfReferenceParityTest on the
Kotlin side.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class StepRef:
    step: int
    top1_id: int
    top1_logit: float
    topk_ids: list[int]
    topk_logits: list[float]


@dataclass
class PromptRef:
    id: str
    text: str
    prompt_token_ids: list[int]
    steps: list[StepRef]


@dataclass
class ReferenceFile:
    model_path: str
    model_filename: str
    n_ctx: int
    temperature: float
    topk: int
    n_steps: int
    llama_cpp_python_version: str
    prompts: list[PromptRef]


def _greedy_step_ref(
    llm,
    prompt_ids: list[int],
    n_steps: int,
    topk: int,
) -> tuple[list[StepRef], list[int]]:
    """Run greedy decoding for n_steps, capturing top-K per step."""
    steps: list[StepRef] = []
    generated: list[int] = []

    # Prefill the prompt. llama.cpp returns logits for the *last* token
    # automatically once `eval` returns; we then sample greedily and feed
    # back one token at a time.
    llm.reset()
    llm.eval(prompt_ids)

    for step in range(n_steps):
        logits = np.asarray(llm.scores[len(prompt_ids) + step - 1], dtype=np.float64)
        top_idx = np.argpartition(-logits, topk)[:topk]
        # Sort top-K by logit descending so position 0 is the argmax.
        top_idx = top_idx[np.argsort(-logits[top_idx])]
        top1_id = int(top_idx[0])
        steps.append(
            StepRef(
                step=step,
                top1_id=top1_id,
                top1_logit=float(logits[top1_id]),
                topk_ids=[int(i) for i in top_idx],
                topk_logits=[float(logits[i]) for i in top_idx],
            )
        )
        generated.append(top1_id)
        llm.eval([top1_id])

    return steps, generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to Qwen3 GGUF file (e.g. Qwen3-1.7B-Q8_0.gguf).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parent / "prompts.json",
        help="Path to prompts.json (default: ./prompts.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSON path under results/.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=512,
        help="llama.cpp context window for prefill+steps (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="llama.cpp seed; greedy at temp=0 should be deterministic regardless.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1

    spec = json.loads(args.prompts.read_text())
    meta = spec["_meta"]
    prompts_spec = spec["prompts"]

    # Imported lazily so `--help` works without llama-cpp-python installed.
    from llama_cpp import Llama, __version__ as lcpp_version

    print(f"Loading {args.model.name} via llama.cpp …")
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        seed=args.seed,
        logits_all=True,  # we need per-step logits, not just the last
        verbose=False,
    )

    refs: list[PromptRef] = []
    for p in prompts_spec:
        # llama.cpp tokenizer is the GGUF-embedded one — same vocab the
        # SKaiNET-transformers CLI sees. Don't add BOS here; the DSL path
        # doesn't auto-prepend either, so both paths see the same prefix.
        token_ids = llm.tokenize(p["text"].encode("utf-8"), add_bos=False)
        steps, _ = _greedy_step_ref(
            llm,
            prompt_ids=token_ids,
            n_steps=meta["steps"],
            topk=meta["topK"],
        )
        refs.append(
            PromptRef(
                id=p["id"],
                text=p["text"],
                prompt_token_ids=[int(t) for t in token_ids],
                steps=steps,
            )
        )
        print(f"  {p['id']}: {len(token_ids)} prompt tokens → top1 sequence "
              f"{[s.top1_id for s in steps]}")

    out = ReferenceFile(
        model_path=str(args.model.resolve()),
        model_filename=args.model.name,
        n_ctx=args.n_ctx,
        temperature=meta["temperature"],
        topk=meta["topK"],
        n_steps=meta["steps"],
        llama_cpp_python_version=lcpp_version,
        prompts=refs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(out), indent=2) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
