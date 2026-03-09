#!/usr/bin/env python3
"""Download Gemma 3n models from HuggingFace Hub.

Usage:
    uv run download_model.py --model ggml-org/gemma-3n-E4B-it-GGUF --quant Q8_0
    uv run download_model.py --model google/gemma-3n-E4B --format safetensors
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def download_gguf(repo_id: str, quant: str, output_dir: Path) -> None:
    """Download a specific GGUF quantization from a repo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching file pattern
    filename = None
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id)
    candidates = [f for f in files if quant in f and f.endswith(".gguf")]

    if not candidates:
        print(f"No GGUF file matching '{quant}' found in {repo_id}")
        print(f"Available files: {[f for f in files if f.endswith('.gguf')]}")
        sys.exit(1)

    filename = candidates[0]
    print(f"Downloading {filename} from {repo_id}...")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
    )
    print(f"Downloaded to: {path}")


def download_safetensors(repo_id: str, output_dir: Path) -> None:
    """Download full SafeTensors model (all shards + config)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} (SafeTensors)...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    print(f"Downloaded to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gemma 3n models")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo ID (e.g. ggml-org/gemma-3n-E4B-it-GGUF)",
    )
    parser.add_argument(
        "--quant",
        default=None,
        help="GGUF quantization type (e.g. Q8_0, F16). Required for GGUF repos.",
    )
    parser.add_argument(
        "--format",
        choices=["gguf", "safetensors"],
        default="gguf",
        help="Model format to download",
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory (default: models/)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.format == "gguf":
        if not args.quant:
            parser.error("--quant is required for GGUF downloads")
        download_gguf(args.model, args.quant, output_dir)
    else:
        download_safetensors(args.model, output_dir)


if __name__ == "__main__":
    main()
