#!/usr/bin/env python3
"""Inspect GGUF model files to discover tensor names, shapes, and metadata.

Usage:
    uv run inspect_model.py path/to/model.gguf
    uv run inspect_model.py path/to/model.gguf --filter altup
    uv run inspect_model.py path/to/model.gguf --metadata-only
"""

import argparse
import sys
from pathlib import Path

from gguf import GGUFReader


def inspect_metadata(reader: GGUFReader) -> None:
    """Print all metadata fields."""
    print("=" * 70)
    print("METADATA")
    print("=" * 70)

    for field in reader.fields.values():
        # Skip internal fields
        if field.name.startswith("GGUF."):
            continue
        try:
            parts = field.parts
            if len(parts) > 0:
                # Try to extract a readable value
                data_part = parts[field.data[0]] if field.data else parts[-1]
                if hasattr(data_part, "tolist"):
                    val = data_part.tolist()
                    if isinstance(val, list) and len(val) > 20:
                        val = f"[{len(val)} items] {val[:5]}..."
                else:
                    val = data_part
                print(f"  {field.name}: {val}")
        except Exception:
            print(f"  {field.name}: <unreadable>")


def inspect_tensors(reader: GGUFReader, name_filter: str | None = None) -> None:
    """Print tensor names, shapes, and types."""
    print()
    print("=" * 70)
    print("TENSORS")
    print("=" * 70)

    total_params = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name_filter and name_filter.lower() not in name.lower():
            continue

        shape = tensor.shape.tolist()
        dtype = tensor.tensor_type.name
        n_params = 1
        for s in shape:
            n_params *= s
        total_params += n_params

        print(f"  {name:<60} {str(shape):<25} {dtype:<10} ({n_params:,} params)")

    print(f"\nTotal tensors: {len(reader.tensors)}")
    print(f"Total params (filtered): {total_params:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GGUF model files")
    parser.add_argument("model_path", help="Path to .gguf file")
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter tensor names (case-insensitive substring match)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only show metadata, skip tensor listing",
    )

    args = parser.parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"File not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_path} ...")
    reader = GGUFReader(str(model_path))

    inspect_metadata(reader)

    if not args.metadata_only:
        inspect_tensors(reader, args.filter)


if __name__ == "__main__":
    main()
