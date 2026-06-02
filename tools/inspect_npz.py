#!/usr/bin/env python3
"""
Inspect NPZ arrays for GD32 Embedded AI quantization.

Examples:
    python tools/inspect_npz.py D:/EDCP/fastestdet_quant_100.npz
    python tools/inspect_npz.py D:/EDCP/fastestdet_quant_100.npz --expect-key data --expect-shape 100,352,352,3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit("Missing dependency: numpy. Install it with: pip install numpy") from exc


def parse_shape(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expect-shape must look like 100,352,352,3") from exc


def format_shape(shape) -> str:
    return "[" + ", ".join(str(item) for item in shape) + "]"


def inspect_npz(path: Path, expect_key: Optional[str], expect_shape: Optional[List[int]]) -> int:
    if not path.exists():
        raise SystemExit(f"NPZ not found: {path}")

    with np.load(path) as data:
        keys = list(data.files)
        print("=== NPZ ===")
        print("path:", path)
        print("keys:", keys)
        if expect_key and expect_key not in keys:
            print(f"ERROR: expected key {expect_key!r}, but it is not present")
            return 1

        status = 0
        for key in keys:
            array = data[key]
            print()
            print(f"--- {key} ---")
            print("shape:", format_shape(array.shape))
            print("dtype:", array.dtype)
            if array.size:
                finite = np.isfinite(array)
                print("finite:", bool(finite.all()))
                print("min:", float(np.nanmin(array)))
                print("max:", float(np.nanmax(array)))
                print("mean:", float(np.nanmean(array)))
            else:
                print("finite: n/a")
                print("min: n/a")
                print("max: n/a")
                print("mean: n/a")

            if expect_key and key == expect_key and expect_shape is not None:
                if list(array.shape) != expect_shape:
                    print(
                        "ERROR: shape mismatch; "
                        f"expected {format_shape(expect_shape)}, got {format_shape(array.shape)}"
                    )
                    status = 1
                else:
                    print("shape_check: ok")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="NPZ file to inspect.")
    parser.add_argument("--expect-key", help="Optional key that must be present.")
    parser.add_argument(
        "--expect-shape",
        type=parse_shape,
        help="Optional shape check, for example 100,352,352,3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return inspect_npz(args.npz, args.expect_key, args.expect_shape)


if __name__ == "__main__":
    raise SystemExit(main())
