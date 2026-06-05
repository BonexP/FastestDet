#!/usr/bin/env python3
"""
Run deterministic FastestDet TFLite vectors for EDCP bare-metal validation.

The generated MCU firmware currently tests four int8 input patterns:

- constant_1
- constant_0
- ramp
- checker

This script runs the same vectors through the quantized TFLite model on a PC and
prints the same summary fields used by the MCU log. It is meant for L3
PC-vs-MCU validation after L1/L2 bare-metal stability has passed.

Usage:
    python tools/run_fastestdet_tflite_vectors.py
    python tools/run_fastestdet_tflite_vectors.py path/to/FastestDet_256.tflite
    python tools/run_fastestdet_tflite_vectors.py --mcu-log "docs/[COM4] 2026-06-04 17-49-41.531.log"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CASE_TO_MCU_NAME = {
    "constant_1": "L1_CONSTANT_1",
    "constant_0": "L2_CONSTANT_0",
    "ramp": "L2_RAMP",
    "checker": "L2_CHECKER",
}

DEFAULT_CASES = tuple(CASE_TO_MCU_NAME.keys())


def repo_default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "components"
        / "EmbeddedAI_q5_int8_256_stack4000"
        / "GD_Embedded_AI"
        / "User_model"
        / "cur_tflite"
        / "FastestDet_256.tflite"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "Missing Python package: numpy. Install it with `pip install numpy`, "
            "or run this script in the training environment that already has numpy."
        ) from exc
    return np


def create_interpreter(model_path: Path) -> Tuple[str, Any]:
    try:
        import tensorflow as tf  # type: ignore

        return "tensorflow", tf.lite.Interpreter(model_path=str(model_path))
    except Exception as tf_exc:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            return "tflite_runtime", Interpreter(model_path=str(model_path))
        except Exception as rt_exc:  # pragma: no cover - depends on host environment
            raise RuntimeError(
                "Missing TFLite runtime. Install one of these options:\n"
                "  pip install tensorflow\n"
                "  pip install tflite-runtime numpy\n\n"
                f"tensorflow import error: {tf_exc}\n"
                f"tflite_runtime import error: {rt_exc}"
            ) from rt_exc


def dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype))


def quantization_text(detail: Dict[str, Any]) -> str:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return f"scale={scale} zero_point={zero_point}"


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def make_input_vector(np: Any, case_name: str, shape: Sequence[int], dtype: Any) -> Any:
    if dtype != np.int8:
        raise RuntimeError(
            f"Expected int8 TFLite input to mirror the MCU buffer, got {dtype_name(dtype)}. "
            "If the model was regenerated, first confirm the generated firmware input dtype."
        )

    size = product(shape)

    if case_name == "constant_1":
        flat = np.ones(size, dtype=np.int8)
    elif case_name == "constant_0":
        flat = np.zeros(size, dtype=np.int8)
    elif case_name == "ramp":
        flat = ((np.arange(size, dtype=np.int32) & 0xFF) - 128).astype(np.int8)
    elif case_name == "checker":
        indexes = np.arange(size, dtype=np.int32)
        flat = np.where((indexes & 1) == 0, 64, -64).astype(np.int8)
    else:
        raise ValueError(f"Unknown case: {case_name}")

    return flat.reshape(tuple(int(v) for v in shape))


def signed_view(np: Any, array: Any) -> Any:
    flat = array.reshape(-1)
    if array.dtype == np.int8:
        return flat
    if array.dtype == np.uint8:
        return flat.view(np.int8)
    raise RuntimeError(
        f"Expected int8/uint8 tensor for byte summary, got {dtype_name(array.dtype)}"
    )


def tensor_summary(np: Any, array: Any) -> Dict[str, Any]:
    signed = signed_view(np, array)
    raw_bytes = array.reshape(-1).view(np.uint8)
    xor_value = 0
    for value in raw_bytes:
        xor_value ^= int(value)

    return {
        "sum": int(np.sum(signed.astype(np.int64))),
        "xor": xor_value,
        "min": int(np.min(signed)),
        "max": int(np.max(signed)),
        "first32": [int(value) for value in signed[:32]],
        "bytes": int(raw_bytes.size),
    }


def parse_first32(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_mcu_log(path: Path) -> Dict[str, Dict[str, Any]]:
    run_re = re.compile(
        r"RUN=(?P<run>\d+)\s+CASE=(?P<case>\S+).*?"
        r"IN_SUM=(?P<in_sum>-?\d+)\s+IN_XOR=0x(?P<in_xor>[0-9A-Fa-f]+)\s+"
        r"OUT_SUM=(?P<out_sum>-?\d+)\s+OUT_XOR=0x(?P<out_xor>[0-9A-Fa-f]+)\s+"
        r"MIN=(?P<min>-?\d+)\s+MAX=(?P<max>-?\d+)"
    )
    first_re = re.compile(r"OUT_FIRST32=(?P<values>[-,\d\s]+)")

    summaries: Dict[str, Dict[str, Any]] = {}
    current_case: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = run_re.search(line)
            if match:
                if int(match.group("run")) != 0:
                    current_case = None
                    continue
                current_case = match.group("case")
                summaries[current_case] = {
                    "in_sum": int(match.group("in_sum")),
                    "in_xor": int(match.group("in_xor"), 16),
                    "out_sum": int(match.group("out_sum")),
                    "out_xor": int(match.group("out_xor"), 16),
                    "min": int(match.group("min")),
                    "max": int(match.group("max")),
                    "first32": [],
                }
                continue

            match = first_re.search(line)
            if match and current_case:
                summaries[current_case]["first32"] = parse_first32(match.group("values"))
                current_case = None

    return summaries


def compare_to_mcu(pc: Dict[str, Any], mcu: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if mcu is None:
        return None
    pc_first32 = pc["output"]["first32"]
    mcu_first32 = mcu.get("first32", [])
    pair_count = min(len(pc_first32), len(mcu_first32))
    first32_abs_diffs = [
        abs(int(pc_first32[i]) - int(mcu_first32[i])) for i in range(pair_count)
    ]
    first32_abs_sum = sum(first32_abs_diffs)
    first32_max_abs = max(first32_abs_diffs) if first32_abs_diffs else None
    first32_mae = (
        float(first32_abs_sum) / float(pair_count) if pair_count > 0 else None
    )

    return {
        "out_sum_match": pc["output"]["sum"] == mcu["out_sum"],
        "out_xor_match": pc["output"]["xor"] == mcu["out_xor"],
        "min_match": pc["output"]["min"] == mcu["min"],
        "max_match": pc["output"]["max"] == mcu["max"],
        "first32_match": pc_first32 == mcu_first32,
        "sum_delta": pc["output"]["sum"] - mcu["out_sum"],
        "min_delta": pc["output"]["min"] - mcu["min"],
        "max_delta": pc["output"]["max"] - mcu["max"],
        "first32_count": pair_count,
        "first32_exact": sum(
            1 for i in range(pair_count) if int(pc_first32[i]) == int(mcu_first32[i])
        ),
        "first32_mae": first32_mae,
        "first32_max_abs": first32_max_abs,
        "mcu": mcu,
    }


def print_summary_line(prefix: str, summary: Dict[str, Any]) -> None:
    print(
        f"{prefix}_SUM={summary['sum']} "
        f"{prefix}_XOR=0x{summary['xor']:02X} "
        f"{prefix}_MIN={summary['min']} "
        f"{prefix}_MAX={summary['max']} "
        f"{prefix}_BYTES={summary['bytes']}"
    )


def print_human(result: Dict[str, Any]) -> None:
    print("=== FILE ===")
    print(result["model_path"])
    print(f"bytes: {result['model_bytes']}")
    print(f"sha256: {result['sha256']}")
    print()

    print("=== RUNTIME ===")
    print(result["runtime"])
    print()

    print("=== TENSORS ===")
    print(
        "INPUT "
        f"name={result['input']['name']} "
        f"dtype={result['input']['dtype']} "
        f"shape={result['input']['shape']} "
        f"{result['input']['quantization']}"
    )
    print(
        "OUTPUT "
        f"name={result['output']['name']} "
        f"dtype={result['output']['dtype']} "
        f"shape={result['output']['shape']} "
        f"{result['output']['quantization']}"
    )
    print()

    for case in result["cases"]:
        print(f"=== CASE {case['pattern']} ===")
        print(f"MCU_CASE={case['mcu_case']}")
        print_summary_line("IN", case["input"])
        print_summary_line("OUT", case["output"])
        print("OUT_FIRST32=" + ",".join(str(value) for value in case["output"]["first32"]))

        compare = case.get("compare")
        if compare:
            status = {
                "sum": compare["out_sum_match"],
                "xor": compare["out_xor_match"],
                "min": compare["min_match"],
                "max": compare["max_match"],
                "first32": compare["first32_match"],
            }
            status_text = " ".join(f"{key}={'OK' if value else 'DIFF'}" for key, value in status.items())
            print(f"MCU_COMPARE {status_text}")
            first32_mae = compare["first32_mae"]
            first32_mae_text = "NA" if first32_mae is None else f"{first32_mae:.3f}"
            first32_max_abs = compare["first32_max_abs"]
            first32_max_abs_text = "NA" if first32_max_abs is None else str(first32_max_abs)
            print(
                "MCU_DIFF "
                f"sum_delta={compare['sum_delta']} "
                f"min_delta={compare['min_delta']} "
                f"max_delta={compare['max_delta']} "
                f"first32_mae={first32_mae_text} "
                f"first32_max_abs={first32_max_abs_text} "
                f"first32_exact={compare['first32_exact']}/{compare['first32_count']}"
            )
        print()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    np = import_numpy()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    runtime_name, interpreter = create_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Expected one input tensor, got {len(input_details)}")
    if len(output_details) != 1:
        raise RuntimeError(f"Expected one output tensor, got {len(output_details)}")

    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = [int(v) for v in input_detail["shape"]]
    input_dtype = input_detail["dtype"]

    mcu_summaries = parse_mcu_log(Path(args.mcu_log)) if args.mcu_log else {}

    cases = []
    for pattern in args.cases:
        vector = make_input_vector(np, pattern, input_shape, input_dtype)
        input_summary = tensor_summary(np, vector)

        interpreter.set_tensor(input_detail["index"], vector)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        output_summary = tensor_summary(np, output)

        mcu_case = CASE_TO_MCU_NAME[pattern]
        case_result = {
            "pattern": pattern,
            "mcu_case": mcu_case,
            "input": input_summary,
            "output": output_summary,
        }
        compare = compare_to_mcu(case_result, mcu_summaries.get(mcu_case))
        if compare:
            case_result["compare"] = compare

        if args.save_output_dir:
            output_dir = Path(args.save_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{pattern}.bin").write_bytes(output.reshape(-1).view(np.uint8).tobytes())

        cases.append(case_result)

    return {
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
        "runtime": runtime_name,
        "input": {
            "name": input_detail.get("name", ""),
            "dtype": dtype_name(input_dtype),
            "shape": input_shape,
            "quantization": quantization_text(input_detail),
        },
        "output": {
            "name": output_detail.get("name", ""),
            "dtype": dtype_name(output_detail["dtype"]),
            "shape": [int(v) for v in output_detail["shape"]],
            "quantization": quantization_text(output_detail),
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic int8 vectors through FastestDet_256.tflite."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=str(repo_default_model_path()),
        help="Path to the quantized TFLite model.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=DEFAULT_CASES,
        default=list(DEFAULT_CASES),
        help="Input vector cases to run.",
    )
    parser.add_argument(
        "--mcu-log",
        help="Optional SuperCom MCU log to compare run-0 summaries against.",
    )
    parser.add_argument(
        "--save-output-dir",
        help="Optional directory for raw PC output bytes, one .bin per case.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of human-readable output.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
