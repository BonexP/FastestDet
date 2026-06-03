#!/usr/bin/env python3
"""
Zero-dependency ONNX inspector for the FastestDet deployment flow.

This script intentionally uses only the Python standard library. It is meant for
machines that do not have onnx, onnxruntime, protobuf, numpy, or OpenCV installed.

Usage:
    python tools/inspect_fastestdet_onnx.py FastestDet.onnx
    python tools/inspect_fastestdet_onnx.py exports/FastestDet_weld_288.onnx --markdown
    python tools/inspect_fastestdet_onnx.py exports/FastestDet_weld_288.onnx --json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TENSOR_TYPES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
}

GD32_MANUAL_OPS = {
    "Conv",
    "DepthConv",
    "AveragePool",
    "MaxPool",
    "Softmax",
    "Reshape",
    "Relu",
    "LeakyRelu",
    "Tanh",
    "FullyConnect",
    "Gemm",
    "Expand",
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Mean",
    "Gather",
    "Pad",
    "Transpose",
    "Concat",
    "QuantizeLinear",
    "DequantizeLinear",
    "Logistic",
    "Sigmoid",
    "Resize",
    "Split",
    "ReduceMax",
}

HIGH_RISK_OPS = {
    "NonMaxSuppression",
    "GridSample",
    "RoiAlign",
    "HardSwish",
    "Swish",
    "SiLU",
    "TopK",
    "ScatterND",
}


def read_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def iter_fields(buf: bytes) -> Iterable[Tuple[int, int, Any]]:
    offset = 0
    size = len(buf)
    while offset < size:
        key, offset = read_varint(buf, offset)
        field_no = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, offset = read_varint(buf, offset)
        elif wire_type == 1:
            value = buf[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = read_varint(buf, offset)
            value = buf[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = buf[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} at {offset}")
        yield field_no, wire_type, value


def decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return repr(raw[:32])


def parse_opset(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {"domain": "", "version": None}
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            result["domain"] = decode_text(value)
        elif field_no == 2 and wire_type == 0:
            result["version"] = value
    return result


def parse_tensor(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": "",
        "dims": [],
        "data_type": None,
        "raw_bytes": 0,
    }
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 0:
            result["dims"].append(value)
        elif field_no == 2 and wire_type == 0:
            result["data_type"] = value
        elif field_no == 8 and wire_type == 2:
            result["name"] = decode_text(value)
        elif field_no == 9 and wire_type == 2:
            result["raw_bytes"] = len(value)
    return result


def parse_dim(buf: bytes) -> Any:
    dim_value = None
    dim_param = None
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 0:
            dim_value = value
        elif field_no == 2 and wire_type == 2:
            dim_param = decode_text(value)
    if dim_value is not None:
        return dim_value
    if dim_param:
        return dim_param
    return "?"


def parse_shape(buf: bytes) -> List[Any]:
    return [
        parse_dim(value)
        for field_no, wire_type, value in iter_fields(buf)
        if field_no == 1 and wire_type == 2
    ]


def parse_tensor_type(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {"elem_type": None, "shape": []}
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 0:
            result["elem_type"] = value
        elif field_no == 2 and wire_type == 2:
            result["shape"] = parse_shape(value)
    return result


def parse_type(buf: bytes) -> Dict[str, Any]:
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            return parse_tensor_type(value)
    return {"elem_type": None, "shape": []}


def parse_value_info(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": "", "elem_type": None, "shape": []}
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            result["name"] = decode_text(value)
        elif field_no == 2 and wire_type == 2:
            result.update(parse_type(value))
    return result


def parse_attribute(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": "", "type": None}
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            result["name"] = decode_text(value)
        elif field_no == 20 and wire_type == 0:
            result["type"] = value
    return result


def parse_node(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "op_type": "",
        "domain": "",
        "name": "",
        "inputs": [],
        "outputs": [],
        "attrs": [],
    }
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            result["inputs"].append(decode_text(value))
        elif field_no == 2 and wire_type == 2:
            result["outputs"].append(decode_text(value))
        elif field_no == 3 and wire_type == 2:
            result["name"] = decode_text(value)
        elif field_no == 4 and wire_type == 2:
            result["op_type"] = decode_text(value)
        elif field_no == 5 and wire_type == 2:
            result["attrs"].append(parse_attribute(value))
        elif field_no == 7 and wire_type == 2:
            result["domain"] = decode_text(value)
    return result


def parse_graph(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": "",
        "nodes": [],
        "inputs": [],
        "outputs": [],
        "initializers": [],
    }
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 2:
            result["nodes"].append(parse_node(value))
        elif field_no == 2 and wire_type == 2:
            result["name"] = decode_text(value)
        elif field_no == 5 and wire_type == 2:
            result["initializers"].append(parse_tensor(value))
        elif field_no == 11 and wire_type == 2:
            result["inputs"].append(parse_value_info(value))
        elif field_no == 12 and wire_type == 2:
            result["outputs"].append(parse_value_info(value))
    return result


def parse_model(buf: bytes) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ir_version": None,
        "producer": "",
        "producer_version": "",
        "opsets": [],
        "graph": None,
    }
    for field_no, wire_type, value in iter_fields(buf):
        if field_no == 1 and wire_type == 0:
            result["ir_version"] = value
        elif field_no == 2 and wire_type == 2:
            result["producer"] = decode_text(value)
        elif field_no == 3 and wire_type == 2:
            result["producer_version"] = decode_text(value)
        elif field_no == 7 and wire_type == 2:
            result["graph"] = parse_graph(value)
        elif field_no == 8 and wire_type == 2:
            result["opsets"].append(parse_opset(value))
    return result


def tensor_type_name(elem_type: Any) -> str:
    if elem_type is None:
        return "unknown"
    return TENSOR_TYPES.get(elem_type, str(elem_type))


def inspect_model(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    parsed = parse_model(raw)
    graph = parsed["graph"]
    if graph is None:
        raise ValueError("ONNX model does not contain a graph")

    initializer_names = {tensor["name"] for tensor in graph["initializers"]}
    real_inputs = [
        value for value in graph["inputs"] if value["name"] not in initializer_names
    ]
    op_counts = collections.Counter(node["op_type"] for node in graph["nodes"])
    domain_counts = collections.Counter(node["domain"] or "ai.onnx" for node in graph["nodes"])
    unknown_ops = sorted(op for op in op_counts if op not in GD32_MANUAL_OPS)
    risky_ops = sorted(op for op in op_counts if op in HIGH_RISK_OPS)
    dynamic_input_dims = any(
        any(not isinstance(dim, int) for dim in value["shape"]) for value in real_inputs
    )

    fastestdet_hint = None
    if len(graph["outputs"]) == 1:
        shape = graph["outputs"][0]["shape"]
        if len(shape) == 4 and isinstance(shape[1], int) and shape[1] >= 6:
            fastestdet_hint = {
                "output_layout": "NCHW",
                "channels": shape[1],
                "class_count_if_5_plus_nc": shape[1] - 5,
                "grid_height": shape[2],
                "grid_width": shape[3],
            }

    return {
        "file": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ir_version": parsed["ir_version"],
        "producer": parsed["producer"],
        "producer_version": parsed["producer_version"],
        "opsets": parsed["opsets"],
        "graph_name": graph["name"],
        "inputs": [
            {
                "name": value["name"],
                "dtype": tensor_type_name(value["elem_type"]),
                "shape": value["shape"],
            }
            for value in real_inputs
        ],
        "outputs": [
            {
                "name": value["name"],
                "dtype": tensor_type_name(value["elem_type"]),
                "shape": value["shape"],
            }
            for value in graph["outputs"]
        ],
        "initializers": {
            "count": len(graph["initializers"]),
            "raw_bytes": sum(tensor["raw_bytes"] for tensor in graph["initializers"]),
        },
        "nodes": {
            "count": len(graph["nodes"]),
            "ops": dict(op_counts.most_common()),
            "domains": dict(domain_counts.most_common()),
            "first_30": [
                {"op_type": node["op_type"], "outputs": node["outputs"][:2]}
                for node in graph["nodes"][:30]
            ],
        },
        "gd32_hints": {
            "unknown_ops_against_manual_list": unknown_ops,
            "high_risk_ops": risky_ops,
            "dynamic_input_dims": dynamic_input_dims,
            "fastestdet_output_hint": fastestdet_hint,
        },
    }


def print_text(report: Dict[str, Any]) -> None:
    print("=== FILE ===")
    print(report["file"])
    print("bytes:", report["bytes"])
    print("sha256:", report["sha256"])
    print("\n=== MODEL ===")
    print("ir_version:", report["ir_version"])
    print("producer:", report["producer"], report["producer_version"])
    print("opsets:", report["opsets"])
    print("graph:", report["graph_name"])
    print("\n=== INPUTS ===")
    for item in report["inputs"]:
        print(item["name"], item["dtype"], item["shape"])
    print("\n=== OUTPUTS ===")
    for item in report["outputs"]:
        print(item["name"], item["dtype"], item["shape"])
    print("\n=== INITIALIZERS ===")
    print("count:", report["initializers"]["count"])
    print("raw_bytes:", report["initializers"]["raw_bytes"])
    print("\n=== NODE OPS ===")
    for op_type, count in report["nodes"]["ops"].items():
        print(op_type, count)
    print("\n=== DOMAINS ===")
    for domain, count in report["nodes"]["domains"].items():
        print(domain, count)
    print("\n=== FIRST 30 NODES ===")
    for node in report["nodes"]["first_30"]:
        print(node["op_type"], "->", node["outputs"])
    print("\n=== GD32 DEPLOYMENT HINTS ===")
    hints = report["gd32_hints"]
    print("unknown_ops_against_manual_list:", hints["unknown_ops_against_manual_list"] or "none")
    print("high_risk_ops:", hints["high_risk_ops"] or "none")
    print("dynamic_input_dims:", hints["dynamic_input_dims"])
    print("fastestdet_output_hint:", hints["fastestdet_output_hint"])
    print("\n=== FASTESTDET MCU POSTPROCESS CONTRACT ===")
    print("input:  float32 NCHW, expected range 0..1 after RGB resize to model size")
    print("output: NCHW feature map; channels are usually 5 + class_count")
    print("classes: channel count - 5")
    print("score: object_score^0.6 * class_score^0.4")
    print("decode: x/y use tanh center offsets; w/h use sigmoid size terms")
    print("filter: threshold first, then class-wise or global NMS on decoded boxes")


def print_markdown(report: Dict[str, Any]) -> None:
    print(f"# ONNX Inspection Report\n")
    print(f"- File: `{report['file']}`")
    print(f"- Bytes: `{report['bytes']}`")
    print(f"- SHA256: `{report['sha256']}`")
    print(f"- Producer: `{report['producer']} {report['producer_version']}`")
    print(f"- Opsets: `{report['opsets']}`")
    print(f"- Graph: `{report['graph_name']}`")
    print("\n## Inputs")
    for item in report["inputs"]:
        print(f"- `{item['name']}`: `{item['dtype']}` `{item['shape']}`")
    print("\n## Outputs")
    for item in report["outputs"]:
        print(f"- `{item['name']}`: `{item['dtype']}` `{item['shape']}`")
    print("\n## Ops")
    for op_type, count in report["nodes"]["ops"].items():
        print(f"- `{op_type}`: {count}")
    print("\n## GD32 Hints")
    hints = report["gd32_hints"]
    print(f"- Unknown ops against manual list: `{hints['unknown_ops_against_manual_list'] or 'none'}`")
    print(f"- High-risk ops: `{hints['high_risk_ops'] or 'none'}`")
    print(f"- Dynamic input dims: `{hints['dynamic_input_dims']}`")
    print(f"- FastestDet output hint: `{hints['fastestdet_output_hint']}`")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the ONNX model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    group.add_argument("--markdown", action="store_true", help="Print a Markdown report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_model(args.model)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.markdown:
        print_markdown(report)
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
