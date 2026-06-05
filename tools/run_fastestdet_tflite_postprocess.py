#!/usr/bin/env python3
"""
Run FastestDet int8 TFLite inference and post-processing.

This script is intended for EDCP/GD32 L4 dry-run validation on a training host.
It mirrors the FastestDet decode path, but runs against the quantized TFLite
model produced for GD32 Embedded AI.

Usage:
    python tools/run_fastestdet_tflite_postprocess.py FastestDet_256.tflite --image sample.jpg
    python tools/run_fastestdet_tflite_postprocess.py FastestDet_256.tflite --image sample.jpg --save-vis result.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CLASS_NAMES = [
    "\u98de\u8fb9",
    "\u5b54\u6d1e",
    "\u96a7\u9053",
    "\u6bdb\u523a",
]


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError("Missing dependency: numpy. Install it with `pip install numpy`.") from exc
    return np


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def import_pil_image() -> Any:
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantization_text(detail: Dict[str, Any]) -> str:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return f"scale={scale} zero_point={zero_point}"


def read_class_names(path: Optional[Path]) -> List[str]:
    if path is None:
        return list(DEFAULT_CLASS_NAMES)
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not names:
        raise RuntimeError(f"No class names found in {path}")
    return names


def cv2_interpolation(cv2: Any, name: str) -> Any:
    return {
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }[name]


def pil_interpolation(Image: Any, name: str) -> Any:
    if name == "nearest":
        return Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    if name == "area":
        return Image.Resampling.BOX if hasattr(Image, "Resampling") else Image.BOX
    return Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def load_image(
    np: Any,
    image_path: Path,
    width: int,
    height: int,
    channel_order: str,
    interpolation: str,
) -> Tuple[Any, Any, Tuple[int, int]]:
    cv2 = import_cv2()
    if cv2 is not None:
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"cv2 could not read image: {image_path}")
        resized = cv2.resize(
            original,
            (width, height),
            interpolation=cv2_interpolation(cv2, interpolation),
        )
        if channel_order == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        original_h, original_w = original.shape[:2]
        return original, resized, (original_w, original_h)

    Image = import_pil_image()
    if Image is None:
        raise RuntimeError("Missing image reader: install opencv-python or Pillow.")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_w, original_h = image.size
        resized_image = image.resize((width, height), pil_interpolation(Image, interpolation))
        original = np.asarray(image)
        resized = np.asarray(resized_image)
    if channel_order == "bgr":
        resized = resized[:, :, ::-1]
    return original, resized, (original_w, original_h)


def quantize_input(np: Any, array: Any, detail: Dict[str, Any]) -> Any:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    dtype = detail["dtype"]
    array = array.astype(np.float32) / 255.0

    if dtype in (np.float32, np.float64):
        return array.astype(dtype)
    if scale == 0:
        raise RuntimeError("Input tensor is quantized but input scale is 0.")

    q = np.round(array / float(scale) + int(zero_point))
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    else:
        raise RuntimeError(f"Unsupported input dtype: {dtype_name(dtype)}")
    return q.astype(dtype)


def dequantize_output(np: Any, array: Any, detail: Dict[str, Any]) -> Any:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if array.dtype in (np.float32, np.float64):
        return array.astype(np.float32)
    if scale == 0:
        raise RuntimeError("Output tensor is quantized but output scale is 0.")
    return (array.astype(np.float32) - int(zero_point)) * float(scale)


def output_to_hwc(output: Any) -> Tuple[Any, str]:
    if output.ndim != 4 or output.shape[0] != 1:
        raise RuntimeError(f"Expected output rank [1,H,W,C] or [1,C,H,W], got {list(output.shape)}")
    if output.shape[-1] >= 6:
        return output[0], "NHWC"
    if output.shape[1] >= 6:
        return output[0].transpose(1, 2, 0), "NCHW"
    raise RuntimeError(f"Could not infer FastestDet output layout from shape {list(output.shape)}")


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def box_iou(a: Detection, b: Detection) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms(detections: Sequence[Detection], threshold: float) -> List[Detection]:
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        should_keep = True
        for kept_det in kept:
            if det.class_id == kept_det.class_id and box_iou(det, kept_det) > threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(det)
    return kept


def decode_fastestdet(
    feature_map: Any,
    image_size: Tuple[int, int],
    class_names: Sequence[str],
    score_threshold: float,
    nms_threshold: float,
) -> Tuple[List[Detection], List[Detection]]:
    height, width, channels = feature_map.shape
    if channels < 6:
        raise RuntimeError(f"FastestDet output channels must be at least 6, got {channels}")
    class_count = channels - 5
    if class_count > len(class_names):
        raise RuntimeError(
            f"Output has {class_count} classes, but only {len(class_names)} class name(s) were provided."
        )

    image_w, image_h = image_size
    candidates: List[Detection] = []
    for gy in range(height):
        for gx in range(width):
            data = feature_map[gy, gx]
            obj_score = clamp01(float(data[0]))
            class_scores = [clamp01(float(value)) for value in data[5 : 5 + class_count]]
            class_id = max(range(class_count), key=lambda idx: class_scores[idx])
            class_score = class_scores[class_id]
            score = (obj_score ** 0.6) * (class_score ** 0.4)
            if score <= score_threshold:
                continue

            x_offset = math.tanh(float(data[1]))
            y_offset = math.tanh(float(data[2]))
            box_w = sigmoid(float(data[3]))
            box_h = sigmoid(float(data[4]))

            cx = (float(gx) + x_offset) / float(width)
            cy = (float(gy) + y_offset) / float(height)
            x1 = (cx - 0.5 * box_w) * image_w
            y1 = (cy - 0.5 * box_h) * image_h
            x2 = (cx + 0.5 * box_w) * image_w
            y2 = (cy + 0.5 * box_h) * image_h
            candidates.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    class_id=class_id,
                    class_name=class_names[class_id],
                )
            )

    return candidates, nms(candidates, nms_threshold)


def draw_detections(image: Any, detections: Sequence[Detection], output_path: Path) -> None:
    cv2 = import_cv2()
    if cv2 is not None:
        canvas = image.copy()
        height, width = canvas.shape[:2]
        for det in detections:
            x1 = max(0, min(width - 1, int(round(det.x1))))
            y1 = max(0, min(height - 1, int(round(det.y1))))
            x2 = max(0, min(width - 1, int(round(det.x2))))
            y2 = max(0, min(height - 1, int(round(det.y2))))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas,
                f"{det.class_id}:{det.score:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"cv2 could not write visualization: {output_path}")
        return

    Image = import_pil_image()
    if Image is None:
        raise RuntimeError("Saving visualization requires opencv-python or Pillow.")
    from PIL import ImageDraw  # type: ignore

    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for det in detections:
        x1 = max(0, min(width - 1, int(round(det.x1))))
        y1 = max(0, min(height - 1, int(round(det.y1))))
        x2 = max(0, min(width - 1, int(round(det.x2))))
        y2 = max(0, min(height - 1, int(round(det.y2))))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"{det.class_id}:{det.score:.2f}", fill=(0, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def print_detection(index: int, det: Detection) -> None:
    print(
        f"DET idx={index} "
        f"cls={det.class_id} "
        f"name={det.class_name} "
        f"score={det.score:.6f} "
        f"x1={det.x1:.2f} y1={det.y1:.2f} x2={det.x2:.2f} y2={det.y2:.2f}"
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    np = import_numpy()
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
    input_shape = [int(value) for value in input_detail["shape"]]
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[-1] != 3:
        raise RuntimeError(f"Expected NHWC image input [1,H,W,3], got {input_shape}")
    input_h, input_w = input_shape[1], input_shape[2]

    original_image, resized_image, original_size = load_image(
        np,
        image_path,
        input_w,
        input_h,
        args.channel_order,
        args.interpolation,
    )
    input_tensor = quantize_input(np, resized_image, input_detail).reshape(tuple(input_shape))
    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_detail["index"])
    output = dequantize_output(np, raw_output, output_detail)
    feature_map, output_layout = output_to_hwc(output)

    class_names = read_class_names(Path(args.class_names).resolve() if args.class_names else None)
    candidates, detections = decode_fastestdet(
        feature_map,
        original_size,
        class_names,
        args.score_threshold,
        args.nms_threshold,
    )

    if args.save_vis:
        draw_detections(original_image, detections, Path(args.save_vis).resolve())

    return {
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "runtime": runtime_name,
        "image_path": str(image_path),
        "image_size": list(original_size),
        "channel_order": args.channel_order,
        "interpolation": args.interpolation,
        "input": {
            "name": input_detail.get("name", ""),
            "dtype": dtype_name(input_detail["dtype"]),
            "shape": input_shape,
            "quantization": quantization_text(input_detail),
        },
        "output": {
            "name": output_detail.get("name", ""),
            "dtype": dtype_name(output_detail["dtype"]),
            "shape": [int(value) for value in output_detail["shape"]],
            "layout": output_layout,
            "quantization": quantization_text(output_detail),
        },
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "candidate_count": len(candidates),
        "det_count": len(detections),
        "detections": [asdict(det) for det in detections],
        "save_vis": str(Path(args.save_vis).resolve()) if args.save_vis else None,
    }


def print_human(result: Dict[str, Any]) -> None:
    print("=== FILE ===")
    print(result["model_path"])
    print(f"bytes: {result['model_bytes']}")
    print(f"sha256: {result['model_sha256']}")
    print()
    print("=== RUNTIME ===")
    print(result["runtime"])
    print()
    print("=== IMAGE ===")
    print(result["image_path"])
    print(
        f"original_size={result['image_size']} "
        f"channel_order={result['channel_order']} "
        f"interpolation={result['interpolation']}"
    )
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
        f"layout={result['output']['layout']} "
        f"{result['output']['quantization']}"
    )
    print()
    print("=== POSTPROCESS ===")
    print(
        f"SCORE_THRESHOLD={result['score_threshold']} "
        f"NMS_THRESHOLD={result['nms_threshold']} "
        f"CANDIDATE_COUNT={result['candidate_count']} "
        f"DET_COUNT={result['det_count']}"
    )
    for index, det_data in enumerate(result["detections"]):
        print_detection(index, Detection(**det_data))
    if result["save_vis"]:
        print(f"VIS={result['save_vis']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FastestDet_256 int8 TFLite image inference and decode detections."
    )
    parser.add_argument("model", help="Path to FastestDet_256.tflite.")
    parser.add_argument("--image", required=True, help="Image path for L4 dry-run inference.")
    parser.add_argument(
        "--class-names",
        help="Optional class names file, one class per line. Defaults to EDCP weld classes.",
    )
    parser.add_argument(
        "--channel-order",
        choices=["bgr", "rgb"],
        default="bgr",
        help="Channel order after resize. Default matches OpenCV/FastestDet training.",
    )
    parser.add_argument(
        "--interpolation",
        choices=["linear", "area", "nearest"],
        default="linear",
        help="Resize interpolation.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--save-vis", help="Optional output image path with drawn detections.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text output.")
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
