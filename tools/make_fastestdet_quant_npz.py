#!/usr/bin/env python3
"""
Create a GD32 Embedded AI quantization NPZ for FastestDet.

FastestDet is trained and exported with OpenCV-style preprocessing:
cv2.imread -> resize to 352x352 -> BGR channels -> float32 / 255.

GD32 Embedded AI converts the ONNX input to TFLite NHWC in the current flow, so
the default output is:
    key:   data
    dtype: float32
    shape: [N, 352, 352, 3]
    range: 0.0 .. 1.0
    channels: BGR

Examples:
    python tools/make_fastestdet_quant_npz.py --images-dir D:/weld/images --output D:/EDCP/fastestdet_quant_100.npz
    python tools/make_fastestdet_quant_npz.py --list D:/dataset/val.txt --output D:/EDCP/fastestdet_quant_100.npz --count 100
    python tools/make_fastestdet_quant_npz.py --yaml configs/coco.yaml --split val --output out.npz
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit("Missing dependency: numpy. Install it with: pip install numpy") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError:
        return None


def import_pil_image():
    try:
        from PIL import Image  # type: ignore

        return Image
    except ImportError:
        return None


def parse_scalar(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    return raw


def parse_fastestdet_yaml(path: Path) -> Dict[str, Dict[str, str]]:
    data: Dict[str, Dict[str, str]] = {}
    section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")) and line.endswith(":"):
            section = line[:-1].strip()
            data.setdefault(section, {})
            continue
        if section and ":" in line:
            key, value = line.split(":", 1)
            data.setdefault(section, {})[key.strip()] = parse_scalar(value)
    return data


def resolve_maybe_relative(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def read_image_list(list_path: Path) -> List[Path]:
    base = list_path.parent
    images: List[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = (base / path).resolve()
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def collect_images_from_dir(images_dir: Path) -> List[Path]:
    images = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda item: str(item).lower())


def select_images(
    images: Sequence[Path],
    count: int,
    shuffle: bool,
    seed: int,
    repeat: bool,
) -> List[Path]:
    if not images:
        raise SystemExit("No image files found.")
    selected = list(images)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if len(selected) >= count:
        return selected[:count]
    if not repeat:
        raise SystemExit(
            f"Only found {len(selected)} images, but --count is {count}. "
            "Add more representative images, lower --count, or pass --repeat."
        )
    output: List[Path] = []
    while len(output) < count:
        output.extend(selected)
    return output[:count]


def cv2_interpolation(cv2, name: str):
    return {
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }[name]


def pil_interpolation(Image, name: str):
    if name == "nearest":
        return Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    if name == "area":
        return Image.Resampling.BOX if hasattr(Image, "Resampling") else Image.BOX
    return Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def load_image_cv2(
    path: Path,
    width: int,
    height: int,
    channel_order: str,
    interpolation: str,
):
    cv2 = import_cv2()
    if cv2 is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cv2 could not read image: {path}")
    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2_interpolation(cv2, interpolation),
    )
    if channel_order == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_pil(
    path: Path,
    width: int,
    height: int,
    channel_order: str,
    interpolation: str,
):
    Image = import_pil_image()
    if Image is None:
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), pil_interpolation(Image, interpolation))
        array = np.asarray(image)
    if channel_order == "bgr":
        array = array[:, :, ::-1]
    return array


def preprocess_image(
    path: Path,
    width: int,
    height: int,
    channel_order: str,
    layout: str,
    interpolation: str,
) -> np.ndarray:
    image = load_image_cv2(path, width, height, channel_order, interpolation)
    if image is None:
        image = load_image_pil(path, width, height, channel_order, interpolation)
    if image is None:
        raise SystemExit("Missing dependency: install opencv-python or Pillow to read images.")

    array = image.astype(np.float32) / 255.0
    if layout == "nchw":
        array = array.transpose(2, 0, 1)
    return np.ascontiguousarray(array, dtype=np.float32)


def build_dataset(
    images: Sequence[Path],
    width: int,
    height: int,
    channel_order: str,
    layout: str,
    interpolation: str,
) -> np.ndarray:
    if layout == "nhwc":
        shape = (len(images), height, width, 3)
    else:
        shape = (len(images), 3, height, width)
    dataset = np.empty(shape, dtype=np.float32)
    for index, path in enumerate(images):
        try:
            dataset[index] = preprocess_image(
                path,
                width=width,
                height=height,
                channel_order=channel_order,
                layout=layout,
                interpolation=interpolation,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to preprocess image #{index}: {path}\n{exc}") from exc
        if (index + 1) % 25 == 0 or index + 1 == len(images):
            print(f"processed {index + 1}/{len(images)} images", file=sys.stderr)
    return dataset


def get_source_images(args: argparse.Namespace) -> Tuple[List[Path], Optional[Path]]:
    if args.images_dir:
        return collect_images_from_dir(args.images_dir), None
    if args.list:
        return read_image_list(args.list), args.list
    if args.yaml:
        data = parse_fastestdet_yaml(args.yaml)
        split_key = "VAL" if args.split == "val" else "TRAIN"
        try:
            list_path = resolve_maybe_relative(data["DATASET"][split_key], args.yaml.parent)
        except KeyError as exc:
            raise SystemExit(f"Could not find DATASET.{split_key} in {args.yaml}") from exc
        return read_image_list(list_path), list_path
    raise SystemExit("Choose one input source: --images-dir, --list, or --yaml.")


def infer_size_from_yaml(args: argparse.Namespace) -> Tuple[int, int]:
    width = args.width
    height = args.height
    if args.yaml and (width is None or height is None):
        data = parse_fastestdet_yaml(args.yaml)
        model = data.get("MODEL", {})
        if width is None and "INPUT_WIDTH" in model:
            width = int(model["INPUT_WIDTH"])
        if height is None and "INPUT_HEIGHT" in model:
            height = int(model["INPUT_HEIGHT"])
    return width or 352, height or 352


def write_manifest(path: Path, images: Sequence[Path], source_list: Optional[Path]) -> None:
    lines = []
    if source_list:
        lines.append(f"# source_list: {source_list}")
    lines.extend(str(path) for path in images)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-dir", type=Path, help="Directory containing calibration images.")
    source.add_argument("--list", type=Path, help="FastestDet train.txt/val.txt image list.")
    source.add_argument("--yaml", type=Path, help="FastestDet yaml; uses DATASET.VAL by default.")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split when --yaml is used.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--key", default="data", help="NPZ array key expected by GD32 examples.")
    parser.add_argument("--count", type=int, default=100, help="Number of calibration images to pack.")
    parser.add_argument("--width", type=int, help="Model input width. Defaults to yaml or 352.")
    parser.add_argument("--height", type=int, help="Model input height. Defaults to yaml or 352.")
    parser.add_argument("--layout", choices=["nhwc", "nchw"], default="nhwc", help="Output tensor layout.")
    parser.add_argument("--channel-order", choices=["bgr", "rgb"], default="bgr", help="Channel order after resize.")
    parser.add_argument("--interpolation", choices=["linear", "area", "nearest"], default="linear", help="Resize method.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidate images before taking --count.")
    parser.add_argument("--seed", type=int, default=20260602, help="Shuffle seed.")
    parser.add_argument("--repeat", action="store_true", help="Repeat images if fewer than --count are available.")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed instead of np.savez.")
    parser.add_argument("--manifest", type=Path, help="Optional text file listing selected image paths.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    width, height = infer_size_from_yaml(args)
    images, source_list = get_source_images(args)
    selected = select_images(
        images,
        count=args.count,
        shuffle=args.shuffle,
        seed=args.seed,
        repeat=args.repeat,
    )
    dataset = build_dataset(
        selected,
        width=width,
        height=height,
        channel_order=args.channel_order,
        layout=args.layout,
        interpolation=args.interpolation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if args.compressed else np.savez
    save(args.output, **{args.key: dataset})
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(args.manifest, selected, source_list)

    print("=== NPZ CREATED ===")
    print("path:", args.output)
    print("key:", args.key)
    print("shape:", list(dataset.shape))
    print("dtype:", dataset.dtype)
    print("min:", float(dataset.min()))
    print("max:", float(dataset.max()))
    print("mean:", float(dataset.mean()))
    print("layout:", args.layout)
    print("channel_order:", args.channel_order)
    print("interpolation:", args.interpolation)
    print("images:", len(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
