#!/usr/bin/env python3
"""
Convert a YOLOv5/Ultralytics-style detection dataset into FastestDet format.

Expected source layout:

    source/
      data.yaml
      train/images/*.jpg
      train/labels/*.txt
      valid/images/*.jpg
      valid/labels/*.txt
      test/images/*.jpg
      test/labels/*.txt

Generated FastestDet layout:

    output/
      category.names
      train/*.jpg + train/*.txt
      val/*.jpg + val/*.txt
      test/*.jpg + test/*.txt
      train.txt
      val.txt
      test.txt
      fastestdet.yaml

Labels are copied as-is because FastestDet uses the same YOLO normalized format:

    class_id center_x center_y width height
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GENERATED_FILES = {
    "category.names",
    "train.txt",
    "val.txt",
    "test.txt",
    "fastestdet.yaml",
}
GENERATED_DIRS = {"train", "val", "test"}


@dataclass
class LabelIssue:
    path: Path
    line_no: int
    message: str


@dataclass
class SplitStats:
    source_name: str
    output_name: str
    image_dir: Path
    label_dir: Path
    total_images: int = 0
    copied_images: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    ignored_non_images: int = 0
    labels_without_images: int = 0
    label_issues: List[LabelIssue] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLOv5/Ultralytics dataset to FastestDet format."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source dataset root containing data.yaml and train/valid/test folders.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Output directory for the generated FastestDet dataset.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Path to data.yaml. Defaults to <src>/data.yaml.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=352,
        help="Square FastestDet model input size. Default: 352.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=None,
        help="FastestDet input width. Overrides --input-size.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=None,
        help="FastestDet input height. Overrides --input-size.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help=(
            "Optional class-name override. Use comma-separated names, for example "
            "'feibian,kongdong,suidao,maoci', or a path to a names file."
        ),
    )
    parser.add_argument(
        "--missing-label",
        choices=("skip", "empty", "error"),
        default="skip",
        help=(
            "How to handle images without a matching .txt label. "
            "skip=do not copy the image, empty=create an empty label, error=stop. "
            "Default: skip."
        ),
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Treat malformed label lines or out-of-range class IDs as errors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite generated FastestDet files/directories under --dst "
            "(train, val, test, *.txt, category.names, fastestdet.yaml)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print what would be generated without copying files.",
    )
    return parser.parse_args()


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def scalar_from_yamlish(value: str) -> Any:
    value = value.strip()
    if not value:
        return None

    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [scalar_from_yamlish(part) for part in inner.split(",")]

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_yaml_fallback(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1

        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace():
            continue

        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$", raw)
        if not match:
            continue

        key, value = match.group(1), match.group(2).strip()
        if key == "names" and not value:
            names: Dict[int, str] = {}
            while i < len(lines):
                child = lines[i]
                child_stripped = child.strip()
                if not child_stripped or child_stripped.startswith("#"):
                    i += 1
                    continue
                if not child[:1].isspace():
                    break
                child_match = re.match(r"^\s*([0-9]+)\s*:\s*(.*)$", child)
                if child_match:
                    idx = int(child_match.group(1))
                    names[idx] = str(scalar_from_yamlish(child_match.group(2)))
                i += 1
            data[key] = names
        else:
            data[key] = scalar_from_yamlish(value)

    return data


def load_data_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        die(f"data yaml not found: {path}")

    try:
        import yaml  # type: ignore
    except ImportError:
        return load_yaml_fallback(path)

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        die(f"expected a mapping in data yaml: {path}")
    return loaded


def load_names_override(value: str) -> List[str]:
    candidate = Path(value)
    if candidate.exists():
        names = [
            line.strip()
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not names:
            die(f"names file is empty: {candidate}")
        return names

    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        die("--names was provided but no names were parsed")
    return names


def normalize_names(data: Dict[str, Any], override: Optional[str]) -> List[str]:
    if override:
        return load_names_override(override)

    raw_names = data.get("names")
    if raw_names is None:
        die("could not find 'names' in data.yaml; use --names as an override")

    if isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    elif isinstance(raw_names, dict):
        parsed: List[Tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                die(f"names contains a non-integer class id: {key!r}")
            parsed.append((idx, str(value)))
        parsed.sort(key=lambda item: item[0])

        expected = list(range(len(parsed)))
        actual = [idx for idx, _ in parsed]
        if actual != expected:
            die(f"names class ids must be contiguous from 0; got {actual}")
        names = [name for _, name in parsed]
    else:
        die("'names' in data.yaml must be a list or a mapping")

    if not names:
        die("no class names found")
    return names


def resolve_dataset_base(src: Path, data_yaml: Path, data: Dict[str, Any]) -> Path:
    raw_path = data.get("path")
    if raw_path is None:
        return src

    base = Path(str(raw_path))
    if base.is_absolute():
        return base
    return (data_yaml.parent / base).resolve()


def resolve_yaml_path(
    raw_value: Any, dataset_base: Path, data_yaml: Path
) -> Optional[Path]:
    if raw_value is None:
        return None

    value = Path(str(raw_value))
    if value.is_absolute():
        return value

    candidate = dataset_base / value
    if candidate.exists():
        return candidate
    return data_yaml.parent / value


def infer_label_dir(image_dir: Path) -> Path:
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels"


def discover_split_dirs(
    src: Path, data_yaml: Path, data: Dict[str, Any]
) -> Dict[str, Tuple[str, Path, Path]]:
    dataset_base = resolve_dataset_base(src, data_yaml, data)
    split_specs: Dict[str, Tuple[str, Sequence[Path], Optional[Any]]] = {
        "train": ("train", (src / "train" / "images",), data.get("train")),
        "val": (
            "val",
            (src / "valid" / "images", src / "val" / "images"),
            data.get("val"),
        ),
        "test": ("test", (src / "test" / "images",), data.get("test")),
    }

    discovered: Dict[str, Tuple[str, Path, Path]] = {}
    for output_name, (source_name, fallback_dirs, yaml_value) in split_specs.items():
        image_dir = resolve_yaml_path(yaml_value, dataset_base, data_yaml)
        if image_dir is None or not image_dir.exists():
            image_dir = next((path for path in fallback_dirs if path.exists()), image_dir)

        if image_dir is None or not image_dir.exists():
            if output_name == "test":
                warn("test split not found; test.txt will not be generated")
                continue
            die(f"{source_name} image directory not found")

        if not image_dir.is_dir():
            die(f"{source_name} image path is not a directory: {image_dir}")

        label_dir = infer_label_dir(image_dir)
        if not label_dir.exists() or not label_dir.is_dir():
            die(f"{source_name} label directory not found: {label_dir}")

        discovered[output_name] = (source_name, image_dir, label_dir)

    return discovered


def ensure_safe_output(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        die("--dst must be different from --src")

    if dry_run:
        return

    dst.mkdir(parents=True, exist_ok=True)
    existing = [path for path in dst.iterdir()]
    if existing and not overwrite:
        die(
            f"output directory is not empty: {dst}. "
            "Use --overwrite to replace generated files."
        )

    if overwrite:
        for dirname in GENERATED_DIRS:
            path = dst / dirname
            if path.exists():
                if not path.is_dir():
                    die(f"refusing to overwrite non-directory path: {path}")
                shutil.rmtree(path)
        for filename in GENERATED_FILES:
            path = dst / filename
            if path.exists():
                if not path.is_file():
                    die(f"refusing to overwrite non-file path: {path}")
                path.unlink()


def list_images_and_ignored(image_dir: Path) -> Tuple[List[Path], int]:
    images: List[Path] = []
    ignored = 0
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        else:
            ignored += 1
    images.sort(key=lambda path: path.name.lower())
    return images, ignored


def validate_label_file(label_path: Path, num_classes: int) -> Tuple[bool, List[LabelIssue]]:
    issues: List[LabelIssue] = []
    has_objects = False

    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        has_objects = True
        parts = stripped.split()
        if len(parts) != 5:
            issues.append(
                LabelIssue(label_path, line_no, f"expected 5 fields, got {len(parts)}")
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            issues.append(LabelIssue(label_path, line_no, "class id is not an integer"))
            continue

        if class_id < 0 or class_id >= num_classes:
            issues.append(
                LabelIssue(
                    label_path,
                    line_no,
                    f"class id {class_id} outside [0, {num_classes - 1}]",
                )
            )

        try:
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(LabelIssue(label_path, line_no, "bbox values are not floats"))
            continue

        coords = (cx, cy, width, height)
        if any(value < 0.0 or value > 1.0 for value in coords):
            issues.append(
                LabelIssue(label_path, line_no, "bbox values should be normalized to 0..1")
            )
        if width <= 0.0 or height <= 0.0:
            issues.append(LabelIssue(label_path, line_no, "bbox width/height must be > 0"))

    return not has_objects, issues


def copy_or_create_empty_label(src_label: Optional[Path], dst_label: Path) -> None:
    if src_label is None:
        dst_label.write_text("", encoding="utf-8")
    else:
        shutil.copy2(src_label, dst_label)


def convert_split(
    output_name: str,
    source_name: str,
    image_dir: Path,
    label_dir: Path,
    dst: Path,
    num_classes: int,
    missing_label_policy: str,
    strict_labels: bool,
    dry_run: bool,
) -> Tuple[SplitStats, List[Path]]:
    stats = SplitStats(
        source_name=source_name,
        output_name=output_name,
        image_dir=image_dir,
        label_dir=label_dir,
    )
    images, ignored = list_images_and_ignored(image_dir)
    stats.total_images = len(images)
    stats.ignored_non_images = ignored

    image_stems = {path.stem for path in images}
    label_stems = {
        path.stem for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt"
    }
    stats.labels_without_images = len(label_stems - image_stems)

    output_dir = dst / output_name
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: List[Path] = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats.missing_labels += 1
            if missing_label_policy == "error":
                die(f"missing label for image: {image_path}")
            if missing_label_policy == "skip":
                continue
            label_path_for_copy: Optional[Path] = None
        else:
            label_path_for_copy = label_path
            empty, issues = validate_label_file(label_path, num_classes)
            if empty:
                stats.empty_labels += 1
            stats.label_issues.extend(issues)
            if issues and strict_labels:
                first = issues[0]
                die(f"label validation failed: {first.path}:{first.line_no}: {first.message}")

        dst_image = output_dir / image_path.name
        dst_label = output_dir / f"{image_path.stem}.txt"
        if not dry_run:
            shutil.copy2(image_path, dst_image)
            copy_or_create_empty_label(label_path_for_copy, dst_label)
        copied_paths.append(dst_image.resolve())
        stats.copied_images += 1

    return stats, copied_paths


def yaml_quote(path: Path) -> str:
    text = str(path.resolve())
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_names(dst: Path, names: Sequence[str], dry_run: bool) -> Path:
    path = dst / "category.names"
    if not dry_run:
        path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path.resolve()


def write_path_list(dst: Path, output_name: str, paths: Sequence[Path], dry_run: bool) -> Path:
    list_path = dst / f"{output_name}.txt"
    if not dry_run:
        content = "\n".join(str(path.resolve()) for path in paths)
        if content:
            content += "\n"
        list_path.write_text(content, encoding="utf-8")
    return list_path.resolve()


def write_fastestdet_yaml(
    dst: Path,
    train_txt: Path,
    val_txt: Path,
    names_path: Path,
    num_classes: int,
    input_width: int,
    input_height: int,
    dry_run: bool,
) -> Path:
    config_path = dst / "fastestdet.yaml"
    content = f"""DATASET:
  TRAIN: {yaml_quote(train_txt)}
  VAL: {yaml_quote(val_txt)}
  NAMES: {yaml_quote(names_path)}
MODEL:
  NC: {num_classes}
  INPUT_WIDTH: {input_width}
  INPUT_HEIGHT: {input_height}
TRAIN:
  LR: 0.001
  THRESH: 0.25
  WARMUP: true
  BATCH_SIZE: 64
  END_EPOCH: 350
  MILESTIONES:
    - 150
    - 250
    - 300
"""
    if not dry_run:
        config_path.write_text(content, encoding="utf-8")
    return config_path.resolve()


def print_issue_samples(issues: Sequence[LabelIssue], limit: int = 10) -> None:
    if not issues:
        return

    warn(f"found {len(issues)} label validation issue(s); showing first {min(limit, len(issues))}:")
    for issue in issues[:limit]:
        warn(f"  {issue.path}:{issue.line_no}: {issue.message}")


def print_summary(
    dst: Path,
    names: Sequence[str],
    stats_by_split: Sequence[SplitStats],
    generated_lists: Dict[str, Path],
    names_path: Path,
    config_path: Optional[Path],
    dry_run: bool,
) -> None:
    prefix = "DRY RUN complete" if dry_run else "Conversion complete"
    print(f"\n{prefix}: {dst.resolve()}")
    print(f"Classes ({len(names)}): {', '.join(names)}")
    print(f"Names file: {names_path}")

    for stats in stats_by_split:
        print(
            f"{stats.output_name}: "
            f"{stats.copied_images}/{stats.total_images} image(s) copied "
            f"from {stats.image_dir}"
        )
        if stats.missing_labels:
            warn(
                f"{stats.output_name}: {stats.missing_labels} image(s) had no matching label"
            )
        if stats.empty_labels:
            warn(f"{stats.output_name}: {stats.empty_labels} empty label file(s)")
        if stats.ignored_non_images:
            warn(
                f"{stats.output_name}: ignored {stats.ignored_non_images} non-image file(s)"
            )
        if stats.labels_without_images:
            warn(
                f"{stats.output_name}: {stats.labels_without_images} label file(s) "
                "had no matching image"
            )
        print_issue_samples(stats.label_issues)

    for split, path in generated_lists.items():
        print(f"{split} list: {path}")
    if config_path is not None:
        print(f"FastestDet yaml: {config_path}")


def main() -> int:
    args = parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    data_yaml = (args.data_yaml or (src / "data.yaml")).resolve()
    input_width = args.input_width or args.input_size
    input_height = args.input_height or args.input_size

    if not src.exists() or not src.is_dir():
        die(f"source dataset directory not found: {src}")
    if input_width <= 0 or input_height <= 0:
        die("input width and height must be positive integers")

    data = load_data_yaml(data_yaml)
    names = normalize_names(data, args.names)
    split_dirs = discover_split_dirs(src, data_yaml, data)
    ensure_safe_output(src, dst, args.overwrite, args.dry_run)

    names_path = write_names(dst, names, args.dry_run)
    generated_lists: Dict[str, Path] = {}
    stats_by_split: List[SplitStats] = []

    for output_name in ("train", "val", "test"):
        if output_name not in split_dirs:
            continue
        source_name, image_dir, label_dir = split_dirs[output_name]
        stats, copied_paths = convert_split(
            output_name=output_name,
            source_name=source_name,
            image_dir=image_dir,
            label_dir=label_dir,
            dst=dst,
            num_classes=len(names),
            missing_label_policy=args.missing_label,
            strict_labels=args.strict_labels,
            dry_run=args.dry_run,
        )
        stats_by_split.append(stats)
        generated_lists[output_name] = write_path_list(
            dst, output_name, copied_paths, args.dry_run
        )

    if "train" not in generated_lists:
        die("train split was not generated")
    if "val" not in generated_lists:
        die("val split was not generated")

    config_path = write_fastestdet_yaml(
        dst=dst,
        train_txt=generated_lists["train"],
        val_txt=generated_lists["val"],
        names_path=names_path,
        num_classes=len(names),
        input_width=input_width,
        input_height=input_height,
        dry_run=args.dry_run,
    )

    print_summary(
        dst=dst,
        names=names,
        stats_by_split=stats_by_split,
        generated_lists=generated_lists,
        names_path=names_path,
        config_path=config_path,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
