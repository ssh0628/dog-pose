#!/usr/bin/env python3
"""Fine-tune an existing side-pose model on the opposite-only dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "side" / "best.pt"
DEFAULT_DATA = (
    PROJECT_ROOT
    / "dataset"
    / "side_opposite_only"
    / "dog_pose_side_opposite_22kpt.yaml"
)
DEFAULT_PROJECT = PROJECT_ROOT / "runs"
DEFAULT_NAME = "side_opposite_only_finetune"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing side best.pt on the opposite-only dataset "
            "as a new Ultralytics YOLO Pose run."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Existing side-model best.pt (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Opposite-only dataset YAML (default: {DEFAULT_DATA}).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--device",
        default=0,
        help="Ultralytics device value, for example 0, 0,1, cpu, or mps.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"Run name (default: {DEFAULT_NAME}).",
    )
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Initial learning rate. The default is conservative for fine-tuning.",
    )
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision during training.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reuse of an existing run directory name.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and image/label pairs without importing Ultralytics.",
    )
    return parser.parse_args()


def yaml_scalar(text: str, key: str) -> str:
    prefix = f"{key}:"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value.strip("'\"")
    raise ValueError(f"Missing '{key}' in dataset YAML")


def resolve_dataset_paths(data_path: Path) -> tuple[Path, Path, Path]:
    text = data_path.read_text(encoding="utf-8")
    root_value = yaml_scalar(text, "path")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = data_path.parent / root

    train = root / yaml_scalar(text, "train")
    val = root / yaml_scalar(text, "val")
    return root.resolve(), train.resolve(), val.resolve()


def corresponding_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an 'images' directory: {image_dir}") from exc
    parts[image_index] = "labels"
    return Path(*parts)


def validate_split(split_name: str, image_dir: Path) -> int:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{split_name} image directory not found: {image_dir}")

    label_dir = corresponding_label_dir(image_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"{split_name} label directory not found: {label_dir}")

    images = {
        path.relative_to(image_dir).with_suffix("")
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.relative_to(label_dir).with_suffix("")
        for path in label_dir.rglob("*.txt")
    }
    if images != labels:
        image_only = sorted(str(path) for path in images - labels)
        label_only = sorted(str(path) for path in labels - images)
        raise ValueError(
            f"{split_name} image/label mismatch: "
            f"image_only={image_only[:5]}, label_only={label_only[:5]}"
        )
    if not images:
        raise ValueError(f"{split_name} split is empty: {image_dir}")
    return len(images)


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt checkpoint: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    dataset_root, train_dir, val_dir = resolve_dataset_paths(data_path)
    train_count = validate_split("train", train_dir)
    val_count = validate_split("val", val_dir)

    print(f"Dataset: {dataset_root}")
    print(f"Pairs: train={train_count}, val={val_count}")
    print(f"Initial weights: {model_path}")
    return model_path, data_path


def load_pose_model(checkpoint: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in this Python environment. "
            "Install the training dependencies before running this script."
        ) from exc

    model = YOLO(str(checkpoint))
    if getattr(model, "task", None) != "pose":
        raise ValueError(f"Checkpoint task must be 'pose', got {model.task!r}")

    model_yaml = getattr(getattr(model, "model", None), "yaml", {})
    kpt_shape = model_yaml.get("kpt_shape") if isinstance(model_yaml, dict) else None
    if kpt_shape and list(kpt_shape) != [22, 3]:
        raise ValueError(f"Checkpoint kpt_shape must be [22, 3], got {kpt_shape}")
    return model


def main() -> int:
    args = parse_args()
    try:
        model_path, data_path = validate_inputs(args)
        if args.check_only:
            print("Input validation passed. No training was started.")
            return 0

        print("Starting a new fine-tuning run from the existing side checkpoint...")
        model = load_pose_model(model_path)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(args.project.expanduser().resolve()),
            name=args.name,
            optimizer=args.optimizer,
            lr0=args.lr0,
            freeze=args.freeze,
            translate=args.translate,
            scale=args.scale,
            seed=args.seed,
            deterministic=True,
            cache=args.cache,
            amp=args.amp,
            exist_ok=args.exist_ok,
        )

        save_dir = Path(model.trainer.save_dir)
        best_path = save_dir / "weights" / "best.pt"
        print(f"Fine-tuning complete. Best checkpoint: {best_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
