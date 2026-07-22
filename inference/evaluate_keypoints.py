#!/usr/bin/env python3
"""Evaluate per-keypoint OKS AP for a 22-keypoint dog pose model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT / "runs" / "side_opposite_only_finetune" / "weights" / "best.pt"
)
DEFAULT_DATA = (
    PROJECT_ROOT
    / "dataset"
    / "side_opposite_only"
    / "dog_pose_side_opposite_22kpt.yaml"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OKS_THRESHOLDS = tuple(value / 100 for value in range(50, 100, 5))

KEYPOINT_NAMES = (
    "left_ear",
    "right_ear",
    "t13_spinous_process",
    "left_dorsal_scapular_spine",
    "left_acromion_greater_tubercle",
    "left_lateral_humeral_epicondyle",
    "left_ulnar_styloid_process",
    "left_fifth_metacarpal_distal",
    "right_dorsal_scapular_spine",
    "right_acromion_greater_tubercle",
    "right_lateral_humeral_epicondyle",
    "right_ulnar_styloid_process",
    "right_fifth_metacarpal_distal",
    "iliac_crest",
    "left_femoral_greater_trochanter",
    "left_femorotibial_joint",
    "left_lateral_malleolus",
    "left_fifth_metatarsal_distal",
    "right_femoral_greater_trochanter",
    "right_femorotibial_joint",
    "right_lateral_malleolus",
    "right_fifth_metatarsal_distal",
)

SIDE_PAIRS = (
    ("ear", 0, 1),
    ("dorsal_scapular_spine", 3, 8),
    ("acromion_greater_tubercle", 4, 9),
    ("lateral_humeral_epicondyle", 5, 10),
    ("ulnar_styloid_process", 6, 11),
    ("fifth_metacarpal_distal", 7, 12),
    ("femoral_greater_trochanter", 14, 18),
    ("femorotibial_joint", 15, 19),
    ("lateral_malleolus", 16, 20),
    ("fifth_metatarsal_distal", 17, 21),
)
SHARED_KEYPOINTS = (2, 13)


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class GroundTruth:
    area: float
    keypoints: tuple[Keypoint, ...]


@dataclass(frozen=True)
class Prediction:
    box_score: float
    keypoints: tuple[Keypoint, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    direction: str
    ground_truths: tuple[GroundTruth, ...]
    predictions: tuple[Prediction, ...]


@dataclass(frozen=True)
class MetricResult:
    gt_count: int
    prediction_count: int
    output_count: int
    output_rate: float | None
    correct_output_count: int
    correct_output_rate: float | None
    confident_error_count: int
    confident_error_rate: float | None
    ap50: float | None
    ap75: float | None
    map50_95: float | None
    recall50: float | None
    recall75: float | None


Selector = Callable[[str], tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate custom single-keypoint OKS AP50, AP75, and mAP50-95 "
            "for every keypoint and for Main/Opposite groups."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--output-conf",
        type=float,
        default=0.25,
        help="Box and keypoint confidence threshold used for output rate.",
    )
    parser.add_argument(
        "--correct-oks",
        type=float,
        default=0.5,
        help="Minimum OKS for a confident output to count as correct.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1 / len(KEYPOINT_NAMES),
        help="Single-keypoint OKS sigma (default: 1/22, matching generic 22-kpt training).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def yaml_scalar(text: str, key: str) -> str:
    prefix = f"{key}:"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip("'\"")
    raise ValueError(f"Missing '{key}' in dataset YAML")


def resolve_image_dir(data_path: Path, split: str) -> Path:
    text = data_path.read_text(encoding="utf-8")
    root = Path(yaml_scalar(text, "path")).expanduser()
    if not root.is_absolute():
        root = data_path.parent / root
    return (root / yaml_scalar(text, split)).resolve()


def corresponding_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain 'images': {image_path}") from exc
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_direction(image_path: Path) -> str:
    directions = [part for part in image_path.parts if part in {"Left", "Right"}]
    if len(directions) != 1:
        raise ValueError(f"Could not determine one direction from path: {image_path}")
    return directions[0]


def parse_ground_truth(label_path: Path, width: int, height: int) -> tuple[GroundTruth, ...]:
    ground_truths = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 + len(KEYPOINT_NAMES) * 3:
            raise ValueError(
                f"Expected 71 values in {label_path}:{line_number}, got {len(values)}"
            )
        _, _, _, box_width, box_height = values[:5]
        area = max(box_width * width * box_height * height, 1.0)
        keypoints = []
        for index in range(len(KEYPOINT_NAMES)):
            offset = 5 + index * 3
            x, y, visibility = values[offset : offset + 3]
            keypoints.append(Keypoint(x * width, y * height, visibility))
        ground_truths.append(GroundTruth(area, tuple(keypoints)))
    return tuple(ground_truths)


def tensor_values(value) -> list:
    if value is None:
        return []
    return value.detach().cpu().tolist()


def parse_predictions(result) -> tuple[Prediction, ...]:
    if result.boxes is None or result.keypoints is None:
        return ()

    box_scores = tensor_values(result.boxes.conf)
    coordinates = tensor_values(result.keypoints.xy)
    confidences = tensor_values(result.keypoints.conf)
    predictions = []
    for prediction_index, (box_score, points) in enumerate(zip(box_scores, coordinates)):
        if len(points) != len(KEYPOINT_NAMES):
            raise ValueError(
                f"Model returned {len(points)} keypoints; expected {len(KEYPOINT_NAMES)}"
            )
        point_confidences = (
            confidences[prediction_index]
            if prediction_index < len(confidences)
            else [1.0] * len(points)
        )
        keypoints = tuple(
            Keypoint(float(x), float(y), float(confidence))
            for (x, y), confidence in zip(points, point_confidences)
        )
        predictions.append(Prediction(float(box_score), keypoints))
    return tuple(predictions)


def single_keypoint_oks(prediction: Keypoint, target: Keypoint, area: float, sigma: float) -> float:
    squared_distance = (prediction.x - target.x) ** 2 + (prediction.y - target.y) ** 2
    denominator = 8 * sigma**2 * max(area, 1.0)
    return math.exp(-squared_distance / denominator)


def interpolated_ap(true_positives: list[int], false_positives: list[int], gt_count: int) -> tuple[float, float]:
    if gt_count == 0:
        return 0.0, 0.0

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []
    for true_positive, false_positive in zip(true_positives, false_positives):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / gt_count)
        precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))

    average_precision = 0.0
    for recall_level in (index / 100 for index in range(101)):
        candidates = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_level
        ]
        average_precision += max(candidates, default=0.0) / 101
    return min(average_precision, 1.0), (recalls[-1] if recalls else 0.0)


def evaluate_selection(
    records: Iterable[ImageRecord],
    selector: Selector,
    sigma: float,
    output_confidence: float = 0.25,
    correct_oks: float = 0.5,
    thresholds: tuple[float, ...] = OKS_THRESHOLDS,
) -> MetricResult:
    ground_truths: dict[tuple[str, int], list[tuple[int, Keypoint, float]]] = {}
    predictions: list[tuple[float, str, int, Keypoint]] = []
    confident_predictions: list[tuple[float, str, int, Keypoint]] = []
    output_counts: dict[tuple[str, int], int] = {}

    for record in records:
        selected_indices = selector(record.direction)
        for keypoint_index in selected_indices:
            target_key = (record.image_id, keypoint_index)
            visible_targets = []
            for ground_truth_index, ground_truth in enumerate(record.ground_truths):
                keypoint = ground_truth.keypoints[keypoint_index]
                if keypoint.visibility > 0:
                    visible_targets.append((ground_truth_index, keypoint, ground_truth.area))
            if not visible_targets:
                continue
            ground_truths[target_key] = visible_targets

            for prediction in record.predictions:
                keypoint = prediction.keypoints[keypoint_index]
                score = prediction.box_score * max(keypoint.visibility, 0.0)
                predictions.append((score, record.image_id, keypoint_index, keypoint))
                if (
                    prediction.box_score >= output_confidence
                    and keypoint.visibility >= output_confidence
                ):
                    output_counts[target_key] = output_counts.get(target_key, 0) + 1
                    confident_predictions.append(
                        (score, record.image_id, keypoint_index, keypoint)
                    )

    gt_count = sum(len(targets) for targets in ground_truths.values())
    output_count = sum(
        min(len(targets), output_counts.get(target_key, 0))
        for target_key, targets in ground_truths.items()
    )
    predictions.sort(key=lambda item: (-item[0], item[1], item[2]))
    confident_predictions.sort(key=lambda item: (-item[0], item[1], item[2]))
    if gt_count == 0:
        return MetricResult(
            gt_count=0,
            prediction_count=len(predictions),
            output_count=output_count,
            output_rate=None,
            correct_output_count=0,
            correct_output_rate=None,
            confident_error_count=0,
            confident_error_rate=None,
            ap50=None,
            ap75=None,
            map50_95=None,
            recall50=None,
            recall75=None,
        )

    confident_matches: set[tuple[str, int, int]] = set()
    correct_output_count = 0
    for _, image_id, keypoint_index, prediction in confident_predictions:
        best_match = None
        best_oks = -1.0
        for ground_truth_index, target, area in ground_truths[
            (image_id, keypoint_index)
        ]:
            match_key = (image_id, keypoint_index, ground_truth_index)
            if match_key in confident_matches:
                continue
            oks = single_keypoint_oks(prediction, target, area, sigma)
            if oks > best_oks:
                best_oks = oks
                best_match = match_key
        if best_match is not None and best_oks >= correct_oks:
            confident_matches.add(best_match)
            correct_output_count += 1

    confident_error_count = len(confident_predictions) - correct_output_count
    confident_error_rate = (
        confident_error_count / len(confident_predictions)
        if confident_predictions
        else None
    )

    aps = []
    recalls = []
    for threshold in thresholds:
        matched: set[tuple[str, int, int]] = set()
        true_positives = []
        false_positives = []
        for _, image_id, keypoint_index, prediction in predictions:
            target_key = (image_id, keypoint_index)
            best_match = None
            best_oks = -1.0
            for ground_truth_index, target, area in ground_truths[target_key]:
                match_key = (image_id, keypoint_index, ground_truth_index)
                if match_key in matched:
                    continue
                oks = single_keypoint_oks(prediction, target, area, sigma)
                if oks > best_oks:
                    best_oks = oks
                    best_match = match_key
            if best_match is not None and best_oks >= threshold:
                matched.add(best_match)
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        average_precision, recall = interpolated_ap(
            true_positives, false_positives, gt_count
        )
        aps.append(average_precision)
        recalls.append(recall)

    threshold_to_index = {threshold: index for index, threshold in enumerate(thresholds)}
    return MetricResult(
        gt_count=gt_count,
        prediction_count=len(predictions),
        output_count=output_count,
        output_rate=output_count / gt_count,
        correct_output_count=correct_output_count,
        correct_output_rate=correct_output_count / gt_count,
        confident_error_count=confident_error_count,
        confident_error_rate=confident_error_rate,
        ap50=aps[threshold_to_index[0.5]],
        ap75=aps[threshold_to_index[0.75]],
        map50_95=sum(aps) / len(aps),
        recall50=recalls[threshold_to_index[0.5]],
        recall75=recalls[threshold_to_index[0.75]],
    )


def fixed_selector(keypoint_index: int) -> Selector:
    return lambda _direction: (keypoint_index,)


def role_selector(role: str) -> Selector:
    def select(direction: str) -> tuple[int, ...]:
        side_index = 1 if direction == "Right" else 0
        if role == "opposite":
            side_index = 1 - side_index
        return tuple(pair[1 + side_index] for pair in SIDE_PAIRS)

    return select


def role_joint_selector(role: str, left_index: int, right_index: int) -> Selector:
    def select(direction: str) -> tuple[int, ...]:
        original_index = right_index if direction == "Right" else left_index
        if role == "original":
            return (original_index,)
        return (left_index if original_index == right_index else right_index,)

    return select


def metric_row(section: str, name: str, keypoint_id: str, result: MetricResult) -> dict:
    return {
        "section": section,
        "keypoint_id": keypoint_id,
        "name": name,
        "gt_count": result.gt_count,
        "prediction_count": result.prediction_count,
        "output_count": result.output_count,
        "output_rate": result.output_rate,
        "correct_output_count": result.correct_output_count,
        "correct_output_rate": result.correct_output_rate,
        "confident_error_count": result.confident_error_count,
        "confident_error_rate": result.confident_error_rate,
        "AP50": result.ap50,
        "AP75": result.ap75,
        "mAP50-95": result.map50_95,
        "Recall50": result.recall50,
        "Recall75": result.recall75,
    }


def calculate_metrics(
    records: tuple[ImageRecord, ...],
    sigma: float,
    output_confidence: float,
    correct_oks: float,
) -> list[dict]:
    rows = []
    for index, name in enumerate(KEYPOINT_NAMES):
        result = evaluate_selection(
            records, fixed_selector(index), sigma, output_confidence, correct_oks
        )
        rows.append(metric_row("anatomical", name, str(index), result))

    for role in ("original", "opposite"):
        result = evaluate_selection(
            records, role_selector(role), sigma, output_confidence, correct_oks
        )
        rows.append(metric_row("role_summary", f"{role}_all", "", result))
        for joint_name, left_index, right_index in SIDE_PAIRS:
            selector = role_joint_selector(role, left_index, right_index)
            result = evaluate_selection(
                records, selector, sigma, output_confidence, correct_oks
            )
            rows.append(metric_row(f"{role}_joint", joint_name, "", result))

    shared_result = evaluate_selection(
        records,
        lambda _direction: SHARED_KEYPOINTS,
        sigma,
        output_confidence,
        correct_oks,
    )
    rows.append(metric_row("role_summary", "shared_all", "", shared_result))
    return rows


def collect_records(model, image_dir: Path, args: argparse.Namespace) -> tuple[ImageRecord, ...]:
    image_paths = sorted(
        path.resolve()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No images found: {image_dir}")

    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )
    records = []
    seen_images = set()
    for result in results:
        image_path = Path(result.path).resolve()
        label_path = corresponding_label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Label not found: {label_path}")
        height, width = result.orig_shape
        records.append(
            ImageRecord(
                image_id=str(image_path),
                direction=image_direction(image_path),
                ground_truths=parse_ground_truth(label_path, width, height),
                predictions=parse_predictions(result),
            )
        )
        seen_images.add(image_path)

    missing = set(image_paths) - seen_images
    if missing:
        raise RuntimeError(f"Inference returned no result for {len(missing)} images")
    return tuple(records)


def output_directory(args: argparse.Namespace, model_path: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    model_name = (
        model_path.parent.parent.name
        if model_path.parent.name == "weights"
        else model_path.stem
    )
    return (
        PROJECT_ROOT
        / "runs"
        / "keypoint_evaluation"
        / f"{model_name}_{args.split}"
    )


def rounded(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def save_results(
    rows: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
    model_path: Path,
    data_path: Path,
    image_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_keypoint_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "metric_definition": "custom single-keypoint OKS AP",
        "model": str(model_path),
        "data": str(data_path),
        "dataset": data_path.parent.name,
        "split": args.split,
        "images": image_count,
        "sigma": args.sigma,
        "oks_thresholds": OKS_THRESHOLDS,
        "prediction_confidence": args.conf,
        "output_confidence": args.output_conf,
        "correct_output_oks": args.correct_oks,
        "rows": rows,
    }
    json_path = output_dir / "per_keypoint_metrics.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nRole summary")
    output_label = f"Out@{args.output_conf:g}"
    correct_label = f"Correct@{args.correct_oks:g}"
    error_label = "ConfErr"
    print(
        f"{'name':<16} {'GT':>5} {output_label:>9} "
        f"{correct_label:>11} {error_label:>8} "
        f"{'AP50':>8} {'AP75':>8} {'mAP50-95':>10}"
    )
    for row in rows:
        if row["section"] != "role_summary":
            continue
        print(
            f"{row['name']:<16} {row['gt_count']:>5} "
            f"{rounded(row['output_rate']):>9} "
            f"{rounded(row['correct_output_rate']):>11} "
            f"{rounded(row['confident_error_rate']):>8} "
            f"{rounded(row['AP50']):>8} {rounded(row['AP75']):>8} "
            f"{rounded(row['mAP50-95']):>10}"
        )
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


def load_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed.") from exc

    model = YOLO(str(model_path))
    if getattr(model, "task", None) != "pose":
        raise ValueError(f"Expected a pose checkpoint, got task={model.task!r}")
    model_yaml = getattr(getattr(model, "model", None), "yaml", {})
    kpt_shape = model_yaml.get("kpt_shape") if isinstance(model_yaml, dict) else None
    if kpt_shape and list(kpt_shape) != [len(KEYPOINT_NAMES), 3]:
        raise ValueError(f"Expected kpt_shape [22, 3], got {kpt_shape}")
    return model


def main() -> int:
    args = parse_args()
    try:
        model_path = args.model.expanduser().resolve()
        data_path = args.data.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
        if args.sigma <= 0:
            raise ValueError("--sigma must be greater than zero")
        if not 0 <= args.output_conf <= 1:
            raise ValueError("--output-conf must be between zero and one")
        if not 0 <= args.correct_oks <= 1:
            raise ValueError("--correct-oks must be between zero and one")

        image_dir = resolve_image_dir(data_path, args.split)
        model = load_model(model_path)
        records = collect_records(model, image_dir, args)
        rows = calculate_metrics(
            records, args.sigma, args.output_conf, args.correct_oks
        )
        save_results(
            rows,
            output_directory(args, model_path),
            args,
            model_path,
            data_path,
            len(records),
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
