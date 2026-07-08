from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Y, Canvas, Frame, Label, Listbox, Scrollbar, StringVar, Tk, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: uv add pillow") from exc


DATA_ROOT = "labeled_data_debug"
DIRECTIONS = ("Front", "Back", "Left", "Right")
OPPOSITE_DIRECTIONS = {"Left": "Right", "Right": "Left"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_SCALE_LIMIT = 0.8
MAIN_COLOR = "#2563eb"
OPPOSITE_COLOR = "#f97316"

KEYPOINT_LABELS = [
    "Ear",
    "T13 Spinous precess",
    "Dorsal scapular spine",
    "Acromion/Greater tubercle",
    "Lateral humeral epicondyle",
    "Ulnar styloid process",
    "Distal lateral aspect of fifth metacarpal bone",
    "Iliac crest",
    "Femoral greater trochanter",
    "Femorotibial joint",
    "Lateral malleolus of the distal tibia",
    "Distal lateral aspect of the fifth metatarsus",
]
KEYPOINT_LABEL_SET = set(KEYPOINT_LABELS)
KEYPOINT_NUMBERS = {label: index + 1 for index, label in enumerate(KEYPOINT_LABELS)}


@dataclass
class LabelRecord:
    direction: str
    stem: str
    image_path: Path | None = None
    label_path: Path | None = None
    annotations: list[dict] = field(default_factory=list)
    opposite_annotations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def display_name(self) -> str:
        status = "OK" if self.ok else "ERR"
        return f"[{status}] {self.direction}/{self.stem} (main {len(self.annotations)}, opp {len(self.opposite_annotations)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate dog pose labels")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Root containing image/{direction} and label/{direction}")
    parser.add_argument("--direction", choices=(*DIRECTIONS, "all"), default="all")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--report-json")
    return parser.parse_args()


def normalized_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        path = folder / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def scan_records(data_root: Path, direction_filter: str = "all") -> list[LabelRecord]:
    directions = DIRECTIONS if direction_filter == "all" else (direction_filter,)
    records: list[LabelRecord] = []

    for direction in directions:
        image_dir = data_root / "image" / direction
        label_dir = data_root / "label" / direction
        stems: set[str] = set()
        if image_dir.exists():
            stems.update(path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if label_dir.exists():
            stems.update(path.stem for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

        for stem in sorted(stems):
            image_path = find_image_by_stem(image_dir, stem) if image_dir.exists() else None
            label_path = label_dir / f"{stem}.json"
            record = LabelRecord(
                direction=direction,
                stem=stem,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
            )
            validate_record(record)
            records.append(record)

    return records


def validate_record(record: LabelRecord) -> None:
    if record.image_path is None:
        record.errors.append("missing image file")
    if record.label_path is None:
        record.errors.append("missing label json")
        return

    try:
        data = json.loads(record.label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        record.errors.append(f"invalid json: {exc}")
        return

    annotation_info = data.get("annotation_info")
    if not isinstance(annotation_info, list):
        record.errors.append("annotation_info must be a list")
    else:
        record.annotations = annotation_info
        validate_annotations(record, annotation_info, "annotation_info")

    opposite_annotation_info = data.get("opposite_annotation_info", [])
    if opposite_annotation_info is None:
        opposite_annotation_info = []
    if not isinstance(opposite_annotation_info, list):
        record.errors.append("opposite_annotation_info must be a list")
    else:
        record.opposite_annotations = opposite_annotation_info
        validate_annotations(record, opposite_annotation_info, "opposite_annotation_info")

    if isinstance(annotation_info, list) and not annotation_info and not record.opposite_annotations:
        record.warnings.append("annotation_info and opposite_annotation_info are both empty")
    if record.direction not in ("Left", "Right") and record.opposite_annotations:
        record.errors.append("opposite_annotation_info is only allowed for Left/Right labels")
    if record.opposite_annotations:
        expected_direction = OPPOSITE_DIRECTIONS.get(record.direction)
        actual_direction = data.get("opposite_direction")
        if actual_direction != expected_direction:
            record.errors.append(f"opposite_direction must be {expected_direction}: {actual_direction}")

    image_info = data.get("image_info")
    if not isinstance(image_info, dict):
        record.warnings.append("image_info is missing or not an object")
        return
    filename = image_info.get("filename")
    if filename and normalized_text(filename) != normalized_text(record.stem):
        record.warnings.append(f"image_info.filename differs from file name: {filename}")


def validate_annotations(record: LabelRecord, annotations: list[dict], field_name: str) -> None:
    for index, item in enumerate(annotations):
        prefix = f"{field_name}[{index}]"
        if not isinstance(item, dict):
            record.errors.append(f"{prefix} must be an object")
            continue
        label = item.get("label")
        if label not in KEYPOINT_LABEL_SET:
            record.errors.append(f"{prefix}.label is invalid: {label}")
        for axis in ("x", "y"):
            try:
                value = float(item.get(axis))
            except (TypeError, ValueError):
                record.errors.append(f"{prefix}.{axis} must be numeric")
                continue
            if not 0 <= value <= 1:
                record.errors.append(f"{prefix}.{axis} out of range: {value}")


class LabelCheckerApp:
    def __init__(self, root: Tk, records: list[LabelRecord]) -> None:
        self.root = root
        self.records = records
        self.filtered_records = records
        self.filter_var = StringVar(value="all")
        self.show_pose_names = False
        self.current_record: LabelRecord | None = None
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_offset = (0, 0)
        self.image_scale = 1.0

        self.root.title("Dog Pose Label Checker")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.build_layout()
        self.apply_filter()

    def build_layout(self) -> None:
        left_panel = Frame(self.root, width=360, padx=8, pady=8)
        left_panel.pack(side=LEFT, fill=Y)

        Label(left_panel, text="Filter", anchor="w").pack(fill="x")
        combo = ttk.Combobox(left_panel, textvariable=self.filter_var, values=("all", "errors", "warnings", *DIRECTIONS), state="readonly")
        combo.pack(fill="x", pady=(0, 8))
        combo.bind("<<ComboboxSelected>>", lambda _: self.apply_filter())

        Label(left_panel, text="Labels", anchor="w").pack(fill="x")
        list_frame = Frame(left_panel)
        list_frame.pack(fill=BOTH, expand=True)
        self.record_list = Listbox(list_frame, exportselection=False)
        self.record_list.pack(side=LEFT, fill=BOTH, expand=True)
        scroll = Scrollbar(list_frame, command=self.record_list.yview)
        scroll.pack(side=RIGHT, fill=Y)
        self.record_list.config(yscrollcommand=scroll.set)
        self.record_list.bind("<<ListboxSelect>>", self.on_record_select)

        center = Frame(self.root)
        center.pack(side=LEFT, fill=BOTH, expand=True)
        toolbar = Frame(center, padx=8, pady=8)
        toolbar.pack(fill="x")
        self.pose_name_button = ttk.Button(toolbar, text="Names: OFF", command=self.toggle_pose_names)
        self.pose_name_button.pack(side=LEFT)
        self.status = Label(toolbar, text="", anchor="w")
        self.status.pack(side=LEFT, fill="x", expand=True, padx=(12, 0))
        self.canvas = Canvas(center, background="#20242a", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _: self.render_image())

        right_panel = Frame(self.root, width=360, padx=8, pady=8)
        right_panel.pack(side=RIGHT, fill=Y)
        Label(right_panel, text="Problems", anchor="w").pack(fill="x")
        self.problem_list = Listbox(right_panel, exportselection=False)
        self.problem_list.pack(fill=BOTH, expand=True)

    def apply_filter(self) -> None:
        value = self.filter_var.get()
        if value == "errors":
            self.filtered_records = [record for record in self.records if record.errors]
        elif value == "warnings":
            self.filtered_records = [record for record in self.records if record.warnings]
        elif value in DIRECTIONS:
            self.filtered_records = [record for record in self.records if record.direction == value]
        else:
            self.filtered_records = self.records

        self.record_list.delete(0, "end")
        for record in self.filtered_records:
            self.record_list.insert("end", record.display_name)

        if self.filtered_records:
            self.record_list.selection_set(0)
            self.load_record(self.filtered_records[0])
        else:
            self.current_record = None
            self.original_image = None
            self.problem_list.delete(0, "end")
            self.status.config(text="No records")
            self.render_image()

    def on_record_select(self, _: object) -> None:
        selection = self.record_list.curselection()
        if selection:
            self.load_record(self.filtered_records[selection[0]])

    def load_record(self, record: LabelRecord) -> None:
        self.current_record = record
        self.problem_list.delete(0, "end")
        for message in record.errors:
            self.problem_list.insert("end", f"ERROR: {message}")
        for message in record.warnings:
            self.problem_list.insert("end", f"WARN: {message}")

        self.original_image = Image.open(record.image_path).convert("RGB") if record.image_path else None
        self.status.config(
            text=f"{record.direction}/{record.stem} - {len(record.annotations)} main / "
            f"{len(record.opposite_annotations)} opposite"
        )
        self.render_image()

    def render_image(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text="No image", fill="#e5e7eb", font=("Arial", 18))
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = min(
            (canvas_width * DISPLAY_SCALE_LIMIT) / self.original_image.width,
            (canvas_height * DISPLAY_SCALE_LIMIT) / self.original_image.height,
            1.0,
        )
        display_width = max(1, int(self.original_image.width * scale))
        display_height = max(1, int(self.original_image.height * scale))
        self.image_scale = scale
        self.image_offset = ((canvas_width - display_width) // 2, (canvas_height - display_height) // 2)
        self.display_image = self.original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(*self.image_offset, anchor="nw", image=self.tk_image)
        self.draw_annotations()

    def draw_annotations(self) -> None:
        if self.current_record is None or self.original_image is None:
            return
        offset_x, offset_y = self.image_offset
        self.draw_annotation_set(self.current_record.annotations, offset_x, offset_y, MAIN_COLOR, "")
        self.draw_annotation_set(self.current_record.opposite_annotations, offset_x, offset_y, OPPOSITE_COLOR, "O")

    def draw_annotation_set(self, annotations: list[dict], offset_x: int, offset_y: int, color: str, prefix: str) -> None:
        for item in annotations:
            try:
                x = offset_x + float(item["x"]) * self.original_image.width * self.image_scale
                y = offset_y + float(item["y"]) * self.original_image.height * self.image_scale
            except (KeyError, TypeError, ValueError):
                continue
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="#ffffff", width=2)
            label = str(item.get("label", "")) if self.show_pose_names else ""
            keypoint_number = KEYPOINT_NUMBERS.get(str(item.get("label", "")), "?")
            display_number = f"{prefix}{keypoint_number}" if prefix else keypoint_number
            self.draw_annotation_label(x, y, display_number, label)

    def draw_annotation_label(self, x: float, y: float, number: int | str, label: str) -> None:
        text = f"{number}. {label}" if label else str(number)
        text_id = self.canvas.create_text(x + 10, y - 10, text=text, fill="#111827", anchor="w", font=("Arial", 11, "bold"))
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            return
        pad = 3
        rect_id = self.canvas.create_rectangle(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad, fill="#ffffff", outline="#d1d5db")
        self.canvas.tag_raise(text_id, rect_id)

    def toggle_pose_names(self) -> None:
        self.show_pose_names = not self.show_pose_names
        self.pose_name_button.config(text=f"Names: {'ON' if self.show_pose_names else 'OFF'}")
        self.render_image()


def summarize(records: list[LabelRecord]) -> dict:
    return {
        "total": len(records),
        "ok": sum(1 for record in records if record.ok),
        "errors": sum(1 for record in records if record.errors),
        "warnings": sum(1 for record in records if record.warnings),
        "records": [
            {
                "direction": record.direction,
                "stem": record.stem,
                "image_path": str(record.image_path) if record.image_path else None,
                "label_path": str(record.label_path) if record.label_path else None,
                "annotation_count": len(record.annotations),
                "opposite_annotation_count": len(record.opposite_annotations),
                "errors": record.errors,
                "warnings": record.warnings,
            }
            for record in records
        ],
    }


def print_summary(summary: dict) -> None:
    print(f"Total: {summary['total']}")
    print(f"OK: {summary['ok']}")
    print(f"With errors: {summary['errors']}")
    print(f"With warnings: {summary['warnings']}")
    for record in summary["records"]:
        if not record["errors"] and not record["warnings"]:
            continue
        print(f"- {record['direction']}/{record['stem']}")
        for message in record["errors"]:
            print(f"  ERROR: {message}")
        for message in record["warnings"]:
            print(f"  WARN: {message}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    records = scan_records(data_root, args.direction)
    summary = summarize(records)
    print_summary(summary)

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report_only:
        return

    root = Tk()
    LabelCheckerApp(root, records)
    root.mainloop()


if __name__ == "__main__":
    main()
