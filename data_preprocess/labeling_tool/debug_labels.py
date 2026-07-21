from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Y, Button, Canvas, Frame, Label, Listbox, Scrollbar, StringVar, Tk, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: uv sync") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "finetuning_data"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAIN_COLOR = "#2563eb"
OPPOSITE_COLOR = "#f97316"
ERROR_COLOR = "#dc2626"
DISPLAY_SCALE_LIMIT = 0.8
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0
ZOOM_STEP = 1.25

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


@dataclass(frozen=True)
class PointRef:
    field_name: str
    index: int


@dataclass
class ProblemRecord:
    label_path: Path
    image_path: Path
    split_name: str
    direction: str
    stem: str
    problems: list[str]

    @property
    def display_name(self) -> str:
        return f"{self.split_name} | {self.direction} | {self.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix duplicate or missing keypoint labels in finetuning JSON files"
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root containing TrainingLabeled and ValidationLabeled",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print only the problematic records without opening the GUI",
    )
    return parser.parse_args()


def normalized(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def annotation_problems(data: dict) -> list[str]:
    problems: list[str] = []
    main = data.get("annotation_info")
    opposite = data.get("opposite_annotation_info")

    if not isinstance(main, list):
        return ["Main annotations are not a list"]
    if not isinstance(opposite, list):
        return ["Opposite annotations are not a list"]

    main_counts = Counter(str(item.get("label", "")) for item in main if isinstance(item, dict))
    opposite_counts = Counter(
        str(item.get("label", "")) for item in opposite if isinstance(item, dict)
    )

    missing = [label for label in KEYPOINT_LABELS if main_counts[label] == 0]
    duplicate_main = [label for label in KEYPOINT_LABELS if main_counts[label] > 1]
    duplicate_opposite = [label for label in KEYPOINT_LABELS if opposite_counts[label] > 1]
    unknown_main = sorted(label for label in main_counts if label not in KEYPOINT_LABEL_SET)
    unknown_opposite = sorted(label for label in opposite_counts if label not in KEYPOINT_LABEL_SET)

    for label in missing:
        problems.append(f"Main missing: {label}")
    for label in duplicate_main:
        problems.append(f"Main duplicate x{main_counts[label]}: {label}")
    for label in duplicate_opposite:
        problems.append(f"Opposite duplicate x{opposite_counts[label]}: {label}")
    for label in unknown_main:
        problems.append(f"Main unknown: {label}")
    for label in unknown_opposite:
        problems.append(f"Opposite unknown: {label}")
    return problems


def find_image(label_path: Path, data_root: Path) -> tuple[Path | None, str, str]:
    relative = label_path.relative_to(data_root)
    if len(relative.parts) < 4:
        return None, "", ""
    split_name = relative.parts[0]
    direction = relative.parts[-2]
    image_dir = data_root / split_name / "image" / direction
    target_stem = normalized(label_path.stem)
    for extension in IMAGE_EXTENSIONS:
        direct = image_dir / f"{label_path.stem}{extension}"
        if direct.exists():
            return direct, split_name, direction
    if image_dir.exists():
        for candidate in image_dir.iterdir():
            if candidate.suffix.lower() in IMAGE_EXTENSIONS and normalized(candidate.stem) == target_stem:
                return candidate, split_name, direction
    return None, split_name, direction


def scan_problem_records(data_root: Path) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    for label_path in sorted(data_root.glob("*Labeled/label/*/*.json")):
        try:
            data = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        problems = annotation_problems(data)
        if not problems:
            continue
        image_path, split_name, direction = find_image(label_path, data_root)
        if image_path is None:
            continue
        records.append(
            ProblemRecord(
                label_path=label_path,
                image_path=image_path,
                split_name=split_name,
                direction=direction,
                stem=label_path.stem,
                problems=problems,
            )
        )
    return records


def print_report(records: list[ProblemRecord]) -> None:
    print(f"Problem records: {len(records)}")
    for record in records:
        print(f"- {record.display_name}")
        for problem in record.problems:
            print(f"  {problem}")


class FinetuningLabelFixer:
    def __init__(self, root: Tk, data_root: Path) -> None:
        self.root = root
        self.data_root = data_root
        self.records: list[ProblemRecord] = []
        self.current_record: ProblemRecord | None = None
        self.current_json: dict | None = None
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_offset = (0, 0)
        self.image_scale = 1.0
        self.zoom_factor = 1.0
        self.show_names = True
        self.selected_point: PointRef | None = None
        self.annotation_refs: list[PointRef] = []
        self.selected_label = StringVar(value=KEYPOINT_LABELS[0])

        self.root.title("Finetuning Label Error Fixer")
        self.root.geometry("1280x780")
        self.root.minsize(1050, 680)
        self.build_layout()
        self.rescan()

    def build_layout(self) -> None:
        left_panel = Frame(self.root, width=330, padx=8, pady=8)
        left_panel.pack(side=LEFT, fill=Y)
        left_panel.pack_propagate(False)

        Label(left_panel, text="Problem Records", anchor="w").pack(fill="x")
        record_frame = Frame(left_panel)
        record_frame.pack(fill=BOTH, expand=True)
        self.record_list = Listbox(record_frame, exportselection=False)
        self.record_list.pack(side=LEFT, fill=BOTH, expand=True)
        record_scroll = Scrollbar(record_frame, command=self.record_list.yview)
        record_scroll.pack(side=RIGHT, fill=Y)
        self.record_list.config(yscrollcommand=record_scroll.set)
        self.record_list.bind("<<ListboxSelect>>", self.on_record_select)

        Button(left_panel, text="Re-scan", command=self.rescan).pack(fill="x", pady=(8, 0))

        center = Frame(self.root)
        center.pack(side=LEFT, fill=BOTH, expand=True)
        toolbar = Frame(center, padx=8, pady=8)
        toolbar.pack(fill="x")
        Button(toolbar, text="Prev Error", command=self.previous_record).pack(side=LEFT)
        Button(toolbar, text="Next Error", command=self.next_record).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Save Fixed JSON", command=self.save_current).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Reload", command=self.reload_current).pack(side=LEFT, padx=(6, 0))
        self.names_button = Button(toolbar, text="Names: ON", command=self.toggle_names)
        self.names_button.pack(side=LEFT, padx=(6, 0))
        self.status = Label(toolbar, text="", anchor="w")
        self.status.pack(side=LEFT, fill="x", expand=True, padx=(12, 0))

        self.canvas = Canvas(center, background="#20242a", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _: self.render_image())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        zoom_controls = Frame(self.canvas, background="#111827")
        Button(zoom_controls, text="-", width=3, command=self.zoom_out).pack(side=LEFT)
        Button(zoom_controls, text="+", width=3, command=self.zoom_in).pack(side=LEFT)
        zoom_controls.place(relx=1.0, x=-12, y=12, anchor="ne")

        right_panel = Frame(self.root, width=390, padx=8, pady=8)
        right_panel.pack(side=RIGHT, fill=Y)
        right_panel.pack_propagate(False)

        Label(right_panel, text="Problems", anchor="w").pack(fill="x")
        self.problem_list = Listbox(right_panel, height=6, exportselection=False)
        self.problem_list.pack(fill="x", pady=(0, 8))

        Label(right_panel, text="Annotations", anchor="w").pack(fill="x")
        annotation_frame = Frame(right_panel)
        annotation_frame.pack(fill=BOTH, expand=True)
        self.annotation_list = Listbox(annotation_frame, exportselection=False)
        self.annotation_list.pack(side=LEFT, fill=BOTH, expand=True)
        annotation_scroll = Scrollbar(annotation_frame, command=self.annotation_list.yview)
        annotation_scroll.pack(side=RIGHT, fill=Y)
        self.annotation_list.config(yscrollcommand=annotation_scroll.set)
        self.annotation_list.bind("<<ListboxSelect>>", self.on_annotation_select)

        Label(right_panel, text="Change Selected Point To", anchor="w").pack(fill="x", pady=(8, 0))
        self.label_combo = ttk.Combobox(
            right_panel,
            textvariable=self.selected_label,
            values=KEYPOINT_LABELS,
            state="readonly",
        )
        self.label_combo.pack(fill="x")
        Button(right_panel, text="Apply Label", command=self.apply_selected_label).pack(fill="x", pady=(6, 0))
        Button(right_panel, text="Use Missing Main Label", command=self.apply_missing_main_label).pack(fill="x", pady=(6, 0))
        Button(right_panel, text="Delete Selected Point", command=self.delete_selected_point).pack(fill="x", pady=(6, 0))

        self.root.bind("<Command-s>", lambda _: self.save_current())
        self.root.bind("<Control-s>", lambda _: self.save_current())

    def rescan(self, preferred_path: Path | None = None) -> None:
        self.records = scan_problem_records(self.data_root)
        self.record_list.delete(0, "end")
        for record in self.records:
            self.record_list.insert("end", record.display_name)

        if not self.records:
            self.current_record = None
            self.current_json = None
            self.original_image = None
            self.selected_point = None
            self.problem_list.delete(0, "end")
            self.annotation_list.delete(0, "end")
            self.status.config(text="No duplicate or missing label problems remain.")
            self.render_image()
            return

        index = 0
        if preferred_path is not None:
            for candidate_index, record in enumerate(self.records):
                if record.label_path == preferred_path:
                    index = candidate_index
                    break
        self.record_list.selection_set(index)
        self.record_list.see(index)
        self.load_record(self.records[index])

    def on_record_select(self, _: object) -> None:
        selection = self.record_list.curselection()
        if selection:
            self.load_record(self.records[selection[0]])

    def load_record(self, record: ProblemRecord) -> None:
        self.current_record = record
        self.current_json = json.loads(record.label_path.read_text(encoding="utf-8-sig"))
        self.original_image = Image.open(record.image_path).convert("RGB")
        self.selected_point = None
        self.zoom_factor = 1.0
        self.refresh_details()
        self.status.config(text=f"{record.split_name} / {record.direction} / {record.stem}")
        self.render_image()

    def reload_current(self) -> None:
        if self.current_record is not None:
            self.load_record(self.current_record)

    def current_annotations(self, field_name: str) -> list[dict]:
        if self.current_json is None:
            return []
        value = self.current_json.get(field_name, [])
        return value if isinstance(value, list) else []

    def current_problems(self) -> list[str]:
        return annotation_problems(self.current_json or {})

    def duplicate_labels(self, field_name: str) -> set[str]:
        counts = Counter(str(item.get("label", "")) for item in self.current_annotations(field_name))
        return {label for label, count in counts.items() if count > 1}

    def refresh_details(self) -> None:
        self.problem_list.delete(0, "end")
        for problem in self.current_problems():
            self.problem_list.insert("end", problem)

        self.annotation_list.delete(0, "end")
        self.annotation_refs = []
        for field_name, title in (
            ("annotation_info", "Main"),
            ("opposite_annotation_info", "Opp"),
        ):
            duplicates = self.duplicate_labels(field_name)
            for index, item in enumerate(self.current_annotations(field_name)):
                label = str(item.get("label", ""))
                marker = "ERROR" if label in duplicates or label not in KEYPOINT_LABEL_SET else ""
                number = KEYPOINT_NUMBERS.get(label, "?")
                self.annotation_refs.append(PointRef(field_name, index))
                self.annotation_list.insert(
                    "end",
                    f"{marker:5} {title} {number}. {label}",
                )

        if self.selected_point is not None:
            try:
                list_index = self.annotation_refs.index(self.selected_point)
            except ValueError:
                self.selected_point = None
            else:
                self.annotation_list.selection_set(list_index)
                self.annotation_list.see(list_index)

    def selected_item(self) -> dict | None:
        if self.selected_point is None:
            return None
        annotations = self.current_annotations(self.selected_point.field_name)
        if not 0 <= self.selected_point.index < len(annotations):
            return None
        return annotations[self.selected_point.index]

    def on_annotation_select(self, _: object) -> None:
        selection = self.annotation_list.curselection()
        if not selection:
            return
        self.select_point(self.annotation_refs[selection[0]])

    def select_point(self, point_ref: PointRef) -> None:
        self.selected_point = point_ref
        item = self.selected_item()
        if item is not None and item.get("label") in KEYPOINT_LABEL_SET:
            self.selected_label.set(str(item["label"]))
        self.refresh_details()
        self.render_image()

    def apply_selected_label(self) -> None:
        item = self.selected_item()
        if item is None:
            messagebox.showinfo("Select a point", "Select an existing point first.")
            return
        item["label"] = self.selected_label.get()
        self.refresh_details()
        self.render_image()

    def apply_missing_main_label(self) -> None:
        if self.selected_point is None or self.selected_point.field_name != "annotation_info":
            messagebox.showinfo("Select a Main point", "Select the incorrectly labeled Main point first.")
            return
        main_counts = Counter(
            str(item.get("label", "")) for item in self.current_annotations("annotation_info")
        )
        missing = [label for label in KEYPOINT_LABELS if main_counts[label] == 0]
        selected = self.selected_item()
        if len(missing) != 1 or selected is None or main_counts[str(selected.get("label", ""))] < 2:
            messagebox.showinfo(
                "No unique suggestion",
                "This action requires exactly one missing Main label and a selected duplicate point.",
            )
            return
        selected["label"] = missing[0]
        self.selected_label.set(missing[0])
        self.refresh_details()
        self.render_image()

    def delete_selected_point(self) -> None:
        if self.selected_point is None:
            messagebox.showinfo("Select a point", "Select an existing point first.")
            return
        annotations = self.current_annotations(self.selected_point.field_name)
        del annotations[self.selected_point.index]
        self.selected_point = None
        self.refresh_details()
        self.render_image()

    def save_current(self) -> None:
        if self.current_record is None or self.current_json is None:
            return
        problems = self.current_problems()
        if problems:
            messagebox.showwarning(
                "Problems remain",
                "Fix every duplicate or missing label before saving:\n\n" + "\n".join(problems),
            )
            return

        order = {label: index for index, label in enumerate(KEYPOINT_LABELS)}
        self.current_json["annotation_info"].sort(key=lambda item: order[str(item["label"])])
        self.current_json["opposite_annotation_info"].sort(
            key=lambda item: order[str(item["label"])]
        )

        backup_path = self.current_record.label_path.with_suffix(".json.bak")
        if not backup_path.exists():
            shutil.copy2(self.current_record.label_path, backup_path)
        self.current_record.label_path.write_text(
            json.dumps(self.current_json, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        fixed_path = self.current_record.label_path
        self.rescan()
        messagebox.showinfo(
            "Saved",
            f"Fixed JSON saved.\nBackup: {backup_path.name}\nRemaining problem files: {len(self.records)}",
        )
        if self.records:
            self.status.config(text=f"Saved {fixed_path.name}. Select the next error.")

    def render_image(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No problem records",
                fill="#e5e7eb",
                font=("Arial", 18),
            )
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        base_scale = min(
            (canvas_width * DISPLAY_SCALE_LIMIT) / self.original_image.width,
            (canvas_height * DISPLAY_SCALE_LIMIT) / self.original_image.height,
            1.0,
        )
        self.image_scale = base_scale * self.zoom_factor
        display_width = max(1, int(self.original_image.width * self.image_scale))
        display_height = max(1, int(self.original_image.height * self.image_scale))
        self.image_offset = (
            (canvas_width - display_width) // 2,
            (canvas_height - display_height) // 2,
        )
        self.display_image = self.original_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(*self.image_offset, anchor="nw", image=self.tk_image)
        self.draw_points()

    def draw_points(self) -> None:
        for field_name, color in (
            ("annotation_info", MAIN_COLOR),
            ("opposite_annotation_info", OPPOSITE_COLOR),
        ):
            duplicates = self.duplicate_labels(field_name)
            for index, item in enumerate(self.current_annotations(field_name)):
                try:
                    x = self.image_offset[0] + float(item["x"]) * self.original_image.width * self.image_scale
                    y = self.image_offset[1] + float(item["y"]) * self.original_image.height * self.image_scale
                except (KeyError, TypeError, ValueError):
                    continue
                point_ref = PointRef(field_name, index)
                selected = point_ref == self.selected_point
                label = str(item.get("label", ""))
                has_error = label in duplicates or label not in KEYPOINT_LABEL_SET
                radius = 8 if selected else 6
                outline = "#ffffff" if selected else ERROR_COLOR if has_error else "#ffffff"
                width = 4 if selected else 3 if has_error else 2
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline=outline,
                    width=width,
                )
                text = f"{KEYPOINT_NUMBERS.get(label, '?')}. {label}" if self.show_names else str(KEYPOINT_NUMBERS.get(label, "?"))
                self.draw_point_label(x, y, text, has_error)

    def draw_point_label(self, x: float, y: float, text: str, has_error: bool) -> None:
        text_id = self.canvas.create_text(
            x + 10,
            y - 10,
            text=text,
            fill="#111827",
            anchor="w",
            font=("Arial", 10, "bold"),
        )
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            return
        rect_id = self.canvas.create_rectangle(
            bbox[0] - 3,
            bbox[1] - 3,
            bbox[2] + 3,
            bbox[3] + 3,
            fill="#fee2e2" if has_error else "#ffffff",
            outline=ERROR_COLOR if has_error else "#d1d5db",
        )
        self.canvas.tag_raise(text_id, rect_id)

    def on_canvas_click(self, event: object) -> None:
        nearest: PointRef | None = None
        nearest_distance = 16.0
        for field_name in ("annotation_info", "opposite_annotation_info"):
            for index, item in enumerate(self.current_annotations(field_name)):
                try:
                    x = self.image_offset[0] + float(item["x"]) * self.original_image.width * self.image_scale
                    y = self.image_offset[1] + float(item["y"]) * self.original_image.height * self.image_scale
                except (KeyError, TypeError, ValueError):
                    continue
                distance = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
                if distance <= nearest_distance:
                    nearest = PointRef(field_name, index)
                    nearest_distance = distance
        if nearest is not None:
            self.select_point(nearest)

    def toggle_names(self) -> None:
        self.show_names = not self.show_names
        self.names_button.config(text=f"Names: {'ON' if self.show_names else 'OFF'}")
        self.render_image()

    def zoom_in(self) -> None:
        self.zoom_factor = min(ZOOM_MAX, self.zoom_factor * ZOOM_STEP)
        self.render_image()

    def zoom_out(self) -> None:
        self.zoom_factor = max(ZOOM_MIN, self.zoom_factor / ZOOM_STEP)
        self.render_image()

    def previous_record(self) -> None:
        selection = self.record_list.curselection()
        if selection and selection[0] > 0:
            index = selection[0] - 1
            self.record_list.selection_clear(0, "end")
            self.record_list.selection_set(index)
            self.record_list.see(index)
            self.load_record(self.records[index])

    def next_record(self) -> None:
        selection = self.record_list.curselection()
        if selection and selection[0] < len(self.records) - 1:
            index = selection[0] + 1
            self.record_list.selection_clear(0, "end")
            self.record_list.selection_set(index)
            self.record_list.see(index)
            self.load_record(self.records[index])


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    records = scan_problem_records(data_root)
    print_report(records)
    if args.report_only:
        return

    root = Tk()
    FinetuningLabelFixer(root, data_root)
    root.mainloop()


if __name__ == "__main__":
    main()
