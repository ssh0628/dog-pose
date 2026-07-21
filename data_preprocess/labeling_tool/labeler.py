from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Y, Button, Canvas, Frame, Label, Listbox, Scrollbar, StringVar, Tk, ttk, messagebox

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: uv add pillow") from exc


INPUT_ROOT = "labeled_data/TrainingLabeled"
OUTPUT_ROOT = "labeled_data_debug"

DIRECTIONS = ("Front", "Back", "Left", "Right")
LABEL_SETS = ("Main", "Opposite")
SIDE_DIRECTIONS = {"Left", "Right"}
OPPOSITE_DIRECTIONS = {"Left": "Right", "Right": "Left"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_SCALE_LIMIT = 0.8
MAIN_COLOR = "#2563eb"
OPPOSITE_COLOR = "#f97316"
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
KEYPOINT_NUMBERS = {label: index + 1 for index, label in enumerate(KEYPOINT_LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dog pose desktop labeler")
    parser.add_argument("--input-root", default=INPUT_ROOT, help="Folder containing raw images to label")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="Output root. Saves to image/{direction} and label/{direction}")
    parser.add_argument("--image-root", dest="input_root", help=argparse.SUPPRESS)
    return parser.parse_args()


def make_empty_label(image_path: Path, width: int, height: int) -> dict:
    return {
        "image_info": {
            "filename": image_path.stem,
            "hospital": "",
            "file_format": image_path.suffix.lstrip(".").lower(),
            "image_size": image_path.stat().st_size,
            "device": "",
            "resolution": f"{width}X{height}",
        },
        "annotation_info": [],
        "opposite_annotation_info": [],
        "opposite_direction": "",
        "pet_medical_record_info": [],
        "sensor_values": [],
        "timestamp": int(time.time()),
        "size": "",
        "severity": 0,
        "age": "",
        "dog_type": "",
    }


class DogPoseLabeler:
    def __init__(self, root: Tk, input_root: Path, output_root: Path) -> None:
        self.root = root
        self.input_root = input_root
        self.output_root = output_root

        self.direction = StringVar(value=DIRECTIONS[0])
        self.label_set = StringVar(value=LABEL_SETS[0])
        self.active_label_set = self.label_set.get()
        self.selected_label = StringVar(value=KEYPOINT_LABELS[0])
        self.images: list[Path] = []
        self.current_image: Path | None = None
        self.current_json: dict | None = None
        self.annotations: list[dict[str, str]] = []
        self.main_annotations: list[dict[str, str]] = []
        self.opposite_annotations: list[dict[str, str]] = []

        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_offset = (0, 0)
        self.image_scale = 1.0
        self.zoom_factor = 1.0
        self.selected_point_index = -1
        self.dragging_point_index = -1
        self.show_pose_names = False

        self.root.title("Dog Pose Labeler")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.build_layout()
        self.load_images()

    def build_layout(self) -> None:
        left_panel = Frame(self.root, width=280, padx=8, pady=8)
        left_panel.pack(side=LEFT, fill=Y)

        Label(left_panel, text="Direction", anchor="w").pack(fill="x")
        direction_combo = ttk.Combobox(left_panel, textvariable=self.direction, values=DIRECTIONS, state="readonly")
        direction_combo.pack(fill="x", pady=(0, 8))
        direction_combo.bind("<<ComboboxSelected>>", lambda _: self.reload_current_label())

        Label(left_panel, text="Label Set", anchor="w").pack(fill="x")
        label_set_combo = ttk.Combobox(left_panel, textvariable=self.label_set, values=LABEL_SETS, state="readonly")
        label_set_combo.pack(fill="x", pady=(0, 8))
        label_set_combo.bind("<<ComboboxSelected>>", lambda _: self.switch_label_set())

        Button(left_panel, text="Refresh", command=self.load_images).pack(fill="x", pady=(0, 8))

        Label(left_panel, text="Images", anchor="w").pack(fill="x")
        list_frame = Frame(left_panel)
        list_frame.pack(fill=BOTH, expand=True)
        self.image_list = Listbox(list_frame, exportselection=False)
        self.image_list.pack(side=LEFT, fill=BOTH, expand=True)
        image_scroll = Scrollbar(list_frame, command=self.image_list.yview)
        image_scroll.pack(side=RIGHT, fill=Y)
        self.image_list.config(yscrollcommand=image_scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self.on_image_select)

        center = Frame(self.root)

        toolbar = Frame(center, padx=8, pady=8)
        toolbar.pack(fill="x")
        Button(toolbar, text="Prev", command=self.previous_image).pack(side=LEFT)
        Button(toolbar, text="Next", command=self.next_image).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Save", command=self.save_label).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Undo", command=self.undo_point).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Clear", command=self.clear_points).pack(side=LEFT, padx=(6, 0))
        self.pose_name_button = Button(toolbar, text="Names: OFF", command=self.toggle_pose_names)
        self.pose_name_button.pack(side=LEFT, padx=(6, 0))
        self.status = Label(toolbar, text="", anchor="w")
        self.status.pack(side=LEFT, fill="x", expand=True, padx=(12, 0))

        self.canvas = Canvas(center, background="#20242a", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _: self.render_image())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        zoom_controls = Frame(self.canvas, background="#111827")
        Button(zoom_controls, text="-", width=3, command=self.zoom_out).pack(side=LEFT)
        Button(zoom_controls, text="+", width=3, command=self.zoom_in).pack(side=LEFT)
        zoom_controls.place(relx=1.0, x=-12, y=12, anchor="ne")

        right_panel = Frame(self.root, width=360, padx=8, pady=8)
        right_panel.pack(side=RIGHT, fill=Y)
        right_panel.pack_propagate(False)

        Label(right_panel, text="Keypoints", anchor="w").pack(fill="x")
        self.keypoint_list = Listbox(right_panel, exportselection=False, height=14)
        self.keypoint_list.pack(fill="x", pady=(0, 8))
        for label in KEYPOINT_LABELS:
            self.keypoint_list.insert("end", f"{KEYPOINT_NUMBERS[label]}. {label}")
        self.keypoint_list.selection_set(0)
        self.keypoint_list.bind("<<ListboxSelect>>", self.on_keypoint_select)

        Label(right_panel, text="Annotations", anchor="w").pack(fill="x")
        point_frame = Frame(right_panel)
        point_frame.pack(fill=BOTH, expand=True)
        self.point_list = Listbox(point_frame, exportselection=False)
        self.point_list.pack(side=LEFT, fill=BOTH, expand=True)
        point_scroll = Scrollbar(point_frame, command=self.point_list.yview)
        point_scroll.pack(side=RIGHT, fill=Y)
        self.point_list.config(yscrollcommand=point_scroll.set)
        self.point_list.bind("<<ListboxSelect>>", self.on_point_select)

        center.pack(side=LEFT, fill=BOTH, expand=True)

        self.root.bind("<Delete>", lambda _: self.delete_selected_point())
        self.root.bind("<BackSpace>", lambda _: self.delete_selected_point())
        self.root.bind("<Command-s>", lambda _: self.save_label())
        self.root.bind("<Control-s>", lambda _: self.save_label())

    def load_images(self) -> None:
        self.images = [
            path for path in sorted(self.input_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        self.image_list.delete(0, "end")
        for path in self.images:
            self.image_list.insert("end", str(path.relative_to(self.input_root)))

        if self.images:
            self.image_list.selection_set(0)
            self.load_image(self.images[0])
        else:
            self.current_image = None
            self.current_json = None
            self.annotations = []
            self.main_annotations = []
            self.opposite_annotations = []
            self.original_image = None
            self.update_point_list()
            self.render_image()
            self.set_status(f"No images in {self.input_root}")

    def on_image_select(self, _: object) -> None:
        selection = self.image_list.curselection()
        if selection:
            self.load_image(self.images[selection[0]])

    def load_image(self, image_path: Path) -> None:
        self.current_image = image_path
        self.original_image = Image.open(image_path).convert("RGB")
        self.selected_point_index = -1
        self.dragging_point_index = -1

        if not self.is_valid_label_set():
            self.current_json = make_empty_label(image_path, self.original_image.width, self.original_image.height)
            self.annotations = []
            self.main_annotations = []
            self.opposite_annotations = []
            self.active_label_set = self.label_set.get()
            self.update_point_list()
            self.render_image()
            self.set_status("Opposite labeling is only available for Left/Right side images.")
            return

        label_path = self.label_path_for(image_path)
        if label_path.exists():
            self.current_json = json.loads(label_path.read_text(encoding="utf-8"))
        else:
            self.current_json = make_empty_label(image_path, self.original_image.width, self.original_image.height)

        self.main_annotations = list(self.current_json.get("annotation_info", []))
        self.opposite_annotations = list(self.current_json.get("opposite_annotation_info", []))
        self.active_label_set = self.label_set.get()
        self.annotations = self.active_annotations()
        self.update_point_list()
        self.render_image()
        self.set_status(
            f"Direction: {self.direction.get()} / Set: {self.label_set.get()} / "
            f"main {len(self.main_annotations)} / opposite {len(self.opposite_annotations)} / "
            f"{image_path.relative_to(self.input_root)}"
        )

    def label_path_for(self, image_path: Path) -> Path:
        return self.output_root / "label" / self.direction.get() / f"{image_path.stem}.json"

    def output_image_path_for(self, image_path: Path) -> Path:
        return self.output_root / "image" / self.direction.get() / image_path.name

    def annotation_key(self) -> str:
        return "opposite_annotation_info" if self.label_set.get() == "Opposite" else "annotation_info"

    def active_annotations(self) -> list[dict[str, str]]:
        return self.opposite_annotations if self.label_set.get() == "Opposite" else self.main_annotations

    def sync_active_annotations(self) -> None:
        if self.active_label_set == "Opposite":
            self.opposite_annotations = self.annotations
        else:
            self.main_annotations = self.annotations

    def opposite_direction(self) -> str:
        return OPPOSITE_DIRECTIONS.get(self.direction.get(), "")

    def is_valid_label_set(self) -> bool:
        return self.label_set.get() != "Opposite" or self.direction.get() in SIDE_DIRECTIONS

    def reload_current_label(self) -> None:
        self.sync_active_annotations()
        if self.current_image is not None:
            self.load_image(self.current_image)

    def switch_label_set(self) -> None:
        if self.current_image is None:
            return
        self.sync_active_annotations()
        if not self.is_valid_label_set():
            self.annotations = []
            self.active_label_set = self.label_set.get()
            self.selected_point_index = -1
            self.dragging_point_index = -1
            self.update_point_list()
            self.render_image()
            self.set_status("Opposite labeling is only available for Left/Right side images.")
            return
        self.annotations = self.active_annotations()
        self.active_label_set = self.label_set.get()
        self.selected_point_index = -1
        self.dragging_point_index = -1
        self.update_point_list()
        self.render_image()
        self.set_status(
            f"Direction: {self.direction.get()} / Set: {self.label_set.get()} / "
            f"main {len(self.main_annotations)} / opposite {len(self.opposite_annotations)}"
        )

    def on_keypoint_select(self, _: object) -> None:
        selection = self.keypoint_list.curselection()
        if selection:
            self.selected_label.set(KEYPOINT_LABELS[selection[0]])

    def on_point_select(self, _: object) -> None:
        selection = self.point_list.curselection()
        if not selection:
            return
        self.selected_point_index = selection[0]
        label = self.annotations[self.selected_point_index]["label"]
        self.selected_label.set(label)
        self.select_keypoint_label(label)
        self.render_image()

    def select_keypoint_label(self, label: str) -> None:
        try:
            index = KEYPOINT_LABELS.index(label)
        except ValueError:
            return
        self.keypoint_list.selection_clear(0, "end")
        self.keypoint_list.selection_set(index)
        self.keypoint_list.see(index)

    def render_image(self) -> None:
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Select an image",
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
        scale = base_scale * self.zoom_factor
        display_width = max(1, int(self.original_image.width * scale))
        display_height = max(1, int(self.original_image.height * scale))
        self.image_scale = scale
        self.image_offset = ((canvas_width - display_width) // 2, (canvas_height - display_height) // 2)

        self.display_image = self.original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(*self.image_offset, anchor="nw", image=self.tk_image)
        self.draw_points()

    def draw_points(self) -> None:
        offset_x, offset_y = self.image_offset
        if self.label_set.get() == "Opposite":
            self.draw_annotation_set(self.main_annotations, offset_x, offset_y, MAIN_COLOR, active=False)
            self.draw_annotation_set(self.opposite_annotations, offset_x, offset_y, OPPOSITE_COLOR, active=True)
        else:
            self.draw_annotation_set(self.opposite_annotations, offset_x, offset_y, OPPOSITE_COLOR, active=False)
            self.draw_annotation_set(self.main_annotations, offset_x, offset_y, MAIN_COLOR, active=True)

    def draw_annotation_set(self, annotations: list[dict[str, str]], offset_x: int, offset_y: int, color: str, active: bool) -> None:
        for index, item in enumerate(annotations):
            try:
                x = offset_x + float(item["x"]) * self.original_image.width * self.image_scale
                y = offset_y + float(item["y"]) * self.original_image.height * self.image_scale
            except (KeyError, TypeError, ValueError):
                continue
            is_selected = active and index == self.selected_point_index
            radius = 7 if is_selected else 5
            outline = "#ffffff" if active else "#e5e7eb"
            width = 2 if active else 1
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=outline, width=width)
            label = str(item.get("label", "")) if self.show_pose_names else ""
            self.draw_point_label(x, y, KEYPOINT_NUMBERS.get(str(item.get("label", "")), "?"), label)

    def draw_point_label(self, x: float, y: float, number: int | str, label: str) -> None:
        text = f"{number}. {label}" if label else str(number)
        text_id = self.canvas.create_text(x + 10, y - 10, text=text, fill="#111827", anchor="w", font=("Arial", 11, "bold"))
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            return
        pad = 3
        rect_id = self.canvas.create_rectangle(
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad,
            fill="#ffffff",
            outline="#d1d5db",
        )
        self.canvas.tag_raise(text_id, rect_id)

    def toggle_pose_names(self) -> None:
        self.show_pose_names = not self.show_pose_names
        self.pose_name_button.config(text=f"Names: {'ON' if self.show_pose_names else 'OFF'}")
        self.render_image()

    def zoom_in(self) -> None:
        self.zoom_factor = min(ZOOM_MAX, self.zoom_factor * ZOOM_STEP)
        self.render_image()

    def zoom_out(self) -> None:
        self.zoom_factor = max(ZOOM_MIN, self.zoom_factor / ZOOM_STEP)
        self.render_image()

    def canvas_to_normalized(self, event_x: int, event_y: int) -> tuple[float, float] | None:
        if self.original_image is None:
            return None
        offset_x, offset_y = self.image_offset
        image_x = (event_x - offset_x) / self.image_scale
        image_y = (event_y - offset_y) / self.image_scale
        if image_x < 0 or image_y < 0 or image_x > self.original_image.width or image_y > self.original_image.height:
            return None
        return image_x / self.original_image.width, image_y / self.original_image.height

    def find_nearest_point(self, event_x: int, event_y: int) -> int:
        if self.original_image is None:
            return -1
        offset_x, offset_y = self.image_offset
        nearest = -1
        nearest_distance = 14
        for index, item in enumerate(self.annotations):
            x = offset_x + float(item["x"]) * self.original_image.width * self.image_scale
            y = offset_y + float(item["y"]) * self.original_image.height * self.image_scale
            distance = ((event_x - x) ** 2 + (event_y - y) ** 2) ** 0.5
            if distance <= nearest_distance:
                nearest = index
                nearest_distance = distance
        return nearest

    def on_canvas_click(self, event: object) -> None:
        if self.current_image is None:
            return
        if not self.is_valid_label_set():
            self.set_status("Opposite labeling is only available for Left/Right side images.")
            return
        hit_index = self.find_nearest_point(event.x, event.y)
        if hit_index >= 0:
            self.selected_point_index = hit_index
            self.dragging_point_index = hit_index
            label = self.annotations[hit_index]["label"]
            self.selected_label.set(label)
            self.select_keypoint_label(label)
            self.update_point_list()
            self.render_image()
            return

        point = self.canvas_to_normalized(event.x, event.y)
        if point is None:
            return
        x, y = point
        self.annotations.append({"x": str(x), "y": str(y), "label": self.selected_label.get()})
        self.selected_point_index = len(self.annotations) - 1
        self.dragging_point_index = self.selected_point_index
        self.update_point_list()
        self.render_image()

    def on_canvas_drag(self, event: object) -> None:
        if self.dragging_point_index < 0:
            return
        point = self.canvas_to_normalized(event.x, event.y)
        if point is None:
            return
        x, y = point
        self.annotations[self.dragging_point_index]["x"] = str(x)
        self.annotations[self.dragging_point_index]["y"] = str(y)
        self.update_point_list(keep_selection=True)
        self.render_image()

    def on_canvas_release(self, _: object) -> None:
        self.dragging_point_index = -1

    def on_canvas_double_click(self, event: object) -> None:
        hit_index = self.find_nearest_point(event.x, event.y)
        if hit_index >= 0:
            del self.annotations[hit_index]
            self.selected_point_index = -1
            self.update_point_list()
            self.render_image()

    def update_point_list(self, keep_selection: bool = False) -> None:
        previous_selection = self.selected_point_index
        self.point_list.delete(0, "end")
        for item in self.annotations:
            x = float(item["x"])
            y = float(item["y"])
            keypoint_number = KEYPOINT_NUMBERS.get(item["label"], "?")
            self.point_list.insert("end", f"{keypoint_number}. {item['label']}  ({x:.4f}, {y:.4f})")
        if keep_selection and 0 <= previous_selection < len(self.annotations):
            self.point_list.selection_set(previous_selection)
            self.point_list.see(previous_selection)

    def save_label(self) -> None:
        if self.current_image is None or self.current_json is None or self.original_image is None:
            return
        if not self.is_valid_label_set():
            messagebox.showwarning("Cannot save", "Opposite labeling is only available for Left/Right side images.")
            return

        self.sync_active_annotations()
        label_path = self.label_path_for(self.current_image)
        output_image_path = self.output_image_path_for(self.current_image)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_json.setdefault("annotation_info", [])
        self.current_json.setdefault("opposite_annotation_info", [])
        self.current_json["annotation_info"] = self.main_annotations
        self.current_json["opposite_annotation_info"] = self.opposite_annotations
        if self.opposite_annotations:
            self.current_json["opposite_direction"] = self.opposite_direction()
        else:
            self.current_json["opposite_direction"] = ""
        self.current_json["image_info"]["filename"] = self.current_image.stem
        self.current_json["image_info"]["file_format"] = self.current_image.suffix.lstrip(".").lower()
        self.current_json["image_info"]["image_size"] = self.current_image.stat().st_size
        self.current_json["image_info"]["resolution"] = f"{self.original_image.width}X{self.original_image.height}"

        shutil.copy2(self.current_image, output_image_path)
        label_path.write_text(json.dumps(self.current_json, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        self.set_status(f"Saved {self.label_set.get()} to {self.annotation_key()}: {label_path}")

    def undo_point(self) -> None:
        if self.annotations:
            self.annotations.pop()
            self.selected_point_index = -1
            self.update_point_list()
            self.render_image()

    def clear_points(self) -> None:
        if not self.annotations:
            return
        if not messagebox.askyesno("Clear annotations", "현재 선택한 라벨 세트의 모든 키포인트를 삭제할까요?"):
            return
        self.annotations.clear()
        self.selected_point_index = -1
        self.update_point_list()
        self.render_image()

    def delete_selected_point(self) -> None:
        if 0 <= self.selected_point_index < len(self.annotations):
            del self.annotations[self.selected_point_index]
            self.selected_point_index = -1
            self.update_point_list()
            self.render_image()

    def previous_image(self) -> None:
        if self.current_image not in self.images:
            return
        index = self.images.index(self.current_image)
        if index > 0:
            self.image_list.selection_clear(0, "end")
            self.image_list.selection_set(index - 1)
            self.image_list.see(index - 1)
            self.load_image(self.images[index - 1])

    def next_image(self) -> None:
        if self.current_image not in self.images:
            return
        index = self.images.index(self.current_image)
        if index < len(self.images) - 1:
            self.image_list.selection_clear(0, "end")
            self.image_list.selection_set(index + 1)
            self.image_list.see(index + 1)
            self.load_image(self.images[index + 1])

    def set_status(self, text: str) -> None:
        self.status.config(text=text)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    root = Tk()
    DogPoseLabeler(root, input_root, output_root)
    root.mainloop()


if __name__ == "__main__":
    main()
