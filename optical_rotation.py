#!/usr/bin/env python3
"""Interactive optical rotation analysis for JPG image sequences.

Choose an image folder, drag a region around the crystal in the first image,
enter the angle settings, and analyze the sequence in one window.
"""

from __future__ import annotations

import csv
import os
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
PREVIEW_MAX_WIDTH = 720
PREVIEW_MAX_HEIGHT = 600


@dataclass(frozen=True)
class Measurement:
    image_path: Path
    angle: float
    mean_intensity: float


@dataclass(frozen=True)
class SinusoidFit:
    coefficients: np.ndarray
    period: float
    harmonics: int
    minimum_angle: float


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def build_angles(
    image_count: int,
    start_angle: float,
    end_angle: float | None,
    angle_step: float | None,
) -> np.ndarray:
    if image_count == 0:
        raise ValueError("No JPG images were found in the selected folder.")
    if end_angle is None and angle_step is None:
        raise ValueError("Enter either an ending angle or an angle increment.")
    if end_angle is not None and angle_step is not None:
        raise ValueError("Use either ending angle or angle increment, not both.")
    if angle_step is not None:
        return start_angle + angle_step * np.arange(image_count, dtype=float)
    return np.linspace(start_angle, end_angle, image_count, dtype=float)


def mean_image_intensity(
    image_path: Path,
    roi: tuple[int, int, int, int] | None,
) -> float:
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        if roi is not None:
            x, y, width, height = roi
            if x + width > grayscale.width or y + height > grayscale.height:
                raise ValueError(
                    f"The selected rectangle is outside {image_path.name}. "
                    "All images must be at least as large as the first image."
                )
            grayscale = grayscale.crop((x, y, x + width, y + height))
        return float(np.asarray(grayscale, dtype=np.float64).mean())


def measure_images(
    image_paths: list[Path],
    angles: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> list[Measurement]:
    return [
        Measurement(path, float(angle), mean_image_intensity(path, roi))
        for path, angle in zip(image_paths, angles, strict=True)
    ]


def build_design_matrix(
    angles: np.ndarray,
    period: float,
    harmonics: int,
) -> np.ndarray:
    angular_frequency = 2 * np.pi / period
    columns = [np.ones_like(angles)]
    for harmonic in range(1, harmonics + 1):
        radians = harmonic * angular_frequency * angles
        columns.extend((np.cos(radians), np.sin(radians)))
    return np.column_stack(columns)


def predict_sinusoid(
    angles: np.ndarray,
    coefficients: np.ndarray,
    period: float,
    harmonics: int,
) -> np.ndarray:
    return build_design_matrix(angles, period, harmonics) @ coefficients


def find_fit_minimum(
    coefficients: np.ndarray,
    period: float,
    harmonics: int,
    lower_bound: float,
    upper_bound: float,
) -> float:
    grid = np.linspace(lower_bound, upper_bound, 20_000)
    values = predict_sinusoid(grid, coefficients, period, harmonics)
    best_index = int(np.argmin(values))

    if best_index in (0, len(grid) - 1):
        return float(grid[best_index])

    nearby_angles = grid[best_index - 1 : best_index + 2]
    nearby_values = values[best_index - 1 : best_index + 2]
    curvature, slope, _ = np.polyfit(nearby_angles, nearby_values, deg=2)
    if np.isclose(curvature, 0):
        return float(grid[best_index])

    refined = float(-slope / (2 * curvature))
    return refined if lower_bound <= refined <= upper_bound else float(grid[best_index])


def fit_sinusoid(
    measurements: list[Measurement], period: float, harmonics: int
) -> SinusoidFit:
    if period <= 0:
        raise ValueError("Fit period must be positive.")
    if harmonics < 1:
        raise ValueError("Harmonics must be at least 1.")
    parameter_count = 1 + 2 * harmonics
    if len(measurements) < parameter_count:
        raise ValueError(
            f"{harmonics} harmonics needs at least {parameter_count} images."
        )

    angles = np.array([measurement.angle for measurement in measurements])
    intensities = np.array([measurement.mean_intensity for measurement in measurements])
    coefficients = np.linalg.lstsq(
        build_design_matrix(angles, period, harmonics), intensities, rcond=None
    )[0]
    if np.allclose(coefficients[1:], 0):
        raise ValueError("The intensity is nearly constant, so a minimum cannot be fitted.")

    return SinusoidFit(
        coefficients=coefficients,
        period=period,
        harmonics=harmonics,
        minimum_angle=find_fit_minimum(
            coefficients,
            period,
            harmonics,
            float(angles.min()),
            float(angles.max()),
        ),
    )


def write_csv(output_path: Path, measurements: list[Measurement]) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "angle", "mean_intensity"])
        for measurement in measurements:
            writer.writerow(
                [
                    measurement.image_path.name,
                    f"{measurement.angle:.12g}",
                    f"{measurement.mean_intensity:.12g}",
                ]
            )


def create_plot(
    measurements: list[Measurement],
    fit: SinusoidFit,
    title: str,
    x_label: str,
    y_label: str,
) -> Figure:
    angles = np.array([measurement.angle for measurement in measurements])
    intensities = np.array([measurement.mean_intensity for measurement in measurements])
    fit_angles = np.linspace(float(angles.min()), float(angles.max()), 600)
    fit_values = predict_sinusoid(
        fit_angles, fit.coefficients, fit.period, fit.harmonics
    )

    figure = Figure(figsize=(8, 5.4), dpi=100)
    axis = figure.add_subplot()
    axis.scatter(angles, intensities, label="Measured intensity", color="#1565c0")
    axis.plot(
        fit_angles,
        fit_values,
        label=f"Sinusoidal fit ({fit.harmonics} harmonics)",
        color="#c62828",
    )
    axis.axvline(
        fit.minimum_angle,
        color="#2e7d32",
        linestyle="--",
        label=f"Minimum: {fit.minimum_angle:.4g} deg",
    )
    axis.set_title(title or "Optical Rotation Analysis")
    axis.set_xlabel(x_label or "Angle (degrees)")
    axis.set_ylabel(y_label or "Mean grayscale intensity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


class OpticalRotationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Optical Rotation Analysis")
        self.root.minsize(1100, 760)

        self.image_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.start_angle_var = tk.StringVar(value="0")
        self.range_mode_var = tk.StringVar(value="step")
        self.range_value_var = tk.StringVar(value="1")
        self.period_var = tk.StringVar(value="180")
        self.harmonics_var = tk.StringVar(value="2")
        self.title_var = tk.StringVar(value="Optical Rotation Analysis")
        self.x_label_var = tk.StringVar(value="Angle (degrees)")
        self.y_label_var = tk.StringVar(value="Mean grayscale intensity")
        self.roi_label_var = tk.StringVar(value="Whole image selected")
        self.status_var = tk.StringVar(value="Choose the folder that contains your JPG images.")

        self.image_paths: list[Path] = []
        self.first_image: Image.Image | None = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.drag_start: tuple[int, int] | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_canvas_id: int | None = None
        self.measurements: list[Measurement] | None = None
        self.fit: SinusoidFit | None = None
        self.plot_figure: Figure | None = None
        self.plot_canvas: FigureCanvasTkAgg | None = None

        self.build_interface()

    def build_interface(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        files = ttk.LabelFrame(outer, text="1. Images and Results", padding=10)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="JPG folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.image_dir_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(files, text="Choose Folder", command=self.choose_image_folder).grid(
            row=0, column=2
        )
        ttk.Button(files, text="Load First Image", command=self.load_first_image).grid(
            row=0, column=3, padx=(6, 0)
        )
        ttk.Label(files, text="Save results in").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(files, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(8, 0)
        )
        ttk.Button(files, text="Choose Folder", command=self.choose_output_folder).grid(
            row=1, column=2, pady=(8, 0)
        )
        ttk.Label(files, text="Creates results.csv and fit.png").grid(
            row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        settings = ttk.LabelFrame(outer, text="2. Analysis Settings", padding=10)
        settings.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column in (1, 3, 5):
            settings.columnconfigure(column, weight=1)

        ttk.Label(settings, text="Starting angle").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.start_angle_var, width=12).grid(
            row=0, column=1, sticky="ew", padx=(6, 16)
        )
        ttk.Radiobutton(
            settings, text="Angle increment", variable=self.range_mode_var,
            value="step", command=self.update_range_label
        ).grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(
            settings, text="Ending angle", variable=self.range_mode_var,
            value="end", command=self.update_range_label
        ).grid(row=0, column=3, sticky="w")
        self.range_value_label = ttk.Label(settings, text="Angle increment")
        self.range_value_label.grid(row=0, column=4, sticky="w", padx=(16, 0))
        ttk.Entry(settings, textvariable=self.range_value_var, width=12).grid(
            row=0, column=5, sticky="ew", padx=(6, 0)
        )

        ttk.Label(settings, text="Fit period").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.period_var, width=12).grid(
            row=1, column=1, sticky="ew", padx=(6, 16), pady=(8, 0)
        )
        ttk.Label(settings, text="Fit harmonics").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(settings, from_=1, to=6, textvariable=self.harmonics_var, width=8).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )
        ttk.Label(settings, text="Rectangle").grid(row=1, column=4, sticky="w", padx=(16, 0), pady=(8, 0))
        ttk.Label(settings, textvariable=self.roi_label_var).grid(
            row=1, column=5, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(settings, text="Graph title").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.title_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(6, 16), pady=(8, 0)
        )
        ttk.Label(settings, text="X-axis title").grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.x_label_var).grid(
            row=2, column=4, sticky="ew", padx=(6, 16), pady=(8, 0)
        )
        ttk.Label(settings, text="Y-axis title").grid(row=2, column=5, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.y_label_var).grid(
            row=2, column=6, sticky="ew", padx=(6, 0), pady=(8, 0)
        )
        settings.columnconfigure(6, weight=1)

        content = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        content.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        preview_frame = ttk.LabelFrame(content, text="3. Drag a Rectangle Around the Crystal", padding=8)
        graph_frame = ttk.LabelFrame(content, text="4. Fit and Results", padding=8)
        content.add(preview_frame, weight=1)
        content.add(graph_frame, weight=1)

        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(
            preview_frame, width=720, height=600, background="#242424", highlightthickness=0
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<ButtonPress-1>", self.start_roi_drag)
        self.preview_canvas.bind("<B1-Motion>", self.drag_roi)
        self.preview_canvas.bind("<ButtonRelease-1>", self.finish_roi_drag)
        ttk.Button(preview_frame, text="Use Whole Image", command=self.use_whole_image).grid(
            row=1, column=0, pady=(8, 0), sticky="w"
        )

        graph_frame.rowconfigure(1, weight=1)
        graph_frame.columnconfigure(0, weight=1)
        actions = ttk.Frame(graph_frame)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(actions, text="Analyze Images", command=self.analyze).pack(side="left")
        ttk.Button(actions, text="Update Graph Labels", command=self.update_graph).pack(side="left", padx=6)
        ttk.Button(actions, text="Save Graph As...", command=self.save_graph_as).pack(side="left")
        self.plot_holder = ttk.Frame(graph_frame)
        self.plot_holder.grid(row=1, column=0, sticky="nsew")
        self.result_label = ttk.Label(graph_frame, text="No analysis yet.", justify="left")
        self.result_label.grid(row=2, column=0, sticky="w", pady=(6, 0))

        ttk.Label(outer, textvariable=self.status_var).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def update_range_label(self) -> None:
        text = "Angle increment" if self.range_mode_var.get() == "step" else "Ending angle"
        self.range_value_label.configure(text=text)

    def choose_image_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose the folder containing JPG images")
        if folder:
            self.image_dir_var.set(folder)
            if not self.output_dir_var.get():
                self.output_dir_var.set(folder)
            self.load_first_image()

    def choose_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose where to save results")
        if folder:
            self.output_dir_var.set(folder)

    def load_first_image(self) -> None:
        image_dir = Path(self.image_dir_var.get()).expanduser()
        if not image_dir.is_dir():
            messagebox.showerror("Image folder", "Choose a valid folder containing JPG images.")
            return
        image_paths = find_images(image_dir)
        if not image_paths:
            messagebox.showerror("Image folder", "No JPG or JPEG images were found in this folder.")
            return
        try:
            with Image.open(image_paths[0]) as image:
                self.first_image = image.convert("RGB").copy()
        except OSError as exc:
            messagebox.showerror("Image folder", f"Could not open the first image.\n\n{exc}")
            return

        self.image_paths = image_paths
        self.roi = None
        self.roi_label_var.set("Whole image selected")
        self.show_preview()
        self.status_var.set(f"Loaded {len(image_paths)} images. Drag a rectangle, or use the whole image.")

    def show_preview(self) -> None:
        if self.first_image is None:
            return
        image = self.first_image
        self.preview_scale = min(PREVIEW_MAX_WIDTH / image.width, PREVIEW_MAX_HEIGHT / image.height, 1.0)
        display_size = (max(1, int(image.width * self.preview_scale)), max(1, int(image.height * self.preview_scale)))
        display_image = image.resize(display_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(display_image)
        canvas_width = max(PREVIEW_MAX_WIDTH, display_size[0])
        canvas_height = max(PREVIEW_MAX_HEIGHT, display_size[1])
        self.preview_canvas.configure(width=canvas_width, height=canvas_height)
        self.preview_canvas.delete("all")
        self.preview_offset = ((canvas_width - display_size[0]) // 2, (canvas_height - display_size[1]) // 2)
        self.preview_canvas.create_image(*self.preview_offset, anchor="nw", image=self.preview_photo)
        if self.roi:
            self.draw_roi(self.roi)

    def clamp_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.first_image is None:
            return x, y
        left, top = self.preview_offset
        right = left + int(self.first_image.width * self.preview_scale)
        bottom = top + int(self.first_image.height * self.preview_scale)
        return min(max(x, left), right), min(max(y, top), bottom)

    def start_roi_drag(self, event: tk.Event) -> None:
        if self.first_image is None:
            return
        self.drag_start = self.clamp_to_image(event.x, event.y)
        if self.roi_canvas_id is not None:
            self.preview_canvas.delete(self.roi_canvas_id)
            self.roi_canvas_id = None

    def drag_roi(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x, y = self.clamp_to_image(event.x, event.y)
        x0, y0 = self.drag_start
        if self.roi_canvas_id is not None:
            self.preview_canvas.delete(self.roi_canvas_id)
        self.roi_canvas_id = self.preview_canvas.create_rectangle(x0, y0, x, y, outline="#ffca28", width=2)

    def finish_roi_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.first_image is None:
            return
        end_x, end_y = self.clamp_to_image(event.x, event.y)
        start_x, start_y = self.drag_start
        self.drag_start = None
        left, right = sorted((start_x, end_x))
        top, bottom = sorted((start_y, end_y))
        if right - left < 4 or bottom - top < 4:
            self.use_whole_image()
            return
        image_x = int(round((left - self.preview_offset[0]) / self.preview_scale))
        image_y = int(round((top - self.preview_offset[1]) / self.preview_scale))
        image_right = int(round((right - self.preview_offset[0]) / self.preview_scale))
        image_bottom = int(round((bottom - self.preview_offset[1]) / self.preview_scale))
        self.roi = (image_x, image_y, image_right - image_x, image_bottom - image_y)
        self.roi_label_var.set(f"x={image_x}, y={image_y}, w={image_right - image_x}, h={image_bottom - image_y}")
        self.draw_roi(self.roi)

    def draw_roi(self, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        if self.roi_canvas_id is not None:
            self.preview_canvas.delete(self.roi_canvas_id)
        left = self.preview_offset[0] + x * self.preview_scale
        top = self.preview_offset[1] + y * self.preview_scale
        right = left + width * self.preview_scale
        bottom = top + height * self.preview_scale
        self.roi_canvas_id = self.preview_canvas.create_rectangle(left, top, right, bottom, outline="#ffca28", width=2)

    def use_whole_image(self) -> None:
        self.roi = None
        self.roi_label_var.set("Whole image selected")
        if self.roi_canvas_id is not None:
            self.preview_canvas.delete(self.roi_canvas_id)
            self.roi_canvas_id = None

    def get_analysis_settings(self) -> tuple[float, float | None, float | None, float, int]:
        start_angle = float(self.start_angle_var.get())
        range_value = float(self.range_value_var.get())
        period = float(self.period_var.get())
        harmonics = int(self.harmonics_var.get())
        if self.range_mode_var.get() == "step":
            return start_angle, None, range_value, period, harmonics
        return start_angle, range_value, None, period, harmonics

    def analyze(self) -> None:
        if not self.image_paths:
            self.load_first_image()
            if not self.image_paths:
                return
        try:
            start, end, step, period, harmonics = self.get_analysis_settings()
            angles = build_angles(len(self.image_paths), start, end, step)
            self.measurements = measure_images(self.image_paths, angles, self.roi)
            self.fit = fit_sinusoid(self.measurements, period, harmonics)
            self.refresh_plot()
            self.write_default_outputs()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Analysis could not run", str(exc))
            return

        measured_minimum = min(self.measurements, key=lambda item: item.mean_intensity)
        self.result_label.configure(text=(
            f"Estimated rotation angle: {self.fit.minimum_angle:.6g} degrees\n"
            f"Lowest measured image: {measured_minimum.image_path.name} at {measured_minimum.angle:.6g} degrees"
        ))
        self.status_var.set("Analysis complete. The graph and results files have been created.")

    def refresh_plot(self) -> None:
        if self.measurements is None or self.fit is None:
            return
        self.plot_figure = create_plot(self.measurements, self.fit, self.title_var.get(), self.x_label_var.get(), self.y_label_var.get())
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=self.plot_holder)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_graph(self) -> None:
        if self.plot_figure is None:
            messagebox.showinfo("Graph", "Run the analysis before updating graph labels.")
            return
        self.refresh_plot()
        self.status_var.set("Graph labels updated.")

    def output_directory(self) -> Path:
        output_text = self.output_dir_var.get().strip()
        if output_text:
            output_dir = Path(output_text).expanduser()
        else:
            output_dir = Path(self.image_dir_var.get()).expanduser()
            self.output_dir_var.set(str(output_dir))
        if not output_dir.is_dir():
            raise ValueError("Choose a valid folder for saving results.")
        return output_dir

    def write_default_outputs(self) -> None:
        if self.measurements is None or self.plot_figure is None:
            return
        output_dir = self.output_directory()
        write_csv(output_dir / "results.csv", self.measurements)
        self.plot_figure.savefig(output_dir / "fit.png", dpi=200)

    def save_graph_as(self) -> None:
        if self.plot_figure is None:
            messagebox.showinfo("Save graph", "Run the analysis before saving a graph.")
            return
        output_path = filedialog.asksaveasfilename(
            title="Save graph", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if output_path:
            self.plot_figure.savefig(output_path, dpi=300, bbox_inches="tight")
            self.status_var.set(f"Saved graph: {output_path}")


def main() -> None:
    root = tk.Tk()
    OpticalRotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
