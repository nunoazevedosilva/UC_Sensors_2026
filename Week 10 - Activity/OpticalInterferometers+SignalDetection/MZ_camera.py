"""
camera_profile_gui.py

Modern Tkinter GUI for real-time camera viewing and intensity-line profiling.

Features:
    - Live camera display
    - Camera selection
    - Grayscale / color toggle
    - Selectable intensity-profile line
    - Real-time intensity profile plot
    - Temporal smoothing filter to reduce flickering

Required packages:

    pip install opencv-python numpy pillow
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class CameraProfileGUI:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.camera = None

        self.grayscale_display = True
        self.running = False

        self.frame = None
        self.gray = None

        self.line_start = None
        self.line_end = None
        self.selected_line = None
        self.drawing = False

        # Temporal smoothing filter
        self.filtered_gray = None
        self.temporal_alpha = 0.15
        self.use_temporal_filter = True

        self.bg_color = "#1e1e1e"
        self.panel_color = "#2b2b2b"
        self.text_color = "#f2f2f2"
        self.accent_color = "#4cc9f0"

        self.root = tk.Tk()
        self.root.title("Digital Holography Camera Viewer")
        self.root.geometry("1150x950")
        self.root.configure(bg=self.bg_color)

        self.setup_style()
        self.setup_gui()
        self.open_camera(self.camera_index)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background=self.panel_color)

        style.configure(
            "TLabel",
            background=self.bg_color,
            foreground=self.text_color,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Title.TLabel",
            background=self.bg_color,
            foreground=self.text_color,
            font=("Segoe UI", 18, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            background=self.bg_color,
            foreground="#b0b0b0",
            font=("Segoe UI", 10),
        )

        style.configure(
            "PanelTitle.TLabel",
            background=self.panel_color,
            foreground=self.text_color,
            font=("Segoe UI", 12, "bold"),
        )

        style.configure(
            "PanelText.TLabel",
            background=self.panel_color,
            foreground="#cccccc",
            font=("Segoe UI", 9),
        )

        style.configure(
            "TButton",
            background="#3a3a3a",
            foreground=self.text_color,
            borderwidth=0,
            padding=(12, 8),
            font=("Segoe UI", 10),
        )

        style.map(
            "TButton",
            background=[
                ("active", "#505050"),
                ("pressed", "#606060"),
            ],
        )

        style.configure(
            "Accent.TButton",
            background=self.accent_color,
            foreground="#000000",
            borderwidth=0,
            padding=(12, 8),
            font=("Segoe UI", 10, "bold"),
        )

        style.map(
            "Accent.TButton",
            background=[
                ("active", "#7bdff2"),
                ("pressed", "#48bfe3"),
            ],
        )

        style.configure(
            "TCheckbutton",
            background=self.panel_color,
            foreground=self.text_color,
            font=("Segoe UI", 10),
        )

        style.map(
            "TCheckbutton",
            background=[
                ("active", self.panel_color),
            ],
            foreground=[
                ("active", self.text_color),
            ],
        )

        style.configure(
            "TCombobox",
            fieldbackground="#3a3a3a",
            background="#3a3a3a",
            foreground=self.text_color,
            arrowcolor=self.text_color,
            padding=5,
        )

    def setup_gui(self):
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=20, pady=(18, 8))

        title = ttk.Label(
            header,
            text="Digital Holography Camera Viewer",
            style="Title.TLabel",
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header,
            text="Live camera feed with selectable intensity profile for interference fringes",
            style="Subtitle.TLabel",
        )
        subtitle.pack(anchor="w", pady=(3, 0))

        # Main layout
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left_panel = ttk.Frame(main)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(main, style="Panel.TFrame")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(18, 0))

        # Camera card
        camera_card = ttk.Frame(left_panel, style="Panel.TFrame")
        camera_card.pack(fill=tk.BOTH, expand=True)

        camera_title = ttk.Label(
            camera_card,
            text="Live camera image",
            style="PanelTitle.TLabel",
        )
        camera_title.pack(anchor="w", padx=14, pady=(12, 6))

        self.camera_label = tk.Label(
            camera_card,
            bg="#111111",
            bd=0,
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.camera_label.pack(padx=14, pady=(0, 14))

        # Profile card
        profile_card = ttk.Frame(left_panel, style="Panel.TFrame")
        profile_card.pack(fill=tk.X, pady=(18, 0))

        profile_title = ttk.Label(
            profile_card,
            text="Intensity profile",
            style="PanelTitle.TLabel",
        )
        profile_title.pack(anchor="w", padx=14, pady=(12, 6))

        self.profile_label = tk.Label(
            profile_card,
            bg="#111111",
            bd=0,
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.profile_label.pack(padx=14, pady=(0, 14))

        # Controls panel
        controls_title = ttk.Label(
            right_panel,
            text="Controls",
            style="PanelTitle.TLabel",
        )
        controls_title.pack(anchor="w", padx=18, pady=(18, 10))

        ttk.Label(
            right_panel,
            text="Camera index",
            style="PanelText.TLabel",
        ).pack(anchor="w", padx=18, pady=(8, 3))

        self.camera_var = tk.IntVar(value=self.camera_index)
        self.camera_box = ttk.Combobox(
            right_panel,
            textvariable=self.camera_var,
            values=[0, 1, 2, 3],
            width=12,
            state="readonly",
        )
        self.camera_box.pack(anchor="w", padx=18, pady=(0, 12))
        self.camera_box.bind("<<ComboboxSelected>>", self.change_camera)

        self.gray_var = tk.BooleanVar(value=self.grayscale_display)
        self.gray_check = ttk.Checkbutton(
            right_panel,
            text="Grayscale display",
            variable=self.gray_var,
            command=self.toggle_grayscale,
        )
        self.gray_check.pack(anchor="w", padx=18, pady=10)

        self.filter_var = tk.BooleanVar(value=self.use_temporal_filter)
        self.filter_check = ttk.Checkbutton(
            right_panel,
            text="Temporal smoothing",
            variable=self.filter_var,
            command=self.toggle_filter,
        )
        self.filter_check.pack(anchor="w", padx=18, pady=10)

        ttk.Label(
            right_panel,
            text="Smoothing strength",
            style="PanelText.TLabel",
        ).pack(anchor="w", padx=18, pady=(12, 3))

        self.alpha_var = tk.DoubleVar(value=self.temporal_alpha)
        self.alpha_slider = ttk.Scale(
            right_panel,
            from_=0.02,
            to=0.8,
            variable=self.alpha_var,
            command=self.update_alpha,
        )
        self.alpha_slider.pack(fill=tk.X, padx=18, pady=(0, 4))

        self.alpha_label = ttk.Label(
            right_panel,
            text=f"alpha = {self.temporal_alpha:.2f}",
            style="PanelText.TLabel",
        )
        self.alpha_label.pack(anchor="w", padx=18, pady=(0, 12))

        self.reset_button = ttk.Button(
            right_panel,
            text="Reset selected line",
            command=self.reset_line,
        )
        self.reset_button.pack(fill=tk.X, padx=18, pady=(15, 8))

        self.quit_button = ttk.Button(
            right_panel,
            text="Quit",
            command=self.close,
            style="Accent.TButton",
        )
        self.quit_button.pack(fill=tk.X, padx=18, pady=8)

        separator = ttk.Separator(right_panel, orient="horizontal")
        separator.pack(fill=tk.X, padx=18, pady=20)

        help_title = ttk.Label(
            right_panel,
            text="How to use",
            style="PanelTitle.TLabel",
        )
        help_title.pack(anchor="w", padx=18, pady=(0, 8))

        help_text = (
            "1. Select a camera index.\n"
            "2. Check that interference fringes are visible.\n"
            "3. Click and drag on the live image.\n"
            "4. The lower panel shows intensity along that line.\n\n"
            "Temporal smoothing reduces flickering by averaging frames over time.\n\n"
            "Smaller alpha = stronger smoothing but slower response.\n"
            "Larger alpha = faster response but more flicker."
        )

        ttk.Label(
            right_panel,
            text=help_text,
            style="PanelText.TLabel",
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 18))

        self.status_label = ttk.Label(
            right_panel,
            text="Status: starting...",
            style="PanelText.TLabel",
            wraplength=220,
        )
        self.status_label.pack(anchor="w", padx=18, pady=(20, 10))

        # Mouse events
        self.camera_label.bind("<ButtonPress-1>", self.mouse_down)
        self.camera_label.bind("<B1-Motion>", self.mouse_drag)
        self.camera_label.bind("<ButtonRelease-1>", self.mouse_up)

    def set_status(self, text):
        self.status_label.configure(text=f"Status: {text}")

    def open_camera(self, index):
        if self.camera is not None:
            self.camera.release()

        self.camera = cv2.VideoCapture(index)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.set_status(f"could not open camera {index}")
            print(f"Could not open camera {index}.")
            return False

        self.camera_index = index
        self.filtered_gray = None
        self.set_status(f"camera {index} opened")
        print(f"Opened camera {index}.")
        return True

    def change_camera(self, event=None):
        new_index = self.camera_var.get()
        self.open_camera(new_index)

    def toggle_grayscale(self):
        self.grayscale_display = self.gray_var.get()

        if self.grayscale_display:
            self.set_status("grayscale display enabled")
        else:
            self.set_status("color display enabled")

    def toggle_filter(self):
        self.use_temporal_filter = self.filter_var.get()

        if not self.use_temporal_filter:
            self.filtered_gray = None
            self.set_status("temporal smoothing disabled")
        else:
            self.set_status("temporal smoothing enabled")

    def update_alpha(self, value=None):
        self.temporal_alpha = float(self.alpha_var.get())
        self.alpha_label.configure(text=f"alpha = {self.temporal_alpha:.2f}")

    def reset_line(self):
        self.line_start = None
        self.line_end = None
        self.selected_line = None
        self.drawing = False
        self.set_status("line reset")

    def mouse_down(self, event):
        self.drawing = True
        self.line_start = (event.x, event.y)
        self.line_end = (event.x, event.y)

    def mouse_drag(self, event):
        if self.drawing:
            self.line_end = (event.x, event.y)

    def mouse_up(self, event):
        self.drawing = False
        self.line_end = (event.x, event.y)
        self.selected_line = (self.line_start, self.line_end)
        self.set_status("line selected")

    def resize_to_width(self, image, target_width):
        h, w = image.shape[:2]
        scale = target_width / w
        new_height = int(h * scale)

        return cv2.resize(
            image,
            (target_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def get_active_line(self, image_shape):
        h, w = image_shape

        if self.drawing and self.line_start is not None and self.line_end is not None:
            return self.line_start, self.line_end

        if self.selected_line is not None:
            return self.selected_line

        return (0, h // 2), (w - 1, h // 2)

    @staticmethod
    def get_line_profile(gray, start, end):
        x1, y1 = start
        x2, y2 = end

        h, w = gray.shape

        x1 = np.clip(x1, 0, w - 1)
        x2 = np.clip(x2, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        y2 = np.clip(y2, 0, h - 1)

        length = int(np.hypot(x2 - x1, y2 - y1))

        if length <= 1:
            return np.array([])

        x_values = np.linspace(x1, x2, length).astype(np.int32)
        y_values = np.linspace(y1, y2, length).astype(np.int32)

        return gray[y_values, x_values]

    def apply_temporal_filter(self, gray):
        if not self.use_temporal_filter:
            self.filtered_gray = None
            return gray

        gray_float = gray.astype(np.float32)

        if self.filtered_gray is None:
            self.filtered_gray = gray_float
        else:
            self.filtered_gray = (
                self.temporal_alpha * gray_float
                + (1.0 - self.temporal_alpha) * self.filtered_gray
            )

        return np.clip(self.filtered_gray, 0, 255).astype(np.uint8)

    def create_profile_image(self, profile, width=640, height=280):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        left = 55
        right = 25
        top = 35
        bottom = 45

        plot_w = width - left - right
        plot_h = height - top - bottom

        # Grid
        for i in range(5):
            y = top + int(i * plot_h / 4)
            cv2.line(panel, (left, y), (left + plot_w, y), (55, 55, 55), 1)

        for i in range(6):
            x = left + int(i * plot_w / 5)
            cv2.line(panel, (x, top), (x, top + plot_h), (45, 45, 45), 1)

        # Axes
        cv2.line(panel, (left, top), (left, top + plot_h), (180, 180, 180), 1)
        cv2.line(panel, (left, top + plot_h), (left + plot_w, top + plot_h), (180, 180, 180), 1)

        if profile.size == 0:
            cv2.putText(
                panel,
                "Select a line on the image",
                (left, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (220, 220, 220),
                2,
            )
            return panel

        profile = profile.astype(np.float32)

        min_val = np.min(profile)
        max_val = np.max(profile)
        mean_val = np.mean(profile)

        if max_val - min_val < 1e-6:
            y_plot = np.ones_like(profile) * (top + plot_h // 2)
        else:
            y_plot = top + plot_h - (
                (profile - min_val) / (max_val - min_val) * plot_h
            )

        x_plot = np.linspace(left, left + plot_w, len(profile)).astype(np.int32)
        y_plot = y_plot.astype(np.int32)

        points = np.column_stack((x_plot, y_plot)).astype(np.int32)

        for i in range(len(points) - 1):
            cv2.line(
                panel,
                tuple(points[i]),
                tuple(points[i + 1]),
                (80, 220, 255),
                2,
            )

        cv2.putText(
            panel,
            "Intensity along selected line",
            (left, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            2,
        )

        cv2.putText(
            panel,
            f"min {min_val:.1f}    mean {mean_val:.1f}    max {max_val:.1f}",
            (left, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
        )

        return panel

    @staticmethod
    def cv_to_tk(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        return ImageTk.PhotoImage(image=image_pil)

    def update(self):
        if self.camera is None:
            self.root.after(30, self.update)
            return

        ret, frame = self.camera.read()

        if not ret:
            self.set_status("could not read frame")
            self.root.after(30, self.update)
            return

        # Resize first, so mouse coordinates match displayed image
        display_width = 760
        frame_resized = self.resize_to_width(frame, display_width)

        gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Apply temporal smoothing to reduce flickering
        gray_filtered = self.apply_temporal_filter(gray_resized)

        self.frame = frame_resized
        self.gray = gray_filtered

        if self.grayscale_display:
            display_frame = cv2.cvtColor(gray_filtered, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = frame_resized.copy()

        active_line = self.get_active_line(gray_filtered.shape)

        # Draw selected line
        cv2.line(display_frame, active_line[0], active_line[1], (0, 0, 255), 2)
        cv2.circle(display_frame, active_line[0], 5, (0, 0, 255), -1)
        cv2.circle(display_frame, active_line[1], 5, (0, 0, 255), -1)

        # Overlay box
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (390, 105), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.45, display_frame, 0.55, 0)

        mode_text = "grayscale" if self.grayscale_display else "color"
        filter_text = "on" if self.use_temporal_filter else "off"

        cv2.putText(
            display_frame,
            f"Camera {self.camera_index}",
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (76, 201, 240),
            2,
        )

        cv2.putText(
            display_frame,
            f"Mode: {mode_text}",
            (25, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )

        cv2.putText(
            display_frame,
            f"Temporal smoothing: {filter_text}, alpha={self.temporal_alpha:.2f}",
            (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )

        profile = self.get_line_profile(
            gray_filtered,
            active_line[0],
            active_line[1],
        )

        profile_panel = self.create_profile_image(
            profile,
            width=display_width,
            height=280,
        )

        camera_image = self.cv_to_tk(display_frame)
        profile_image = self.cv_to_tk(profile_panel)

        self.camera_label.configure(image=camera_image)
        self.camera_label.image = camera_image

        self.profile_label.configure(image=profile_image)
        self.profile_label.image = profile_image

        self.root.after(15, self.update)

    def run(self):
        self.running = True
        self.update()
        self.root.mainloop()

    def close(self):
        self.running = False

        if self.camera is not None:
            self.camera.release()

        self.root.destroy()


def run_camera_gui(camera_index=1):
    app = CameraProfileGUI(camera_index=camera_index)
    app.run()


if __name__ == "__main__":
    run_camera_gui()