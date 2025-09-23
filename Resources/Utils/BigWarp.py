#!/usr/bin/env python3
"""
IT WORKS with the transformation matrix!!!!
This version supports multi-image workflows in the GUI.
"""

import sys
import os
import math
import json
import csv  # Aggiunto per l'esportazione degli angoli
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Install with: pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    print("Error: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    print("Error: PIL or tkinter not found. Install with: pip install Pillow")
    sys.exit(1)


@dataclass
class Landmark:
    """Represents a landmark point with coordinates in both images."""
    fixed_x: float
    fixed_y: float
    moving_x: float
    moving_y: float
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fixed_x': self.fixed_x,
            'fixed_y': self.fixed_y,
            'moving_x': self.moving_x,
            'moving_y': self.moving_y,
            'active': self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Landmark':
        return cls(**data)


class ThinPlateSpline:
    """
    Implementation of Thin Plate Splines for deformable transformations.
    Based on Bookstein's theory for image registration.
    """
    
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray, regularization: float = 0.0):
        """
        Initializes the TPS with source and target points.
        """
        self.source_points = np.array(source_points)
        self.target_points = np.array(target_points)
        self.regularization = regularization
        self.n_points = len(source_points)
        self._compute_coefficients()

    def _radial_basis_function(self, r: np.ndarray) -> np.ndarray:
        """Radial basis function for TPS: r^2 * ln(r)"""
        r_safe = np.where(r == 0, 1e-10, r)
        return r**2 * np.log(r_safe)
    
    def _compute_coefficients(self):
        """Calculates the w and a coefficients of the TPS transformation."""
        n = self.n_points
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = np.linalg.norm(self.source_points[i] - self.source_points[j])
                    K[i, j] = self._radial_basis_function(r)
        
        K += self.regularization * np.eye(n)
        P = np.ones((n, 3))
        P[:, 1:] = self.source_points
        
        top = np.hstack([K, P])
        bottom = np.hstack([P.T, np.zeros((3, 3))])
        L = np.vstack([top, bottom])
        
        Y_x = np.hstack([self.target_points[:, 0], np.zeros(3)])
        Y_y = np.hstack([self.target_points[:, 1], np.zeros(3)])
        
        try:
            coeffs_x = np.linalg.solve(L, Y_x)
            coeffs_y = np.linalg.solve(L, Y_y)
        except np.linalg.LinAlgError:
            coeffs_x = np.linalg.pinv(L) @ Y_x
            coeffs_y = np.linalg.pinv(L) @ Y_y
        
        self.w_x = coeffs_x[:n]
        self.a_x = coeffs_x[n:]
        self.w_y = coeffs_y[:n]
        self.a_y = coeffs_y[n:]
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transforms a single point using TPS."""
        x, y = point
        f_x = self.a_x[0] + self.a_x[1] * x + self.a_x[2] * y
        f_y = self.a_y[0] + self.a_y[1] * x + self.a_y[2] * y
        
        for i in range(self.n_points):
            r = np.linalg.norm(point - self.source_points[i])
            if r > 0:
                rbf = self._radial_basis_function(r)
                f_x += self.w_x[i] * rbf
                f_y += self.w_y[i] * rbf
        
        return np.array([f_x, f_y])
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transforms an array of points."""
        return np.array([self.transform_point(p) for p in points])
    
    def transform_image(self, image: np.ndarray, output_shape: Optional[Tuple[int, int]] = None, 
                          progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Transforms an image using TPS with inverse mapping.
        """
        if output_shape is None:
            output_shape = image.shape[:2]
        
        h_out, w_out = output_shape
        y_coords, x_coords = np.mgrid[0:h_out, 0:w_out]
        output_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # We need the inverse transformation for image warping.
        # So we create a TPS from target to source.
        inverse_tps = ThinPlateSpline(self.target_points, self.source_points, self.regularization)
        input_coords = inverse_tps.transform_points(output_coords)
        
        if len(image.shape) == 2:
            transformed = self._interpolate_image(image, input_coords, output_shape, progress_callback)
        else:
            transformed = np.zeros((*output_shape, image.shape[2]))
            for c in range(image.shape[2]):
                callback_for_channel = progress_callback if c == 0 else None
                transformed[:, :, c] = self._interpolate_image(image[:, :, c], input_coords, output_shape, callback_for_channel)
        
        return transformed.astype(image.dtype)
    
    def _interpolate_image(self, image: np.ndarray, coords: np.ndarray, output_shape: Tuple[int, int],
                           progress_callback: Optional[callable] = None) -> np.ndarray:
        """Interpolates image values using bilinear interpolation."""
        h, w = image.shape
        h_out, w_out = output_shape
        
        result = np.zeros(output_shape)
        total_pixels = len(coords)
        
        for i, (x, y) in enumerate(coords):
            if progress_callback and (i % 5000 == 0 or i == total_pixels - 1):
                progress_callback(i + 1, total_pixels)
            
            out_y = i // w_out
            out_x = i % w_out
            
            if 0 <= x < w-1 and 0 <= y < h-1:
                x0, x1 = int(x), int(x) + 1
                y0, y1 = int(y), int(y) + 1
                
                dx = x - x0
                dy = y - y0
                
                value = (image[y0, x0] * (1-dx) * (1-dy) +
                         image[y0, x1] * dx * (1-dy) +
                         image[y1, x0] * (1-dx) * dy +
                         image[y1, x1] * dx * dy)
                
                result[out_y, out_x] = value
        
        return result
    
    def get_affine_parameters(self) -> np.ndarray:
        """
        Returns the affine part of the TPS transformation as a 3x3 matrix.
        """
        affine_matrix = np.array([
            [self.a_x[1], self.a_x[2], self.a_x[0]],
            [self.a_y[1], self.a_y[2], self.a_y[0]],
            [0,           0,           1]
        ])
        return affine_matrix

# ==============================================================================
# === MODIFIED BigWarpGUI Class for Multi-Image Support ========================
# ==============================================================================
class BigWarpGUI:
    """Main graphical interface for BigWarp Python."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BigWarp Python - Deformable Registration (Multi-Image)")
        self.root.geometry("1200x800")
        
        # --- State variables MODIFIED for multi-image support ---
        self.fixed_image = None
        
        self.moving_images: Dict[str, np.ndarray] = {}
        self.landmarks_sets: Dict[str, List[Landmark]] = {}
        self.results: Dict[str, Dict[str, Any]] = {} # To store tps, transformed_image, angle etc.
        
        self.active_moving_image_key: Optional[str] = None
        
        # Interaction mode
        self.mode = "select"  # "select", "add_fixed", "add_moving"
        self.selected_landmark = None
        self.temp_fixed_point = None

        self.setup_ui()
        self.setup_plots()
    
    def setup_ui(self):
        """Configures the user interface with multi-image controls."""
        
        top_toolbar = ttk.Frame(self.root)
        bottom_toolbar = ttk.Frame(self.root)
        self.main_frame = ttk.Frame(self.root)
        self.landmark_table = ttk.Treeview(self.root, columns=("fixed_x", "fixed_y", "moving_x", "moving_y"), show="headings", height=5)
        status_frame = ttk.Frame(self.root)

        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.landmark_table.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        bottom_toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
        top_toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        top_toolbar.columnconfigure(0, weight=1)
        top_toolbar.columnconfigure(1, weight=1)
        bottom_toolbar.columnconfigure(0, weight=1)
        bottom_toolbar.columnconfigure(1, weight=1)

        # --- Gruppo 1: Load Data ---
        load_frame = ttk.LabelFrame(top_toolbar, text="1. Load Data")
        load_frame.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        
        load_buttons_frame = ttk.Frame(load_frame)
        load_buttons_frame.pack(pady=2, fill=tk.X)
        ttk.Button(load_buttons_frame, text="Load Fixed Image", command=self.load_fixed_image).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(load_buttons_frame, text="Load Moving Images", command=self.load_moving_images).pack(side=tk.LEFT, padx=3, pady=3)
        
        # --- NEW: Combobox for active image selection ---
        ttk.Label(load_buttons_frame, text="Active Moving Image:").pack(side=tk.LEFT, padx=(10, 2))
        self.moving_image_selector = ttk.Combobox(load_buttons_frame, state="readonly", width=30)
        self.moving_image_selector.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.X, expand=True)
        self.moving_image_selector.bind("<<ComboboxSelected>>", self.on_moving_image_selected)

        # Gruppo 2: Edit Landmarks
        edit_frame = ttk.LabelFrame(top_toolbar, text="2. Edit Landmarks")
        edit_frame.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        edit_buttons_frame = ttk.Frame(edit_frame)
        edit_buttons_frame.pack(pady=2)
        ttk.Button(edit_buttons_frame, text="Add Landmark", command=lambda: self.set_mode("add_fixed")).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(edit_buttons_frame, text="Select", command=lambda: self.set_mode("select")).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(edit_buttons_frame, text="Delete Selected", command=self.delete_selected_landmark).pack(side=tk.LEFT, padx=3, pady=3)

        # Gruppo 3: Transform
        transform_frame = ttk.LabelFrame(bottom_toolbar, text="3. Transform (for active image)")
        transform_frame.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        transform_buttons_frame = ttk.Frame(transform_frame)
        transform_buttons_frame.pack(pady=2)
        ttk.Button(transform_buttons_frame, text="Compute TPS", command=self.compute_tps).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(transform_buttons_frame, text="Apply Transform", command=self.apply_transformation).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(transform_buttons_frame, text="Show Parameters", command=self.show_transform_parameters).pack(side=tk.LEFT, padx=3, pady=3)
        
        # Gruppo 4: Save / Export
        save_frame = ttk.LabelFrame(bottom_toolbar, text="4. Save / Export")
        save_frame.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        save_buttons_frame = ttk.Frame(save_frame)
        save_buttons_frame.pack(pady=2)
        ttk.Button(save_buttons_frame, text="Save Landmarks", command=self.save_landmarks).pack(side=tk.LEFT, padx=3, pady=3)
        ttk.Button(save_buttons_frame, text="Export Result", command=self.export_result).pack(side=tk.LEFT, padx=3, pady=3)
        # --- NEW: Button to export all angles ---
        ttk.Button(save_buttons_frame, text="Export All Angles", command=self.export_all_angles).pack(side=tk.LEFT, padx=3, pady=3)

        # Table and status bar setup
        self.landmark_table.heading("fixed_x", text="Fixed X")
        self.landmark_table.heading("fixed_y", text="Fixed Y")
        self.landmark_table.heading("moving_x", text="Moving X")
        self.landmark_table.heading("moving_y", text="Moving Y")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load images to begin.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate')

    def setup_plots(self):
        self.fig = Figure(figsize=(15, 5))
        self.ax_fixed = self.fig.add_subplot(131)
        self.ax_moving = self.fig.add_subplot(132)
        self.ax_result = self.fig.add_subplot(133)
        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def on_moving_image_selected(self, event=None):
        """Handles the selection of a new active moving image from the combobox."""
        selected_key = self.moving_image_selector.get()
        if selected_key and selected_key in self.moving_images:
            self.active_moving_image_key = selected_key
            self.status_var.set(f"Active image: {self.active_moving_image_key}")
            self.update_landmark_table()
            self.update_display()

    def load_fixed_image(self):
        filename = filedialog.askopenfilename(title="Select Fixed Image")
        if filename:
            try:
                image = Image.open(filename)
                self.fixed_image = np.array(image.convert('RGB'))
                self.status_var.set(f"Fixed image loaded: {os.path.basename(filename)}")
                self.update_display()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")

    def load_moving_images(self):
        """Loads one or more moving images."""
        filenames = filedialog.askopenfilenames(
            title="Select Moving Image(s)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tiff *.bmp"), ("All files", "*.*")]
        )
        
        if not filenames:
            return

        loaded_count = 0
        for fn in filenames:
            try:
                image = Image.open(fn)
                key = os.path.basename(fn)
                if key in self.moving_images:
                    key = f"{Path(fn).stem}_{hash(fn)}{Path(fn).suffix}"

                self.moving_images[key] = np.array(image.convert('RGB'))
                self.landmarks_sets[key] = []
                self.results[key] = {}
                loaded_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image {os.path.basename(fn)}: {e}")
        
        if loaded_count > 0:
            self.status_var.set(f"{loaded_count} moving image(s) loaded.")
            self.moving_image_selector['values'] = list(self.moving_images.keys())
            if self.active_moving_image_key is None and self.moving_images:
                first_key = list(self.moving_images.keys())[0]
                self.moving_image_selector.set(first_key)
                self.on_moving_image_selected()

    def update_landmark_table(self):
        """Updates the landmark table for the active image."""
        for row in self.landmark_table.get_children():
            self.landmark_table.delete(row)

        if not self.active_moving_image_key:
            return
            
        active_landmarks = self.landmarks_sets.get(self.active_moving_image_key, [])
        for landmark in active_landmarks:
            self.landmark_table.insert("", "end", values=(
                f"{landmark.fixed_x:.1f}", f"{landmark.fixed_y:.1f}", 
                f"{landmark.moving_x:.1f}", f"{landmark.moving_y:.1f}"
            ))

    def on_canvas_click(self, event):
        if event.inaxes is None or event.xdata is None: return
            
        if not self.active_moving_image_key and self.mode in ["add_moving", "select"]:
             messagebox.showwarning("Warning", "Please load and select a moving image first.")
             return

        x, y = event.xdata, event.ydata

        if self.mode == "select":
            self.select_landmark_at(x, y, event.inaxes)
        elif self.mode == "add_fixed" and event.inaxes == self.ax_fixed:
            if self.fixed_image is not None:
                self.temp_fixed_point = (x, y)
                self.set_mode("add_moving")
                self.update_display()
        elif self.mode == "add_moving" and event.inaxes == self.ax_moving:
            if self.moving_images.get(self.active_moving_image_key) is not None and self.temp_fixed_point is not None:
                landmark = Landmark(self.temp_fixed_point[0], self.temp_fixed_point[1], x, y)
                active_landmarks = self.landmarks_sets[self.active_moving_image_key]
                active_landmarks.append(landmark)
                self.temp_fixed_point = None
                self.set_mode("select")
                self.status_var.set(f"Landmark added to '{self.active_moving_image_key}'. Total: {len(active_landmarks)}")
                self.update_display()
                self.update_landmark_table()

    def select_landmark_at(self, x: float, y: float, axes):
        if not self.active_moving_image_key: return
        active_landmarks = self.landmarks_sets.get(self.active_moving_image_key, [])
        if not active_landmarks: return
        
        min_dist, closest_lm = float('inf'), None
        for lm in active_landmarks:
            dist = 0
            if axes == self.ax_fixed:
                dist = math.hypot(lm.fixed_x - x, lm.fixed_y - y)
            elif axes == self.ax_moving:
                dist = math.hypot(lm.moving_x - x, lm.moving_y - y)
            
            if dist < min_dist and dist < 20: # Selection threshold
                min_dist, closest_lm = dist, lm
        
        self.selected_landmark = closest_lm
        self.update_display()

    def delete_selected_landmark(self):
        if not self.active_moving_image_key or not self.selected_landmark: return
        
        active_landmarks = self.landmarks_sets[self.active_moving_image_key]
        if self.selected_landmark in active_landmarks:
            active_landmarks.remove(self.selected_landmark)
            self.selected_landmark = None
            self.update_display()
            self.update_landmark_table()

    def compute_tps(self):
        if not self.active_moving_image_key: return messagebox.showwarning("Warning", "No active moving image.")
        key = self.active_moving_image_key
        active_landmarks = self.landmarks_sets[key]

        if len(active_landmarks) < 3: return messagebox.showwarning("Warning", "At least 3 landmarks needed.")
        
        try:
            source_pts = np.array([[lm.moving_x, lm.moving_y] for lm in active_landmarks])
            target_pts = np.array([[lm.fixed_x, lm.fixed_y] for lm in active_landmarks])
            tps = ThinPlateSpline(source_pts, target_pts, regularization=0.001)
            
            self.results[key]['tps'] = tps
            
            affine_matrix = tps.get_affine_parameters()
            M = affine_matrix[0:2, 0:2]
            rotation_rad = math.atan2(M[1, 0], M[0, 0])
            rotation_degrees = math.degrees(rotation_rad)
            
            self.results[key]['affine_matrix'] = affine_matrix
            self.results[key]['rotation_degrees'] = rotation_degrees
            
            messagebox.showinfo("Success", f"TPS computed for {key}.\nRotation Angle: {rotation_degrees:.2f}°")
        except Exception as e:
            messagebox.showerror("Error", f"Error computing TPS for '{key}': {e}")

    def apply_transformation(self):
        if not self.active_moving_image_key: return messagebox.showwarning("Warning", "No active image.")
        key = self.active_moving_image_key
        if 'tps' not in self.results.get(key, {}): return messagebox.showwarning("Warning", "Compute TPS first.")
        
        tps, moving_image = self.results[key]['tps'], self.moving_images[key]
        
        self.status_var.set(f"Applying transformation to '{key}'...")
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, expand=True, padx=(5,0))
        
        try:
            output_shape = self.fixed_image.shape[:2] if self.fixed_image is not None else moving_image.shape[:2]
            transformed_image = tps.transform_image(moving_image, output_shape, self._update_progress)
            self.results[key]['transformed_image'] = transformed_image
            self.status_var.set(f"Transformation applied to '{key}'")
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Error applying transformation: {e}")
        finally:
            self.progress_bar.pack_forget()

    def update_display(self):
        self.ax_fixed.clear()
        self.ax_moving.clear()
        self.ax_result.clear()
        key = self.active_moving_image_key
        
        # Fixed image and its landmarks
        if self.fixed_image is not None:
            self.ax_fixed.imshow(self.fixed_image)
            self.ax_fixed.set_title("Fixed Image")
            if key and key in self.landmarks_sets:
                for i, lm in enumerate(self.landmarks_sets[key]):
                    color = 'red' if lm == self.selected_landmark else 'yellow'
                    self.ax_fixed.plot(lm.fixed_x, lm.fixed_y, 'o', ms=8, c=color)
                    self.ax_fixed.text(lm.fixed_x + 5, lm.fixed_y + 5, str(i + 1), c='white')
            if self.mode == "add_moving" and self.temp_fixed_point:
                self.ax_fixed.plot(self.temp_fixed_point[0], self.temp_fixed_point[1], 'r+', ms=12, mew=2)

        # Moving image and result
        if key and key in self.moving_images:
            self.ax_moving.imshow(self.moving_images[key])
            self.ax_moving.set_title(f"Moving: {key}")
            for i, lm in enumerate(self.landmarks_sets[key]):
                color = 'red' if lm == self.selected_landmark else 'cyan'
                self.ax_moving.plot(lm.moving_x, lm.moving_y, 'o', ms=8, c=color)
                self.ax_moving.text(lm.moving_x + 5, lm.moving_y + 5, str(i + 1), c='white')

            if 'transformed_image' in self.results.get(key, {}):
                self.ax_result.imshow(self.results[key]['transformed_image'])
                self.ax_result.set_title(f"Result for: {key}")
            else:
                self.ax_result.set_title("Result")
        
        for ax in [self.ax_fixed, self.ax_moving, self.ax_result]:
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()
    
    def export_all_angles(self):
        if not self.results: return messagebox.showwarning("Warning", "No transformations computed.")

        filename = filedialog.asksaveasfilename(
            title="Export All Rotation Angles",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All files", "*.*")]
        )
        if not filename: return

        export_data = [{'filename': k, 'rotation_angle_degrees': v['rotation_degrees']} 
                       for k, v in self.results.items() if 'rotation_degrees' in v]

        if not export_data: return messagebox.showinfo("Info", "No rotation angles calculated.")

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'rotation_angle_degrees'])
                writer.writeheader()
                writer.writerows(export_data)
            messagebox.showinfo("Success", f"Exported {len(export_data)} results.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export data: {e}")

    def show_transform_parameters(self):
        """
        Calculates and displays the parameters of the affine part of the TPS transformation.
        """
        if self.tps is None:
            messagebox.showwarning("Warning", "Compute the TPS transformation first.")
            return

        # Get the 3x3 affine matrix
        affine_matrix = self.tps.get_affine_parameters()
        
        # Extract the 2x2 submatrix for rotation/scaling/shear analysis
        M = affine_matrix[0:2, 0:2]

        # Decompose M to extract parameters
        # 1. Scaling
        scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)

        # 2. Rotation
        # atan2 is more robust than atan for calculating the angle
        rotation_rad = math.atan2(M[1, 0], M[0, 0])
        rotation_deg = math.degrees(rotation_rad)

        # 3. Shear (calculated from the deviation from orthogonality)
        det_M = np.linalg.det(M)
        if abs(scale_x) > 1e-6:
             shear_rad = math.atan2(-M[0, 1] * scale_x**2 + M[0, 0] * M[1, 1] * scale_x, det_M * scale_x)
             shear_deg = math.degrees(shear_rad)
        else:
             shear_deg = 0

        # 4. Translation
        translation_x = affine_matrix[0, 2]
        translation_y = affine_matrix[1, 2]

        # Prepare the text to display
        info_text = (
            "Parameters of the Affine Component of the TPS Transformation\n"
            "----------------------------------------------------------\n\n"
            f"Rotation Angle: {rotation_deg:.2f} degrees\n"
            f"Scaling (X, Y): ({scale_x:.3f}, {scale_y:.3f})\n"
            f"Shear: {shear_deg:.2f} degrees\n"
            f"Translation (X, Y): ({translation_x:.1f}, {translation_y:.1f}) pixels\n\n"
            "Affine Matrix (3x3):\n"
            f"[[{affine_matrix[0,0]:.4f}, {affine_matrix[0,1]:.4f}, {affine_matrix[0,2]:.2f}],\n"
            f" [{affine_matrix[1,0]:.4f}, {affine_matrix[1,1]:.4f}, {affine_matrix[1,2]:.2f}],\n"
            f" [{affine_matrix[2,0]:.4f}, {affine_matrix[2,1]:.4f}, {affine_matrix[2,2]:.2f}]]\n\n"
            "Note: These parameters describe only the 'global' transformation.\n"
            "Local deformation is handled by the non-linear coefficients of the TPS."
        )

        # Create a new window to show the information
        info_window = tk.Toplevel(self.root)
        info_window.title("Transformation Parameters")
        info_window.geometry("500x350")
        
        text_widget = tk.Text(info_window, wrap='word', font=("Courier New", 10))
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled') # Make the text non-editable
        
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=5)
        
    def save_landmarks(self):
        if not self.active_moving_image_key: return messagebox.showwarning("Warning", "No active image.")
        key = self.active_moving_image_key
        if not self.landmarks_sets[key]: return messagebox.showwarning("Warning", f"No landmarks for '{key}'.")
        
        filename = filedialog.asksaveasfilename(
            title=f"Save Landmarks for {key}",
            defaultextension=".json", initialfile=f"landmarks_{Path(key).stem}.json"
        )
        if filename:
            try:
                data = {'landmarks': [lm.to_dict() for lm in self.landmarks_sets[key]]}
                with open(filename, 'w') as f: json.dump(data, f, indent=2)
            except Exception as e: messagebox.showerror("Error", f"Could not save: {e}")

    def export_result(self):
        if not self.active_moving_image_key or 'transformed_image' not in self.results.get(self.active_moving_image_key, {}):
            return messagebox.showwarning("Warning", "No result to export for active image.")

        key = self.active_moving_image_key
        img = self.results[key]['transformed_image']
        filename = filedialog.asksaveasfilename(
            title=f"Export Result for {key}",
            defaultextension=".png", initialfile=f"result_{Path(key).stem}.png"
        )
        if filename:
            try: Image.fromarray(img.astype(np.uint8)).save(filename)
            except Exception as e: messagebox.showerror("Error", f"Could not export: {e}")

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "add_moving": self.temp_fixed_point = None
        self.update_display()
        
    def _update_progress(self, current_step: int, total_steps: int):
        if total_steps > 0:
            if self.progress_bar['maximum'] != total_steps: self.progress_bar['maximum'] = total_steps
            self.progress_bar['value'] = current_step
            self.root.update_idletasks()

    def run(self):
        self.root.mainloop()

# ==============================================================================
# === Unchanged CLI and Main execution parts ===================================
# ==============================================================================
class BigWarpCLI:
    """Command-line interface for BigWarp Python."""
    
    def __init__(self):
        self.landmarks = []
        self.tps = None
    
    def load_image(self, filename: str) -> np.ndarray:
        try:
            image = Image.open(filename)
            return np.array(image.convert('RGB'))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def load_landmarks_from_file(self, filename: str) -> bool:
        try:
            with open(filename, 'r') as f: data = json.load(f)
            self.landmarks = [Landmark.from_dict(lm) for lm in data['landmarks']]
            print(f"Loaded {len(self.landmarks)} landmarks from {filename}")
            return True
        except Exception as e:
            print(f"Error loading landmarks: {e}")
            return False
    
    def compute_transformation(self, regularization: float = 0.001) -> bool:
        if len(self.landmarks) < 3:
            print("Error: at least 3 landmarks are needed")
            return False
        
        try:
            source = np.array([[lm.moving_x, lm.moving_y] for lm in self.landmarks])
            target = np.array([[lm.fixed_x, lm.fixed_y] for lm in self.landmarks])
            self.tps = ThinPlateSpline(source, target, regularization)
            print(f"TPS computed with {len(self.landmarks)} landmarks")
            return True
        except Exception as e:
            print(f"Error computing TPS: {e}")
            return False
    
    def transform_image(self, moving_image_path: str, output_path: str, 
                        reference_shape: Optional[Tuple[int, int]] = None):
        moving_img = self.load_image(moving_image_path)
        if moving_img is None: return

        print("Applying transformation...")
        
        def cli_progress(current, total):
            if total > 0:
                percent = (current / total) * 100
                sys.stdout.write(f"\rProgress: {percent:.1f}%")
                sys.stdout.flush()

        transformed_img = self.tps.transform_image(
            moving_img, output_shape=reference_shape, progress_callback=cli_progress
        )
        print("\nTransformation complete.")

        try:
            Image.fromarray(transformed_img.astype(np.uint8)).save(output_path)
            print(f"Result saved to: {output_path}")
        except Exception as e:
            print(f"Error saving result: {e}")

def print_help():
    """Prints the program's help text."""
    print("""
BigWarp Python - Deformable Registration Tool
=============================================
USAGE:
    python bigwarp.py             # Start graphical interface
    python bigwarp.py --gui       # Start graphical interface
    python bigwarp.py --cli [opts] # Command-line mode
    
CLI OPTIONS:
    --landmarks FILE.json         # Input landmarks
    --moving IMAGE                # Image to transform
    --fixed IMAGE                 # Reference image for output dimensions
    --output IMAGE                # Output file for transformed image
""")

def main():
    if len(sys.argv) == 1 or '--gui' in sys.argv:
        print("Starting BigWarp Python graphical interface...")
        app = BigWarpGUI()
        app.run()
        return

    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return

    if '--cli' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description="BigWarp CLI")
        parser.add_argument('--landmarks', required=True, type=str)
        parser.add_argument('--moving', required=True, type=str)
        parser.add_argument('--output', required=True, type=str)
        parser.add_argument('--fixed', type=str)
        parser.add_argument('--regularization', type=float, default=0.001)
        # Filter out '--cli' before parsing
        cli_args = [arg for arg in sys.argv[1:] if arg != '--cli']
        args = parser.parse_args(cli_args)
        
        cli = BigWarpCLI()
        if not cli.load_landmarks_from_file(args.landmarks): return
        if not cli.compute_transformation(args.regularization): return
        
        ref_shape = None
        if args.fixed:
            fixed_img = cli.load_image(args.fixed)
            if fixed_img is not None:
                ref_shape = fixed_img.shape[:2]
        
        cli.transform_image(args.moving, args.output, ref_shape)

if __name__ == "__main__":
    print("=" * 60)
    print("BigWarp Python - Deformable Image Registration")
    print("=" * 60)
    
    # Simple dependency check
    try:
        import numpy, matplotlib, PIL, tkinter
        print("✓ All dependencies seem to be present.")
        print()
        main()
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e.name}")
        print("Please install required packages (numpy, matplotlib, Pillow).")
        sys.exit(1)

#Mi serve salvare il rotation angle all'interno del file json inserito in ParsePathology
