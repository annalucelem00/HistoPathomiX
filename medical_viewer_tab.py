#!/usr/bin/env python3
"""
Medical Image Viewer Tab for RadPathFusion
Medical viewer with controls for MRI, segmentation, and histology.
This version includes dynamic Multi-Planar Reconstruction (MPR) and
voxel spacing correction, with a robust crosshair implementation.
Enhanced with direction matrix and LabelMap support.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QSpinBox,
    QGroupBox, QFrame, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# Import viewer modules
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib is not available. 2D visualization will be limited.")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK is not available.")

class ImageData:
    """Container for image data"""
    def __init__(self, name: str = "", data: Any = None, visible: bool = True):
        self.name = name; self.data = data; self.visible = visible
        self.opacity = 1.0; self.color_map = "gray"; self.window_level = (0, 0)
        self.spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def get_array(self) -> Optional[np.ndarray]:
        if self.data is None: return None
        if SITK_AVAILABLE and isinstance(self.data, sitk.Image): return sitk.GetArrayFromImage(self.data)
        elif isinstance(self.data, np.ndarray): return self.data
        return None

    def get_slice_count(self) -> int:
        array = self.get_array()
        if array is None: return 0
        return array.shape[0] if len(array.shape) >= 3 else 1

    def get_slice(self, slice_idx: int) -> Optional[np.ndarray]:
        array = self.get_array()
        if array is None: return None
        if len(array.shape) >= 3 and 0 <= slice_idx < array.shape[0]: return array[slice_idx]
        elif len(array.shape) == 2 and slice_idx == 0: return array
        return None

class MedicalCanvas(FigureCanvas):
    """Matplotlib canvas for medical visualization with dynamic MPR"""
    slice_changed = pyqtSignal(int)

    def __init__(self):
        self.figure = Figure(figsize=(12, 4), facecolor='black')
        super().__init__(self.figure)
        self.setParent(None)
        
        self.figure.patch.set_facecolor('black')
        self.axial_ax, self.coronal_ax, self.sagittal_ax, self.info_ax = None, None, None, None
        self.setup_subplots()
        
        self.mr_data = ImageData("MRI")
        self.segmentation_data = ImageData("Segmentation")
        self.histology_data = ImageData("Histology")
        
        self.current_slice = 0
        self.max_slices = 0
        
        self.mpr_coords = {'x': 0, 'y': 0, 'z': 0}
        self.crosshair_lines = {}
        self._crosshairs_initialized = False

        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)

    def setup_subplots(self):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2)
        self.axial_ax = self.figure.add_subplot(gs[0, 0])
        self.coronal_ax = self.figure.add_subplot(gs[0, 1])
        self.sagittal_ax = self.figure.add_subplot(gs[1, 0])
        self.info_ax = self.figure.add_subplot(gs[1, 1])
        
        for ax, title in zip([self.axial_ax, self.coronal_ax, self.sagittal_ax, self.info_ax],
                             ["Axial", "Coronal", "Sagittal", "Information"]):
            ax.set_title(title, color='white', fontsize=10)
            ax.set_facecolor('black')
            if ax != self.info_ax: ax.set_xticks([]); ax.set_yticks([])
            else: ax.axis('off')
        self.figure.tight_layout(pad=1.5)

    def init_crosshairs(self):
        """Creates the Line2D artists for the crosshairs once."""
        if self._crosshairs_initialized: return
        self.crosshair_lines['axial_h'] = self.axial_ax.plot([0], [0], 'g-', lw=0.5, alpha=0.7)[0]
        self.crosshair_lines['axial_v'] = self.axial_ax.plot([0], [0], 'g-', lw=0.5, alpha=0.7)[0]
        self.crosshair_lines['coronal_h'] = self.coronal_ax.plot([0], [0], 'b-', lw=0.5, alpha=0.7)[0]
        self.crosshair_lines['coronal_v'] = self.coronal_ax.plot([0], [0], 'b-', lw=0.5, alpha=0.7)[0]
        self.crosshair_lines['sagittal_h'] = self.sagittal_ax.plot([0], [0], 'r-', lw=0.5, alpha=0.7)[0]
        self.crosshair_lines['sagittal_v'] = self.sagittal_ax.plot([0], [0], 'r-', lw=0.5, alpha=0.7)[0]
        self._crosshairs_initialized = True
        
    def load_mr_data(self, data: Any, name: str = "MRI"):
        # Apply target direction if data is SimpleITK Image
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            target_direction = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
            data.SetDirection(target_direction)
            self.mr_data.spacing = data.GetSpacing()
        
        self.mr_data.data = data
        self.mr_data.name = name
        self.update_max_slices()
        self.init_mpr_coords()
        self.init_crosshairs() # Initialize crosshairs after data is loaded
        self.update_display()

    def load_segmentation_data(self, data: Any, name: str = "Segmentation"):
        # Convert to LabelMap if data is SimpleITK Image
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            # Cast to appropriate label type if needed
            if data.GetPixelID() != sitk.sitkUInt8:
                data = sitk.Cast(data, sitk.sitkUInt8)
            
            # Create LabelMap from the image
            labelmap_filter = sitk.LabelImageToLabelMapFilter()
            labelmap = labelmap_filter.Execute(data)
            
            # Convert back to label image (maintains LabelMap properties)
            labelmap_to_image = sitk.LabelMapToLabelImageFilter()
            data = labelmap_to_image.Execute(labelmap)
        
        self.segmentation_data.data = data
        self.segmentation_data.name = name
        self.update_max_slices()
        self.update_display()

    def load_histology_data(self, data: Any, name: str = "Histology"):
        self.histology_data.data = data; self.histology_data.name = name
        self.update_max_slices(); self.update_display()

    def update_max_slices(self):
        slice_counts = [d.get_slice_count() for d in [self.mr_data, self.segmentation_data] if d.data is not None]
        self.max_slices = max(slice_counts) if slice_counts else 0
        if self.current_slice >= self.max_slices and self.max_slices > 0:
            self.current_slice = self.max_slices - 1

    def init_mpr_coords(self):
        array = self.mr_data.get_array()
        if array is not None and len(array.shape) >= 3:
            self.mpr_coords = {'z': array.shape[0] // 2, 'y': array.shape[1] // 2, 'x': array.shape[2] // 2}
            self.current_slice = self.mpr_coords['z']

    def set_slice(self, slice_idx: int):
        if 0 <= slice_idx < self.max_slices:
            self.current_slice = slice_idx; self.mpr_coords['z'] = slice_idx
            self.update_display()

    def update_display(self):
        if not MATPLOTLIB_AVAILABLE: return
        # Clear images on axes, but not the axes themselves (to keep crosshairs)
        for ax in [self.axial_ax, self.coronal_ax, self.sagittal_ax]:
            for img in ax.get_images(): img.remove()
        
        self.display_axial_view()
        self.display_coronal_view()
        self.display_sagittal_view()
        self.update_crosshairs()
        self.update_info_panel()
        self.draw()

    def display_axial_view(self):
        self.axial_ax.set_title(f"Axial - Slice {self.current_slice + 1}/{self.max_slices}", color='white', fontsize=10)
        aspect = self.mr_data.spacing[1] / self.mr_data.spacing[0]
        if self.mr_data.visible and self.mr_data.data is not None:
            mr_slice = self.mr_data.get_slice(self.current_slice)
            if mr_slice is not None:
                mr_slice = np.rot90(mr_slice, k=2)
                if self.mr_data.window_level != (0, 0): mr_slice = self.apply_window_level(mr_slice, *self.mr_data.window_level)
                self.axial_ax.imshow(mr_slice, cmap='gray', alpha=self.mr_data.opacity, aspect=aspect)
        if self.segmentation_data.visible and self.segmentation_data.data is not None:
            seg_slice = self.segmentation_data.get_slice(self.current_slice)
            if seg_slice is not None:
                seg_slice = np.rot90(seg_slice, k=2); mask = seg_slice > 0
                if np.any(mask):
                    overlay = np.zeros((*seg_slice.shape, 4)); overlay[mask] = [1, 0, 0, self.segmentation_data.opacity]
                    self.axial_ax.imshow(overlay, aspect=aspect)
        if self.histology_data.visible and self.histology_data.data is not None:
            hist_slice = self.histology_data.get_slice(self.current_slice)
            if hist_slice is not None: self.axial_ax.imshow(hist_slice, alpha=self.histology_data.opacity)

    def display_coronal_view(self):
        self.coronal_ax.set_title(f"Coronal - Row {self.mpr_coords['y']}", color='white', fontsize=10)
        array = self.mr_data.get_array()
        if self.mr_data.visible and array is not None and len(array.shape) >= 3 and 0 <= self.mpr_coords['y'] < array.shape[1]:
            coronal_slice = np.rot90(array[:, self.mpr_coords['y'], :], k=2)
            self.coronal_ax.imshow(coronal_slice, cmap='gray', aspect=self.mr_data.spacing[2]/self.mr_data.spacing[0])

    def display_sagittal_view(self):
        self.sagittal_ax.set_title(f"Sagittal - Col {self.mpr_coords['x']}", color='white', fontsize=10)
        array = self.mr_data.get_array()
        if self.mr_data.visible and array is not None and len(array.shape) >= 3 and 0 <= self.mpr_coords['x'] < array.shape[2]:
            sagittal_slice = np.rot90(array[:, :, self.mpr_coords['x']], k=2)
            self.sagittal_ax.imshow(sagittal_slice, cmap='gray', aspect=self.mr_data.spacing[2]/self.mr_data.spacing[1])

    def update_crosshairs(self):
        if not self._crosshairs_initialized: return
        array = self.mr_data.get_array()
        if array is None or len(array.shape) < 3: return
        
        z, y, x = self.mpr_coords['z'], self.mpr_coords['y'], self.mpr_coords['x']
        s_z, s_y, s_x = array.shape
        disp_x, disp_y = x, s_y - 1 - y
        
        self.crosshair_lines['axial_h'].set_data([-0.5, s_x-0.5], [disp_y, disp_y])
        self.crosshair_lines['axial_v'].set_data([disp_x, disp_x], [-0.5, s_y-0.5])
        self.crosshair_lines['coronal_h'].set_data([-0.5, s_x-0.5], [s_z-1-z, s_z-1-z])
        self.crosshair_lines['coronal_v'].set_data([x, x], [-0.5, s_z-0.5])
        self.crosshair_lines['sagittal_h'].set_data([-0.5, s_y-0.5], [s_z-1-z, s_z-1-z])
        self.crosshair_lines['sagittal_v'].set_data([y, y], [-0.5, s_z-0.5])

    def update_info_panel(self):
        self.info_ax.clear(); self.info_ax.set_facecolor('black'); self.info_ax.axis('off')
        info_text = (f"Slice (Z): {self.current_slice + 1}/{self.max_slices}\n"
                     f"Coords (X,Y): ({self.mpr_coords['x']}, {self.mpr_coords['y']})\n\n"
                     f"Spacing (X,Y,Z): ({self.mr_data.spacing[0]:.2f}, "
                     f"{self.mr_data.spacing[1]:.2f}, {self.mr_data.spacing[2]:.2f})\n\nLayers:\n")
        for d in [self.mr_data, self.segmentation_data, self.histology_data]:
            if d.visible and d.data is not None: info_text += f"✓ {d.name} (α={d.opacity:.2f})\n"
        self.info_ax.text(0.05, 0.95, info_text, transform=self.info_ax.transAxes,
                         verticalalignment='top', color='white', fontsize=9, fontfamily='monospace')

    def apply_window_level(self, image: np.ndarray, window: float, level: float) -> np.ndarray:
        if window <= 0: return image
        min_val, max_val = level - window / 2, level + window / 2
        return np.clip((image - min_val) / (max_val - min_val), 0, 1)

    def on_scroll(self, event):
        if event.inaxes:
            step = 1 if event.button == 'up' else -1
            new_slice = np.clip(self.current_slice + step, 0, self.max_slices - 1 if self.max_slices > 0 else 0)
            if new_slice != self.current_slice: self.set_slice(new_slice); self.slice_changed.emit(new_slice)

    def on_click(self, event):
        if event.inaxes and event.xdata is not None and event.ydata is not None: self.update_mpr_from_event(event)

    def on_mouse_move(self, event):
        if event.inaxes and event.button == 1 and event.xdata is not None and event.ydata is not None: self.update_mpr_from_event(event)

    def update_mpr_from_event(self, event):
        array = self.mr_data.get_array()
        if array is None or len(array.shape) < 3: return
        s_z, s_y, s_x = array.shape
        if event.inaxes == self.axial_ax:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.mpr_coords['x'] = np.clip(x, 0, s_x - 1)
            self.mpr_coords['y'] = np.clip(s_y - 1 - y, 0, s_y - 1)
        elif event.inaxes == self.coronal_ax:
            x, z = int(round(event.xdata)), int(round(event.ydata))
            self.mpr_coords['x'] = np.clip(x, 0, s_x - 1)
            new_slice = np.clip(s_z - 1 - z, 0, self.max_slices - 1)
            if new_slice != self.current_slice: self.set_slice(new_slice); self.slice_changed.emit(new_slice)
        elif event.inaxes == self.sagittal_ax:
            y, z = int(round(event.xdata)), int(round(event.ydata))
            self.mpr_coords['y'] = np.clip(y, 0, s_y - 1)
            new_slice = np.clip(s_z - 1 - z, 0, self.max_slices - 1)
            if new_slice != self.current_slice: self.set_slice(new_slice); self.slice_changed.emit(new_slice)
        self.update_display()


class MedicalViewer(QWidget):
    """Main widget for medical visualization"""
    data_loaded = pyqtSignal(str, object)
    mr_file_loaded = pyqtSignal(str, str)     # (nome_file, percorso_file)
    seg_file_loaded = pyqtSignal(str, str)    # (nome_file, percorso_file)
    status_message_changed = pyqtSignal(str, int) # (messaggio, timeout)
    
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, 0)
        
        if MATPLOTLIB_AVAILABLE:
            self.canvas = MedicalCanvas()
            self.canvas.slice_changed.connect(self.on_slice_changed)
            layout.addWidget(self.canvas, 1)
        else:
            layout.addWidget(QLabel("Matplotlib is not available", alignment=Qt.AlignmentFlag.AlignCenter), 1)
        self.setLayout(layout)
        
    def create_control_panel(self):
        panel = QFrame(); panel.setFrameStyle(QFrame.Shape.StyledPanel); panel.setMaximumWidth(300)
        layout = QVBoxLayout()
        load_group = QGroupBox("Data Loading")
        load_layout = QVBoxLayout()
        self.load_mr_btn = QPushButton("Load MRI"); self.load_mr_btn.clicked.connect(self.load_mr_data)
        self.load_seg_btn = QPushButton("Load Segmentation"); self.load_seg_btn.clicked.connect(self.load_segmentation_data)
        load_layout.addWidget(self.load_mr_btn); load_layout.addWidget(self.load_seg_btn); 
        load_group.setLayout(load_layout); layout.addWidget(load_group)
        
        visibility_group = QGroupBox("Layer Visibility & Opacity")
        visibility_layout = QVBoxLayout()
        for name, checked, value in [("MRI", True, 100), ("Segmentation", True, 80), ("Histology", True, 70)]:
            h_layout = QHBoxLayout()
            cb = QCheckBox(name); cb.setChecked(checked)
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, 100); slider.setValue(value)
            cb.stateChanged.connect(self.update_visibility); slider.valueChanged.connect(self.update_opacity)
            h_layout.addWidget(cb); h_layout.addWidget(slider)
            visibility_layout.addLayout(h_layout)
            setattr(self, f"{name.lower()}_visible_cb", cb); setattr(self, f"{name.lower()}_opacity_slider", slider)
        visibility_group.setLayout(visibility_layout); layout.addWidget(visibility_group)
        
        slice_group = QGroupBox("Slice Control")
        slice_layout = QVBoxLayout()
        slice_control_layout = QHBoxLayout()
        self.slice_spin = QSpinBox(); self.slice_spin.setMinimum(1); self.slice_spin.valueChanged.connect(self.on_slice_spin_changed)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal); self.slice_slider.setMinimum(1); self.slice_slider.valueChanged.connect(self.on_slice_slider_changed)
        slice_control_layout.addWidget(QLabel("Slice:")); slice_control_layout.addWidget(self.slice_spin)
        slice_layout.addLayout(slice_control_layout); slice_layout.addWidget(self.slice_slider)
        slice_group.setLayout(slice_layout); layout.addWidget(slice_group)
        
        wl_group = QGroupBox("Window/Level (MRI)")
        wl_layout = QGridLayout()
        self.window_spin = QDoubleSpinBox(); self.window_spin.setRange(1, 10000); self.window_spin.setValue(400)
        self.level_spin = QDoubleSpinBox(); self.level_spin.setRange(-1000, 3000); self.level_spin.setValue(200)
        self.window_spin.valueChanged.connect(self.update_window_level)
        self.level_spin.valueChanged.connect(self.update_window_level)
        wl_layout.addWidget(QLabel("Window:"), 0, 0); wl_layout.addWidget(self.window_spin, 0, 1)
        wl_layout.addWidget(QLabel("Level:"), 1, 0); wl_layout.addWidget(self.level_spin, 1, 1)
        wl_group.setLayout(wl_layout); layout.addWidget(wl_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def load_data_file(self, data_type: str, title: str, file_filter: str):
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if not file_path or not self.canvas: return

        try:
            data, array_data = None, None
            if data_type == 'histology' and not file_path.endswith(('.nii', '.nii.gz', '.mhd')):
                data = imread(file_path)
            else:
                if not SITK_AVAILABLE:
                    QMessageBox.warning(self, "Error", "SimpleITK is needed for this file type."); return
                data = sitk.ReadImage(file_path)
            
            array_data = sitk.GetArrayFromImage(data) if SITK_AVAILABLE and isinstance(data, sitk.Image) else data
            load_method = getattr(self.canvas, f"load_{data_type}_data")
            load_method(data, os.path.basename(file_path))

            if data_type == 'mr': self.update_slice_controls()
            self.data_loaded.emit(data_type, array_data)
            QMessageBox.information(self, "Success", f"{data_type.title()} loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {data_type} file: {e}")

    def load_mr_data(self): self.load_data_file("mr", "Load MRI", "Medical Images (*.dcm *.nii *.nii.gz *.mhd)")
    def load_segmentation_data(self): self.load_data_file("segmentation", "Load Segmentation", "Medical Images (*.nii *.nii.gz *.seg.nrrd)")

    def update_visibility(self):
        if self.canvas:
            self.canvas.mr_data.visible = self.mri_visible_cb.isChecked()
            self.canvas.segmentation_data.visible = self.segmentation_visible_cb.isChecked()
            self.canvas.histology_data.visible = self.histology_visible_cb.isChecked()
            self.canvas.update_display()

    def update_opacity(self):
        if self.canvas:
            self.canvas.mr_data.opacity = self.mri_opacity_slider.value() / 100.0
            self.canvas.segmentation_data.opacity = self.segmentation_opacity_slider.value() / 100.0
            self.canvas.histology_data.opacity = self.histology_opacity_slider.value() / 100.0
            self.canvas.update_display()

    def update_window_level(self):
        if self.canvas:
            self.canvas.mr_data.window_level = (self.window_spin.value(), self.level_spin.value())
            self.canvas.update_display()
            
    def update_slice_controls(self):
        if self.canvas and self.canvas.max_slices > 0:
            self.slice_spin.setMaximum(self.canvas.max_slices)
            self.slice_slider.setMaximum(self.canvas.max_slices)
            self.slice_spin.setValue(self.canvas.current_slice + 1)
            self.slice_slider.setValue(self.canvas.current_slice + 1)
            
    def on_slice_spin_changed(self, value):
        if self.canvas:
            self.slice_slider.blockSignals(True); self.slice_slider.setValue(value); self.slice_slider.blockSignals(False)
            self.canvas.set_slice(value - 1)
            
    def on_slice_slider_changed(self, value):
        if self.canvas:
            self.slice_spin.blockSignals(True); self.slice_spin.setValue(value); self.slice_spin.blockSignals(False)
            self.canvas.set_slice(value - 1)
            
    def on_slice_changed(self, slice_idx):
        value = slice_idx + 1
        self.slice_spin.blockSignals(True); self.slice_slider.blockSignals(True)
        self.slice_spin.setValue(value); self.slice_slider.setValue(value)
        self.slice_spin.blockSignals(False); self.slice_slider.blockSignals(False)
        
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QMainWindow
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Medical Viewer Test")
            self.setGeometry(100, 100, 1200, 800)
            self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
            self.viewer_tab = MedicalViewer(); self.tabs.addTab(self.viewer_tab, "Medical Viewer")

    app = QApplication(sys.argv)
    main_window = MainWindow(); main_window.show()
    sys.exit(app.exec())