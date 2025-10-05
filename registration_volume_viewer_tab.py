# File: registration_viewer_tab.py

import os
import numpy as np
from typing import Optional
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QCheckBox, QGroupBox, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK non è installato. Le funzionalità di caricamento immagini saranno limitate.")


class ImageCanvas(QLabel):
    """Canvas nativo PyQt6 per visualizzare immagini con overlay"""
    
    slice_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(400, 400)
        
        # Data storage
        self.mr_data: Optional[np.ndarray] = None
        self.histology_data: Optional[np.ndarray] = None
        self.mask_data: Optional[np.ndarray] = None
        
        # Display settings
        self.current_slice = 0
        self.max_slices = 0
        self.mr_opacity = 1.0
        self.histology_opacity = 0.7
        self.mask_opacity = 0.5
        self.show_mr = True
        self.show_histology = True
        self.show_mask = True
        
        # High-res data
        self.highres_histology: Optional[np.ndarray] = None
        self.highres_mask: Optional[np.ndarray] = None
        
        # Window/Level for MR
        self.window = 400
        self.level = 200
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for slice navigation"""
        if self.max_slices > 0:
            delta = event.angleDelta().y()
            step = -1 if delta > 0 else 1
            new_slice = np.clip(self.current_slice + step, 0, self.max_slices - 1)
            if new_slice != self.current_slice:
                self.set_slice(new_slice)
                self.slice_changed.emit(new_slice)
        event.accept()
        
    def load_mr_volume(self, data):
        """Load MR volume data (low resolution)"""
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            self.mr_data = sitk.GetArrayFromImage(data)
        else:
            self.mr_data = data
        self.update_max_slices()
        self.update_display()
        
    def load_histology_volume(self, data):
        """Load histology volume data (low resolution)"""
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            self.histology_data = sitk.GetArrayFromImage(data)
        else:
            self.histology_data = data
        self.update_max_slices()
        self.update_display()
        
    def load_mask_volume(self, data):
        """Load mask volume data (low resolution)"""
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            self.mask_data = sitk.GetArrayFromImage(data)
        else:
            self.mask_data = data
        self.update_display()
        
    def load_highres_slice(self, histology_slice: np.ndarray, mask_slice: Optional[np.ndarray] = None):
        """Load high-resolution data for current slice"""
        self.highres_histology = histology_slice
        self.highres_mask = mask_slice
        self.update_display()
        
    def clear_highres(self):
        """Clear high-resolution data"""
        self.highres_histology = None
        self.highres_mask = None
        self.update_display()
        
    def update_max_slices(self):
        """Update maximum slice count"""
        slice_counts = []
        if self.mr_data is not None: 
            slice_counts.append(self.mr_data.shape[0])
        if self.histology_data is not None: 
            slice_counts.append(self.histology_data.shape[0])
            
        self.max_slices = max(slice_counts) if slice_counts else 0
        
        if self.current_slice >= self.max_slices and self.max_slices > 0:
            self.set_slice(self.max_slices - 1)
            
    def set_slice(self, slice_idx: int):
        """Set current slice index"""
        if 0 <= slice_idx < self.max_slices:
            self.current_slice = slice_idx
            self.update_display()
            
    def apply_window_level(self, image: np.ndarray) -> np.ndarray:
        """Apply window/level to MR image"""
        if self.window <= 0: 
            return image
        min_val = self.level - self.window / 2
        max_val = self.level + self.window / 2
        return np.clip((image - min_val) / (max_val - min_val), 0, 1)
        
    def update_display(self):
        """Update the display with current settings"""
        use_highres = self.highres_histology is not None
        
        if use_highres:
            composite = self._create_highres_composite()
        else:
            composite = self._create_lowres_composite()
            
        if composite is not None:
            self._display_composite(composite)
        else:
            self.setText(f"Slice {self.current_slice + 1}/{self.max_slices}\nNo data to display")
            
    def _create_lowres_composite(self) -> Optional[np.ndarray]:
        """Create low-resolution composite image"""
        composite = None
        
        # Start with MR as base
        if self.show_mr and self.mr_data is not None and self.current_slice < self.mr_data.shape[0]:
            mr_slice = self.apply_window_level(self.mr_data[self.current_slice])
            # Convert to RGB
            composite = np.stack([mr_slice, mr_slice, mr_slice], axis=-1)
            composite = (composite * self.mr_opacity).astype(np.float32)
        
        # Add histology overlay
        if self.show_histology and self.histology_data is not None and self.current_slice < self.histology_data.shape[0]:
            hist_slice = self.histology_data[self.current_slice]
            
            # Ensure RGB
            if hist_slice.ndim == 2:
                hist_slice = np.stack([hist_slice, hist_slice, hist_slice], axis=-1)
            elif hist_slice.ndim == 3 and hist_slice.shape[2] == 1:
                hist_slice = np.repeat(hist_slice, 3, axis=2)
                
            # Normalize if needed
            if hist_slice.max() > 1.0:
                hist_slice = hist_slice / 255.0
            
            if composite is None:
                composite = (hist_slice * self.histology_opacity).astype(np.float32)
            else:
                # Blend with existing composite
                alpha = self.histology_opacity
                composite = composite * (1 - alpha) + hist_slice * alpha
        
        # Add mask overlay
        if self.show_mask and self.mask_data is not None and self.current_slice < self.mask_data.shape[0]:
            mask_slice = self.mask_data[self.current_slice] > 0
            
            if composite is None:
                composite = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
            
            if np.any(mask_slice):
                # Red overlay for mask
                composite[mask_slice, 0] = composite[mask_slice, 0] * (1 - self.mask_opacity) + self.mask_opacity
                composite[mask_slice, 1] *= (1 - self.mask_opacity)
                composite[mask_slice, 2] *= (1 - self.mask_opacity)
        
        if composite is not None:
            composite = np.clip(composite * 255, 0, 255).astype(np.uint8)
            
        return composite
        
    def _create_highres_composite(self) -> Optional[np.ndarray]:
        """Create high-resolution composite image"""
        composite = None
        
        # Start with MR base (low-res) - resize to match highres if needed
        if self.show_mr and self.mr_data is not None and self.current_slice < self.mr_data.shape[0]:
            mr_slice = self.apply_window_level(self.mr_data[self.current_slice])
            
            # Simple resize using numpy (nearest neighbor)
            if self.highres_histology is not None:
                mr_slice = self._resize_array(mr_slice, self.highres_histology.shape[:2])
            
            composite = np.stack([mr_slice, mr_slice, mr_slice], axis=-1)
            composite = (composite * self.mr_opacity).astype(np.float32)
        
        # Add high-res histology
        if self.show_histology and self.highres_histology is not None:
            hist_slice = self.highres_histology
            
            if hist_slice.ndim == 2:
                hist_slice = np.stack([hist_slice, hist_slice, hist_slice], axis=-1)
            elif hist_slice.ndim == 3 and hist_slice.shape[2] == 1:
                hist_slice = np.repeat(hist_slice, 3, axis=2)
                
            # Normalize if needed
            if hist_slice.max() > 1.0:
                hist_slice = hist_slice.astype(np.float32) / 255.0
            
            if composite is None:
                composite = (hist_slice * self.histology_opacity).astype(np.float32)
            else:
                # Ensure same shape
                if composite.shape[:2] != hist_slice.shape[:2]:
                    composite = self._resize_array(composite, hist_slice.shape[:2])
                
                alpha = self.histology_opacity
                composite = composite * (1 - alpha) + hist_slice * alpha
        
        # Add high-res mask
        if self.show_mask and self.highres_mask is not None:
            mask_binary = self.highres_mask > 0
            
            if composite is None:
                composite = np.zeros((*mask_binary.shape, 3), dtype=np.float32)
            elif composite.shape[:2] != mask_binary.shape:
                composite = self._resize_array(composite, mask_binary.shape)
            
            if np.any(mask_binary):
                composite[mask_binary, 0] = composite[mask_binary, 0] * (1 - self.mask_opacity) + self.mask_opacity
                composite[mask_binary, 1] *= (1 - self.mask_opacity)
                composite[mask_binary, 2] *= (1 - self.mask_opacity)
        
        if composite is not None:
            composite = np.clip(composite * 255, 0, 255).astype(np.uint8)
            
        return composite
    
    def _resize_array(self, array: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Simple array resize using nearest neighbor (no scipy needed)"""
        if array.shape[:2] == target_shape:
            return array
            
        h_ratio = target_shape[0] / array.shape[0]
        w_ratio = target_shape[1] / array.shape[1]
        
        # Create index arrays
        y_indices = (np.arange(target_shape[0]) / h_ratio).astype(int)
        x_indices = (np.arange(target_shape[1]) / w_ratio).astype(int)
        
        # Clip to valid range
        y_indices = np.clip(y_indices, 0, array.shape[0] - 1)
        x_indices = np.clip(x_indices, 0, array.shape[1] - 1)
        
        # Resize
        if array.ndim == 2:
            return array[y_indices[:, None], x_indices[None, :]]
        else:  # 3D array (with color channels)
            return array[y_indices[:, None], x_indices[None, :], :]
    
    def _display_composite(self, composite: np.ndarray):
        """Display composite image using QPixmap"""
        height, width = composite.shape[:2]
        
        # Convert to QImage
        if composite.shape[2] == 3:  # RGB
            bytes_per_line = 3 * width
            q_image = QImage(composite.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:  # RGBA
            bytes_per_line = 4 * width
            q_image = QImage(composite.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        
        # Convert to pixmap and display
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)


class RegistrationResultsViewer(QWidget):
    """Main widget for viewing registration results"""
    
    def __init__(self):
        super().__init__()
        self.canvas: Optional[ImageCanvas] = None
        self.pathology_volume = None  # Reference to PathologyVolume object
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        main_layout = QHBoxLayout(self)
        
        # Crea un'area scrollabile per il pannello di controllo
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMaximumWidth(370)
        
        control_panel = self.create_control_panel()
        scroll_area.setWidget(control_panel)
        
        main_layout.addWidget(scroll_area, 0)
        
        self.canvas = ImageCanvas()
        self.canvas.slice_changed.connect(self.on_slice_changed_from_canvas)
        main_layout.addWidget(self.canvas, 1)
            
    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout(load_group)
        self.load_mr_btn = QPushButton("Load MR Volume (Low-Res)")
        self.load_mr_btn.clicked.connect(self.load_mr_volume_dialog)
        self.load_histology_btn = QPushButton("Load Histology Volume (Low-Res)")
        self.load_histology_btn.clicked.connect(self.load_histology_volume_dialog)
        self.load_mask_btn = QPushButton("Load Mask Volume (Low-Res)")
        self.load_mask_btn.clicked.connect(self.load_mask_volume_dialog)
        load_layout.addWidget(self.load_mr_btn)
        load_layout.addWidget(self.load_histology_btn)
        load_layout.addWidget(self.load_mask_btn)
        layout.addWidget(load_group)
        
        visibility_group = QGroupBox("Layer Visibility & Opacity")
        visibility_layout = QVBoxLayout(visibility_group)
        self.mr_visible_cb = QCheckBox("MR Background")
        self.mr_visible_cb.setChecked(True)
        self.mr_visible_cb.stateChanged.connect(self.update_visibility_and_opacity)
        self.histology_visible_cb = QCheckBox("Histology Overlay")
        self.histology_visible_cb.setChecked(True)
        self.histology_visible_cb.stateChanged.connect(self.update_visibility_and_opacity)
        self.mask_visible_cb = QCheckBox("Tumor Mask")
        self.mask_visible_cb.setChecked(True)
        self.mask_visible_cb.stateChanged.connect(self.update_visibility_and_opacity)
        
        self.mr_opacity_slider = self._create_opacity_slider("MR:", 100)
        self.hist_opacity_slider = self._create_opacity_slider("Hist:", 70)
        self.mask_opacity_slider = self._create_opacity_slider("Mask:", 50)
        
        visibility_layout.addWidget(self.mr_visible_cb)
        visibility_layout.addLayout(self.mr_opacity_slider)
        visibility_layout.addWidget(self.histology_visible_cb)
        visibility_layout.addLayout(self.hist_opacity_slider)
        visibility_layout.addWidget(self.mask_visible_cb)
        visibility_layout.addLayout(self.mask_opacity_slider)
        layout.addWidget(visibility_group)
        
        slice_group = QGroupBox("Slice Navigation")
        slice_layout = QVBoxLayout(slice_group)
        slice_control_layout = QHBoxLayout()
        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(1)
        self.slice_spin.valueChanged.connect(self.on_slice_spin_changed)
        slice_control_layout.addWidget(QLabel("Slice:"))
        slice_control_layout.addWidget(self.slice_spin)
        slice_layout.addLayout(slice_control_layout)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(1)
        self.slice_slider.valueChanged.connect(self.on_slice_slider_changed)
        slice_layout.addWidget(self.slice_slider)
        layout.addWidget(slice_group)
        
        highres_group = QGroupBox("High-Resolution View")
        highres_layout = QVBoxLayout(highres_group)
        self.load_highres_btn = QPushButton("Load High-Res for Current Slice")
        self.load_highres_btn.clicked.connect(self.load_current_highres_slice)
        self.load_highres_btn.setEnabled(False)
        self.clear_highres_btn = QPushButton("Return to Low-Res View")
        self.clear_highres_btn.clicked.connect(self.clear_highres)
        self.clear_highres_btn.setEnabled(False)
        self.highres_status = QLabel("Mode: Low-Resolution")
        self.highres_status.setStyleSheet("QLabel { color: gray; }")
        highres_layout.addWidget(self.load_highres_btn)
        highres_layout.addWidget(self.clear_highres_btn)
        highres_layout.addWidget(self.highres_status)
        layout.addWidget(highres_group)
        
        wl_group = QGroupBox("MR Window/Level")
        wl_layout = QVBoxLayout(wl_group)
        self.window_spin = self._create_wl_spinbox("Window:", 1, 10000, 400)
        self.level_spin = self._create_wl_spinbox("Level:", -1000, 3000, 200)
        wl_layout.addLayout(self.window_spin)
        wl_layout.addLayout(self.level_spin)
        layout.addWidget(wl_group)
        
        return panel

    def _create_opacity_slider(self, label_text, value):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.valueChanged.connect(self.update_visibility_and_opacity)
        label = QLabel(f"{value}%")
        slider.label = label
        layout.addWidget(slider)
        layout.addWidget(label)
        return layout

    def _create_wl_spinbox(self, label_text, min_val, max_val, value):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.valueChanged.connect(self.update_window_level)
        layout.addWidget(spinbox)
        return layout

    def load_mr_volume_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load MR Volume", "", 
                                                   "Images (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path: 
            self.load_volume_from_path(file_path, 'mr')

    def load_histology_volume_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Histology Volume", "", 
                                                   "Images (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path: 
            self.load_volume_from_path(file_path, 'histology')

    def load_mask_volume_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Mask Volume", "", 
                                                   "Images (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path: 
            self.load_volume_from_path(file_path, 'mask')

    def load_volume_from_path(self, path, volume_type):
        if not self.canvas or not SITK_AVAILABLE:
            QMessageBox.warning(self, "Error", "SimpleITK or Canvas not available.")
            return
        try:
            data = sitk.ReadImage(path)
            if volume_type == 'mr': 
                self.canvas.load_mr_volume(data)
            elif volume_type == 'histology':
                self.canvas.load_histology_volume(data)
                self.load_highres_btn.setEnabled(self.pathology_volume is not None)
            elif volume_type == 'mask': 
                self.canvas.load_mask_volume(data)
            self.update_slice_controls()
            QMessageBox.information(self, "Success", f"{volume_type.upper()} volume loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {path}: {e}")
            
    def load_current_highres_slice(self):
        if not self.canvas or not self.pathology_volume or not hasattr(self.pathology_volume, 'pathologySlices'):
            QMessageBox.warning(self, "Warning", "Pathology data not loaded or incomplete.")
            return
        
        current_slice = self.canvas.current_slice
        try:
            ps = self.pathology_volume.pathologySlices[current_slice]
            highres_img = ps.loadRgbImage()
            if highres_img:
                highres_array = sitk.GetArrayFromImage(highres_img)[0]
                highres_mask_array = None
                highres_mask = ps.loadMask(0)
                if highres_mask:
                    highres_mask_array = sitk.GetArrayFromImage(highres_mask)[0]
                    
                self.canvas.load_highres_slice(highres_array, highres_mask_array)
                self.clear_highres_btn.setEnabled(True)
                self.highres_status.setText(f"Mode: High-Res (Slice {current_slice + 1})")
                self.highres_status.setStyleSheet("QLabel { color: green; }")
            else:
                QMessageBox.warning(self, "Warning", "Failed to load high-res image for this slice.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load high-res data: {e}")
            
    def clear_highres(self):
        if self.canvas:
            self.canvas.clear_highres()
            self.clear_highres_btn.setEnabled(False)
            self.highres_status.setText("Mode: Low-Resolution")
            self.highres_status.setStyleSheet("QLabel { color: gray; }")
            
    def update_visibility_and_opacity(self):
        if not self.canvas: 
            return
        self.canvas.show_mr = self.mr_visible_cb.isChecked()
        self.canvas.show_histology = self.histology_visible_cb.isChecked()
        self.canvas.show_mask = self.mask_visible_cb.isChecked()
        
        self.canvas.mr_opacity = self.mr_opacity_slider.itemAt(1).widget().value() / 100.0
        self.canvas.histology_opacity = self.hist_opacity_slider.itemAt(1).widget().value() / 100.0
        self.canvas.mask_opacity = self.mask_opacity_slider.itemAt(1).widget().value() / 100.0
        
        self.mr_opacity_slider.itemAt(2).widget().setText(f"{int(self.canvas.mr_opacity*100)}%")
        self.hist_opacity_slider.itemAt(2).widget().setText(f"{int(self.canvas.histology_opacity*100)}%")
        self.mask_opacity_slider.itemAt(2).widget().setText(f"{int(self.canvas.mask_opacity*100)}%")

        self.canvas.update_display()
            
    def update_window_level(self):
        if self.canvas:
            self.canvas.window = self.window_spin.itemAt(1).widget().value()
            self.canvas.level = self.level_spin.itemAt(1).widget().value()
            self.canvas.update_display()
            
    def update_slice_controls(self):
        if self.canvas and self.canvas.max_slices > 0:
            self.slice_spin.setMaximum(self.canvas.max_slices)
            self.slice_slider.setMaximum(self.canvas.max_slices)
            self.slice_spin.setValue(self.canvas.current_slice + 1)
            self.slice_slider.setValue(self.canvas.current_slice + 1)
            
    def on_slice_spin_changed(self, value):
        if self.canvas:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(value)
            self.slice_slider.blockSignals(False)
            self.canvas.set_slice(value - 1)
            if self.canvas.highres_histology is not None: 
                self.clear_highres()
            
    def on_slice_slider_changed(self, value):
        if self.canvas:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(value)
            self.slice_spin.blockSignals(False)
            self.canvas.set_slice(value - 1)
            if self.canvas.highres_histology is not None: 
                self.clear_highres()
            
    def on_slice_changed_from_canvas(self, slice_idx):
        value = slice_idx + 1
        self.slice_spin.blockSignals(True)
        self.slice_slider.blockSignals(True)
        self.slice_spin.setValue(value)
        self.slice_slider.setValue(value)
        self.slice_spin.blockSignals(False)
        self.slice_slider.blockSignals(False)
        if self.canvas and self.canvas.highres_histology is not None: 
            self.clear_highres()


# Esempio di test (se eseguito come script autonomo)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import sys
    
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("Registration Results Viewer Test")
    main_window.setGeometry(100, 100, 1200, 800)
    
    viewer_tab = RegistrationResultsViewer()
    
    main_window.setCentralWidget(viewer_tab)
    main_window.show()
    
    sys.exit(app.exec())