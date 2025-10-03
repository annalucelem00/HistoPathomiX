#PROVA DI CODICE PER IL TAB DI VISUALIZZAZIONE DELLA REGISTRAZIONE


#!/usr/bin/env python3
"""
Registration Results Viewer Tab
Optimized viewer for low-resolution registration results with optional high-res slice viewing
"""

import os
import numpy as np
from typing import Optional, Tuple
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QCheckBox, QGroupBox, QSpinBox, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK not available")


class RegistrationCanvas(FigureCanvas):
    """Canvas for displaying registration results with overlay controls"""
    
    slice_changed = pyqtSignal(int)
    
    def __init__(self):
        self.figure = Figure(figsize=(10, 8), facecolor='black')
        super().__init__(self.figure)
        
        self.figure.patch.set_facecolor('black')
        self.main_ax = None
        self.setup_plot()
        
        # Data storage
        self.mr_data = None
        self.histology_data = None
        self.mask_data = None
        
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
        self.highres_histology = None
        self.highres_mask = None
        
        # Window/Level for MR
        self.window = 400
        self.level = 200
        
        # Mouse interaction
        self.mpl_connect('scroll_event', self.on_scroll)
        
    def setup_plot(self):
        """Setup the main plotting area"""
        self.figure.clear()
        self.main_ax = self.figure.add_subplot(111)
        self.main_ax.set_facecolor('black')
        self.main_ax.set_title('Registration Results', color='white', fontsize=12)
        self.main_ax.set_xticks([])
        self.main_ax.set_yticks([])
        self.figure.tight_layout()
        
    def load_mr_volume(self, data: np.ndarray):
        """Load MR volume data (low resolution)"""
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            self.mr_data = sitk.GetArrayFromImage(data)
        else:
            self.mr_data = data
            
        self.update_max_slices()
        self.update_display()
        
    def load_histology_volume(self, data: np.ndarray):
        """Load histology volume data (low resolution)"""
        if SITK_AVAILABLE and isinstance(data, sitk.Image):
            self.histology_data = sitk.GetArrayFromImage(data)
        else:
            self.histology_data = data
            
        self.update_max_slices()
        self.update_display()
        
    def load_mask_volume(self, data: np.ndarray):
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
            self.current_slice = self.max_slices - 1
            
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
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Clear previous images
        for img in self.main_ax.get_images():
            img.remove()
            
        self.main_ax.set_title(
            f'Registration Results - Slice {self.current_slice + 1}/{self.max_slices}',
            color='white', fontsize=12
        )
        
        # Check if we should display high-res or low-res
        use_highres = self.highres_histology is not None
        
        if use_highres:
            self._display_highres()
        else:
            self._display_lowres()
            
        self.draw()
        
    def _display_lowres(self):
        """Display low-resolution overlay"""
        # Display MR background
        if self.show_mr and self.mr_data is not None:
            mr_slice = self.mr_data[self.current_slice]
            mr_slice = self.apply_window_level(mr_slice)
            self.main_ax.imshow(
                mr_slice, 
                cmap='gray', 
                alpha=self.mr_opacity,
                interpolation='bilinear'
            )
            
        # Display histology overlay
        if self.show_histology and self.histology_data is not None:
            hist_slice = self.histology_data[self.current_slice]
            
            # Handle RGB or grayscale
            if len(hist_slice.shape) == 2:
                self.main_ax.imshow(
                    hist_slice,
                    cmap='hot',
                    alpha=self.histology_opacity,
                    interpolation='bilinear'
                )
            else:
                self.main_ax.imshow(
                    hist_slice,
                    alpha=self.histology_opacity,
                    interpolation='bilinear'
                )
                
        # Display mask overlay
        if self.show_mask and self.mask_data is not None:
            mask_slice = self.mask_data[self.current_slice]
            mask_binary = mask_slice > 0
            
            if np.any(mask_binary):
                overlay = np.zeros((*mask_slice.shape, 4))
                overlay[mask_binary] = [1, 0, 0, self.mask_opacity]  # Red
                self.main_ax.imshow(overlay, interpolation='nearest')
                
    def _display_highres(self):
        """Display high-resolution slice"""
        # Display MR background (still low-res for context)
        if self.show_mr and self.mr_data is not None:
            mr_slice = self.mr_data[self.current_slice]
            mr_slice = self.apply_window_level(mr_slice)
            self.main_ax.imshow(
                mr_slice,
                cmap='gray',
                alpha=self.mr_opacity,
                interpolation='bilinear'
            )
            
        # Display high-res histology
        if self.show_histology and self.highres_histology is not None:
            if len(self.highres_histology.shape) == 2:
                self.main_ax.imshow(
                    self.highres_histology,
                    cmap='hot',
                    alpha=self.histology_opacity,
                    interpolation='bilinear'
                )
            else:
                self.main_ax.imshow(
                    self.highres_histology,
                    alpha=self.histology_opacity,
                    interpolation='bilinear'
                )
                
        # Display high-res mask
        if self.show_mask and self.highres_mask is not None:
            mask_binary = self.highres_mask > 0
            
            if np.any(mask_binary):
                overlay = np.zeros((*self.highres_mask.shape, 4))
                overlay[mask_binary] = [1, 0, 0, self.mask_opacity]
                self.main_ax.imshow(overlay, interpolation='nearest')
                
    def on_scroll(self, event):
        """Handle mouse scroll for slice navigation"""
        if event.inaxes:
            step = 1 if event.button == 'up' else -1
            new_slice = np.clip(
                self.current_slice + step,
                0,
                self.max_slices - 1 if self.max_slices > 0 else 0
            )
            
            if new_slice != self.current_slice:
                self.set_slice(new_slice)
                self.slice_changed.emit(new_slice)


class RegistrationResultsViewer(QWidget):
    """Main widget for viewing registration results"""
    
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.pathology_volume = None  # Reference to PathologyVolume object
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        main_layout = QHBoxLayout()
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)
        
        # Canvas
        if MATPLOTLIB_AVAILABLE:
            self.canvas = RegistrationCanvas()
            self.canvas.slice_changed.connect(self.on_slice_changed)
            main_layout.addWidget(self.canvas, 1)
        else:
            main_layout.addWidget(
                QLabel("Matplotlib not available", alignment=Qt.AlignmentFlag.AlignCenter),
                1
            )
            
        self.setLayout(main_layout)
        
    def create_control_panel(self):
        """Create the control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(350)
        
        layout = QVBoxLayout()
        
        # Load data group
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout()
        
        self.load_mr_btn = QPushButton("Load MR Volume (Low-Res)")
        self.load_mr_btn.clicked.connect(self.load_mr_volume)
        
        self.load_histology_btn = QPushButton("Load Histology Volume (Low-Res)")
        self.load_histology_btn.clicked.connect(self.load_histology_volume)
        
        self.load_mask_btn = QPushButton("Load Mask Volume (Low-Res)")
        self.load_mask_btn.clicked.connect(self.load_mask_volume)
        
        load_layout.addWidget(self.load_mr_btn)
        load_layout.addWidget(self.load_histology_btn)
        load_layout.addWidget(self.load_mask_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Visibility group
        visibility_group = QGroupBox("Layer Visibility")
        visibility_layout = QVBoxLayout()
        
        self.mr_visible_cb = QCheckBox("MR Background")
        self.mr_visible_cb.setChecked(True)
        self.mr_visible_cb.stateChanged.connect(self.update_visibility)
        
        self.histology_visible_cb = QCheckBox("Histology Overlay")
        self.histology_visible_cb.setChecked(True)
        self.histology_visible_cb.stateChanged.connect(self.update_visibility)
        
        self.mask_visible_cb = QCheckBox("Tumor Mask")
        self.mask_visible_cb.setChecked(True)
        self.mask_visible_cb.stateChanged.connect(self.update_visibility)
        
        visibility_layout.addWidget(self.mr_visible_cb)
        visibility_layout.addWidget(self.histology_visible_cb)
        visibility_layout.addWidget(self.mask_visible_cb)
        
        visibility_group.setLayout(visibility_layout)
        layout.addWidget(visibility_group)
        
        # Opacity group
        opacity_group = QGroupBox("Layer Opacity")
        opacity_layout = QVBoxLayout()
        
        # MR opacity
        mr_opacity_layout = QHBoxLayout()
        mr_opacity_layout.addWidget(QLabel("MR:"))
        self.mr_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mr_opacity_slider.setRange(0, 100)
        self.mr_opacity_slider.setValue(100)
        self.mr_opacity_slider.valueChanged.connect(self.update_opacity)
        self.mr_opacity_label = QLabel("100%")
        mr_opacity_layout.addWidget(self.mr_opacity_slider)
        mr_opacity_layout.addWidget(self.mr_opacity_label)
        opacity_layout.addLayout(mr_opacity_layout)
        
        # Histology opacity
        hist_opacity_layout = QHBoxLayout()
        hist_opacity_layout.addWidget(QLabel("Histology:"))
        self.hist_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hist_opacity_slider.setRange(0, 100)
        self.hist_opacity_slider.setValue(70)
        self.hist_opacity_slider.valueChanged.connect(self.update_opacity)
        self.hist_opacity_label = QLabel("70%")
        hist_opacity_layout.addWidget(self.hist_opacity_slider)
        hist_opacity_layout.addWidget(self.hist_opacity_label)
        opacity_layout.addLayout(hist_opacity_layout)
        
        # Mask opacity
        mask_opacity_layout = QHBoxLayout()
        mask_opacity_layout.addWidget(QLabel("Mask:"))
        self.mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(50)
        self.mask_opacity_slider.valueChanged.connect(self.update_opacity)
        self.mask_opacity_label = QLabel("50%")
        mask_opacity_layout.addWidget(self.mask_opacity_slider)
        mask_opacity_layout.addWidget(self.mask_opacity_label)
        opacity_layout.addLayout(mask_opacity_layout)
        
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)
        
        # Slice control
        slice_group = QGroupBox("Slice Navigation")
        slice_layout = QVBoxLayout()
        
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
        
        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)
        
        # High-res viewing
        highres_group = QGroupBox("High-Resolution View")
        highres_layout = QVBoxLayout()
        
        self.load_highres_btn = QPushButton("Load High-Res Slice")
        self.load_highres_btn.clicked.connect(self.load_highres_slice)
        self.load_highres_btn.setEnabled(False)
        
        self.clear_highres_btn = QPushButton("Clear High-Res")
        self.clear_highres_btn.clicked.connect(self.clear_highres)
        self.clear_highres_btn.setEnabled(False)
        
        self.highres_status = QLabel("Low-res mode")
        self.highres_status.setStyleSheet("QLabel { color: gray; }")
        
        highres_layout.addWidget(self.load_highres_btn)
        highres_layout.addWidget(self.clear_highres_btn)
        highres_layout.addWidget(self.highres_status)
        
        highres_group.setLayout(highres_layout)
        layout.addWidget(highres_group)
        
        # Window/Level for MR
        wl_group = QGroupBox("MR Window/Level")
        wl_layout = QVBoxLayout()
        
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10000)
        self.window_spin.setValue(400)
        self.window_spin.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(self.window_spin)
        wl_layout.addLayout(window_layout)
        
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 3000)
        self.level_spin.setValue(200)
        self.level_spin.valueChanged.connect(self.update_window_level)
        level_layout.addWidget(self.level_spin)
        wl_layout.addLayout(level_layout)
        
        wl_group.setLayout(wl_layout)
        layout.addWidget(wl_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        return panel
        
    def load_mr_volume(self):
        """Load MR volume"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load MR Volume", "",
            "Medical Images (*.nii *.nii.gz *.mhd *.nrrd)"
        )
        
        if file_path and self.canvas:
            try:
                if SITK_AVAILABLE:
                    data = sitk.ReadImage(file_path)
                    self.canvas.load_mr_volume(data)
                    self.update_slice_controls()
                    QMessageBox.information(self, "Success", "MR volume loaded")
                else:
                    QMessageBox.warning(self, "Error", "SimpleITK required")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                
    def load_histology_volume(self):
        """Load histology volume"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Histology Volume", "",
            "Medical Images (*.nii *.nii.gz *.mhd *.nrrd)"
        )
        
        if file_path and self.canvas:
            try:
                if SITK_AVAILABLE:
                    data = sitk.ReadImage(file_path)
                    self.canvas.load_histology_volume(data)
                    self.update_slice_controls()
                    self.load_highres_btn.setEnabled(True)
                    QMessageBox.information(self, "Success", "Histology volume loaded")
                else:
                    QMessageBox.warning(self, "Error", "SimpleITK required")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                
    def load_mask_volume(self):
        """Load mask volume"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask Volume", "",
            "Medical Images (*.nii *.nii.gz *.mhd *.nrrd)"
        )
        
        if file_path and self.canvas:
            try:
                if SITK_AVAILABLE:
                    data = sitk.ReadImage(file_path)
                    self.canvas.load_mask_volume(data)
                    QMessageBox.information(self, "Success", "Mask volume loaded")
                else:
                    QMessageBox.warning(self, "Error", "SimpleITK required")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                
    def load_highres_slice(self):
        """Load high-resolution version of current slice"""
        if not self.canvas or not self.pathology_volume:
            QMessageBox.warning(
                self, "Warning",
                "Load pathology data first or set pathology_volume reference"
            )
            return
            
        current_slice = self.canvas.current_slice
        
        try:
            # Load high-res slice from PathologyVolume
            if hasattr(self.pathology_volume, 'pathologySlices'):
                ps = self.pathology_volume.pathologySlices[current_slice]
                
                # Load high-res image
                highres_img = ps.loadRgbImage()
                if highres_img:
                    highres_array = sitk.GetArrayFromImage(highres_img)
                    
                    # Load high-res mask
                    highres_mask = ps.loadMask(0)
                    highres_mask_array = None
                    if highres_mask:
                        highres_mask_array = sitk.GetArrayFromImage(highres_mask)
                        
                    self.canvas.load_highres_slice(highres_array, highres_mask_array)
                    self.clear_highres_btn.setEnabled(True)
                    self.highres_status.setText(f"High-res: Slice {current_slice + 1}")
                    self.highres_status.setStyleSheet("QLabel { color: green; }")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to load high-res image")
            else:
                QMessageBox.warning(self, "Warning", "Pathology slices not available")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load high-res: {e}")
            
    def clear_highres(self):
        """Clear high-resolution display"""
        if self.canvas:
            self.canvas.clear_highres()
            self.clear_highres_btn.setEnabled(False)
            self.highres_status.setText("Low-res mode")
            self.highres_status.setStyleSheet("QLabel { color: gray; }")
            
    def update_visibility(self):
        """Update layer visibility"""
        if self.canvas:
            self.canvas.show_mr = self.mr_visible_cb.isChecked()
            self.canvas.show_histology = self.histology_visible_cb.isChecked()
            self.canvas.show_mask = self.mask_visible_cb.isChecked()
            self.canvas.update_display()
            
    def update_opacity(self):
        """Update layer opacity"""
        if self.canvas:
            self.canvas.mr_opacity = self.mr_opacity_slider.value() / 100.0
            self.canvas.histology_opacity = self.hist_opacity_slider.value() / 100.0
            self.canvas.mask_opacity = self.mask_opacity_slider.value() / 100.0
            
            self.mr_opacity_label.setText(f"{self.mr_opacity_slider.value()}%")
            self.hist_opacity_label.setText(f"{self.hist_opacity_slider.value()}%")
            self.mask_opacity_label.setText(f"{self.mask_opacity_slider.value()}%")
            
            self.canvas.update_display()
            
    def update_window_level(self):
        """Update MR window/level"""
        if self.canvas:
            self.canvas.window = self.window_spin.value()
            self.canvas.level = self.level_spin.value()
            self.canvas.update_display()
            
    def update_slice_controls(self):
        """Update slice navigation controls"""
        if self.canvas and self.canvas.max_slices > 0:
            self.slice_spin.setMaximum(self.canvas.max_slices)
            self.slice_slider.setMaximum(self.canvas.max_slices)
            self.slice_spin.setValue(self.canvas.current_slice + 1)
            self.slice_slider.setValue(self.canvas.current_slice + 1)
            
    def on_slice_spin_changed(self, value):
        """Handle slice spinbox change"""
        if self.canvas:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(value)
            self.slice_slider.blockSignals(False)
            self.canvas.set_slice(value - 1)
            
    def on_slice_slider_changed(self, value):
        """Handle slice slider change"""
        if self.canvas:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(value)
            self.slice_spin.blockSignals(False)
            self.canvas.set_slice(value - 1)
            
    def on_slice_changed(self, slice_idx):
        """Handle slice change from canvas"""
        value = slice_idx + 1
        self.slice_spin.blockSignals(True)
        self.slice_slider.blockSignals(True)
        self.slice_spin.setValue(value)
        self.slice_slider.setValue(value)
        self.slice_spin.blockSignals(False)
        self.slice_slider.blockSignals(False)
        
        # Clear high-res when changing slices
        if self.canvas.highres_histology is not None:
            self.clear_highres()


# Integration example
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
    import sys
    
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("Registration Results Viewer Test")
    main_window.setGeometry(100, 100, 1200, 800)
    
    tabs = QTabWidget()
    viewer_tab = RegistrationResultsViewer()
    tabs.addTab(viewer_tab, "Registration Results")
    
    main_window.setCentralWidget(tabs)
    main_window.show()
    
    sys.exit(app.exec())