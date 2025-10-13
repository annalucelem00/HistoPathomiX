# File: registration_viewer_tab.py
"""
Fixed Registration Results Viewer
Replicates the 3D Slicer visualization behavior for standalone Python
"""

import os
import numpy as np
from typing import Optional, Dict
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QGroupBox, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK is not installed. Image loading functionalities will be limited.")


class ImageCanvas(QLabel):
    """Canvas to display medical images with proper orientation and overlay."""
    slice_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setMinimumSize(400, 400)

        # Raw SITK images (with proper spacing/origin/direction)
        self.mr_image: Optional[sitk.Image] = None
        self.histology_image: Optional[sitk.Image] = None
        self.mask_image: Optional[sitk.Image] = None

        # Display settings
        self.current_slice = 0
        self.max_slices = 0
        self.mr_opacity = 1.0
        self.histology_opacity = 0.7
        self.mask_opacity = 0.5
        self.show_mr = True
        self.show_histology = True
        self.show_mask = True

        # High-res data (original pathology slices)
        self.highres_histology: Optional[np.ndarray] = None
        self.highres_mask: Optional[np.ndarray] = None
        self.highres_transform: Optional[sitk.Transform] = None

        # Window/Level for MR
        self.window = 400
        self.level = 200

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for slice navigation."""
        if self.max_slices > 0:
            delta = event.angleDelta().y()
            step = 1 if delta > 0 else -1
            new_slice = np.clip(self.current_slice + step, 0, self.max_slices - 1)
            if new_slice != self.current_slice:
                self.set_slice(new_slice)
                self.slice_changed.emit(new_slice)
        event.accept()

    def load_mr_volume(self, mr_image: sitk.Image):
        """Load MR volume - keep as SITK image to preserve physical space."""
        self.mr_image = sitk.Cast(mr_image, sitk.sitkFloat32)
        self.max_slices = self.mr_image.GetSize()[2]
        logging.info(f"Loaded MR volume: size={self.mr_image.GetSize()}, "
                    f"spacing={self.mr_image.GetSpacing()}, "
                    f"origin={self.mr_image.GetOrigin()}")

    def load_histology_volume(self, histology_image: sitk.Image):
        """Load registered histology volume."""
        self.histology_image = histology_image
        hist_size = self.histology_image.GetSize()
        mr_size = self.mr_image.GetSize() if self.mr_image else (0, 0, 0)
        
        logging.info(f"Loaded histology volume: size={hist_size}, "
                    f"spacing={self.histology_image.GetSpacing()}")
        
        # Warn if histology has fewer slices than MR
        if mr_size[2] > hist_size[2]:
            logging.warning(f"Histology has {hist_size[2]} slices but MR has {mr_size[2]} slices. "
                          f"Some slices will show only MR.")

    def load_mask_volume(self, mask_image: sitk.Image):
        """Load registered mask volume."""
        self.mask_image = mask_image
        mask_size = self.mask_image.GetSize()
        mr_size = self.mr_image.GetSize() if self.mr_image else (0, 0, 0)
        
        logging.info(f"Loaded mask volume: size={mask_size}")
        
        # Warn if mask has fewer slices than MR
        if mr_size[2] > mask_size[2]:
            logging.warning(f"Mask has {mask_size[2]} slices but MR has {mr_size[2]} slices. "
                          f"Some slices will show no mask.")

    def load_highres_slice(self, histology_img: sitk.Image, mask_img: Optional[sitk.Image],
                          transform: sitk.Transform):
        """
        Load high-resolution original histology slice with its transform.
        
        Args:
            histology_img: Original high-res RGB histology image
            mask_img: Original high-res mask (if available)
            transform: The transform from pathology space to MR space
        """
        self.highres_histology = histology_img
        self.highres_mask = mask_img
        self.highres_transform = transform
        self.update_display()

    def clear_highres(self):
        """Clear high-resolution data to return to low-res view."""
        self.highres_histology = None
        self.highres_mask = None
        self.highres_transform = None
        self.update_display()

    def clear_all_data(self):
        """Clear all loaded image data."""
        self.mr_image = None
        self.histology_image = None
        self.mask_image = None
        self.highres_histology = None
        self.highres_mask = None
        self.highres_transform = None
        self.max_slices = 0
        self.current_slice = 0
        self.setPixmap(QPixmap())
        self.setText("Load data to begin viewing.")

    def set_slice(self, slice_idx: int):
        """Set the current slice index and update the display."""
        if 0 <= slice_idx < self.max_slices:
            self.current_slice = slice_idx
            self.update_display()

    def apply_window_level(self, image_array: np.ndarray) -> np.ndarray:
        """Apply window/level to MR data and normalize to [0,1]."""
        if self.window <= 0:
            return np.clip(image_array / 255.0, 0, 1)
        
        min_val = self.level - self.window / 2.0
        max_val = self.level + self.window / 2.0
        
        windowed = (image_array - min_val) / (max_val - min_val)
        return np.clip(windowed, 0, 1)

    def extract_2d_slice(self, image_3d: sitk.Image, slice_idx: int) -> Optional[sitk.Image]:
        """Extract a 2D slice from a 3D volume maintaining physical space info."""
        size = list(image_3d.GetSize())
        
        # Check if slice index is within bounds
        if slice_idx >= size[2] or slice_idx < 0:
            logging.warning(f"Slice index {slice_idx} out of bounds for image with {size[2]} slices")
            return None
        
        size[2] = 0  # Extract single slice
        index = [0, 0, slice_idx]
        
        try:
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)
            slice_2d = extractor.Execute(image_3d)
            return slice_2d
        except Exception as e:
            logging.error(f"Failed to extract slice {slice_idx}: {e}")
            return None

    def resample_to_reference(self, moving_image: sitk.Image, 
                             reference_image: sitk.Image,
                             transform: Optional[sitk.Transform] = None,
                             interpolator = sitk.sitkLinear,
                             default_value = 0.0) -> sitk.Image:
        """
        Resample moving image to reference image space.
        This replicates the Slicer resampling behavior.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        
        if transform:
            resampler.SetTransform(transform)
        else:
            resampler.SetTransform(sitk.Transform())
        
        return resampler.Execute(moving_image)

    def sitk_to_numpy_display(self, image_2d: sitk.Image) -> np.ndarray:
        """
        Convert SITK 2D image to numpy array for display.
        Applies proper orientation for visualization (flip Y axis for screen display).
        """
        array = sitk.GetArrayFromImage(image_2d)
        
        # SITK array is in (row, col) which is (Y, X) in image coordinates
        # Flip Y axis to match standard screen display (origin at top-left)
        array = np.flipud(array)
        
        return array

    def update_display(self):
        """Render and display the composite image for the current slice."""
        if self.mr_image is None:
            self.setText(f"Slice {self.current_slice + 1}/{self.max_slices}\nNo MR data loaded")
            return

        try:
            # Extract the reference MR slice
            mr_slice_2d = self.extract_2d_slice(self.mr_image, self.current_slice)
            if mr_slice_2d is None:
                self.setText(f"Error: Could not extract MR slice {self.current_slice + 1}")
                return
            
            mr_array = self.sitk_to_numpy_display(mr_slice_2d)
            
            # Apply window/level and create RGB base
            mr_windowed = self.apply_window_level(mr_array)
            composite = np.stack([mr_windowed] * 3, axis=-1)
            
            if self.show_mr:
                composite = composite * self.mr_opacity
            else:
                composite = composite * 0.0
            
            # Overlay histology
            if self.show_histology:
                if self.highres_histology is not None:
                    # High-res mode: resample original histology to MR slice space
                    try:
                        hist_resampled = self.resample_to_reference(
                            self.highres_histology,
                            mr_slice_2d,
                            self.highres_transform,
                            interpolator=sitk.sitkLinear,
                            default_value=255.0
                        )
                        hist_array = self.sitk_to_numpy_display(hist_resampled)
                    except Exception as e:
                        logging.warning(f"Failed to resample high-res histology: {e}")
                        hist_array = None
                    
                elif self.histology_image is not None:
                    # Low-res mode: use registered volume
                    # Check if slice exists in histology volume
                    hist_size = self.histology_image.GetSize()
                    if self.current_slice < hist_size[2]:
                        hist_slice_2d = self.extract_2d_slice(self.histology_image, self.current_slice)
                        if hist_slice_2d is not None:
                            try:
                                hist_resampled = self.resample_to_reference(
                                    hist_slice_2d,
                                    mr_slice_2d,
                                    interpolator=sitk.sitkLinear,
                                    default_value=255.0
                                )
                                hist_array = self.sitk_to_numpy_display(hist_resampled)
                            except Exception as e:
                                logging.warning(f"Failed to resample histology slice: {e}")
                                hist_array = None
                        else:
                            hist_array = None
                    else:
                        # No histology for this slice
                        hist_array = None
                else:
                    hist_array = None
                
                if hist_array is not None:
                    # Handle RGB histology
                    if hist_array.ndim == 2:
                        # Grayscale - convert to RGB
                        hist_rgb = np.stack([hist_array] * 3, axis=-1)
                    else:
                        hist_rgb = hist_array
                    
                    # Normalize to [0, 1]
                    if hist_rgb.max() > 1.0:
                        hist_rgb = hist_rgb / 255.0
                    
                    # Alpha blending
                    alpha = self.histology_opacity
                    composite = composite * (1 - alpha) + hist_rgb * alpha
            
            # Overlay mask
            if self.show_mask:
                if self.highres_mask is not None:
                    # High-res mask
                    try:
                        mask_resampled = self.resample_to_reference(
                            self.highres_mask,
                            mr_slice_2d,
                            self.highres_transform,
                            interpolator=sitk.sitkNearestNeighbor,
                            default_value=0.0
                        )
                        mask_array = self.sitk_to_numpy_display(mask_resampled)
                    except Exception as e:
                        logging.warning(f"Failed to resample high-res mask: {e}")
                        mask_array = None
                    
                elif self.mask_image is not None:
                    # Low-res mask
                    # Check if slice exists in mask volume
                    mask_size = self.mask_image.GetSize()
                    if self.current_slice < mask_size[2]:
                        mask_slice_2d = self.extract_2d_slice(self.mask_image, self.current_slice)
                        if mask_slice_2d is not None:
                            try:
                                mask_resampled = self.resample_to_reference(
                                    mask_slice_2d,
                                    mr_slice_2d,
                                    interpolator=sitk.sitkNearestNeighbor,
                                    default_value=0.0
                                )
                                mask_array = self.sitk_to_numpy_display(mask_resampled)
                            except Exception as e:
                                logging.warning(f"Failed to resample mask slice: {e}")
                                mask_array = None
                        else:
                            mask_array = None
                    else:
                        # No mask for this slice
                        mask_array = None
                else:
                    mask_array = None
                
                if mask_array is not None:
                    mask_binary = mask_array > 0
                    if np.any(mask_binary):
                        # Red overlay for mask
                        composite[mask_binary, 0] = (composite[mask_binary, 0] * (1 - self.mask_opacity) + 
                                                     self.mask_opacity)
                        composite[mask_binary, 1] *= (1 - self.mask_opacity)
                        composite[mask_binary, 2] *= (1 - self.mask_opacity)
            
            # Convert to displayable format
            display_array = np.clip(composite * 255, 0, 255).astype(np.uint8)
            self._display_numpy_array(display_array)
            
        except Exception as e:
            logging.error(f"Error updating display: {e}", exc_info=True)
            self.setText(f"Error displaying slice {self.current_slice + 1}")

    def _display_numpy_array(self, array: np.ndarray):
        """Convert numpy array to QPixmap and display."""
        if array.ndim == 2:
            # Grayscale
            array = np.stack([array] * 3, axis=-1)
        
        height, width = array.shape[:2]
        bytes_per_line = 3 * width
        
        q_image = QImage(array.data, width, height, bytes_per_line, 
                        QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


class RegistrationResultsViewer(QWidget):
    """Main widget for viewing registration results."""
    
    def __init__(self):
        super().__init__()
        self.canvas: Optional[ImageCanvas] = None
        self.pathology_volume = None
        self.registered_mask_paths: Dict[str, str] = {}
        self.current_mask_name = None
        self.init_ui()

    def load_data_from_registration(self, results: dict):
        """
        Load all data from registration results.
        
        Expected results structure:
        {
            'fixed_volume': path_to_mr.nii.gz,
            'registered_rgb': path_to_registered_histology.mha,
            'registered_masks': {'region0': path_to_mask.mha, ...},
            'pathology_volume': PathologyVolume object,
            'output_dir': registration output directory
        }
        """
        logging.info("Viewer: Loading registration results...")
        
        if not self.canvas or not SITK_AVAILABLE:
            QMessageBox.critical(self, "Error", 
                               "Viewer canvas or SimpleITK is not available.")
            return

        try:
            self.canvas.clear_all_data()
            self.pathology_volume = None
            self.registered_mask_paths = {}

            # 1. Load MR volume (fixed/reference)
            fixed_path = results.get('fixed_volume')
            if not fixed_path or not os.path.exists(fixed_path):
                raise FileNotFoundError(f"MR volume not found: {fixed_path}")
            
            mr_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            self.canvas.load_mr_volume(mr_image)
            logging.info(f"Loaded MR volume: {os.path.basename(fixed_path)}")

            # 2. Load registered histology volume
            histology_path = results.get('registered_rgb')
            if histology_path and os.path.exists(histology_path):
                histology_image = sitk.ReadImage(histology_path)
                self.canvas.load_histology_volume(histology_image)
                logging.info(f"Loaded histology: {os.path.basename(histology_path)}")
            else:
                logging.warning("Registered histology volume not found")

            # 3. Load registered masks
            self.registered_mask_paths = results.get('registered_masks', {})
            if self.registered_mask_paths:
                # Load first mask by default
                first_mask_name = list(self.registered_mask_paths.keys())[0]
                self.load_mask(first_mask_name)
                
                # Update mask selector
                self.mask_selector.clear()
                self.mask_selector.addItems(list(self.registered_mask_paths.keys()))
                self.mask_selector.setCurrentText(first_mask_name)
            else:
                logging.warning("No registered masks found")

            # 4. Store pathology volume for high-res access
            self.pathology_volume = results.get('pathology_volume')
            if self.pathology_volume:
                self.load_highres_btn.setEnabled(True)
                logging.info("PathologyVolume object loaded for high-res viewing")
            else:
                self.load_highres_btn.setEnabled(False)
                logging.warning("PathologyVolume not available")

            # 5. Update UI
            self.update_slice_controls()
            self.canvas.set_slice(0)
            
            QMessageBox.information(self, "Success", 
                                  "Registration results loaded successfully!")

        except Exception as e:
            logging.error(f"Failed to load registration results: {e}", exc_info=True)
            QMessageBox.critical(self, "Loading Error", 
                               f"Error loading results:\n{str(e)}")

    def load_mask(self, mask_name: str):
        """Load a specific mask by name."""
        if mask_name not in self.registered_mask_paths:
            logging.warning(f"Mask '{mask_name}' not found")
            return
        
        mask_path = self.registered_mask_paths[mask_name]
        if not os.path.exists(mask_path):
            logging.warning(f"Mask file not found: {mask_path}")
            return
        
        try:
            mask_image = sitk.ReadImage(mask_path)
            self.canvas.load_mask_volume(mask_image)
            self.current_mask_name = mask_name
            logging.info(f"Loaded mask: {mask_name}")
        except Exception as e:
            logging.error(f"Failed to load mask {mask_name}: {e}")

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        
        # Control panel (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMaximumWidth(370)
        control_panel = self.create_control_panel()
        scroll_area.setWidget(control_panel)
        main_layout.addWidget(scroll_area, 0)
        
        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.slice_changed.connect(self.on_slice_changed_from_canvas)
        main_layout.addWidget(self.canvas, 1)

    def create_control_panel(self):
        """Create the control panel with all UI elements."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # === Visibility & Opacity ===
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
        
        self.mr_opacity_slider = self._create_opacity_control("MR:", 100)
        self.hist_opacity_slider = self._create_opacity_control("Hist:", 70)
        self.mask_opacity_slider = self._create_opacity_control("Mask:", 50)
        
        visibility_layout.addWidget(self.mr_visible_cb)
        visibility_layout.addLayout(self.mr_opacity_slider)
        visibility_layout.addWidget(self.histology_visible_cb)
        visibility_layout.addLayout(self.hist_opacity_slider)
        visibility_layout.addWidget(self.mask_visible_cb)
        visibility_layout.addLayout(self.mask_opacity_slider)
        
        layout.addWidget(visibility_group)
        
        # === Mask Selection ===
        mask_group = QGroupBox("Mask Selection")
        mask_layout = QVBoxLayout(mask_group)
        
        mask_selector_layout = QHBoxLayout()
        mask_selector_layout.addWidget(QLabel("Region:"))
        self.mask_selector = QComboBox()
        self.mask_selector.currentTextChanged.connect(self.on_mask_selection_changed)
        mask_selector_layout.addWidget(self.mask_selector)
        mask_layout.addLayout(mask_selector_layout)
        
        layout.addWidget(mask_group)
        
        # === Slice Navigation ===
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
        
        # === High-Resolution View ===
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
        
        # === Window/Level ===
        wl_group = QGroupBox("MR Window/Level")
        wl_layout = QVBoxLayout(wl_group)
        
        self.window_spin = self._create_wl_spinbox("Window:", 1, 10000, 400)
        self.level_spin = self._create_wl_spinbox("Level:", -1000, 3000, 200)
        
        wl_layout.addLayout(self.window_spin)
        wl_layout.addLayout(self.level_spin)
        
        layout.addWidget(wl_group)
        
        layout.addStretch()
        return panel

    def _create_opacity_control(self, label_text: str, value: int) -> QHBoxLayout:
        """Create an opacity slider with label."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.valueChanged.connect(self.update_visibility_and_opacity)
        
        value_label = QLabel(f"{value}%")
        value_label.setMinimumWidth(40)
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        # Store references for later access
        slider.value_label = value_label
        
        return layout

    def _create_wl_spinbox(self, label_text: str, min_val: int, 
                          max_val: int, value: int) -> QHBoxLayout:
        """Create a window/level spinbox with label."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.valueChanged.connect(self.update_window_level)
        
        layout.addWidget(spinbox)
        return layout

    def on_mask_selection_changed(self, mask_name: str):
        """Handle mask selection change."""
        if mask_name and mask_name != self.current_mask_name:
            self.load_mask(mask_name)
            self.canvas.update_display()

    def load_current_highres_slice(self):
        """Load high-resolution data for the current slice."""
        if not self.canvas or not self.pathology_volume:
            QMessageBox.warning(self, "Warning", 
                              "Pathology data not available.")
            return

        current_slice = self.canvas.current_slice
        
        if not hasattr(self.pathology_volume, 'pathologySlices'):
            QMessageBox.warning(self, "Warning", 
                              "PathologyVolume structure not recognized.")
            return
        
        if current_slice >= len(self.pathology_volume.pathologySlices):
            QMessageBox.warning(self, "Warning", 
                              f"Slice {current_slice+1} out of bounds.")
            return

        try:
            ps = self.pathology_volume.pathologySlices[current_slice]
            
            # Load original high-res RGB image
            highres_rgb = ps.loadRgbImage()
            if not highres_rgb:
                QMessageBox.warning(self, "Warning", 
                                  "Failed to load high-res image.")
                return
            
            # Load original high-res mask
            highres_mask = None
            try:
                highres_mask = ps.loadMask(0)
            except:
                logging.warning(f"No mask available for slice {current_slice}")
            
            # Get the transform
            transform = ps.transform if hasattr(ps, 'transform') and ps.transform else sitk.Transform()
            
            # Load into canvas
            self.canvas.load_highres_slice(highres_rgb, highres_mask, transform)
            
            self.clear_highres_btn.setEnabled(True)
            self.highres_status.setText(f"Mode: High-Res (Slice {current_slice + 1})")
            self.highres_status.setStyleSheet("QLabel { color: green; }")
            
        except Exception as e:
            logging.error(f"Failed to load high-res slice: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", 
                               f"Failed to load high-res data:\n{str(e)}")

    def clear_highres(self):
        """Return to low-resolution view."""
        if self.canvas:
            self.canvas.clear_highres()
            self.clear_highres_btn.setEnabled(False)
            self.highres_status.setText("Mode: Low-Resolution")
            self.highres_status.setStyleSheet("QLabel { color: gray; }")

    def update_visibility_and_opacity(self):
        """Update visibility and opacity settings."""
        if not self.canvas:
            return
        
        self.canvas.show_mr = self.mr_visible_cb.isChecked()
        self.canvas.show_histology = self.histology_visible_cb.isChecked()
        self.canvas.show_mask = self.mask_visible_cb.isChecked()
        
        # Get slider values
        mr_slider = self.mr_opacity_slider.itemAt(1).widget()
        hist_slider = self.hist_opacity_slider.itemAt(1).widget()
        mask_slider = self.mask_opacity_slider.itemAt(1).widget()
        
        self.canvas.mr_opacity = mr_slider.value() / 100.0
        self.canvas.histology_opacity = hist_slider.value() / 100.0
        self.canvas.mask_opacity = mask_slider.value() / 100.0
        
        # Update labels
        mr_slider.value_label.setText(f"{mr_slider.value()}%")
        hist_slider.value_label.setText(f"{hist_slider.value()}%")
        mask_slider.value_label.setText(f"{mask_slider.value()}%")
        
        self.canvas.update_display()

    def update_window_level(self):
        """Update window/level settings."""
        if self.canvas:
            window_spin = self.window_spin.itemAt(1).widget()
            level_spin = self.level_spin.itemAt(1).widget()
            
            self.canvas.window = window_spin.value()
            self.canvas.level = level_spin.value()
            self.canvas.update_display()

    def update_slice_controls(self):
        """Update slice navigation controls."""
        if self.canvas and self.canvas.max_slices > 0:
            self.slice_spin.setMaximum(self.canvas.max_slices)
            self.slice_slider.setMaximum(self.canvas.max_slices)
            self.slice_spin.setValue(self.canvas.current_slice + 1)
            self.slice_slider.setValue(self.canvas.current_slice + 1)
        else:
            self.slice_spin.setMaximum(1)
            self.slice_slider.setMaximum(1)

    def on_slice_spin_changed(self, value):
        """Handle slice spinbox changes."""
        if self.canvas:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(value)
            self.slice_slider.blockSignals(False)
            
            if self.canvas.current_slice != value - 1:
                self.canvas.set_slice(value - 1)
                # Clear high-res when changing slices
                if self.canvas.highres_histology is not None:
                    self.clear_highres()

    def on_slice_slider_changed(self, value):
        """Handle slice slider changes."""
        if self.canvas:
            self.slice_spin.blockSignals(True)
            self.slice_spin.setValue(value)
            self.slice_spin.blockSignals(False)
            
            if self.canvas.current_slice != value - 1:
                self.canvas.set_slice(value - 1)
                # Clear high-res when changing slices
                if self.canvas.highres_histology is not None:
                    self.clear_highres()

    def on_slice_changed_from_canvas(self, slice_idx):
        """Handle slice changes from canvas (mouse wheel)."""
        value = slice_idx + 1
        
        self.slice_spin.blockSignals(True)
        self.slice_slider.blockSignals(True)
        
        self.slice_spin.setValue(value)
        self.slice_slider.setValue(value)
        
        self.slice_spin.blockSignals(False)
        self.slice_slider.blockSignals(False)
        
        # Clear high-res when changing slices
        if self.canvas and self.canvas.highres_histology is not None:
            self.clear_highres()


# ============================================================================
# Testing and standalone execution
# ============================================================================

if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import sys
    
    app = QApplication(sys.argv)
    
    main_window = QMainWindow()
    main_window.setWindowTitle("Registration Results Viewer - Fixed Version")
    main_window.setGeometry(100, 100, 1400, 900)
    
    viewer_tab = RegistrationResultsViewer()
    main_window.setCentralWidget(viewer_tab)
    
    main_window.show()
    
    # Example: Load test data (if available)
    # Uncomment and modify paths as needed:
    """
    test_results = {
        'fixed_volume': '/path/to/mr_volume.nii.gz',
        'registered_rgb': '/path/to/registered_histology.mha',
        'registered_masks': {
            'region0': '/path/to/registered_mask_region0.mha',
            'region1': '/path/to/registered_mask_region1.mha'
        },
        'pathology_volume': None,  # PathologyVolume object if available
        'output_dir': '/path/to/output'
    }
    viewer_tab.load_data_from_registration(test_results)
    """
    
    sys.exit(app.exec())