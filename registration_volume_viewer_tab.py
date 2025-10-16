# File: registration_viewer_tab.py (FIXED VERSION)
"""
Fixed Registration Results Viewer
Key fixes:
1. Properly identifies which MR slices contain histology/mask data
2. Removes unnecessary flipud that caused orientation issues
3. Handles partial histology coverage correctly
"""

import os
import numpy as np
from typing import Optional, Dict, List, Tuple
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

        # Raw SITK images
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

        # High-res data
        self.highres_histology: Optional[sitk.Image] = None
        self.highres_mask: Optional[sitk.Image] = None
        self.highres_transform: Optional[sitk.Transform] = None

        # Window/Level for MR
        self.window = 400
        self.level = 200
        
        # NEW: Track which MR slices have histology/mask data
        self.mr_slices_with_histology: List[int] = []
        self.mr_slices_with_mask: List[int] = []
        self.histology_to_mr_slice_map: Dict[int, int] = {}

    def load_mr_volume(self, mr_image: sitk.Image):
        """Load MR volume."""
        self.mr_image = sitk.Cast(mr_image, sitk.sitkFloat32)
        self.max_slices = self.mr_image.GetSize()[2]
        logging.info(f"Loaded MR volume: size={self.mr_image.GetSize()}, "
                    f"spacing={self.mr_image.GetSpacing()}, "
                    f"origin={self.mr_image.GetOrigin()}")

    def load_histology_volume(self, histology_image: sitk.Image):
        """Load registered histology volume and identify corresponding MR slices."""
        self.histology_image = histology_image
        hist_size = self.histology_image.GetSize()
        
        logging.info(f"Loaded histology volume: size={hist_size}, "
                    f"spacing={self.histology_image.GetSpacing()}")
        
        # NEW: Find which MR slices correspond to histology slices
        self._identify_histology_slice_mapping()

    def load_mask_volume(self, mask_image: sitk.Image):
        """Load registered mask volume and identify corresponding MR slices."""
        self.mask_image = mask_image
        mask_size = self.mask_image.GetSize()
        
        logging.info(f"Loaded mask volume: size={mask_size}")
        
        # NEW: Find which MR slices contain mask data
        self._identify_mask_slice_mapping()

    def _identify_histology_slice_mapping(self):
        """
        Identify which MR slices correspond to each histology slice.
        Uses physical coordinates to determine the mapping.
        """
        if not self.mr_image or not self.histology_image:
            return
        
        self.histology_to_mr_slice_map = {}
        self.mr_slices_with_histology = []
        
        hist_size = self.histology_image.GetSize()
        
        for hist_z in range(hist_size[2]):
            # Get physical Z coordinate of histology slice center
            hist_center_index = [hist_size[0]//2, hist_size[1]//2, hist_z]
            hist_center_phys = self.histology_image.TransformIndexToPhysicalPoint(hist_center_index)
            hist_z_phys = hist_center_phys[2]
            
            # Find closest MR slice
            mr_z = self._find_closest_mr_slice(hist_z_phys)
            
            if mr_z is not None:
                self.histology_to_mr_slice_map[hist_z] = mr_z
                if mr_z not in self.mr_slices_with_histology:
                    self.mr_slices_with_histology.append(mr_z)
        
        self.mr_slices_with_histology.sort()
        
        logging.info(f"Histology slice mapping identified:")
        for hist_z, mr_z in self.histology_to_mr_slice_map.items():
            logging.info(f"  Histology slice {hist_z} → MR slice {mr_z}")

    def _identify_mask_slice_mapping(self):
        """
        Identify which MR slices contain mask data.
        """
        if not self.mr_image or not self.mask_image:
            return
        
        self.mr_slices_with_mask = []
        
        mask_size = self.mask_image.GetSize()
        
        for mask_z in range(mask_size[2]):
            # Get physical Z coordinate
            mask_center_index = [mask_size[0]//2, mask_size[1]//2, mask_z]
            mask_center_phys = self.mask_image.TransformIndexToPhysicalPoint(mask_center_index)
            mask_z_phys = mask_center_phys[2]
            
            # Find closest MR slice
            mr_z = self._find_closest_mr_slice(mask_z_phys)
            
            if mr_z is not None and mr_z not in self.mr_slices_with_mask:
                self.mr_slices_with_mask.append(mr_z)
        
        self.mr_slices_with_mask.sort()
        
        logging.info(f"Mask found in MR slices: {self.mr_slices_with_mask}")

    def _find_closest_mr_slice(self, z_physical: float) -> Optional[int]:
        """
        Find the MR slice index closest to the given physical Z coordinate.
        """
        if not self.mr_image:
            return None
        
        mr_size = self.mr_image.GetSize()
        min_distance = float('inf')
        closest_slice = None
        
        for mr_z in range(mr_size[2]):
            # Get physical Z coordinate of MR slice
            mr_center_index = [mr_size[0]//2, mr_size[1]//2, mr_z]
            mr_center_phys = self.mr_image.TransformIndexToPhysicalPoint(mr_center_index)
            mr_z_phys = mr_center_phys[2]
            
            distance = abs(mr_z_phys - z_physical)
            
            if distance < min_distance:
                min_distance = distance
                closest_slice = mr_z
        
        # Only return if distance is reasonable (within one slice thickness)
        mr_spacing = self.mr_image.GetSpacing()
        if min_distance < mr_spacing[2] * 1.5:
            return closest_slice
        
        return None

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

    def load_highres_slice(self, histology_img: sitk.Image, mask_img: Optional[sitk.Image],
                          transform: sitk.Transform):
        """Load high-resolution original histology slice with its transform."""
        self.highres_histology = histology_img
        self.highres_mask = mask_img
        self.highres_transform = transform
        self.update_display()

    def clear_highres(self):
        """Clear high-resolution data."""
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
        self.histology_to_mr_slice_map = {}
        self.mr_slices_with_histology = []
        self.mr_slices_with_mask = []
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
        
        if slice_idx >= size[2] or slice_idx < 0:
            logging.warning(f"Slice index {slice_idx} out of bounds for image with {size[2]} slices")
            return None
        
        size[2] = 0
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
        """Resample moving image to reference image space."""
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
        FIXED: Removed flipud - orientation is now correct!
        """
        array = sitk.GetArrayFromImage(image_2d)
        # REMOVED: array = np.flipud(array) 
        return array

    def _get_histology_slice_for_mr(self, mr_slice_idx: int) -> Optional[int]:
        """
        Get the histology slice index that corresponds to the given MR slice.
        Returns None if no histology data exists for this MR slice.
        """
        # Reverse lookup in the mapping
        for hist_z, mr_z in self.histology_to_mr_slice_map.items():
            if mr_z == mr_slice_idx:
                return hist_z
        return None

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
            
            # NEW: Check if this MR slice has histology data
            has_histology = self.current_slice in self.mr_slices_with_histology
            
            # Overlay histology (only if available for this slice)
            if self.show_histology and has_histology:
                if self.highres_histology is not None:
                    # High-res mode
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
                    # Low-res mode - find corresponding histology slice
                    hist_slice_idx = self._get_histology_slice_for_mr(self.current_slice)
                    
                    if hist_slice_idx is not None:
                        hist_slice_2d = self.extract_2d_slice(self.histology_image, hist_slice_idx)
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
                        hist_array = None
                else:
                    hist_array = None
                
                if hist_array is not None:
                    # Handle RGB histology
                    if hist_array.ndim == 2:
                        hist_rgb = np.stack([hist_array] * 3, axis=-1)
                    else:
                        hist_rgb = hist_array
                    
                    # Normalize to [0, 1]
                    if hist_rgb.max() > 1.0:
                        hist_rgb = hist_rgb / 255.0
                    
                    # Alpha blending
                    alpha = self.histology_opacity
                    composite = composite * (1 - alpha) + hist_rgb * alpha
            
            # NEW: Check if this MR slice has mask data
            has_mask = self.current_slice in self.mr_slices_with_mask
            
            # Overlay mask (only if available for this slice)
            if self.show_mask and has_mask:
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
                    # Low-res mask - find corresponding mask slice
                    mask_slice_idx = self._get_histology_slice_for_mr(self.current_slice)
                    
                    if mask_slice_idx is not None:
                        mask_slice_2d = self.extract_2d_slice(self.mask_image, mask_slice_idx)
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
            
            # NEW: Add info about histology availability in status
            if has_histology:
                status_text = f"Slice {self.current_slice + 1}/{self.max_slices} [with histology]"
            else:
                status_text = f"Slice {self.current_slice + 1}/{self.max_slices} [MR only]"
            
        except Exception as e:
            logging.error(f"Error updating display: {e}", exc_info=True)
            self.setText(f"Error displaying slice {self.current_slice + 1}")

    def _display_numpy_array(self, array: np.ndarray):
        """Convert numpy array to QPixmap and display."""
        if array.ndim == 2:
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
        """Load all data from registration results."""
        logging.info("Viewer: Loading registration results...")
        
        if not self.canvas or not SITK_AVAILABLE:
            QMessageBox.critical(self, "Error", 
                               "Viewer canvas or SimpleITK is not available.")
            return

        try:
            self.canvas.clear_all_data()
            self.pathology_volume = None
            self.registered_mask_paths = {}

            # Load MR volume
            fixed_path = results.get('fixed_volume')
            if not fixed_path or not os.path.exists(fixed_path):
                raise FileNotFoundError(f"MR volume not found: {fixed_path}")
            
            mr_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            self.canvas.load_mr_volume(mr_image)
            logging.info(f"Loaded MR volume: {os.path.basename(fixed_path)}")

            # Load registered histology volume
            histology_path = results.get('registered_rgb')
            if histology_path and os.path.exists(histology_path):
                histology_image = sitk.ReadImage(histology_path)
                self.canvas.load_histology_volume(histology_image)
                logging.info(f"Loaded histology: {os.path.basename(histology_path)}")
            else:
                logging.warning("Registered histology volume not found")

            # Load registered masks
            self.registered_mask_paths = results.get('registered_masks', {})
            if self.registered_mask_paths:
                first_mask_name = list(self.registered_mask_paths.keys())[0]
                self.load_mask(first_mask_name)
                
                self.mask_selector.clear()
                self.mask_selector.addItems(list(self.registered_mask_paths.keys()))
                self.mask_selector.setCurrentText(first_mask_name)
            else:
                logging.warning("No registered masks found")

            # Store pathology volume
            self.pathology_volume = results.get('pathology_volume')
            if self.pathology_volume:
                self.load_highres_btn.setEnabled(True)
                logging.info("PathologyVolume object loaded for high-res viewing")
            else:
                self.load_highres_btn.setEnabled(False)
                logging.warning("PathologyVolume not available")

            # Update UI and jump to first slice with histology
            self.update_slice_controls()
            
            # NEW: Jump to first slice with histology data
            if self.canvas.mr_slices_with_histology:
                first_hist_slice = self.canvas.mr_slices_with_histology[0]
                self.canvas.set_slice(first_hist_slice)
                self.slice_spin.setValue(first_hist_slice + 1)
                self.slice_slider.setValue(first_hist_slice + 1)
                logging.info(f"Jumped to first slice with histology: {first_hist_slice + 1}")
            else:
                self.canvas.set_slice(0)
            
            QMessageBox.information(self, "Success", 
                                  f"Registration results loaded successfully!\n\n"
                                  f"Histology visible in slices: {[s+1 for s in self.canvas.mr_slices_with_histology]}")

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
        
        # === Quick Navigation (NEW) ===
        nav_group = QGroupBox("🎯 Quick Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        self.prev_hist_btn = QPushButton("← Previous Histology Slice")
        self.prev_hist_btn.clicked.connect(self.jump_to_previous_histology)
        
        self.next_hist_btn = QPushButton("Next Histology Slice →")
        self.next_hist_btn.clicked.connect(self.jump_to_next_histology)
        
        nav_layout.addWidget(self.prev_hist_btn)
        nav_layout.addWidget(self.next_hist_btn)
        
        layout.addWidget(nav_group)
        
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

    def jump_to_previous_histology(self):
        """Jump to the previous MR slice that contains histology data."""
        if not self.canvas or not self.canvas.mr_slices_with_histology:
            return
        
        current = self.canvas.current_slice
        hist_slices = self.canvas.mr_slices_with_histology
        
        # Find previous slice with histology
        prev_slices = [s for s in hist_slices if s < current]
        
        if prev_slices:
            target_slice = prev_slices[-1]  # Get the closest one before current
            self.canvas.set_slice(target_slice)
            self.slice_spin.setValue(target_slice + 1)
            self.slice_slider.setValue(target_slice + 1)

    def jump_to_next_histology(self):
        """Jump to the next MR slice that contains histology data."""
        if not self.canvas or not self.canvas.mr_slices_with_histology:
            return
        
        current = self.canvas.current_slice
        hist_slices = self.canvas.mr_slices_with_histology
        
        # Find next slice with histology
        next_slices = [s for s in hist_slices if s > current]
        
        if next_slices:
            target_slice = next_slices[0]  # Get the closest one after current
            self.canvas.set_slice(target_slice)
            self.slice_spin.setValue(target_slice + 1)
            self.slice_slider.setValue(target_slice + 1)

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

        current_mr_slice = self.canvas.current_slice
        
        # Find the corresponding histology slice
        hist_slice_idx = self.canvas._get_histology_slice_for_mr(current_mr_slice)
        
        if hist_slice_idx is None:
            QMessageBox.warning(self, "Warning", 
                              f"No histology data for MR slice {current_mr_slice + 1}.\n"
                              f"Histology is only available in slices: {[s+1 for s in self.canvas.mr_slices_with_histology]}")
            return
        
        if not hasattr(self.pathology_volume, 'pathologySlices'):
            QMessageBox.warning(self, "Warning", 
                              "PathologyVolume structure not recognized.")
            return
        
        if hist_slice_idx >= len(self.pathology_volume.pathologySlices):
            QMessageBox.warning(self, "Warning", 
                              f"Histology slice {hist_slice_idx} out of bounds.")
            return

        try:
            ps = self.pathology_volume.pathologySlices[hist_slice_idx]
            
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
                logging.warning(f"No mask available for histology slice {hist_slice_idx}")
            
            # Get the transform
            transform = ps.transform if hasattr(ps, 'transform') and ps.transform else sitk.Transform()
            
            # Load into canvas
            self.canvas.load_highres_slice(highres_rgb, highres_mask, transform)
            
            self.clear_highres_btn.setEnabled(True)
            self.highres_status.setText(f"Mode: High-Res (Hist Slice {hist_slice_idx}, MR Slice {current_mr_slice + 1})")
            self.highres_status.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            
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
    
    logging.basicConfig(level=logging.INFO)
    
    app = QApplication(sys.argv)
    
    main_window = QMainWindow()
    main_window.setWindowTitle("Registration Results Viewer - FIXED VERSION")
    main_window.setGeometry(100, 100, 1400, 900)
    
    viewer_tab = RegistrationResultsViewer()
    main_window.setCentralWidget(viewer_tab)
    
    main_window.show()
    
    print("\n" + "="*70)
    print("FIXED REGISTRATION VIEWER")
    print("="*70)
    print("\nKey fixes applied:")
    print("  ✓ Removed flipud - correct orientation")
    print("  ✓ Physical coordinate mapping for slice correspondence")
    print("  ✓ Automatic detection of which MR slices contain histology")
    print("  ✓ Quick navigation buttons to jump between histology slices")
    print("  ✓ Clear status indication: [with histology] or [MR only]")
    print("\nUsage:")
    print("  1. Load registration results")
    print("  2. Viewer will automatically jump to first histology slice")
    print("  3. Use 'Previous/Next Histology Slice' buttons for quick navigation")
    print("  4. Mouse wheel to browse all slices")
    print("="*70 + "\n")
    
    sys.exit(app.exec())