"""
Registration Tab - RadPathFusion Integration
Completo con logica backend e GUI PyQt6
"""
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QTextEdit, QMessageBox,
    QComboBox, QCheckBox, QProgressBar, QSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("⚠️ Warning: SimpleITK not available. Registration will not work.")
try: 
    from Resources.Utils.ImageStack import PathologyVolume
    from Resources.Utils.parse_json import ParsePathJsonLogic
    from Resources.Utils.image_registration import RegisterImages
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    print("⚠️ Warning: RadPathFusion backend not available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegistrationWorker(QThread):
    """
    Worker thread per eseguire la registrazione RadPathFusion in background.
    """
    progress_update = pyqtSignal(str)
    step_complete = pyqtSignal(str, int)  # step_name, progress_percentage
    registration_complete = pyqtSignal(dict)
    registration_failed = pyqtSignal(str)
    
    def __init__(self, config: dict, pathology_volume=None):
        super().__init__()
        self.config = config
        #self.pathology_volume = pathology_volume
        self.pathology_volume = None
        self.should_abort = False
        
    def run(self):
        try:
            # === SETUP ===
            self.progress_update.emit("=" * 70)
            self.progress_update.emit("🚀 RADPATHFUSION REGISTRATION STARTED")
            self.progress_update.emit("=" * 70)
            
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # === STEP 1: LOAD RADIOLOGY DATA ===
            self.progress_update.emit("\n📥 STEP 1/5: Loading Radiology Data...")
            self.step_complete.emit("Loading radiology", 10)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Load fixed volume (MRI/CT)
            fixed_volume = sitk.ReadImage(self.config['fixed_volume'], sitk.sitkFloat32)
            self.progress_update.emit(f"   ✓ Fixed volume loaded: {fixed_volume.GetSize()}")
            
            # Load fixed mask (organ segmentation)
            fixed_mask = None
            if self.config.get('fixed_mask'):
                fixed_mask = sitk.ReadImage(self.config['fixed_mask'])
                self.progress_update.emit(f"   ✓ Fixed mask loaded: {fixed_mask.GetSize()}")
            
            self.step_complete.emit("Radiology loaded", 20)
            
            # === STEP 2: PREPARE PATHOLOGY VOLUME ===
            self.progress_update.emit("\n🔬 STEP 2/5: Preparing Pathology Volume...")
            self.step_complete.emit("Preparing pathology", 30)
            
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")

            
            # Use pathology volume from previous tab or load from JSON
            if self.pathology_volume:
                logic = self.pathology_volume
                self.progress_update.emit("   ✓ Using pathology volume from previous tab")
            else:
                logic = PathologyVolume()
                logic.verbose = True
                logic.setPath(self.config['json_path'])
                success = logic.initComponents()
                if not success:
                    raise RuntimeError("Failed to load pathology JSON")
                self.progress_update.emit(f"   ✓ Pathology volume loaded: {logic.noSlices} slices")
            
            # Set imaging constraints
            logic.imaging_constraint = fixed_volume
            logic.imaging_constraint_mask = fixed_mask
            logic.imaging_constraint_filename = self.config['fixed_volume']
            logic.imaging_constraint_mask_filename = self.config.get('fixed_mask')
            
            # Get constraint region
            # The method 'get_constraint' is missing from the PathologyVolume class.
            # We re-implement its logic here directly using SimpleITK.
            self.progress_update.emit("   Calculating constraint region from mask...")
            if not fixed_mask:
                raise ValueError("A fixed mask is required to determine the constraint region.")

            # Use LabelShapeStatistics to find the bounding box of the mask's largest component
            stats_filter = sitk.LabelShapeStatisticsImageFilter()
            # Ensure the mask is treated as a labeled image
            connected_component_image = sitk.ConnectedComponent(fixed_mask != 0)
            stats_filter.Execute(connected_component_image)
            
            labels = stats_filter.GetLabels()
            if not labels:
                raise ValueError("The provided mask is empty. Cannot determine a bounding box.")
            
            # We will use the first (and likely only) label in the mask
            label_to_use = labels[0]
            bounding_box = stats_filter.GetBoundingBox(label_to_use)
            # The bounding_box is a tuple: (start_x, start_y, start_z, size_x, size_y, size_z)

            # The constraint_range is the range of z-slices defined by the bounding box
            start_z, size_z = bounding_box[2], bounding_box[5]
            constraint_range = range(start_z, start_z + size_z)

            # Crop the fixed volume using the bounding box to create the constraint image `im_c`
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetSize([bounding_box[3], bounding_box[4], bounding_box[5]])
            roi_filter.SetIndex([bounding_box[0], bounding_box[1], bounding_box[2]])
            im_c = roi_filter.Execute(fixed_volume)
            # ========================== FIX END ==========================
            self.progress_update.emit(f"   ✓ Constraint region: {im_c.GetSize()}")
            self.progress_update.emit(f"   ✓ Slices to register: {len(constraint_range)}")
            
            # Update volume properties
            logic.volume_size = im_c.GetSize()
            logic.volume_spacing = im_c.GetSpacing()
            logic.volume_origin = im_c.GetOrigin()
            logic.volume_direction = im_c.GetDirection()
            
            self.step_complete.emit("Pathology prepared", 40)
            
            # === STEP 3: REGISTRATION PIPELINE ===
            self.progress_update.emit("\n🎯 STEP 3/5: Running Registration Pipeline...")
            self.progress_update.emit(f"   Registration type: {self.config['registration_type']}")
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Set registration parameters
            logic.do_affine = 'affine' in self.config['registration_type']
            logic.do_deformable = 'bspline' in self.config['registration_type'] or 'demons' in self.config['registration_type']
            logic.do_reconstruct = False  # Already reconstructed
            logic.fast_execution = not self.config.get('high_quality', True)
            logic.discard_orientation = self.config.get('discard_orientation', False)
            
            # Load reference volumes
            self.progress_update.emit("   📊 Loading reference volumes...")
            logic.store_volume = True
            ref = logic.loadRgbVolume()
            ref_mask = logic.loadMask(0)
            
            logic.ref_w_constraints = ref
            logic.msk_ref_w_constraints = ref_mask
            
            self.progress_update.emit(f"   ✓ Reference RGB volume: {ref.GetSize()}")
            self.progress_update.emit(f"   ✓ Reference mask volume: {ref_mask.GetSize()}")
            
            # Save intermediate if requested
            if self.config.get('save_intermediate', True):
                ref_path = os.path.join(output_dir, 'reference_volume.nii.gz')
                sitk.WriteImage(ref, ref_path)
                self.progress_update.emit(f"   💾 Reference saved: {ref_path}")
            
            self.step_complete.emit("Registration started", 50)
            
            # Register each slice to constraint
            total_slices = min(logic.noSlices, len(constraint_range))
            for idx, (imov, ifix) in enumerate(zip(range(total_slices), constraint_range)):
                if self.should_abort:
                    raise RuntimeError("Registration aborted by user.")
                
                progress = 50 + int((idx / total_slices) * 30)
                self.step_complete.emit(f"Registering slice {idx+1}/{total_slices}", progress)
                self.progress_update.emit(f"\n   🔄 Registering slice {imov+1}/{total_slices} → MRI slice {ifix+1}")
                
                mov_ps = logic.pathologySlices[imov] # <-- Changed _s to S
                mov_ps.do_affine = logic.do_affine
                mov_ps.do_deformable = logic.do_deformable
                mov_ps.fast_execution = logic.fast_execution
                
                # Register to constraint
                mov_ps.registerToConstraint(
                    im_c[:, :, idx],
                    ref,
                    ref_mask,
                    ref,
                    ref_mask,
                    ifix
                )
                
                self.progress_update.emit(f"      ✓ Slice {imov+1} registered")
            
            self.progress_update.emit("\n   ✅ All slices registered successfully")
            self.step_complete.emit("Registration complete", 80)
            
            # === STEP 4: GENERATE OUTPUT VOLUMES ===
            self.progress_update.emit("\n📤 STEP 4/5: Generating Output Volumes...")
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Generate final registered RGB volume
            self.progress_update.emit("   🎨 Creating registered RGB volume...")
            registered_rgb = logic.loadRgbVolume()
            
            # Save registered RGB volume
            registered_rgb_path = os.path.join(output_dir, 'registered_histology_rgb.nii.gz')
            sitk.WriteImage(registered_rgb, registered_rgb_path)
            self.progress_update.emit(f"   ✓ Registered RGB saved: {registered_rgb_path}")
            
            # Generate registered masks for each region
            mask_paths = {}
            for idx_region in range(logic.noRegions):
                if self.should_abort:
                    raise RuntimeError("Registration aborted by user.")
                
                region_id = logic.regionIDs[idx_region]
                self.progress_update.emit(f"   🎭 Creating mask for region: {region_id}")
                
                registered_mask = logic.loadMask(idx_region)
                mask_path = os.path.join(output_dir, f'registered_mask_{region_id}.nii.gz')
                sitk.WriteImage(registered_mask, mask_path)
                mask_paths[region_id] = mask_path
                self.progress_update.emit(f"      ✓ Mask saved: {mask_path}")
            
            self.step_complete.emit("Outputs generated", 90)
            
            # === STEP 5: CALCULATE METRICS ===
            self.progress_update.emit("\n📊 STEP 5/5: Calculating Quality Metrics...")
            
            metrics = {
                'total_slices': total_slices,
                'registered_slices': total_slices,
                'registration_type': self.config['registration_type'],
                'do_affine': logic.do_affine,
                'do_deformable': logic.do_deformable,
                'volume_spacing': list(registered_rgb.GetSpacing()),
                'volume_size': list(registered_rgb.GetSize()),
                'regions': list(logic.regionIDs)
            }
            
            # Save metrics
            metrics_path = os.path.join(output_dir, 'registration_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            self.progress_update.emit(f"   ✓ Metrics saved: {metrics_path}")
            
            # Create checkerboard if requested
            checkerboard_path = None
            if self.config.get('create_checkerboard', False):
                self.progress_update.emit("\n   🏁 Creating checkerboard visualization...")
                try:
                    # Convert RGB to grayscale for checkerboard
                    select = sitk.VectorIndexSelectionCastImageFilter()
                    select.SetIndex(0)
                    gray = select.Execute(registered_rgb) / 3.0
                    select.SetIndex(1)
                    gray += select.Execute(registered_rgb) / 3.0
                    select.SetIndex(2)
                    gray += select.Execute(registered_rgb) / 3.0
                    
                    # Create checkerboard with fixed volume
                    checker = sitk.CheckerBoard(
                        fixed_volume,
                        sitk.Cast(gray, sitk.sitkFloat32),
                        [8, 8, 1]
                    )
                    checkerboard_path = os.path.join(output_dir, 'checkerboard.nii.gz')
                    sitk.WriteImage(checker, checkerboard_path)
                    self.progress_update.emit(f"   ✓ Checkerboard saved: {checkerboard_path}")
                except Exception as e:
                    self.progress_update.emit(f"   ⚠️ Checkerboard creation failed: {e}")
            
            self.step_complete.emit("Metrics calculated", 100)
            
            # === FINALIZE ===
            final_results = {
                'fixed_volume': self.config['fixed_volume'],
                'registered_rgb': registered_rgb_path,
                'registered_masks': mask_paths,
                'metrics_file': metrics_path,
                'output_dir': output_dir,
                'metrics': metrics,
                'pathology_volume': logic,  # Pass the logic object for viewer
                'checkerboard': checkerboard_path
            }
            
            self.progress_update.emit("\n" + "=" * 70)
            self.progress_update.emit("✅ REGISTRATION COMPLETED SUCCESSFULLY!")
            self.progress_update.emit("=" * 70)
            self.progress_update.emit(f"\n📁 Results saved to: {output_dir}")
            self.progress_update.emit(f"   • Registered RGB: {os.path.basename(registered_rgb_path)}")
            self.progress_update.emit(f"   • Masks: {len(mask_paths)} region(s)")
            self.progress_update.emit(f"   • Metrics: {os.path.basename(metrics_path)}")
            if checkerboard_path:
                self.progress_update.emit(f"   • Checkerboard: {os.path.basename(checkerboard_path)}")
            self.progress_update.emit("=" * 70)
            
            self.registration_complete.emit(final_results)
            
        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            self.registration_failed.emit(str(e))
    
    def abort(self):
        """Interrompe il worker"""
        self.should_abort = True


class RegistrationTab(QWidget):
    """
    Tab per la registrazione RadPathFusion con GUI completa
    """
    
    # Signals
    registration_succeeded = pyqtSignal(dict)  # Emette risultati per il viewer tab
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self, pathology_volume=None):
        super().__init__()
        
        #self.pathology_volume = pathology_volume  
        self.pathology_volume = None # Reference from previous tab
        self.worker = None
        
        self.loaded_files = {
            'fixed_volume': None,
            'fixed_mask': None,
            'json_path': None
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # === Input Group ===
        input_group = self.create_input_group()
        content_layout.addWidget(input_group)
        
        # === Configuration Group ===
        config_group = self.create_config_group()
        content_layout.addWidget(config_group)
        
        # === Progress Group ===
        progress_group = self.create_progress_group()
        content_layout.addWidget(progress_group)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # === Control Buttons ===
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)
        
    @pyqtSlot(str, str, str)
    def update_from_viewer(self, data_type: str, name: str, path: str):
        """
        Slot to receive file paths from the Medical Viewer tab.
        Updates the appropriate input field in the Registration tab.
        """
        logger.info(f"RegistrationTab received '{name}' for type '{data_type}' from viewer.")
        if data_type == 'mr':
            self.fixed_volume_input.setText(path)
            self.loaded_files['fixed_volume'] = path
            self.log_message(f"✓ Fixed volume auto-loaded from viewer: {name}")
        elif data_type == 'segmentation':
            self.fixed_mask_input.setText(path)
            self.loaded_files['fixed_mask'] = path
            self.log_message(f"✓ Fixed mask auto-loaded from viewer: {name}")    
        
    
    def create_input_group(self):
        """Create input files group"""
        group = QGroupBox("📁 Input Files")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                border: 2px solid #142d4c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Fixed Volume (MRI/CT)
        fixed_layout = QHBoxLayout()
        fixed_label = QLabel("Fixed Volume (MRI/CT):")
        fixed_label.setMinimumWidth(180)
        fixed_layout.addWidget(fixed_label)
        
        self.fixed_volume_input = QLineEdit()
        self.fixed_volume_input.setPlaceholderText("Select fixed radiology volume...")
        self.fixed_volume_input.setReadOnly(True)
        fixed_layout.addWidget(self.fixed_volume_input, 1)
        
        self.fixed_browse_btn = QPushButton("Browse...")
        self.fixed_browse_btn.clicked.connect(lambda: self.browse_file('fixed_volume'))
        self.fixed_browse_btn.setMinimumWidth(100)
        fixed_layout.addWidget(self.fixed_browse_btn)
        
        layout.addLayout(fixed_layout)
        
        # Fixed Mask (Organ Segmentation)
        mask_layout = QHBoxLayout()
        mask_label = QLabel("Fixed Mask (Organ Seg):")
        mask_label.setMinimumWidth(180)
        mask_layout.addWidget(mask_label)
        
        self.fixed_mask_input = QLineEdit()
        self.fixed_mask_input.setPlaceholderText("Select organ segmentation mask...")
        self.fixed_mask_input.setReadOnly(True)
        mask_layout.addWidget(self.fixed_mask_input, 1)
        
        self.mask_browse_btn = QPushButton("Browse...")
        self.mask_browse_btn.clicked.connect(lambda: self.browse_file('fixed_mask'))
        self.mask_browse_btn.setMinimumWidth(100)
        mask_layout.addWidget(self.mask_browse_btn)
        
        layout.addLayout(mask_layout)
        
        # Pathology JSON - <<< MODIFICATO PER GESTIONE AUTOMATICA E MANUALE >>>
        json_layout = QHBoxLayout()
        json_label = QLabel("Pathology JSON:")
        json_label.setMinimumWidth(180)
        json_layout.addWidget(json_label)
        
        self.json_input = QLineEdit()
        self.json_input.setPlaceholderText("Load from parser tab or browse...")
        self.json_input.setReadOnly(True)
        json_layout.addWidget(self.json_input, 1)
        
        self.json_browse_btn = QPushButton("Browse...")
        self.json_browse_btn.clicked.connect(lambda: self.browse_file('json_path'))
        self.json_browse_btn.setMinimumWidth(100)
        self.json_browse_btn.setEnabled(True) # Abilitato di default
        json_layout.addWidget(self.json_browse_btn)
        
        layout.addLayout(json_layout)    
        # Info label
        info_label = QLabel(
            "ℹ️ The pathology JSON is automatically loaded from the 'Pathology Parser' tab. "
        "You can also browse for a file manually."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #7f8c8d; font-size: 9pt; padding: 5px;")
        layout.addWidget(info_label)
        
        group.setLayout(layout)
        return group
    
    @pyqtSlot(str)
    def update_pathology_json_path(self, json_path: str):
        """
        Slot to receive and update the pathology JSON file path from the Parser Tab.
        This method updates the UI and internal state.
        """
        if json_path and os.path.exists(json_path):
            self.log_message(f"Pathology JSON path received from parser: {os.path.basename(json_path)}")
            self.json_input.setText(json_path)
            self.json_input.setStyleSheet("background-color: #d5f4e6;") # Green background
            self.loaded_files['json_path'] = json_path
            self.json_browse_btn.setEnabled(False) # Disable manual browsing
            self.log_message("✓ Pathology JSON path updated. Manual browse disabled.")
        else:
            # Gestisce il caso in cui il percorso venga cancellato o sia invalido
            self.json_input.clear()
            self.json_input.setPlaceholderText("Load from parser tab or browse...")
            self.json_input.setStyleSheet("") # Reset style
            self.loaded_files['json_path'] = None
            self.json_browse_btn.setEnabled(True) # Riabilita la selezione manuale
            self.log_message("Pathology JSON path cleared. Manual browse enabled.")

        
    def create_config_group(self):
        """Create configuration group"""
        group = QGroupBox("⚙️ Configuration")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                border: 2px solid #142d4c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setMinimumWidth(180)
        output_layout.addWidget(output_label)
        
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select output directory for results...")
        output_layout.addWidget(self.output_input, 1)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        self.output_browse_btn.setMinimumWidth(100)
        output_layout.addWidget(self.output_browse_btn)
        
        layout.addLayout(output_layout)
        
        # Registration type
        reg_type_layout = QHBoxLayout()
        reg_type_label = QLabel("Registration Type:")
        reg_type_label.setMinimumWidth(180)
        reg_type_layout.addWidget(reg_type_label)
        
        self.reg_type_combo = QComboBox()
        self.reg_type_combo.addItems([
            "affine",
            "affine+bspline",
            "affine+demons"
        ])
        self.reg_type_combo.setCurrentText("affine+bspline")
        self.reg_type_combo.setToolTip(
            "affine: 12 DOF linear registration (fast)\n"
            "affine+bspline: Linear + deformable B-spline (recommended)\n"
            "affine+demons: Linear + Demons deformable (alternative)"
        )
        reg_type_layout.addWidget(self.reg_type_combo, 1)
        
        layout.addLayout(reg_type_layout)
        
        # Advanced options
        options_label = QLabel("Options:")
        options_label.setMinimumWidth(180)
        options_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(options_label)
        
        self.high_quality_cb = QCheckBox("High quality mode (slower, better results)")
        self.high_quality_cb.setChecked(True)
        self.high_quality_cb.setToolTip("Use higher iterations and finer sampling")
        layout.addWidget(self.high_quality_cb)
        
        self.save_intermediate_cb = QCheckBox("Save intermediate results")
        self.save_intermediate_cb.setChecked(True)
        self.save_intermediate_cb.setToolTip("Save reference volume before registration")
        layout.addWidget(self.save_intermediate_cb)
        
        self.checkerboard_cb = QCheckBox("Generate checkerboard visualization")
        self.checkerboard_cb.setChecked(True)
        self.checkerboard_cb.setToolTip("Create checkerboard for quality assessment")
        layout.addWidget(self.checkerboard_cb)
        
        self.discard_orientation_cb = QCheckBox("Discard image orientation")
        self.discard_orientation_cb.setChecked(False)
        self.discard_orientation_cb.setToolTip("Ignore DICOM orientation metadata")
        layout.addWidget(self.discard_orientation_cb)
        
        group.setLayout(layout)
        return group
    
    def create_progress_group(self):
        """Create progress and log group"""
        group = QGroupBox("📊 Progress & Log")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                border: 2px solid #27ae60;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Progress bar
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to start registration")
        self.progress_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addLayout(progress_layout)
        
        # Log output
        log_label = QLabel("Registration Log:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_output)
        
        group.setLayout(layout)
        return group
    
    def create_button_layout(self):
        """Create control buttons"""
        layout = QHBoxLayout()
        layout.setSpacing(10)
        
        self.run_btn = QPushButton("▶️  Run Registration")
        self.run_btn.clicked.connect(self.run_registration)
        #self.run_btn.setEnabled(SITK_AVAILABLE and BACKEND_AVAILABLE)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        layout.addWidget(self.run_btn)
        
        self.abort_btn = QPushButton("⏹  Abort")
        self.abort_btn.clicked.connect(self.abort_registration)
        self.abort_btn.setEnabled(False)
        self.abort_btn.setMinimumHeight(40)
        self.abort_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        layout.addWidget(self.abort_btn)
        
        self.clear_log_btn = QPushButton("🗑️  Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setMinimumHeight(40)
        self.clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                font-size: 10pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        layout.addWidget(self.clear_log_btn)
        
        layout.addStretch()
        
        return layout
    
    def show_error_banner(self, message):
        """Show error banner at top"""
        error_label = QLabel(message)
        error_label.setStyleSheet("""
            QLabel {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                font-size: 10pt;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.layout().insertWidget(0, error_label)

    def receive_pathology_volume(self, updated_volume_object):
        """
        This slot receives the updated PathologyVolume object from the parser tab.
        Its main purpose is to make the in-memory object available to the registration worker.
        """
        self.log_message("PathologyVolume object received from parser tab.")
        self.pathology_volume = updated_volume_object
        # The JSON path in the UI is updated separately by the `update_pathology_json_path` slot
        # to ensure consistency, especially after a "Save As..." operation.
    
    # Sostituisci QUESTA funzione nella classe RegistrationTab

    def browse_file(self, file_type: str):
        """Browse for input file with context-aware filters."""
        
        # Definiamo i componenti del filtro
        medical_filter = "Medical Images (*.nii *.nii.gz *.mha *.mhd *.nrrd)"
        json_filter = "JSON Files (*.json)"
        all_filter = "All Files (*.*)"

        # Determiniamo l'ordine dei filtri (e quindi quello di default) 
        # in base al tipo di file che stiamo cercando.
        if file_type == 'json_path':
            # Se cerchiamo un JSON, mettiamo il filtro JSON per primo
            filters = f"{json_filter};;{medical_filter};;{all_filter}"
        else: 
            # Altrimenti, il default è per le immagini medicali
            filters = f"{medical_filter};;{json_filter};;{all_filter}"

        # Usiamo la stringa di filtri che abbiamo appena creato
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_type.replace('_', ' ').title()}",
            "",
            filters  # <--- MODIFICA CHIAVE
        )
        
        if file_path:
            self.loaded_files[file_type] = file_path
            
            if file_type == 'fixed_volume':
                self.fixed_volume_input.setText(file_path)
                self.fixed_volume_input.setStyleSheet("background-color: #d5f4e6;")
                self.log_message(f"✓ Fixed volume loaded: {os.path.basename(file_path)}")
            elif file_type == 'fixed_mask':
                self.fixed_mask_input.setText(file_path)
                self.fixed_mask_input.setStyleSheet("background-color: #d5f4e6;")
                self.log_message(f"✓ Fixed mask loaded: {os.path.basename(file_path)}")
            elif file_type == 'json_path':
                # Questa logica viene eseguita solo se l'utente clicca il pulsante "Browse" per il JSON
                self.json_input.setText(file_path)
                self.json_input.setStyleSheet("background-color: #d5f4e6;")
                self.log_message(f"✓ Pathology JSON manually selected: {os.path.basename(file_path)}")
                # Se l'utente seleziona un JSON manualmente, significa che sta sovrascrivendo
                # quello ricevuto dal parser tab. Disconnettiamo l'oggetto in memoria per sicurezza.
                self.pathology_volume = None
                self.log_message("⚠️ Manually selected JSON. In-memory PathologyVolume object from parser is now disconnected.")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if dir_path:
            self.output_input.setText(dir_path)
            self.log_message(f"✓ Output directory set: {dir_path}")
    
    def log_message(self, message: str):
        """Add message to log with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_output.append(formatted_msg)
        logger.info(message)
    
    def clear_log(self):
        """Clear the log output"""
        self.log_output.clear()
        self.log_message("Log cleared")
    
    def validate_inputs(self) -> Tuple[bool, str]:
        """Validate inputs before running registration"""
        
        # Check fixed volume
        if not self.loaded_files.get('fixed_volume'):
            return False, "Please select a fixed volume (MRI/CT)"
        
        if not os.path.exists(self.loaded_files['fixed_volume']):
            return False, f"Fixed volume not found: {self.loaded_files['fixed_volume']}"
        
        # Check fixed mask
        if not self.loaded_files.get('fixed_mask'):
            return False, "Please select a fixed mask (organ segmentation)"
        
        if not os.path.exists(self.loaded_files['fixed_mask']):
            return False, f"Fixed mask not found: {self.loaded_files['fixed_mask']}"
        
        # Check pathology data
        if not self.pathology_volume and not self.loaded_files.get('json_path'):
            return False, "Pathology data not available. Load from previous tab or browse JSON file."
        
        if self.loaded_files.get('json_path') and not os.path.exists(self.loaded_files['json_path']):
            return False, f"Pathology JSON not found: {self.loaded_files['json_path']}"
        
        # Check output directory
        if not self.output_input.text():
            return False, "Please select an output directory"
        
        return True, ""
    
    def run_registration(self):
        """Start registration process"""
        
        # Validate inputs
        valid, error_msg = self.validate_inputs()
        if not valid:
            QMessageBox.warning(self, "Invalid Input", error_msg)
            self.log_message(f"❌ Validation failed: {error_msg}")
            return
        
        # Prepare configuration
        config = {
            'fixed_volume': self.loaded_files['fixed_volume'],
            'fixed_mask': self.loaded_files['fixed_mask'],
            'json_path': self.loaded_files.get('json_path') or (
                str(self.pathology_volume.path) if self.pathology_volume else None
            ),
            'output_dir': self.output_input.text(),
            'registration_type': self.reg_type_combo.currentText(),
            'high_quality': self.high_quality_cb.isChecked(),
            'save_intermediate': self.save_intermediate_cb.isChecked(),
            'create_checkerboard': self.checkerboard_cb.isChecked(),
            'discard_orientation': self.discard_orientation_cb.isChecked()
        }
        
        # Log configuration
        self.log_message("=" * 70)
        self.log_message("🚀 STARTING RADPATHFUSION REGISTRATION")
        self.log_message("=" * 70)
        self.log_message(f"Fixed volume: {os.path.basename(config['fixed_volume'])}")
        self.log_message(f"Fixed mask: {os.path.basename(config['fixed_mask'])}")
        self.log_message(f"Pathology JSON: {os.path.basename(config['json_path'])}")
        self.log_message(f"Output directory: {config['output_dir']}")
        self.log_message(f"Registration type: {config['registration_type']}")
        self.log_message(f"High quality: {config['high_quality']}")
        self.log_message("=" * 70)
        
        # Update UI state
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Initializing registration...")
        
        # Disable input controls
        self.fixed_browse_btn.setEnabled(False)
        self.mask_browse_btn.setEnabled(False)
        self.json_browse_btn.setEnabled(False)
        self.output_browse_btn.setEnabled(False)
        
        # Create and start worker
        self.worker = RegistrationWorker(config, self.pathology_volume)
        self.worker.progress_update.connect(self.log_message)
        self.worker.step_complete.connect(self.on_step_complete)
        self.worker.registration_complete.connect(self.on_registration_complete)
        self.worker.registration_failed.connect(self.on_registration_failed)
        self.worker.start()
    
    def abort_registration(self):
        """Abort registration process"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Abort Registration",
                "Are you sure you want to abort the registration?\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.log_message("⚠️ Aborting registration...")
                self.worker.abort()
                self.worker.wait()
                self.reset_ui()
                self.log_message("❌ Registration aborted by user")
    
    @pyqtSlot(str, int)
    def on_step_complete(self, step_name: str, progress: int):
        """Update progress bar when step completes"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(step_name)
    
    @pyqtSlot(dict)
    def on_registration_complete(self, results: dict):
        """Handle successful registration completion"""
        self.log_message("\n" + "=" * 70)
        self.log_message("✅ REGISTRATION COMPLETED SUCCESSFULLY!")
        self.log_message("=" * 70)
        self.log_message(f"\n📁 Results Summary:")
        self.log_message(f"   • Fixed volume: {os.path.basename(results['fixed_volume'])}")
        self.log_message(f"   • Registered RGB: {os.path.basename(results['registered_rgb'])}")
        self.log_message(f"   • Registered masks: {len(results['registered_masks'])} region(s)")
        
        for region_id, mask_path in results['registered_masks'].items():
            self.log_message(f"      - {region_id}: {os.path.basename(mask_path)}")
        
        self.log_message(f"   • Metrics: {os.path.basename(results['metrics_file'])}")
        
        if results.get('checkerboard'):
            self.log_message(f"   • Checkerboard: {os.path.basename(results['checkerboard'])}")
        
        self.log_message(f"\n📊 Registration Metrics:")
        metrics = results['metrics']
        self.log_message(f"   • Total slices: {metrics['total_slices']}")
        self.log_message(f"   • Registered slices: {metrics['registered_slices']}")
        self.log_message(f"   • Registration type: {metrics['registration_type']}")
        self.log_message(f"   • Affine: {metrics['do_affine']}")
        self.log_message(f"   • Deformable: {metrics['do_deformable']}")
        self.log_message(f"   • Output size: {metrics['volume_size']}")
        self.log_message(f"   • Output spacing: {[f'{s:.4f}' for s in metrics['volume_spacing']]}")
        
        self.log_message("\n" + "=" * 70)
        
        # Update UI
        self.reset_ui()
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ Registration completed successfully!")
        self.progress_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        
        # Emit signal to viewer tab with results
        self.registration_succeeded.emit(results)
        
        # Show success dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Registration Completed")
        msg.setText("✅ Registration completed successfully!")
        msg.setInformativeText(
            f"Type: {metrics['registration_type']}\n"
            f"Slices: {metrics['registered_slices']}/{metrics['total_slices']}\n"
            f"Regions: {len(results['registered_masks'])}\n\n"
            f"Results saved to:\n{results['output_dir']}\n\n"
            f"The results are now available in the Viewer tab."
        )
        msg.setDetailedText(
            f"Output files:\n"
            f"- Registered RGB: {results['registered_rgb']}\n"
            f"- Masks: {', '.join(results['registered_masks'].keys())}\n"
            f"- Metrics: {results['metrics_file']}\n"
            + (f"- Checkerboard: {results['checkerboard']}\n" if results.get('checkerboard') else "")
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    @pyqtSlot(str)
    def on_registration_failed(self, error_msg: str):
        """Handle registration failure"""
        self.log_message("\n" + "=" * 70)
        self.log_message("❌ REGISTRATION FAILED")
        self.log_message("=" * 70)
        self.log_message(f"Error: {error_msg}")
        self.log_message("=" * 70)
        
        # Update UI
        self.reset_ui()
        self.progress_bar.setValue(0)
        self.progress_label.setText("❌ Registration failed")
        self.progress_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Registration Failed",
            f"Registration failed with error:\n\n{error_msg}\n\n"
            "Check the log output for details."
        )
    
    def reset_ui(self):
        """Reset UI to ready state"""
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        
        # Re-enable input controls
        self.fixed_browse_btn.setEnabled(True)
        self.mask_browse_btn.setEnabled(True)
        self.json_browse_btn.setEnabled(self.pathology_volume is None)
        self.output_browse_btn.setEnabled(True)
    
    @pyqtSlot(object)
    def set_pathology_volume(self, pathology_volume):
        """Receive pathology volume from previous tab"""
        self.pathology_volume = pathology_volume
        
        if pathology_volume and hasattr(pathology_volume, 'path'):
            self.json_input.setText(str(pathology_volume.path))
            self.json_input.setStyleSheet("background-color: #d5f4e6;")
            self.loaded_files['json_path'] = str(pathology_volume.path)
            self.json_browse_btn.setEnabled(False)
            self.log_message(f"✓ Pathology volume received from previous tab: {pathology_volume.noSlices} slices")
        else:
            self.json_browse_btn.setEnabled(True)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QWidget()
    window.setWindowTitle("RadPathFusion Registration Tab - Test")
    window.setGeometry(100, 100, 1000, 800)
    
    layout = QVBoxLayout(window)
    
    # Create registration tab
    reg_tab = RegistrationTab()
    
    # Connect signal to see results
    def on_registration_complete(results):
        print("\n" + "="*70)
        print("REGISTRATION RESULTS RECEIVED:")
        print("="*70)
        print(f"Fixed volume: {results['fixed_volume']}")
        print(f"Registered RGB: {results['registered_rgb']}")
        print(f"Masks: {list(results['registered_masks'].keys())}")
        print(f"Metrics: {results['metrics']}")
        print("="*70)
    
    reg_tab.registration_succeeded.connect(on_registration_complete)
    
    layout.addWidget(reg_tab)
    
    window.show()
    sys.exit(app.exec())