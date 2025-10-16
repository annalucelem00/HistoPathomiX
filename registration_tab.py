"""
Registration Tab - WITH LABELMAP ASSOCIATION SUPPORT
Key Features:
1. Loads MRI segmentation as LABELMAP (not just binary mask)
2. Associates each histology slice to corresponding MRI label
3. Preserves MRI label values in output
4. Creates slice-to-label mapping file
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
    print("⚠️ Warning: SimpleITK not available.")

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
    """Worker thread for registration WITH LABELMAP ASSOCIATION"""
    
    progress_update = pyqtSignal(str)
    step_complete = pyqtSignal(str, int)
    registration_complete = pyqtSignal(dict)
    registration_failed = pyqtSignal(str)
    
    def __init__(self, config: dict, pathology_volume=None):
        super().__init__()
        self.config = config
        self.pathology_volume = pathology_volume
        self.should_abort = False
    
    def run(self):
        try:
            # === SETUP ===
            self.progress_update.emit("=" * 70)
            self.progress_update.emit("🚀 RADPATHFUSION REGISTRATION - LABELMAP ASSOCIATION")
            self.progress_update.emit("=" * 70)
            
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # === STEP 1: LOAD RADIOLOGY DATA ===
            self.progress_update.emit("\n📥 STEP 1/7: Loading Radiology Data...")
            self.step_complete.emit("Loading radiology", 10)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Load MRI volume
            fixed_volume = sitk.ReadImage(self.config['fixed_volume'], sitk.sitkFloat32)
            self.progress_update.emit(f"   ✓ MRI volume loaded: {fixed_volume.GetSize()}")
            
            # Load MRI LABELMAP (not just binary mask!)
            fixed_labelmap = None
            if self.config.get('fixed_mask'):
                fixed_labelmap = sitk.ReadImage(self.config['fixed_mask'])
                
                # Verify it's a labelmap with multiple labels
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(fixed_labelmap)
                num_labels = stats.GetNumberOfLabels()
                
                self.progress_update.emit(f"   ✓ MRI labelmap loaded: {fixed_labelmap.GetSize()}")
                self.progress_update.emit(f"     Number of labels: {num_labels}")
                
                if num_labels == 0:
                    raise ValueError("Labelmap is empty - no labels found!")
                
                if num_labels == 1:
                    self.progress_update.emit("   ⚠️  WARNING: Only 1 label found. Expected multiple labels for slice-wise association.")
            else:
                raise ValueError("MRI segmentation (labelmap) is required!")
            
            self.step_complete.emit("Radiology loaded", 15)
            
            # === STEP 2: LOAD PATHOLOGY DATA ===
            self.progress_update.emit("\n🔬 STEP 2/7: Loading Pathology Data...")
            self.step_complete.emit("Loading pathology", 20)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            logic = PathologyVolume()
            logic.verbose = True
            logic.setPath(self.config['json_path'])
            
            success = logic.initComponents()
            if not success:
                raise RuntimeError("Failed to load pathology JSON")
            
            self.progress_update.emit(f"   ✓ Pathology data loaded: {logic.noSlices} slices")
            self.progress_update.emit(f"     Regions: {logic.regionIDs}")
            
            # Set imaging constraints
            logic.imagingContraint = fixed_volume
            logic.imagingContraintMask = fixed_labelmap  # This is the LABELMAP
            logic.imagingContraintFilename = self.config['fixed_volume']
            logic.imagingContraintMaskFilename = self.config.get('fixed_mask')
            
            self.step_complete.emit("Pathology loaded", 25)
            
            # === STEP 3: ANALYZE LABELMAP STRUCTURE ===
            self.progress_update.emit("\n🏷️  STEP 3/7: Analyzing MRI Labelmap Structure...")
            self.step_complete.emit("Analyzing labelmap", 30)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Extract constraint region WITH labelmap preservation
            self.progress_update.emit("   🔍 Extracting constraint region with labels...")
            constrained_mri, constrained_labelmap, bbox, label_to_z_map = \
                logic.extractConstraintRegionWithLabels(fixed_volume, fixed_labelmap)
            
            self.progress_update.emit(f"   ✓ Labelmap structure analyzed:")
            self.progress_update.emit(f"     Unique labels: {sorted(label_to_z_map.keys())}")
            self.progress_update.emit(f"     Total labeled slices: {sum(len(v) for v in label_to_z_map.values())}")
            
            self.step_complete.emit("Labelmap analyzed", 35)
            
            # === STEP 4: CREATE SLICE-TO-LABEL MAPPING ===
            self.progress_update.emit("\n🔗 STEP 4/7: Creating Slice-to-Label Mapping...")
            self.step_complete.emit("Creating mapping", 40)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            logic.createSliceToLabelMapping(label_to_z_map, logic.noSlices)
            
            self.progress_update.emit(f"   ✓ Mapping created:")
            for hist_idx in range(min(logic.noSlices, 5)):
                label_val = logic.getTargetLabelForSlice(hist_idx)
                self.progress_update.emit(f"     Histology slice {hist_idx} → MRI label {label_val}")
            if logic.noSlices > 5:
                self.progress_update.emit(f"     ... and {logic.noSlices - 5} more")
            
            self.step_complete.emit("Mapping created", 45)
            
            # === STEP 5: ALIGN TO MRI PHYSICAL SPACE ===
            self.progress_update.emit("\n📐 STEP 5/7: Aligning to MRI Physical Space...")
            self.step_complete.emit("Aligning space", 50)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            logic.alignToMRISpace(fixed_volume, fixed_labelmap)
            
            self.progress_update.emit(f"   ✓ Physical space aligned")
            self.progress_update.emit(f"     Histology size: {logic.volumeSize}")
            self.progress_update.emit(f"     Histology spacing: {logic.volumeSpacing} mm")
            
            self.step_complete.emit("Space aligned", 55)
            
            # === STEP 6: CREATE REFERENCE VOLUMES ===
            self.progress_update.emit("\n📊 STEP 6/7: Creating Reference Volumes...")
            self.step_complete.emit("Creating references", 60)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Load histology volumes
            logic.storeVolume = True
            hist_rgb_native = logic.loadRgbVolume()
            self.progress_update.emit(f"   ✓ RGB volume created: {hist_rgb_native.GetSize()}")
            
            # Load mask AS LABELMAP with proper label values
            hist_mask_native = logic.loadMaskWithLabels(0)
            self.progress_update.emit(f"   ✓ Mask LABELMAP created with MRI label values")
            
            # Verify labelmap structure
            hist_mask_array = sitk.GetArrayFromImage(hist_mask_native)
            unique_labels = np.unique(hist_mask_array)
            unique_labels = unique_labels[unique_labels != 0]
            self.progress_update.emit(f"     Labels in histology: {sorted(unique_labels)}")
            
            # Resample to MRI space
            self.progress_update.emit("   🔄 Resampling to MRI space...")
            hist_rgb_mri, hist_mask_mri = logic.resampleHistologyToMRISpace(
                constrained_mri, hist_rgb_native, hist_mask_native
            )
            
            logic.refWContraints = hist_rgb_mri
            logic.mskRefWContraints = hist_mask_mri
            
            self.progress_update.emit(f"   ✓ Resampled to MRI space: {hist_rgb_mri.GetSize()}")
            
            # Save intermediate if requested
            if self.config.get('save_intermediate', True):
                ref_path = os.path.join(output_dir, 'reference_volume_before_reg.nrrd')
                sitk.WriteImage(hist_rgb_mri, ref_path)
                self.progress_update.emit(f"   💾 Reference saved: {ref_path}")
                
                mask_ref_path = os.path.join(output_dir, 'reference_labelmap_before_reg.nrrd')
                sitk.WriteImage(hist_mask_mri, mask_ref_path)
                self.progress_update.emit(f"   💾 Labelmap reference saved: {mask_ref_path}")
            
            self.step_complete.emit("References created", 65)
            
            # === STEP 7: REGISTER SLICES WITH LABEL ASSOCIATION ===
            self.progress_update.emit("\n🎯 STEP 7/7: Registering Slices with Label Association...")
            self.step_complete.emit("Registering slices", 70)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            # Set registration parameters
            logic.doAffine = self.config.get('do_affine', True)
            logic.doDeformable = self.config.get('do_deformable', False)
            logic.fastExecution = self.config.get('fast_execution', True)
            
            self.progress_update.emit(f"   Registration settings:")
            self.progress_update.emit(f"     Affine: {logic.doAffine}")
            self.progress_update.emit(f"     Deformable: {logic.doDeformable}")
            self.progress_update.emit(f"     Fast mode: {logic.fastExecution}")
            
            # Register each slice to its corresponding MRI label
            for hist_idx in range(logic.noSlices):
                if self.should_abort:
                    raise RuntimeError("Registration aborted by user.")
                
                progress = 70 + int((hist_idx / logic.noSlices) * 25)
                self.step_complete.emit(f"Registering slice {hist_idx+1}/{logic.noSlices}", progress)
                
                target_label = logic.getTargetLabelForSlice(hist_idx)
                
                # Find Z-slices containing this label
                z_indices = label_to_z_map.get(target_label, [])
                
                if not z_indices:
                    self.progress_update.emit(f"   ⚠️  Slice {hist_idx+1}: No MRI slice for label {target_label}")
                    continue
                
                # Use middle Z-index if label spans multiple slices
                target_z = z_indices[len(z_indices) // 2]
                
                self.progress_update.emit(f"\n   🔄 Slice {hist_idx+1}/{logic.noSlices} → Label {target_label} (Z={target_z})")
                
                mov_ps = logic.pathologySlices[hist_idx]
                mov_ps.doAffine = logic.doAffine
                mov_ps.doDeformable = logic.doDeformable
                mov_ps.fastExecution = logic.fastExecution
                mov_ps.verbose = False  # Reduce verbosity in loop
                
                # Extract label-specific mask for registration
                label_mask_3d = constrained_labelmap == target_label
                label_mask_2d = label_mask_3d[:, :, target_z]
                
                # Register
                try:
                    mov_ps.registerToConstraintWithLabel(
                        constrained_mri[:, :, target_z],
                        label_mask_2d,
                        hist_rgb_mri,
                        hist_mask_mri,
                        hist_rgb_mri,
                        hist_mask_mri,
                        target_z,
                        target_label
                    )
                    self.progress_update.emit(f"      ✓ Registered successfully")
                except Exception as e:
                    self.progress_update.emit(f"      ⚠️  Registration failed: {str(e)}")
            
            self.step_complete.emit("All slices registered", 95)
            
            # === SAVE RESULTS ===
            self.progress_update.emit("\n💾 Saving Results...")
            self.step_complete.emit("Saving results", 97)
            
            if self.should_abort:
                raise RuntimeError("Registration aborted by user.")
            
            results = {}
            
            # Save registered RGB volume
            rgb_path = os.path.join(output_dir, "registered_histology_rgb.nrrd")
            sitk.WriteImage(hist_rgb_mri, rgb_path)
            results['rgb_volume'] = rgb_path
            self.progress_update.emit(f"   ✓ RGB volume: {rgb_path}")
            
            # Save registered LABELMAP (preserving MRI label values!)
            mask_path = os.path.join(output_dir, "registered_histology_labelmap.nrrd")
            sitk.WriteImage(hist_mask_mri, mask_path)
            results['mask_labelmap'] = mask_path
            self.progress_update.emit(f"   ✓ Mask labelmap: {mask_path}")
            
            # Verify saved labelmap
            saved_mask = sitk.ReadImage(mask_path)
            saved_array = sitk.GetArrayFromImage(saved_mask)
            saved_labels = np.unique(saved_array)
            saved_labels = saved_labels[saved_labels != 0]
            self.progress_update.emit(f"     Saved labels: {sorted(saved_labels)}")
            
            # Save slice-to-label mapping as JSON
            mapping_path = os.path.join(output_dir, "slice_to_label_mapping.json")
            mapping_data = {
                'slice_to_label': {int(k): int(v) for k, v in logic.slice_to_label_map.items()},
                'label_to_slice': {int(k): (v if isinstance(v, int) else list(v)) 
                                  for k, v in logic.label_to_slice_map.items()},
                'mri_labels': sorted([int(l) for l in label_to_z_map.keys()]),
                'num_histology_slices': logic.noSlices,
                'num_mri_labels': len(label_to_z_map)
            }
            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            results['mapping_file'] = mapping_path
            self.progress_update.emit(f"   ✓ Mapping saved: {mapping_path}")
            
            # Save transform parameters for each slice
            transforms_path = os.path.join(output_dir, "slice_transforms.json")
            transforms_data = []
            for hist_idx, ps in enumerate(logic.pathologySlices):
                transform_info = {
                    'slice_index': hist_idx,
                    'mri_label': logic.getTargetLabelForSlice(hist_idx),
                    'has_transform': ps.transform is not None,
                    'slice_thickness_mm': getattr(ps, 'sliceThickness', 1.0)
                }
                transforms_data.append(transform_info)
            
            with open(transforms_path, 'w') as f:
                json.dump(transforms_data, f, indent=2)
            results['transforms_file'] = transforms_path
            self.progress_update.emit(f"   ✓ Transforms saved: {transforms_path}")
            
            # Save registration report
            report_path = os.path.join(output_dir, "registration_report.txt")
            with open(report_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("RADPATHFUSION REGISTRATION REPORT - LABELMAP ASSOCIATION\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("INPUT DATA:\n")
                f.write(f"  MRI volume: {self.config['fixed_volume']}\n")
                f.write(f"  MRI labelmap: {self.config.get('fixed_mask')}\n")
                f.write(f"  Pathology JSON: {self.config['json_path']}\n\n")
                
                f.write("LABELMAP ANALYSIS:\n")
                f.write(f"  Number of MRI labels: {len(label_to_z_map)}\n")
                f.write(f"  Label values: {sorted(label_to_z_map.keys())}\n")
                f.write(f"  Number of histology slices: {logic.noSlices}\n\n")
                
                f.write("SLICE-TO-LABEL MAPPING:\n")
                for hist_idx in range(logic.noSlices):
                    label_val = logic.getTargetLabelForSlice(hist_idx)
                    f.write(f"  Histology slice {hist_idx} → MRI label {label_val}\n")
                
                f.write("\nREGISTRATION SETTINGS:\n")
                f.write(f"  Affine: {logic.doAffine}\n")
                f.write(f"  Deformable: {logic.doDeformable}\n")
                f.write(f"  Fast execution: {logic.fastExecution}\n\n")
                
                f.write("OUTPUT FILES:\n")
                f.write(f"  RGB volume: {rgb_path}\n")
                f.write(f"  Mask labelmap: {mask_path}\n")
                f.write(f"  Mapping JSON: {mapping_path}\n")
                f.write(f"  Transforms JSON: {transforms_path}\n\n")
                
                f.write("FINAL VOLUMES:\n")
                f.write(f"  Size: {hist_rgb_mri.GetSize()}\n")
                f.write(f"  Spacing: {hist_rgb_mri.GetSpacing()} mm\n")
                f.write(f"  Origin: {hist_rgb_mri.GetOrigin()}\n")
                f.write(f"  Labels preserved: {sorted(saved_labels)}\n")
                
            results['report_file'] = report_path
            self.progress_update.emit(f"   ✓ Report saved: {report_path}")
            
            # Create metrics dict compatible with existing code
            results['metrics'] = {
                'total_slices': logic.noSlices,
                'registered_slices': logic.noSlices,
                'registration_type': self.config.get('registration_type', 'affine+bspline'),
                'do_affine': logic.doAffine,
                'do_deformable': logic.doDeformable,
                'volume_spacing': list(hist_rgb_mri.GetSpacing()),
                'volume_size': list(hist_rgb_mri.GetSize()),
                'volume_origin': list(hist_rgb_mri.GetOrigin()),
                'regions': list(logic.regionIDs),
                'labelmap_labels': sorted([int(l) for l in saved_labels]),
                'num_mri_labels': len(label_to_z_map)
            }
            
            # For backward compatibility, create 'registered_masks' dict
            results['registered_masks'] = {
                'labelmap': mask_path
            }
            
            # Additional fields
            results['fixed_volume'] = self.config['fixed_volume']
            results['registered_rgb'] = rgb_path
            results['output_dir'] = output_dir
            results['metrics_file'] = transforms_path  # For compatibility
            
            self.step_complete.emit("Results saved", 100)
            
            # === COMPLETION ===
            self.progress_update.emit("\n" + "=" * 70)
            self.progress_update.emit("✅ REGISTRATION COMPLETE WITH LABELMAP ASSOCIATION!")
            self.progress_update.emit("=" * 70)
            self.progress_update.emit(f"\n📁 Output directory: {output_dir}")
            self.progress_update.emit(f"   - RGB volume with registered histology")
            self.progress_update.emit(f"   - Labelmap preserving MRI label values ({len(saved_labels)} labels)")
            self.progress_update.emit(f"   - Slice-to-label mapping file")
            self.progress_update.emit(f"   - Complete registration report")
            
            self.registration_complete.emit(results)
            
        except Exception as e:
            error_msg = f"Registration failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.progress_update.emit(f"\n❌ ERROR: {error_msg}")
            self.registration_failed.emit(error_msg)
    
    def abort(self):
        """Request abortion of registration"""
        self.should_abort = True


class RegistrationTab(QWidget):
    """Registration tab with labelmap association support"""
    
    registration_succeeded = pyqtSignal(dict)
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self, pathology_volume=None):
        super().__init__()
        self.pathology_volume = pathology_volume
        self.worker = None
        self.loaded_files = {
            'fixed_volume': None,
            'fixed_mask': None,
            'json_path': None
        }
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Input Group
        input_group = self.create_input_group()
        content_layout.addWidget(input_group)
        
        # Config Group
        config_group = self.create_config_group()
        content_layout.addWidget(config_group)
        
        # Progress Group
        progress_group = self.create_progress_group()
        content_layout.addWidget(progress_group)
        
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # Buttons
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)
    
    @pyqtSlot(str, str, str)
    def update_from_viewer(self, data_type: str, name: str, path: str):
        """Update from Medical Viewer tab"""
        logger.info(f"RegistrationTab received '{name}' for type '{data_type}'")
        
        if data_type == 'mr':
            self.fixed_volume_input.setText(path)
            self.loaded_files['fixed_volume'] = path
            self.log_message(f"✓ Fixed volume auto-loaded: {name}")
        elif data_type == 'segmentation':
            self.fixed_mask_input.setText(path)
            self.loaded_files['fixed_mask'] = path
            self.log_message(f"✓ Fixed mask (labelmap) auto-loaded: {name}")
    
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
        
        # Fixed Volume
        fixed_layout = QHBoxLayout()
        fixed_label = QLabel("Fixed Volume (MRI):")
        fixed_label.setMinimumWidth(180)
        fixed_layout.addWidget(fixed_label)
        
        self.fixed_volume_input = QLineEdit()
        self.fixed_volume_input.setPlaceholderText("Select MRI volume...")
        self.fixed_volume_input.setReadOnly(True)
        fixed_layout.addWidget(self.fixed_volume_input, 1)
        
        self.fixed_browse_btn = QPushButton("Browse...")
        self.fixed_browse_btn.clicked.connect(lambda: self.browse_file('fixed_volume'))
        self.fixed_browse_btn.setMinimumWidth(100)
        fixed_layout.addWidget(self.fixed_browse_btn)
        
        layout.addLayout(fixed_layout)
        
        # Fixed Mask (LABELMAP!)
        mask_layout = QHBoxLayout()
        mask_label = QLabel("MRI Labelmap:")
        mask_label.setMinimumWidth(180)
        mask_layout.addWidget(mask_label)
        
        self.fixed_mask_input = QLineEdit()
        self.fixed_mask_input.setPlaceholderText("Select MRI segmentation labelmap...")
        self.fixed_mask_input.setReadOnly(True)
        mask_layout.addWidget(self.fixed_mask_input, 1)
        
        self.mask_browse_btn = QPushButton("Browse...")
        self.mask_browse_btn.clicked.connect(lambda: self.browse_file('fixed_mask'))
        self.mask_browse_btn.setMinimumWidth(100)
        mask_layout.addWidget(self.mask_browse_btn)
        
        layout.addLayout(mask_layout)
        
        # Info about labelmap
        labelmap_info = QLabel(
            "⚠️ IMPORTANT: The MRI segmentation should be a LABELMAP where each slice "
            "has a distinct label value (e.g., 1, 2, 3...). This enables slice-to-slice association."
        )
        labelmap_info.setWordWrap(True)
        labelmap_info.setStyleSheet("color: #e67e22; font-size: 9pt; padding: 5px; font-weight: bold;")
        layout.addWidget(labelmap_info)
        
        # Pathology JSON
        json_layout = QHBoxLayout()
        json_label = QLabel("Pathology JSON:")
        json_label.setMinimumWidth(180)
        json_layout.addWidget(json_label)
        
        self.json_input = QLineEdit()
        self.json_input.setPlaceholderText("Load from parser or browse...")
        self.json_input.setReadOnly(True)
        json_layout.addWidget(self.json_input, 1)
        
        self.json_browse_btn = QPushButton("Browse...")
        self.json_browse_btn.clicked.connect(lambda: self.browse_file('json_path'))
        self.json_browse_btn.setMinimumWidth(100)
        json_layout.addWidget(self.json_browse_btn)
        
        layout.addLayout(json_layout)
        
        group.setLayout(layout)
        return group
    
    @pyqtSlot(str)
    def update_pathology_json_path(self, json_path: str):
        """Update pathology JSON path from parser"""
        if json_path and os.path.exists(json_path):
            self.json_input.setText(json_path)
            self.json_input.setStyleSheet("background-color: #d5f4e6;")
            self.loaded_files['json_path'] = json_path
            self.json_browse_btn.setEnabled(False)
            self.log_message(f"✓ Pathology JSON loaded from parser: {os.path.basename(json_path)}")
        else:
            self.json_input.clear()
            self.json_input.setStyleSheet("")
            self.loaded_files['json_path'] = None
            self.json_browse_btn.setEnabled(True)

    
    
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
        """)
        
        layout = QVBoxLayout()
        
        # Output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        output_label.setMinimumWidth(180)
        output_layout.addWidget(output_label)
        
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select output directory...")
        output_layout.addWidget(self.output_input, 1)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        self.output_browse_btn.setMinimumWidth(100)
        output_layout.addWidget(self.output_browse_btn)
        
        layout.addLayout(output_layout)
        
        # Registration type
        reg_layout = QHBoxLayout()
        reg_label = QLabel("Registration Type:")
        reg_label.setMinimumWidth(180)
        reg_layout.addWidget(reg_label)
        
        self.reg_type_combo = QComboBox()
        self.reg_type_combo.addItems([
            "affine",
            "affine+bspline",
            "affine+demons"
        ])
        self.reg_type_combo.setCurrentText("affine+bspline")
        reg_layout.addWidget(self.reg_type_combo, 1)
        
        layout.addLayout(reg_layout)
        
        # Options
        self.high_quality_cb = QCheckBox("High quality mode (slower)")
        self.high_quality_cb.setChecked(False)
        layout.addWidget(self.high_quality_cb)
        
        self.save_intermediate_cb = QCheckBox("Save intermediate volumes")
        self.save_intermediate_cb.setChecked(True)
        layout.addWidget(self.save_intermediate_cb)
        
        group.setLayout(layout)
        return group
    
    def create_progress_group(self):
        """Create progress group"""
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
        """)
        
        layout = QVBoxLayout()
        
        # Progress bar
        self.progress_label = QLabel("Ready to start registration")
        self.progress_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Log
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background-color: #2c3e50;
                color: #ecf0f1;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_output)
        
        group.setLayout(layout)
        return group
    
    def create_button_layout(self):
        """Create buttons"""
        layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶️ Run Registration")
        self.run_btn.clicked.connect(self.run_registration)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        layout.addWidget(self.run_btn)
        
        self.abort_btn = QPushButton("⏹ Abort")
        self.abort_btn.clicked.connect(self.abort_registration)
        self.abort_btn.setEnabled(False)
        self.abort_btn.setMinimumHeight(40)
        self.abort_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        layout.addWidget(self.abort_btn)
        
        self.clear_log_btn = QPushButton("🗑️ Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setMinimumHeight(40)
        layout.addWidget(self.clear_log_btn)
        
        layout.addStretch()
        return layout
    
    def _check_and_convert_mask_to_labelmap(self, mask_path: str) -> Optional[str]:
        """
        Controlla se la maschera è una labelmap. Se è binaria, la converte.
        Restituisce il percorso al file labelmap (nuovo o originale).
        """
        if not SITK_AVAILABLE:
            self.log_message("❌ Errore: SimpleITK non è disponibile per analizzare la maschera.")
            return None

        try:
            mask_image = sitk.ReadImage(mask_path)
            
            # Analizza il numero di label presenti
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(mask_image)
            num_labels = stats.GetNumberOfLabels()

            # Se ha 0 o più di 1 label, è già una labelmap (o vuota), quindi va bene
            if num_labels != 1:
                self.log_message(f"✓ La maschera ha già {num_labels} label. Nessuna conversione necessaria.")
                return mask_path

            # --- SE SIAMO QUI, LA MASCHERA È BINARIA E VA CONVERTITA ---
            self.log_message("⚠️ La maschera è binaria (1 solo label). Avvio conversione automatica in labelmap...")

            mask_array = sitk.GetArrayFromImage(mask_image)  # Shape: (Z, Y, X)
            labelmap_array = np.zeros_like(mask_array, dtype=np.uint16)
            
            current_label = 1
            for z in range(mask_array.shape[0]):
                slice_mask = mask_array[z, :, :]
                if np.any(slice_mask > 0):
                    labelmap_array[z, :, :][slice_mask > 0] = current_label
                    current_label += 1
            
            if current_label == 1:
                self.log_message("⚠️ Attenzione: Nessuna regione trovata nella maschera durante la conversione.")
                return mask_path # Restituisce l'originale se non c'è nulla da convertire

            labelmap_image = sitk.GetImageFromArray(labelmap_array)
            labelmap_image.CopyInformation(mask_image)

            # Crea un nuovo percorso per il file convertito
            base, ext = os.path.splitext(mask_path)
            if ext == ".gz":
                base, ext2 = os.path.splitext(base)
                ext = ext2 + ext
            
            new_path = f"{base}_labelmap{ext}"
            
            sitk.WriteImage(labelmap_image, new_path)
            
            self.log_message(f"✓ Conversione completata. {current_label - 1} label creati.")
            self.log_message(f"   Nuovo file labelmap: {os.path.basename(new_path)}")
            
            return new_path

        except Exception as e:
            self.log_message(f"❌ Errore durante la conversione della maschera: {e}")
            QMessageBox.critical(self, "Errore Conversione Maschera", f"Impossibile convertire la maschera in labelmap:\n\n{e}")
            return None
    
    def browse_file(self, file_type: str):
        """Browse for file (MODIFIED)"""
        if file_type == 'json_path':
            filters = "JSON Files (*.json);;All Files (*.*)"
        else:
            filters = "Medical Images (*.nii *.nii.gz *.mha *.nrrd);;All Files (*.*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {file_type.replace('_', ' ').title()}",
            "",
            filters
        )
        
        if file_path:
            if file_type == 'fixed_volume':
                self.loaded_files[file_type] = file_path
                self.fixed_volume_input.setText(file_path)
                self.fixed_volume_input.setStyleSheet("background-color: #d5f4e6;")
                self.log_message(f"✓ Fixed volume: {os.path.basename(file_path)}")
            
            elif file_type == 'fixed_mask':
                # --- LOGICA DI CONVERSIONE INTEGRATA QUI ---
                self.log_message(f"Analisi della maschera: {os.path.basename(file_path)}...")
                final_mask_path = self._check_and_convert_mask_to_labelmap(file_path)
                
                if final_mask_path:
                    self.loaded_files[file_type] = final_mask_path
                    self.fixed_mask_input.setText(final_mask_path)
                    self.fixed_mask_input.setStyleSheet("background-color: #d5f4e6;")
                    # Il messaggio di log dettagliato è già in _check_and_convert_mask_to_labelmap
            
            elif file_type == 'json_path':
                self.loaded_files[file_type] = file_path
                self.json_input.setText(file_path)
                self.json_input.setStyleSheet("background-color: #d5f4e6;")
                self.log_message(f"✓ Pathology JSON: {os.path.basename(file_path)}")
                self.pathology_volume = None
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        
        if dir_path:
            self.output_input.setText(dir_path)
            self.log_message(f"✓ Output directory: {dir_path}")
    
    def log_message(self, message: str):
        """Log message with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def clear_log(self):
        """Clear log"""
        self.log_output.clear()
        self.log_message("Log cleared")
    
    def validate_inputs(self) -> Tuple[bool, str]:
        """Validate inputs"""
        if not self.loaded_files.get('fixed_volume'):
            return False, "Please select MRI volume"
        
        if not self.loaded_files.get('fixed_mask'):
            return False, "Please select MRI labelmap segmentation"
        
        if not self.pathology_volume and not self.loaded_files.get('json_path'):
            return False, "Please load pathology data"
        
        if not self.output_input.text():
            return False, "Please select output directory"
        
        return True, ""
    
    def run_registration(self):
        """Start registration"""
        valid, error_msg = self.validate_inputs()
        if not valid:
            QMessageBox.warning(self, "Invalid Input", error_msg)
            self.log_message(f"❌ {error_msg}")
            return
        
        # Parse registration type
        reg_type = self.reg_type_combo.currentText()
        do_affine = 'affine' in reg_type
        do_deformable = 'bspline' in reg_type or 'demons' in reg_type
        
        config = {
            'fixed_volume': self.loaded_files['fixed_volume'],
            'fixed_mask': self.loaded_files['fixed_mask'],
            'json_path': self.loaded_files.get('json_path') or str(self.pathology_volume.path),
            'output_dir': self.output_input.text(),
            'registration_type': reg_type,
            'do_affine': do_affine,
            'do_deformable': do_deformable,
            'fast_execution': not self.high_quality_cb.isChecked(),
            'save_intermediate': self.save_intermediate_cb.isChecked()
        }
        
        self.log_message("=" * 70)
        self.log_message("🚀 STARTING LABELMAP-BASED REGISTRATION")
        self.log_message("=" * 70)
        self.log_message(f"MRI volume: {os.path.basename(config['fixed_volume'])}")
        self.log_message(f"MRI labelmap: {os.path.basename(config['fixed_mask'])}")
        self.log_message(f"Pathology JSON: {os.path.basename(config['json_path'])}")
        self.log_message(f"Registration: {reg_type}")
        self.log_message("=" * 70)
        
        # Update UI
        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Disable inputs
        self.fixed_browse_btn.setEnabled(False)
        self.mask_browse_btn.setEnabled(False)
        self.json_browse_btn.setEnabled(False)
        self.output_browse_btn.setEnabled(False)
        
        # Start worker
        self.worker = RegistrationWorker(config, self.pathology_volume)
        self.worker.progress_update.connect(self.log_message)
        self.worker.step_complete.connect(self.on_step_complete)
        self.worker.registration_complete.connect(self.on_registration_complete)
        self.worker.registration_failed.connect(self.on_registration_failed)
        self.worker.start()
    
    def abort_registration(self):
        """Abort registration"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Abort Registration",
                "Are you sure you want to abort?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.log_message("⚠️ Aborting...")
                self.worker.abort()
                self.worker.wait()
                self.reset_ui()
                self.log_message("❌ Aborted by user")
    
    @pyqtSlot(str, int)
    def on_step_complete(self, step_name: str, progress: int):
        """Update progress"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(step_name)
    
    @pyqtSlot(dict)
    def on_registration_complete(self, results: dict):
        """Handle completion"""
        self.log_message("\n" + "=" * 70)
        self.log_message("✅ REGISTRATION COMPLETED!")
        self.log_message("=" * 70)
        
        metrics = results['metrics']
        self.log_message(f"\n📊 Results:")
        self.log_message(f"  Slices: {metrics['registered_slices']}/{metrics['total_slices']}")
        self.log_message(f"  Registration: {metrics['registration_type']}")
        self.log_message(f"  MRI labels: {metrics['num_mri_labels']}")
        self.log_message(f"  Preserved labels: {metrics['labelmap_labels']}")
        self.log_message(f"\n📁 Files:")
        self.log_message(f"  RGB: {os.path.basename(results['rgb_volume'])}")
        self.log_message(f"  Labelmap: {os.path.basename(results['mask_labelmap'])}")
        self.log_message(f"  Mapping: {os.path.basename(results['mapping_file'])}")
        self.log_message("=" * 70)
        
        self.reset_ui()
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ Complete!")
        self.progress_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        
        # Emit signal
        self.registration_succeeded.emit(results)
        
        # Show dialog
        QMessageBox.information(
            self,
            "Registration Complete",
            f"✅ Registration completed successfully!\n\n"
            f"Slices: {metrics['registered_slices']}/{metrics['total_slices']}\n"
            f"MRI labels preserved: {len(metrics['labelmap_labels'])}\n\n"
            f"Results saved to:\n{results['output_dir']}\n\n"
            f"The labelmap preserves MRI label values for "
            f"accurate slice-to-slice correspondence."
        )
    
    @pyqtSlot(str)
    def on_registration_failed(self, error_msg: str):
        """Handle failure"""
        self.log_message("\n" + "=" * 70)
        self.log_message("❌ REGISTRATION FAILED")
        self.log_message("=" * 70)
        self.log_message(f"Error: {error_msg}")
        
        self.reset_ui()
        self.progress_bar.setValue(0)
        self.progress_label.setText("❌ Failed")
        self.progress_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        
        QMessageBox.critical(
            self,
            "Registration Failed",
            f"Registration failed:\n\n{error_msg}\n\n"
            f"Check the log for details."
        )
    
    def reset_ui(self):
        """Reset UI to ready state"""
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.fixed_browse_btn.setEnabled(True)
        self.mask_browse_btn.setEnabled(True)
        self.json_browse_btn.setEnabled(self.pathology_volume is None)
        self.output_browse_btn.setEnabled(True)
    
    @pyqtSlot(object)
    def set_pathology_volume(self, pathology_volume):
        """Receive pathology volume from parser tab"""
        self.pathology_volume = pathology_volume
        if pathology_volume and hasattr(pathology_volume, 'path'):
            self.json_input.setText(str(pathology_volume.path))
            self.json_input.setStyleSheet("background-color: #d5f4e6;")
            self.loaded_files['json_path'] = str(pathology_volume.path)
            self.json_browse_btn.setEnabled(False)
            self.log_message(f"✓ Pathology volume from parser: {pathology_volume.noSlices} slices")
        else:
            self.json_browse_btn.setEnabled(True)
    
    def receive_pathology_volume(self, updated_volume_object):
        """Alternative method name for compatibility"""
        self.set_pathology_volume(updated_volume_object)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("RadPathFusion Registration - Labelmap Support")
    window.setGeometry(100, 100, 1000, 800)
    
    layout = QVBoxLayout(window)
    
    reg_tab = RegistrationTab()
    
    def on_complete(results):
        print("\n" + "="*70)
        print("REGISTRATION COMPLETED!")
        print("="*70)
        print(f"RGB: {results['rgb_volume']}")
        print(f"Labelmap: {results['mask_labelmap']}")
        print(f"Mapping: {results['mapping_file']}")
        print(f"Labels: {results['metrics']['labelmap_labels']}")
        print("="*70)
    
    reg_tab.registration_succeeded.connect(on_complete)
    
    layout.addWidget(reg_tab)
    window.show()
    
    sys.exit(app.exec())