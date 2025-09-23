#!/usr/bin/env python3
"""
Enhanced integration between BigWarp and ParsePathology for automatic rotation angles transfer
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel
from PyQt6.QtCore import pyqtSignal

from Resources.Utils.ImageStack_3 import PathologyVolume
from Resources.Utils.ImageRegistration import RegisterImages
from Resources.Utils.RegisterVolumesElastix import RegisterVolumesElastix
from Resources.Utils.ParsePathJsonUtils import ParsePathJsonUtils
from Resources.Utils.dicom_thickness import DicomSliceInfo,PathologyParser,DicomAnalyzer
from Resources.Utils.dicom_creator import create_dicom
from standalone_gui_4 import BigWarpTab, PathologyParser, MainWindow

import SimpleITK as sitk
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QSpinBox, QCheckBox,
    QComboBox, QTableWidget, QTableWidgetItem, QSplitter,
    QScrollArea, QFrame, QDialog, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

try:
    from Resources.Utils.BigWarp import BigWarpGUI, Landmark, ThinPlateSpline
    BIGWARP_AVAILABLE = True
except ImportError:
    BIGWARP_AVAILABLE = False
    print("Warning: BigWarp non trovato. Il tab di registrazione deformabile non sarà disponibile.")

try:
    from medical_viewer_tab import MedicalViewer
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. The medical viewer will not be available.")


class RotationDataManager:
    """Class to manage rotation data between BigWarp and ParsePathology"""
    
    def __init__(self):
        self.rotation_data: Dict[str, float] = {}
        self.image_mapping: Dict[str, str] = {}  # Maps BigWarp filenames to PathologySlice IDs
    
    def add_rotation_data(self, filename: str, rotation_angle: float):
        """Add rotation data for a specific image"""
        self.rotation_data[filename] = rotation_angle
    
    def get_rotation_angle(self, filename: str) -> Optional[float]:
        """Get rotation angle for a specific image"""
        return self.rotation_data.get(filename)
    
    def export_to_csv(self, filepath: str):
        """Export rotation data to CSV"""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'rotation_angle_degrees'])
            writer.writeheader()
            for filename, angle in self.rotation_data.items():
                writer.writerow({'filename': filename, 'rotation_angle_degrees': angle})
    
    def import_from_csv(self, filepath: str):
        """Import rotation data from CSV"""
        import csv
        self.rotation_data.clear()
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rotation_data[row['filename']] = float(row['rotation_angle_degrees'])

# Enhanced BigWarpTab with data sharing capabilities
class EnhancedBigWarpTab(BigWarpTab):
    """Enhanced BigWarp tab with rotation data sharing"""
    
    # Signal to emit rotation data
    rotation_data_ready = pyqtSignal(dict)  # {filename: rotation_angle}
    
    def __init__(self):
        super().__init__()
        self.rotation_manager = RotationDataManager()
        self.add_data_sharing_controls()
    
    def add_data_sharing_controls(self):
        """Add controls for data sharing with ParsePathology"""
        # Add to existing layout
        layout = self.layout()
        
        # Data sharing group
        sharing_group = QGroupBox("Data Sharing with ParsePathology")
        sharing_layout = QVBoxLayout()
        
        buttons_layout = QHBoxLayout()
        
        self.export_angles_btn = QPushButton("Export Rotation Angles (CSV)")
        self.export_angles_btn.clicked.connect(self.export_rotation_angles)
        
        self.send_to_pathology_btn = QPushButton("Send Angles to ParsePathology")
        self.send_to_pathology_btn.clicked.connect(self.send_angles_to_pathology)
        self.send_to_pathology_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.export_angles_btn)
        buttons_layout.addWidget(self.send_to_pathology_btn)
        buttons_layout.addStretch()
        
        self.angles_status_label = QLabel("No rotation angles calculated yet")
        self.angles_status_label.setStyleSheet("QLabel { color: gray; }")
        
        sharing_layout.addLayout(buttons_layout)
        sharing_layout.addWidget(self.angles_status_label)
        
        sharing_group.setLayout(sharing_layout)
        
        # Insert before the last stretch
        layout.insertWidget(layout.count() - 1, sharing_group)
    
    def import_result(self):
        """Enhanced import_result that also manages rotation data"""
        # Call parent method
        super().import_result()
        
        # If we have rotation angle, store it
        if hasattr(self, 'rotation_angle'):
            # Try to get the filename from the imported JSON
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select the corresponding image file for this transformation", 
                "", "Image files (*.png *.jpg *.jpeg *.tiff *.bmp)"
            )
            if filename:
                image_name = os.path.basename(filename)
                self.rotation_manager.add_rotation_data(image_name, self.rotation_angle)
                self.angles_status_label.setText(f"Rotation angle stored for: {image_name}")
                self.send_to_pathology_btn.setEnabled(True)
    
    def export_rotation_angles(self):
        """Export rotation angles to CSV file"""
        if not self.rotation_manager.rotation_data:
            QMessageBox.warning(self, "Warning", "No rotation angles to export.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Rotation Angles", "", "CSV files (*.csv)"
        )
        if filepath:
            try:
                self.rotation_manager.export_to_csv(filepath)
                QMessageBox.information(self, "Success", f"Rotation angles exported to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def send_angles_to_pathology(self):
        """Send rotation angles to ParsePathology tab"""
        if self.rotation_manager.rotation_data:
            self.rotation_data_ready.emit(self.rotation_manager.rotation_data)
            QMessageBox.information(self, "Data Sent", "Rotation angles sent to ParsePathology tab")

# Enhanced ParsePathology with rotation angles integration
class EnhancedPathologyParser(PathologyParser):
    """Enhanced PathologyParser with rotation angles integration"""
    
    def __init__(self):
        super().__init__()
        self.rotation_angles_data: Dict[str, float] = {}
        self.add_rotation_controls()
    
    def add_rotation_controls(self):
        """Add controls for rotation angles management"""
        layout = self.layout()
        
        # Rotation angles group
        rotation_group = QGroupBox("Rotation Angles from BigWarp")
        rotation_layout = QVBoxLayout()
        
        buttons_layout = QHBoxLayout()
        
        self.import_angles_btn = QPushButton("Import Rotation Angles (CSV)")
        self.import_angles_btn.clicked.connect(self.import_rotation_angles)
        
        self.apply_angles_btn = QPushButton("Apply Angles to Slices")
        self.apply_angles_btn.clicked.connect(self.apply_rotation_angles)
        self.apply_angles_btn.setEnabled(False)
        
        self.auto_match_btn = QPushButton("Auto-Match by Filename")
        self.auto_match_btn.clicked.connect(self.auto_match_filenames)
        self.auto_match_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.import_angles_btn)
        buttons_layout.addWidget(self.apply_angles_btn)
        buttons_layout.addWidget(self.auto_match_btn)
        buttons_layout.addStretch()
        
        self.rotation_status_label = QLabel("No rotation angles loaded")
        self.rotation_status_label.setStyleSheet("QLabel { color: gray; }")
        
        rotation_layout.addLayout(buttons_layout)
        rotation_layout.addWidget(self.rotation_status_label)
        
        rotation_group.setLayout(rotation_layout)
        
        # Insert after the volume settings group
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QGroupBox) and widget.title() == "Volume Info":
                layout.insertWidget(i + 1, rotation_group)
                break
    
    def receive_rotation_data(self, rotation_data: Dict[str, float]):
        """Receive rotation data from BigWarp tab"""
        self.rotation_angles_data.update(rotation_data)
        self.rotation_status_label.setText(f"Received {len(rotation_data)} rotation angles from BigWarp")
        self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
        self.apply_angles_btn.setEnabled(True)
        self.auto_match_btn.setEnabled(True)
    
    def import_rotation_angles(self):
        """Import rotation angles from CSV file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Rotation Angles", "", "CSV files (*.csv)"
        )
        if filepath:
            try:
                import csv
                self.rotation_angles_data.clear()
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.rotation_angles_data[row['filename']] = float(row['rotation_angle_degrees'])
                
                self.rotation_status_label.setText(f"Loaded {len(self.rotation_angles_data)} rotation angles")
                self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
                self.apply_angles_btn.setEnabled(True)
                self.auto_match_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import rotation angles: {e}")
    
    def auto_match_filenames(self):
        """Automatically match rotation angles to slices by filename"""
        if not self.pathology_volume or not self.rotation_angles_data:
            QMessageBox.warning(self, "Warning", "Load both pathology data and rotation angles first")
            return
        
        matches_found = 0
        
        try:
            for i, ps in enumerate(self.pathology_volume.pathologySlices):
                slice_filename = os.path.basename(ps.rgbImageFn)
                
                # Try exact match first
                if slice_filename in self.rotation_angles_data:
                    angle = self.rotation_angles_data[slice_filename]
                    ps.doRotate = angle
                    matches_found += 1
                    continue
                
                # Try fuzzy matching (without extension, with different extensions)
                base_name = Path(slice_filename).stem
                for angle_filename, angle in self.rotation_angles_data.items():
                    angle_base = Path(angle_filename).stem
                    if base_name == angle_base:
                        ps.doRotate = angle
                        matches_found += 1
                        break
            
            self.update_slices_display()
            self.update_volume_info_display()
            
            QMessageBox.information(
                self, "Auto-Match Results", 
                f"Successfully matched {matches_found} out of {len(self.pathology_volume.pathologySlices)} slices"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during auto-matching: {e}")
    
    def apply_rotation_angles(self):
        """Manually apply rotation angles with user selection"""
        if not self.pathology_volume or not self.rotation_angles_data:
            QMessageBox.warning(self, "Warning", "Load both pathology data and rotation angles first")
            return
        
        # Show dialog for manual matching
        self.show_manual_matching_dialog()
    
    def show_manual_matching_dialog(self):
        """Show dialog for manual matching of rotation angles to slices"""
        from PyQt6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Match Rotation Angles to Slices")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Table for matching
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels([
            "Slice Index", "Slice Filename", "Available Angles", "Current Rotation"
        ])
        
        table.setRowCount(len(self.pathology_volume.pathologySlices))
        
        for i, ps in enumerate(self.pathology_volume.pathologySlices):
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
            
            # Combo box with available angles
            from PyQt6.QtWidgets import QComboBox
            combo = QComboBox()
            combo.addItem("No change", None)
            for filename, angle in self.rotation_angles_data.items():
                combo.addItem(f"{filename} ({angle:.1f}°)", angle)
            table.setCellWidget(i, 2, combo)
            
            current_rotation = getattr(ps, 'doRotate', 0.0)
            table.setItem(i, 3, QTableWidgetItem(f"{current_rotation:.1f}°"))
        
        layout.addWidget(table)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply Selected")
        cancel_btn = QPushButton("Cancel")
        
        def apply_selections():
            try:
                for i in range(table.rowCount()):
                    combo = table.cellWidget(i, 2)
                    selected_angle = combo.currentData()
                    if selected_angle is not None:
                        self.pathology_volume.pathologySlices[i].doRotate = selected_angle
                
                self.update_slices_display()
                self.update_volume_info_display()
                dialog.accept()
                QMessageBox.information(self, "Success", "Rotation angles applied successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error applying rotations: {e}")
        
        apply_btn.clicked.connect(apply_selections)
        cancel_btn.clicked.connect(dialog.reject)
        
        buttons_layout.addWidget(apply_btn)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        dialog.setLayout(layout)
        dialog.exec()

# Enhanced MainWindow with proper signal connections
class EnhancedMainWindow(MainWindow):
    """Enhanced MainWindow with rotation angles integration"""
    
    def init_ui(self):
        """Override to use enhanced tabs"""
        # Call parent but replace tabs
        super().init_ui()
        
        # Replace BigWarp tab if available
        if BIGWARP_AVAILABLE:
            # Remove old tab
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "BigWarp Registration":
                    self.tabs.removeTab(i)
                    break
            
            # Add enhanced BigWarp tab
            self.bigwarp_tab = EnhancedBigWarpTab()
            self.tabs.insertTab(0, self.bigwarp_tab, "BigWarp Registration")
        
        # Replace PathologyParser tab
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Parse Pathology":
                self.tabs.removeTab(i)
                break
        
        # Add enhanced PathologyParser tab
        self.pathology_tab = EnhancedPathologyParser()
        insert_index = 1 if BIGWARP_AVAILABLE else 0
        self.tabs.insertTab(insert_index, self.pathology_tab, "Parse Pathology")
        
        # Connect signals for rotation data transfer
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab.rotation_data_ready.connect(self.pathology_tab.receive_rotation_data)
        
        # Reconnect other signals
        if MATPLOTLIB_AVAILABLE:
            self.pathology_tab.data_loaded.connect(self.send_data_to_viewer)
            if BIGWARP_AVAILABLE:
                self.bigwarp_tab.data_loaded.connect(self.send_data_to_viewer)

# Additional utility functions for JSON manipulation
class PathologyJSONManager:
    """Utility class for manipulating pathology JSON files"""
    
    @staticmethod
    def update_rotation_angles_in_json(json_path: str, rotation_data: Dict[str, float], output_path: str = None):
        """Update rotation angles in a pathology JSON file"""
        if output_path is None:
            output_path = json_path
        
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            if 'slices' not in json_data:
                raise ValueError("Invalid pathology JSON format: missing 'slices' key")
            
            updated_count = 0
            
            for slice_data in json_data['slices']:
                if 'rgbImageFn' in slice_data:
                    filename = os.path.basename(slice_data['rgbImageFn'])
                    
                    # Try exact match
                    if filename in rotation_data:
                        slice_data['doRotate'] = rotation_data[filename]
                        updated_count += 1
                        continue
                    
                    # Try fuzzy match
                    base_name = Path(filename).stem
                    for angle_filename, angle in rotation_data.items():
                        angle_base = Path(angle_filename).stem
                        if base_name == angle_base:
                            slice_data['doRotate'] = angle
                            updated_count += 1
                            break
            
            # Save updated JSON
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return updated_count
            
        except Exception as e:
            raise Exception(f"Error updating JSON file: {e}")
    
    @staticmethod
    def batch_update_from_csv(json_files: List[str], csv_path: str, output_dir: str = None):
        """Batch update multiple JSON files from a CSV of rotation angles"""
        import csv
        
        # Read rotation data
        rotation_data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rotation_data[row['filename']] = float(row['rotation_angle_degrees'])
        
        results = {}
        
        for json_file in json_files:
            try:
                if output_dir:
                    output_path = os.path.join(output_dir, os.path.basename(json_file))
                else:
                    output_path = json_file
                
                updated_count = PathologyJSONManager.update_rotation_angles_in_json(
                    json_file, rotation_data, output_path
                )
                results[json_file] = updated_count
                
            except Exception as e:
                results[json_file] = f"Error: {e}"
        
        return results