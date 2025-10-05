# File: pathology_parser_tab.py

import os
import csv
from pathlib import Path
from typing import Dict, List

import SimpleITK as sitk
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLineEdit, QPushButton,
    QSpinBox, QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QScrollArea, QDialog, QCheckBox, QInputDialog, QMessageBox,
    QFileDialog
)
from PyQt6.QtCore import pyqtSignal

# Importa le utility necessarie dalla cartella Resources
from Resources.Utils.ImageStack_3 import PathologyVolume

# Gestisci l'importazione opzionale di Pillow
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("Warning: Pillow (PIL) not found. Vertical flip functionality will be disabled.")


class PathologyParser(QWidget):
    
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        self.pathology_volume = None
        self.rectum_distance = 5.0
        self.rotation_angles_data: Dict[str, float] = {}
        self.flip_data: Dict[str, Dict] = {}
        self.init_ui()
        
    def init_ui(self):
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        
        input_group = QGroupBox("Input files")
        input_layout = QVBoxLayout()
        
        json_layout = QHBoxLayout()
        self.json_input = QLineEdit()
        self.json_input.setPlaceholderText("Select pathology JSON file")
        json_browse = QPushButton("Browse")
        json_browse.clicked.connect(self.browse_json)
        json_layout.addWidget(QLabel("JSON file:"))
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(json_browse)
        input_layout.addLayout(json_layout)
        
        volume_layout = QHBoxLayout()
        self.volume_input = QLineEdit()
        self.volume_input.setPlaceholderText("Output Volume name")
        volume_layout.addWidget(QLabel("Output Volume:"))
        volume_layout.addWidget(self.volume_input)
        input_layout.addLayout(volume_layout)
        
        mask_layout = QHBoxLayout()
        self.mask_input = QLineEdit()
        self.mask_input.setPlaceholderText("Output mask name")
        
        self.mask_id_spin = QSpinBox()
        self.mask_id_spin.setRange(0, 100)
        mask_layout.addWidget(QLabel("Output Mask:"))
        mask_layout.addWidget(self.mask_input)
        mask_layout.addWidget(QLabel("<b>Mask ID:</b>"))
        mask_layout.addWidget(self.mask_id_spin)
        input_layout.addLayout(mask_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        volume_settings_group = QGroupBox("Volume Info")
        volume_settings_layout = QVBoxLayout()

        rectum_layout = QHBoxLayout()
        rectum_layout.addWidget(QLabel("Distance from anal verge (mm):"))
        self.rectum_distance_spin = QSpinBox()
        self.rectum_distance_spin.setRange(0, 100)
        self.rectum_distance_spin.setSingleStep(1)
        self.rectum_distance_spin.setValue(int(self.rectum_distance))
        self.rectum_distance_spin.valueChanged.connect(self.update_rectum_distance)
        rectum_layout.addWidget(self.rectum_distance_spin)
        rectum_layout.addStretch()
        volume_settings_layout.addLayout(rectum_layout)
        
        self.volume_info = QTextEdit()
        self.volume_info.setReadOnly(True)
        self.volume_info.setPlaceholderText("Volume information will appear here after loading JSON...")
        volume_settings_layout.addWidget(self.volume_info)
        
        volume_settings_group.setLayout(volume_settings_layout)
        layout.addWidget(volume_settings_group)
        
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        primary_actions = QHBoxLayout()
        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.clicked.connect(self.load_json)
        self.load_volume_btn = QPushButton("Load Volume")
        self.load_volume_btn.clicked.connect(self.load_volume)
        self.load_volume_btn.setEnabled(False)
        self.load_mask_btn = QPushButton("Load Mask")
        self.load_mask_btn.clicked.connect(self.load_mask)
        self.load_mask_btn.setEnabled(False)
        primary_actions.addWidget(self.load_json_btn)
        primary_actions.addWidget(self.load_volume_btn)
        primary_actions.addWidget(self.load_mask_btn)
        actions_layout.addLayout(primary_actions)
        
        secondary_actions = QHBoxLayout()
        actions_layout.addLayout(secondary_actions)
        
        export_actions = QHBoxLayout()
        self.send_to_viewer_btn = QPushButton("Send to Viewer")
        self.send_to_viewer_btn.clicked.connect(self.send_to_viewer)
        self.send_to_viewer_btn.setEnabled(False)
        self.save_json_btn = QPushButton("Save JSON")
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_json_btn.setEnabled(False)
        self.export_volume_btn = QPushButton("Export Volume")
        self.export_volume_btn.clicked.connect(self.export_volume)
        self.export_volume_btn.setEnabled(False)
        export_actions.addWidget(self.send_to_viewer_btn)
        export_actions.addWidget(self.save_json_btn)
        export_actions.addWidget(self.export_volume_btn)
        actions_layout.addLayout(export_actions)
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        external_data_group = QGroupBox("Data from BigWarp")
        external_data_layout = QVBoxLayout()
        grid_layout = QHBoxLayout()

        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout()
        self.import_angles_btn = QPushButton("Import Angles (CSV)")
        self.import_angles_btn.clicked.connect(self.import_rotation_angles)
        self.apply_angles_btn = QPushButton("Apply Angles")
        self.apply_angles_btn.clicked.connect(self.apply_rotation_angles)
        self.apply_angles_btn.setEnabled(False)
        self.rotation_status_label = QLabel("No rotation angles loaded")
        self.rotation_status_label.setStyleSheet("QLabel { color: gray; }")
        rotation_layout.addWidget(self.import_angles_btn)
        rotation_layout.addWidget(self.apply_angles_btn)
        rotation_layout.addWidget(self.rotation_status_label)
        rotation_group.setLayout(rotation_layout)
        grid_layout.addWidget(rotation_group)

        flip_group = QGroupBox("Flip")
        flip_layout = QVBoxLayout()
        self.import_flips_btn = QPushButton("Import Flips (CSV)")
        self.import_flips_btn.clicked.connect(self.import_flips)
        self.apply_flips_btn = QPushButton("Apply Flips")
        self.apply_flips_btn.clicked.connect(self.apply_flips)
        self.apply_flips_btn.setEnabled(False)
        self.flip_status_label = QLabel("No flip data loaded")
        self.flip_status_label.setStyleSheet("QLabel { color: gray; }")
        flip_layout.addWidget(self.import_flips_btn)
        flip_layout.addWidget(self.apply_flips_btn)
        flip_layout.addWidget(self.flip_status_label)
        flip_group.setLayout(flip_layout)
        grid_layout.addWidget(flip_group)

        external_data_layout.addLayout(grid_layout)
        external_data_group.setLayout(external_data_layout)
        layout.addWidget(external_data_group)
        
        info_group = QGroupBox("JSON Info")
        info_layout = QVBoxLayout()
        self.json_info = QTextEdit()
        self.json_info.setReadOnly(True)
        self.json_info.setMaximumHeight(150)
        info_layout.addWidget(self.json_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
       
        slices_group = QGroupBox("Slice Management")
        slices_layout = QVBoxLayout()
        
        self.slices_table = QTableWidget()
        self.slices_table.setColumnCount(9)
        self.slices_table.setHorizontalHeaderLabels([
            "ID", "Slice#", "File", "Thickness(mm)", "Z Pos(mm)", "V Flip", "H Flip", "Rotation(°)", "Regions"
        ])
        
        self.slices_table.setColumnWidth(0, 50)
        self.slices_table.setColumnWidth(1, 60)
        self.slices_table.setColumnWidth(2, 200)
        self.slices_table.setColumnWidth(3, 100)
        self.slices_table.setColumnWidth(4, 100)
        self.slices_table.setColumnWidth(5, 50)
        self.slices_table.setColumnWidth(6, 50)
        self.slices_table.setColumnWidth(7, 80)
        self.slices_table.setColumnWidth(8, 80)
        
        self.slices_table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self.slices_table.setMinimumHeight(400)
        self.slices_table.itemChanged.connect(self.on_slice_item_changed)
        self.slices_table.doubleClicked.connect(self.edit_slice_dialog)
        
        slices_layout.addWidget(self.slices_table, 1)
        
        table_controls = QHBoxLayout()
        self.renumber_btn = QPushButton("Renumber Slices")
        self.renumber_btn.clicked.connect(self.renumber_slices)
        self.update_masks_btn = QPushButton("Update Masks")
        self.update_masks_btn.clicked.connect(self.update_masks)
        self.edit_slice_btn = QPushButton("Edit Selected")
        self.edit_slice_btn.clicked.connect(self.edit_selected_slice)
        self.update_thickness_btn = QPushButton("Update Thickness")
        self.update_thickness_btn.clicked.connect(self.update_slice_thickness)
        
        table_controls.addWidget(self.renumber_btn)
        table_controls.addWidget(self.update_masks_btn)
        table_controls.addWidget(self.edit_slice_btn)
        table_controls.addWidget(self.update_thickness_btn)
        table_controls.addStretch()
        
        slices_layout.addLayout(table_controls)
        slices_group.setLayout(slices_layout)
        layout.addWidget(slices_group)
        
        layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(container_widget)
        scroll_area.setWidgetResizable(True)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        
        self.loaded_volume = None
        self.loaded_mask = None

    def update_rectum_distance(self):
        self.rectum_distance = float(self.rectum_distance_spin.value())
        if self.pathology_volume:
            try:
                self.pathology_volume.updateRectumDistance(self.rectum_distance)
                self.update_slices_display()
                self.update_volume_info_display()
                QMessageBox.information(self, "Updated", f"Rectum distance updated to {self.rectum_distance} mm")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not update rectum distance: {e}")
        
    def update_volume_info_display(self):
        if not self.pathology_volume:
            self.volume_info.clear()
            return
            
        try:
            info_lines = []
            info_lines.append(f"Number of slices: {self.pathology_volume.noSlices}")
            info_lines.append(f"Number of regions: {getattr(self.pathology_volume, 'noRegions', 'N/A')}")
            info_lines.append(f"Volume size: {getattr(self.pathology_volume, 'volumeSize', 'N/A')}")
            
            if hasattr(self.pathology_volume, 'totalVolumeThickness'):
                info_lines.append(f"Total thickness: {self.pathology_volume.totalVolumeThickness:.2f} mm")
            
            info_lines.append(f"Rectum distance: {self.rectum_distance:.2f} mm")
            
            if hasattr(self.pathology_volume, 'slicePositions') and self.pathology_volume.slicePositions:
                z_min = min(self.pathology_volume.slicePositions)
                z_max = max(self.pathology_volume.slicePositions)
                info_lines.append(f"Z position range: {z_min:.2f} - {z_max:.2f} mm")
                info_lines.append(f"Regions: {', '.join(map(str, self.pathology_volume.regionIDs))}")
                              
            self.volume_info.setText('\n'.join(info_lines))
            
        except Exception as e:
            self.volume_info.setText(f"Error updating info: {e}")
        
    def browse_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON file", "", "JSON files (*.json)"
        )
        if file_path:
            self.json_input.setText(file_path)
            
    def load_json(self):
        json_path = self.json_input.text()
        if not json_path or not os.path.exists(json_path):
            QMessageBox.warning(self, "Error", "Select a valid JSON file")
            return

        try:
            self.pathology_volume = PathologyVolume()
            self.pathology_volume.setPath(json_path)
            
            if not self.pathology_volume.initComponents():
                raise Exception("PathologyVolume class initialization error")
            
            if hasattr(self.pathology_volume, 'pathologySlices'):
                for ps in self.pathology_volume.pathologySlices:
                    ps.doHorizontalFlip = getattr(ps, 'doFlip', 0) == 1
                    ps.doVerticalFlip = False
                
            if hasattr(self.pathology_volume, 'rectumDistance'):
                self.rectum_distance = self.pathology_volume.rectumDistance
                self.rectum_distance_spin.setValue(int(self.rectum_distance))
                
            self.update_json_info_display()
            self.update_slices_display()
            self.update_volume_info_display()

            self.load_volume_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)
            self.save_json_btn.setEnabled(True)
            
            QMessageBox.information(self, "Success", "JSON loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading JSON: {e}")

    def receive_rotation_data(self, rotation_data: Dict[str, float]):
        """Receive rotation data from BigWarp tab"""
        self.rotation_angles_data.update(rotation_data)
        self.rotation_status_label.setText(f"Received {len(rotation_data)} rotation angles from BigWarp")
        self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
        self.apply_angles_btn.setEnabled(True)
        
    def import_rotation_angles(self):
        """Import rotation angles from CSV file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Rotation Angles", "", "CSV files (*.csv)"
        )
        if filepath:
            try:
                self.rotation_angles_data.clear()
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row['filename']
                        angle = float(row['rotation_angle_degrees'])
                        self.rotation_angles_data[filename] = angle
                
                self.rotation_status_label.setText(f"Loaded {len(self.rotation_angles_data)} rotation angles")
                self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
                self.apply_angles_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import rotation angles: {e}")

    def import_flips(self):
        """Import flip data from a CSV file generated by BigWarp."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Flips", "", "CSV files (*.csv)")
        if filepath:
            try:
                self.flip_data.clear()
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row['filename']
                        v_flip = row.get('vertical_flip', 'False').strip().lower() == 'true'
                        h_flip = row.get('horizontal_flip', 'False').strip().lower() == 'true'
                        self.flip_data[filename] = {'vertical': v_flip, 'horizontal': h_flip}
                
                self.flip_status_label.setText(f"Loaded {len(self.flip_data)} flip states")
                self.flip_status_label.setStyleSheet("QLabel { color: green; }")
                self.apply_flips_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import flip data: {e}")
                
    def apply_flips(self):
        """Apply all loaded flip states automatically."""
        if not self.flip_data or not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "Load JSON and flip data first.")
            return
        self.auto_match_and_apply_flips()

    def auto_match_filenames(self):
        """Automatically match rotation angles to slices by filename"""
        if not self.pathology_volume or not self.rotation_angles_data:
            QMessageBox.warning(self, "Warning", "Load both pathology data and rotation angles first")
            return
        
        if not hasattr(self.pathology_volume, 'pathologySlices') or not self.pathology_volume.pathologySlices:
            QMessageBox.warning(self, "Warning", "No pathology slices found. Load JSON first.")
            return
        
        matches_found = 0
        try:
            for ps in self.pathology_volume.pathologySlices:
                slice_filename = os.path.basename(ps.rgbImageFn)
                
                if slice_filename in self.rotation_angles_data:
                    ps.doRotate = self.rotation_angles_data[slice_filename]
                    matches_found += 1
                    continue
                
                base_name = Path(slice_filename).stem
                for angle_filename, angle in self.rotation_angles_data.items():
                    if base_name == Path(angle_filename).stem:
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

    def apply_vertical_flips(self):
        if not self.pathology_volume or not hasattr(self.pathology_volume, 'pathologySlices'):
            QMessageBox.warning(self, "Warning", "No pathology data loaded.")
            return

        slices_to_flip = [ps for ps in self.pathology_volume.pathologySlices if getattr(ps, 'doVerticalFlip', False)]

        if not slices_to_flip:
            QMessageBox.information(self, "No Action", "No slices are marked for vertical flip.")
            return

        reply = QMessageBox.question(
            self, "Confirm Destructive Action",
            f"You are about to permanently overwrite {len(slices_to_flip)} image file(s) with their vertically flipped version.\n\n"
            "This action cannot be undone.\n\nAre you sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply != QMessageBox.StandardButton.Yes: return

        flipped_count, error_count, error_messages = 0, 0, []
        for ps in slices_to_flip:
            image_path = Path(ps.rgbImageFn)
            if not image_path.exists():
                error_count += 1
                error_messages.append(f"File not found: {image_path.name}")
                continue
            
            try:
                with Image.open(image_path) as img:
                    flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    flipped_img.save(image_path)
                ps.doVerticalFlip = False
                flipped_count += 1
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error flipping {image_path.name}: {e}")

        self.update_slices_display()

        summary_message = f"Successfully flipped and overwrote {flipped_count} image(s)."
        if error_count > 0:
            summary_message += f"\n\nFailed to process {error_count} image(s):\n" + "\n".join(error_messages)
            QMessageBox.critical(self, "Operation Finished with Errors", summary_message)
        else:
            QMessageBox.information(self, "Operation Successful", summary_message)

    def auto_match_and_apply_flips(self):
        """Automatically matches flip data to slices by filename and applies it."""
        if not self.pathology_volume or not self.flip_data:
            QMessageBox.warning(self, "Warning", "Load both pathology data and flip data first.")
            return

        if not hasattr(self.pathology_volume, 'pathologySlices'):
            QMessageBox.warning(self, "Warning", "No pathology slices found. Load JSON first.")
            return

        matches_found = 0
        try:
            for ps in self.pathology_volume.pathologySlices:
                slice_filename = os.path.basename(ps.rgbImageFn)
                flip_values = self.flip_data.get(slice_filename)
                
                if not flip_values:
                    base_name = Path(slice_filename).stem
                    for flip_filename, values in self.flip_data.items():
                        if Path(flip_filename).stem == base_name:
                            flip_values = values
                            break
                
                if flip_values:
                    if hasattr(ps, 'doVerticalFlip') and hasattr(ps, 'doHorizontalFlip'):
                        ps.doVerticalFlip = flip_values.get('vertical', False)
                        ps.doHorizontalFlip = flip_values.get('horizontal', False)
                        matches_found += 1
            
            self.update_slices_display()
            self.update_volume_info_display()

            QMessageBox.information(
                self, "Auto-Match Results",
                f"Successfully matched and applied flip data to {matches_found} out of {len(self.pathology_volume.pathologySlices)} slices."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while applying flip data: {e}")

    def apply_rotation_angles(self):
        """Apply all loaded rotation angles automatically"""
        if not self.rotation_angles_data:
            QMessageBox.warning(self, "Warning", "No rotation angles loaded. Import angles first.")
            return
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "No pathology volume loaded. Load JSON first.")
            return
        self.auto_match_filenames()

    def update_json_info_display(self):
        if not self.pathology_volume: return
        info_text = f"File: {os.path.basename(self.json_input.text())}\n"
        info_text += f"Number of slices: {self.pathology_volume.noSlices}\n"
        if hasattr(self.pathology_volume, 'regionIDs'):
            info_text += f"Regions: {', '.join(map(str, self.pathology_volume.regionIDs))}\n"
        self.json_info.setText(info_text)
            
    def update_slices_display(self):
        if not self.pathology_volume or not hasattr(self.pathology_volume, 'pathologySlices'):
            self.slices_table.setRowCount(0)
            return
        
        self.slices_table.blockSignals(True)
        try:
            self.slices_table.setRowCount(self.pathology_volume.noSlices)
            for i, ps in enumerate(self.pathology_volume.pathologySlices):
                thickness = getattr(ps, 'sliceThickness', 1.0)
                z_pos = getattr(ps, 'zPosition', 0.0)
                flip_v = getattr(ps, 'doVerticalFlip', False)
                flip_h = getattr(ps, 'doHorizontalFlip', False)
                rotate = getattr(ps, 'doRotate', 0.0)
                
                self.slices_table.setItem(i, 0, QTableWidgetItem(str(ps.id)))
                self.slices_table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
                self.slices_table.setItem(i, 2, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
                self.slices_table.setItem(i, 3, QTableWidgetItem(f"{thickness:.2f}"))
                self.slices_table.setItem(i, 4, QTableWidgetItem(f"{z_pos:.2f}"))
                self.slices_table.setItem(i, 5, QTableWidgetItem(str(int(flip_v))))
                self.slices_table.setItem(i, 6, QTableWidgetItem(str(int(flip_h))))
                self.slices_table.setItem(i, 7, QTableWidgetItem(f"{rotate:.1f}"))
                self.slices_table.setItem(i, 8, QTableWidgetItem("0"))
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error updating slice display: {e}")
        finally:
            self.slices_table.blockSignals(False)

    def on_slice_item_changed(self, item):
        if not self.pathology_volume: return
        row, col = item.row(), item.column()
        ps = self.pathology_volume.pathologySlices[row]
        try:
            if col == 3: ps.sliceThickness = float(item.text())
            elif col == 5: ps.doVerticalFlip = bool(int(item.text()))
            elif col == 6: ps.doHorizontalFlip = bool(int(item.text()))
            elif col == 7: ps.doRotate = float(item.text())
            self.update_volume_info_display()
        except (ValueError, TypeError):
            QMessageBox.warning(self, "Error", "Invalid input value.")
            self.update_slices_display()
            
    def edit_slice_dialog(self):
        current_row = self.slices_table.currentRow()
        if current_row < 0:
            return
        self.edit_selected_slice()
        
    def edit_selected_slice(self):
        current_row = self.slices_table.currentRow()
        if current_row < 0 or not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "Please select a slice to edit")
            return
        self.open_slice_editor(current_row)
        
    def open_slice_editor(self, slice_idx):
        if not self.pathology_volume or slice_idx >= len(self.pathology_volume.pathologySlices): return
        ps = self.pathology_volume.pathologySlices[slice_idx]
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Slice {slice_idx + 1}")
        dialog.setModal(True); dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        form_layout = QVBoxLayout()
        
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Slice Thickness (mm):"))
        thickness_spin = QSpinBox()
        thickness_spin.setRange(1, 100)
        thickness_spin.setValue(int(getattr(ps, 'sliceThickness', 1.0) * 10))
        thickness_layout.addWidget(thickness_spin)
        form_layout.addLayout(thickness_layout)
        
        v_flip_layout = QHBoxLayout()
        v_flip_layout.addWidget(QLabel("Vertical Flip:"))
        v_flip_cb = QCheckBox(); v_flip_cb.setChecked(getattr(ps, 'doVerticalFlip', False))
        v_flip_layout.addWidget(v_flip_cb); v_flip_layout.addStretch()
        form_layout.addLayout(v_flip_layout)
        
        h_flip_layout = QHBoxLayout()
        h_flip_layout.addWidget(QLabel("Horizontal Flip:"))
        h_flip_cb = QCheckBox(); h_flip_cb.setChecked(getattr(ps, 'doHorizontalFlip', False))
        h_flip_layout.addWidget(h_flip_cb); h_flip_layout.addStretch()
        form_layout.addLayout(h_flip_layout)
        
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation (degrees):"))
        rotation_spin = QSpinBox()
        rotation_spin.setRange(-180, 180)
        rotation_spin.setValue(int(getattr(ps, 'doRotate', 0.0)))
        rotation_layout.addWidget(rotation_spin)
        form_layout.addLayout(rotation_layout)
        
        z_pos_layout = QHBoxLayout()
        z_pos_layout.addWidget(QLabel("Z Position (mm):"))
        z_pos_label = QLabel(f"{getattr(ps, 'zPosition', 0.0):.2f}")
        z_pos_layout.addWidget(z_pos_label)
        z_pos_layout.addStretch()
        form_layout.addLayout(z_pos_layout)
        
        layout.addLayout(form_layout)
    
        button_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply"); cancel_btn = QPushButton("Cancel")
        
        def apply_changes():
            ps.sliceThickness = thickness_spin.value() / 10.0
            ps.doVerticalFlip = v_flip_cb.isChecked()
            ps.doHorizontalFlip = h_flip_cb.isChecked()
            ps.doRotate = float(rotation_spin.value())
            self.update_slices_display(); self.update_volume_info_display()
            dialog.accept()
        
        apply_btn.clicked.connect(apply_changes)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(apply_btn); button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def update_slice_thickness(self):
        current_row = self.slices_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a slice")
            return
        current_thickness = float(self.slices_table.item(current_row, 3).text()) if self.slices_table.item(current_row, 3) else 1.0
        thickness, ok = QInputDialog.getDouble(self, "Update Thickness", "Enter new thickness (mm):", current_thickness, 0.1, 10.0, 1)
        if ok:
            ps = self.pathology_volume.pathologySlices[current_row]
            ps.sliceThickness = thickness
            self.update_slices_display(); self.update_volume_info_display()

    def save_json(self):
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "No pathology volume loaded")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Pathology JSON", "", "JSON files (*.json)")
        if file_path:
            original_flips = []
            try:
                for ps in self.pathology_volume.pathologySlices:
                    original_flips.append(getattr(ps, 'doFlip', 0))
                    ps.doFlip = 1 if getattr(ps, 'doHorizontalFlip', False) else 0
                
                if hasattr(self.pathology_volume, 'saveJson'):
                    if self.pathology_volume.saveJson(file_path):
                        QMessageBox.information(self, "Success", "JSON file saved successfully!")
                    else:
                        QMessageBox.warning(self, "Warning", "Failed to save JSON file")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving JSON: {e}")
            finally:
                if original_flips:
                    for i, ps in enumerate(self.pathology_volume.pathologySlices):
                        ps.doFlip = original_flips[i]
                        
    def export_volume(self):
        if not self.loaded_volume:
            QMessageBox.warning(self, "Warning", "No volume available for export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Volume", "", "NRRD (*.nrrd);;NIfTI (*.nii);;MHD (*.mhd)")
        if file_path:
            try:
                sitk.WriteImage(self.loaded_volume, file_path)
                QMessageBox.information(self, "Success", "Volume exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export volume: {e}")
    
    def load_volume(self):
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "Load a JSON file first.")
            return
        try:
            self.loaded_volume = self.pathology_volume.loadRgbVolume()
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Volume", "Histological volume loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading volume: {e}")
         
    def load_mask(self):
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "Load a JSON file first.")
            return
        try:
            mask_id = self.mask_id_spin.value()
            self.loaded_mask = self.pathology_volume.loadMask(idxMask=mask_id)
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Mask", f"Mask (ID: {mask_id}) loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading mask: {e}")
        
    def send_to_viewer(self):
        if self.loaded_volume is not None:
            self.data_loaded.emit("histology", sitk.GetArrayFromImage(self.loaded_volume))
        if self.loaded_mask is not None:
            self.data_loaded.emit("segmentation", sitk.GetArrayFromImage(self.loaded_mask))
        QMessageBox.information(self, "Viewer", "Data sent to the medical viewer")
    
    def renumber_slices(self):
        for i in range(self.slices_table.rowCount()):
            self.slices_table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
        QMessageBox.information(self, "Renumbering", "Slices renumbered correctly.")
        
    def update_masks(self):
        QMessageBox.information(self, "Update", "Mask numbering updated.")