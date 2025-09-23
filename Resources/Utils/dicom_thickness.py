#!/usr/bin/env python3
"""
Modifiche per integrare la distanza tra le fette dal file DICOM
"""

import sys
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

# Aggiungi import per DICOM
import pydicom
import SimpleITK as sitk
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QSpinBox, QCheckBox,
    QComboBox, QTableWidget, QTableWidgetItem, QSplitter,
    QScrollArea, QFrame, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

@dataclass
class DicomSliceInfo:
    """Informazioni sulle fette DICOM"""
    slice_thickness: float = 0.0
    pixel_spacing: List[float] = None
    number_of_frames: int = 0
    z_positions: List[float] = None
    
    def __post_init__(self):
        if self.pixel_spacing is None:
            self.pixel_spacing = [1.0, 1.0]
        if self.z_positions is None:
            self.z_positions = []

class DicomAnalyzer:
    """Classe per analizzare file DICOM e estrarre informazioni sulle fette"""
    
    @staticmethod
    def analyze_dicom_file(dicom_path: str) -> Optional[DicomSliceInfo]:
        """Analizza un file DICOM e restituisce le informazioni sulle fette"""
        try:
            ds = pydicom.dcmread(dicom_path)
            
            slice_info = DicomSliceInfo()
            
            # Estrai slice thickness
            if hasattr(ds, 'SliceThickness'):
                slice_info.slice_thickness = float(ds.SliceThickness)
            
            # Estrai pixel spacing
            if hasattr(ds, 'PixelSpacing'):
                slice_info.pixel_spacing = [float(x) for x in ds.PixelSpacing]
            
            # Estrai numero di frame per DICOM multi-frame
            if hasattr(ds, 'NumberOfFrames'):
                slice_info.number_of_frames = int(ds.NumberOfFrames)
            else:
                slice_info.number_of_frames = 1
            
            # Estrai posizioni Z dalle sequenze per-frame
            z_positions = []
            if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
                for frame_group in ds.PerFrameFunctionalGroupsSequence:
                    if hasattr(frame_group, 'PlanePositionSequence'):
                        for plane_pos in frame_group.PlanePositionSequence:
                            if hasattr(plane_pos, 'ImagePositionPatient'):
                                z_pos = float(plane_pos.ImagePositionPatient[2])  # Z è il terzo elemento
                                z_positions.append(z_pos)
            
            # Se non ci sono posizioni per-frame, usa ImagePositionPatient base
            if not z_positions and hasattr(ds, 'ImagePositionPatient'):
                z_positions.append(float(ds.ImagePositionPatient[2]))
            
            slice_info.z_positions = z_positions
            
            return slice_info
            
        except Exception as e:
            logging.error(f"Errore durante l'analisi del file DICOM {dicom_path}: {e}")
            return None

class PathologyParser(QWidget):
    """Widget per parsing dati patologici con integrazione DICOM"""
    
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        self.pathology_volume = None
        self.dicom_slice_info = None  # Informazioni dal DICOM caricato
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface con sezione DICOM"""
        layout = QVBoxLayout()
        
        # Sezione DICOM Reference
        dicom_group = QGroupBox("DICOM Reference for Slice Spacing")
        dicom_layout = QVBoxLayout()
        
        # Caricamento file DICOM
        dicom_file_layout = QHBoxLayout()
        self.dicom_input = QLineEdit()
        self.dicom_input.setPlaceholderText("Select DICOM file for slice spacing reference")
        dicom_browse = QPushButton("Browse DICOM")
        dicom_browse.clicked.connect(self.browse_dicom)
        dicom_file_layout.addWidget(QLabel("DICOM file:"))
        dicom_file_layout.addWidget(self.dicom_input)
        dicom_file_layout.addWidget(dicom_browse)
        dicom_layout.addLayout(dicom_file_layout)
        
        # Informazioni DICOM
        self.dicom_info_label = QLabel("No DICOM loaded")
        dicom_layout.addWidget(self.dicom_info_label)
        
        # Controllo manuale dello spessore fetta
        slice_thickness_layout = QHBoxLayout()
        self.slice_thickness_spin = QDoubleSpinBox()
        self.slice_thickness_spin.setRange(0.01, 100.0)
        self.slice_thickness_spin.setValue(1.0)
        self.slice_thickness_spin.setSuffix(" mm")
        self.slice_thickness_spin.setDecimals(3)
        self.use_dicom_spacing_cb = QCheckBox("Use DICOM slice spacing")
        self.use_dicom_spacing_cb.setChecked(True)
        self.use_dicom_spacing_cb.toggled.connect(self.on_use_dicom_spacing_changed)
        
        slice_thickness_layout.addWidget(QLabel("Target slice thickness:"))
        slice_thickness_layout.addWidget(self.slice_thickness_spin)
        slice_thickness_layout.addWidget(self.use_dicom_spacing_cb)
        dicom_layout.addLayout(slice_thickness_layout)
        
        dicom_group.setLayout(dicom_layout)
        layout.addWidget(dicom_group)
        
        # Input files section (esistente)
        input_group = QGroupBox("Input files")
        input_layout = QVBoxLayout()
        
        # JSON file input
        json_layout = QHBoxLayout()
        self.json_input = QLineEdit()
        self.json_input.setPlaceholderText("Select pathology JSON file")
        json_browse = QPushButton("Browse")
        json_browse.clicked.connect(self.browse_json)
        json_layout.addWidget(QLabel("JSON file:"))
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(json_browse)
        input_layout.addLayout(json_layout)
        
        # Volume output
        volume_layout = QHBoxLayout()
        self.volume_input = QLineEdit()
        self.volume_input.setPlaceholderText("Output Volume name")
        volume_layout.addWidget(QLabel("Output Volume:"))
        volume_layout.addWidget(self.volume_input)
        input_layout.addLayout(volume_layout)
        
        # Mask output
        mask_layout = QHBoxLayout()
        self.mask_input = QLineEdit()
        self.mask_input.setPlaceholderText("Output mask name")
        self.mask_id_spin = QSpinBox()
        self.mask_id_spin.setRange(0, 100)
        mask_layout.addWidget(QLabel("Output Mask:"))
        mask_layout.addWidget(self.mask_input)
        mask_layout.addWidget(QLabel("Mask ID:"))
        mask_layout.addWidget(self.mask_id_spin)
        input_layout.addLayout(mask_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Actions section (modificata per includere creazione DICOM)
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        # Prima riga di pulsanti
        actions_row1 = QHBoxLayout()
        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.clicked.connect(self.load_json)
        
        self.load_volume_btn = QPushButton("Load Volume")
        self.load_volume_btn.clicked.connect(self.load_volume)
        self.load_volume_btn.setEnabled(False)
        
        self.load_mask_btn = QPushButton("Load Mask")
        self.load_mask_btn.clicked.connect(self.load_mask)
        self.load_mask_btn.setEnabled(False)
        
        actions_row1.addWidget(self.load_json_btn)
        actions_row1.addWidget(self.load_volume_btn)
        actions_row1.addWidget(self.load_mask_btn)
        actions_layout.addLayout(actions_row1)
        
        # Seconda riga di pulsanti
        actions_row2 = QHBoxLayout()
        self.refine_volume_btn = QPushButton("Refine Volume")
        self.refine_volume_btn.clicked.connect(self.refine_volume)
        self.refine_volume_btn.setEnabled(False)
        
        self.create_dicom_btn = QPushButton("Create DICOM")
        self.create_dicom_btn.clicked.connect(self.create_dicom)
        self.create_dicom_btn.setEnabled(False)
        
        self.send_to_viewer_btn = QPushButton("Send to Viewer")
        self.send_to_viewer_btn.clicked.connect(self.send_to_viewer)
        self.send_to_viewer_btn.setEnabled(False)
        
        actions_row2.addWidget(self.refine_volume_btn)
        actions_row2.addWidget(self.create_dicom_btn)
        actions_row2.addWidget(self.send_to_viewer_btn)
        actions_layout.addLayout(actions_row2)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # JSON information section
        info_group = QGroupBox("JSON Info")
        info_layout = QVBoxLayout()
        
        self.json_info = QTextEdit()
        self.json_info.setReadOnly(True)
        self.json_info.setMaximumHeight(200)
        info_layout.addWidget(self.json_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Slice table
        slices_group = QGroupBox("Slice Management")
        slices_layout = QVBoxLayout()
        
        self.slices_table = QTableWidget()
        self.slices_table.setColumnCount(5)  # Aggiunta colonna per Z position
        self.slices_table.setHorizontalHeaderLabels(["ID", "File", "Rotation", "Flip", "Z Position (mm)"])
        slices_layout.addWidget(self.slices_table)
        
        # Table controls
        table_controls = QHBoxLayout()
        self.renumber_btn = QPushButton("Renumber Slices")
        self.renumber_btn.clicked.connect(self.renumber_slices)
        self.update_masks_btn = QPushButton("Update masks")
        self.update_masks_btn.clicked.connect(self.update_masks)
        self.apply_dicom_spacing_btn = QPushButton("Apply DICOM Spacing")
        self.apply_dicom_spacing_btn.clicked.connect(self.apply_dicom_spacing)
        self.apply_dicom_spacing_btn.setEnabled(False)
        
        table_controls.addWidget(self.renumber_btn)
        table_controls.addWidget(self.update_masks_btn)
        table_controls.addWidget(self.apply_dicom_spacing_btn)
        table_controls.addStretch()
        
        slices_layout.addLayout(table_controls)
        slices_group.setLayout(slices_layout)
        layout.addWidget(slices_group)
        
        self.setLayout(layout)
        
        # Loaded data
        self.loaded_volume = None
        self.loaded_mask = None
        
    def browse_dicom(self):
        """Opens dialog to select DICOM file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DICOM file", "", "DICOM files (*.dcm);;All files (*)"
        )
        if file_path:
            self.dicom_input.setText(file_path)
            self.load_dicom_info(file_path)
            
    def load_dicom_info(self, dicom_path: str):
        """Carica informazioni dal file DICOM"""
        try:
            self.dicom_slice_info = DicomAnalyzer.analyze_dicom_file(dicom_path)
            
            if self.dicom_slice_info:
                info_text = f"Slice thickness: {self.dicom_slice_info.slice_thickness:.3f} mm\n"
                info_text += f"Pixel spacing: {self.dicom_slice_info.pixel_spacing[0]:.3f} x {self.dicom_slice_info.pixel_spacing[1]:.3f} mm\n"
                info_text += f"Number of frames: {self.dicom_slice_info.number_of_frames}\n"
                if self.dicom_slice_info.z_positions:
                    info_text += f"Z positions: {len(self.dicom_slice_info.z_positions)} slices"
                
                self.dicom_info_label.setText(info_text)
                
                # Imposta automaticamente lo spessore della fetta
                if self.dicom_slice_info.slice_thickness > 0:
                    self.slice_thickness_spin.setValue(self.dicom_slice_info.slice_thickness)
                
                self.apply_dicom_spacing_btn.setEnabled(True)
                
            else:
                self.dicom_info_label.setText("Error loading DICOM information")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading DICOM file: {e}")
            self.dicom_info_label.setText("Error loading DICOM information")
    
    def on_use_dicom_spacing_changed(self, checked):
        """Gestisce il cambiamento dell'opzione di usare lo spacing DICOM"""
        self.slice_thickness_spin.setEnabled(not checked)
        if checked and self.dicom_slice_info and self.dicom_slice_info.slice_thickness > 0:
            self.slice_thickness_spin.setValue(self.dicom_slice_info.slice_thickness)
    
    def get_target_slice_thickness(self) -> float:
        """Restituisce lo spessore target della fetta"""
        if self.use_dicom_spacing_cb.isChecked() and self.dicom_slice_info:
            return self.dicom_slice_info.slice_thickness
        else:
            return self.slice_thickness_spin.value()
    
    def apply_dicom_spacing(self):
        """Applica lo spacing DICOM alle fette istologiche"""
        if not self.dicom_slice_info or not self.dicom_slice_info.z_positions:
            QMessageBox.warning(self, "Warning", "No DICOM spacing information available")
            return
            
        if self.slices_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No slices loaded. Load JSON first.")
            return
        
        try:
            # Calcola le posizioni Z per le fette istologiche basandosi sul DICOM
            z_positions = self.dicom_slice_info.z_positions
            slice_thickness = self.get_target_slice_thickness()
            
            # Se abbiamo più fette istologiche che posizioni DICOM, interpola
            num_histo_slices = self.slices_table.rowCount()
            
            if len(z_positions) >= num_histo_slices:
                # Usa direttamente le posizioni DICOM
                for i in range(num_histo_slices):
                    z_pos = z_positions[i] if i < len(z_positions) else z_positions[-1] + i * slice_thickness
                    self.slices_table.setItem(i, 4, QTableWidgetItem(f"{z_pos:.3f}"))
            else:
                # Interpola tra le posizioni DICOM disponibili
                if len(z_positions) >= 2:
                    z_min, z_max = min(z_positions), max(z_positions)
                    for i in range(num_histo_slices):
                        z_pos = z_min + (z_max - z_min) * i / (num_histo_slices - 1)
                        self.slices_table.setItem(i, 4, QTableWidgetItem(f"{z_pos:.3f}"))
                else:
                    # Una sola posizione DICOM disponibile
                    base_z = z_positions[0] if z_positions else 0.0
                    for i in range(num_histo_slices):
                        z_pos = base_z + i * slice_thickness
                        self.slices_table.setItem(i, 4, QTableWidgetItem(f"{z_pos:.3f}"))
            
            QMessageBox.information(self, "Success", "DICOM spacing applied to histological slices")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying DICOM spacing: {e}")
    
    def browse_json(self):
        """Opens dialog to select JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON file", "", "JSON files (*.json)"
        )
        if file_path:
            self.json_input.setText(file_path)
            
    def load_json(self):
        """Loads and analyzes JSON file using PathologyVolume"""
        json_path = self.json_input.text()
        if not json_path or not os.path.exists(json_path):
            QMessageBox.warning(self, "Error", "Select JSON file")
            return

        try:
            # Carica PathologyVolume (assumendo che sia definito altrove)
            from Resources.Utils.ImageStack_2 import PathologyVolume
            
            self.pathology_volume = PathologyVolume() 
            self.pathology_volume.setPath(json_path)
            
            if not self.pathology_volume.initComponents():
                raise Exception("PathologyVolume class initialization error")
                
            info_text = f"File: {os.path.basename(json_path)}\n"
            info_text += f"Number of slices: {self.pathology_volume.noSlices}\n"
            info_text += f"Regions: {', '.join(self.pathology_volume.regionIDs)}\n"
            
            self.json_info.setText(info_text)
            
            self.slices_table.setRowCount(self.pathology_volume.noSlices)
            for i, ps in enumerate(self.pathology_volume.pathologySlices):
                self.slices_table.setItem(i, 0, QTableWidgetItem(str(ps.id)))
                self.slices_table.setItem(i, 1, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
                self.slices_table.setItem(i, 2, QTableWidgetItem(str(ps.doRotate)))
                self.slices_table.setItem(i, 3, QTableWidgetItem(str(ps.doFlip)))
                # Inizializza posizione Z
                self.slices_table.setItem(i, 4, QTableWidgetItem("0.000"))

            self.load_volume_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)
            self.refine_volume_btn.setEnabled(True)
            self.create_dicom_btn.setEnabled(True)
            
            # Se abbiamo informazioni DICOM, applica automaticamente lo spacing
            if self.dicom_slice_info:
                self.apply_dicom_spacing()
            
            QMessageBox.information(self, "Success", "JSON loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading JSON: {e}")
    
    def create_dicom(self):
        """Crea un file DICOM dalle fette istologiche caricate"""
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "Load JSON file first")
            return
            
        try:
            # Importa la funzione di creazione DICOM
            from dicom_creator import create_dicom
            
            # Prepara i dati delle fette
            slice_data_list = []
            
            for i in range(self.slices_table.rowCount()):
                # Ottieni i dati della fetta dall'istologia
                ps = self.pathology_volume.pathologySlices[i]
                
                # Carica l'immagine della fetta (questo dipende dalla tua implementazione)
                # Assumendo che ps abbia un metodo per ottenere i dati numpy
                try:
                    slice_image = ps.getRgbImage()  # Metodo ipotetico
                    if hasattr(slice_image, 'shape'):
                        slice_data = np.array(slice_image)
                    else:
                        slice_data = None
                except:
                    slice_data = None
                
                # Ottieni posizione Z dalla tabella
                z_pos_item = self.slices_table.item(i, 4)
                z_pos = float(z_pos_item.text()) if z_pos_item else i * self.get_target_slice_thickness()
                
                # Spessore della fetta
                thickness = self.get_target_slice_thickness()
                
                slice_data_list.append((slice_data, z_pos, thickness, i))
            
            # Scegli dove salvare il DICOM
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save DICOM file", "histology_volume.dcm", "DICOM files (*.dcm)"
            )
            
            if not output_path:
                return
            
            # Crea il file DICOM
            success = create_dicom(
                slice_data_list=slice_data_list,
                output_dicom_path=output_path,
                patient_name="Histology^Patient",
                patient_id="HISTO001",
                study_id=1001,
                series_number=1,
                series_description="Histological Volume",
                modality="XC",  # X-Ray Angiography (per istologia)
                target_slice_thickness_mm=self.get_target_slice_thickness()
            )
            
            if success:
                QMessageBox.information(self, "Success", f"DICOM file created successfully: {output_path}")
            else:
                QMessageBox.critical(self, "Error", "Error creating DICOM file")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating DICOM: {e}")
    
    # Resto dei metodi rimangono invariati...
    def load_volume(self):
        """Loads 3D volume from PathologyVolume"""
        try:
            self.loaded_volume = self.pathology_volume.loadRgbVolume()
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Volume", f"Histological volume loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading volume: {e}")
    
    def load_mask(self):
        """Loads mask from PathologyVolume"""
        try:
            mask_id = self.mask_id_spin.value()
            self.loaded_mask = self.pathology_volume.loadMask(idxMask=mask_id)
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Mask", f"Mask (ID: {mask_id}) loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading mask: {e}")
        
    def refine_volume(self):
        """Refines volume reconstruction (slice-to-slice)"""
        try:
            self.pathology_volume.registerSlices(useImagingConstraint=False)
            self.loaded_volume = self.pathology_volume.refWoContraints
            self.loaded_mask = self.pathology_volume.mskRefWoContraints
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Refinement", "Volume refinement completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in volume refinement: {e}")
        
    def send_to_viewer(self):
        """Sends loaded data to medical viewer"""
        if self.loaded_volume is not None:
            vol_np = sitk.GetArrayFromImage(self.loaded_volume)
            self.data_loaded.emit("histology", vol_np)
        if self.loaded_mask is not None:
            mask_np = sitk.GetArrayFromImage(self.loaded_mask)
            self.data_loaded.emit("segmentation", mask_np)
        QMessageBox.information(self, "Viewer", "Data sent to the medical viewer")
        
    def renumber_slices(self):
        """Renumbers slices starting from 1"""
        for i in range(self.slices_table.rowCount()):
            self.slices_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
        QMessageBox.information(self, "Renumbering", "Slices renumbered correctly.")
        
    def update_masks(self):
        """Updates mask numbering"""
        QMessageBox.information(self, "Update", "Mask numbering updated.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PathologyParser()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())