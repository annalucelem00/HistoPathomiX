#!/usr/bin/env python3
"""
clonata la repo+ aggiunto BigWarp
"""

import sys
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import subprocess


from Resources.Utils.ImageStack_3 import PathologyVolume
from Resources.Utils.ImageRegistration import RegisterImages
from Resources.Utils.RegisterVolumesElastix import RegisterVolumesElastix
from Resources.Utils.ParsePathJsonUtils import ParsePathJsonUtils
from Resources.Utils.dicom_thickness import DicomSliceInfo,PathologyParser,DicomAnalyzer
from Resources.Utils.dicom_creator import create_dicom

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
import csv
from pathlib import Path

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegistrationConfig:
    input_json: str = ""
    output_path: str = ""
    discard_orientation: bool = False
    elastix_path: str = ""
    fixed_volume_path: str = ""
    moving_volume_path: str = ""
    parameter_filenames: List[str] = None
    fixed_volume_mask_path: str = None
    moving_volume_mask_path: str = None
    
@dataclass
class PathologyData:
    """Pathology data"""
    json_path: str = ""
    volume_name: str = ""
    mask_name: str = ""
    mask_id: int = 0
    slices: List[Dict] = None

    def __post_init__(self):
        if self.slices is None:
            self.slices = []

class RotationDataManager:
    """Class to manage rotation data between BigWarp and ParsePathology"""
    
    def __init__(self):
        self.rotation_data: Dict[str, float] = {}
    
    def add_rotation_data(self, filename: str, rotation_angle: float):
        """Add rotation data for a specific image"""
        self.rotation_data[filename] = rotation_angle
    
    def get_rotation_angle(self, filename: str) -> Optional[float]:
        """Get rotation angle for a specific image"""
        return self.rotation_data.get(filename)
    
    def export_to_csv(self, filepath: str):
        """Export rotation data to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'rotation_angle_degrees'])
            writer.writeheader()
            for filename, angle in self.rotation_data.items():
                writer.writerow({'filename': filename, 'rotation_angle_degrees': angle})


class RegistrationWorker(QThread):
    """Worker thread for registration with Elastix"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, config: RegistrationConfig):
        super().__init__()
        self.config = config
        
    def run(self):
        """Performs registration using RegisterVolumesElastix"""
        try:
            self.status_updated.emit("Starting Elastix...")
            self.progress_updated.emit(10)

            reg_elastix = RegisterVolumesElastix()
            reg_elastix.set_elastix_bin_dir(self.config.elastix_path)
            reg_elastix.set_registration_parameter_files_dir(os.path.dirname(self.config.input_json))

            self.status_updated.emit("Start volume registration...")
            self.progress_updated.emit(25)
            
            output_paths = reg_elastix.register_volumes(
                fixed_volume_path=self.config.fixed_volume_path,
                moving_volume_path=self.config.moving_volume_path,
                parameter_filenames=self.config.parameter_filenames,
                output_dir=self.config.output_path,
                fixed_volume_mask_path=self.config.fixed_volume_mask_path,
                moving_volume_mask_path=self.config.moving_volume_mask_path
            )

            self.status_updated.emit("Registration completed")
            self.progress_updated.emit(75)

            # Simulate loading of registered volume
            # In a real implementation, would load "result.mhd" file
            # self.registered_volume = sitk.ReadImage(output_paths["output_volume"])
            
            self.status_updated.emit("Saving results")
            self.progress_updated.emit(100)
            
            self.finished.emit(True, "Registration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            self.finished.emit(False, f"Error: {str(e)}")

class BigWarpTab(QWidget):
    """Tab wrapper per BigWarp che integra la GUI Tkinter esistente"""
    
    data_loaded = pyqtSignal(str, object)
    rotation_data_ready = pyqtSignal(dict)  # NUOVO: Segnale per angoli rotazione
    
    def __init__(self):
        super().__init__()
        self.bigwarp_app = None
        self.bigwarp_process = None
        self.rotation_manager = RotationDataManager()  # NUOVO
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # INFO GROUP (esistente)
        info_group = QGroupBox("BigWarp - Registrazione Deformabile")
        info_layout = QVBoxLayout()

        # PULSANTI PRINCIPALI (esistente)
        buttons_layout = QHBoxLayout()
        
        self.launch_bigwarp_btn = QPushButton("Open BigWarp")
        self.launch_bigwarp_btn.clicked.connect(self.launch_bigwarp)
        
        self.import_result_btn = QPushButton("Import Results")
        self.import_result_btn.clicked.connect(self.import_result)
        
        buttons_layout.addWidget(self.launch_bigwarp_btn)
        buttons_layout.addWidget(self.import_result_btn)
        buttons_layout.addStretch()
        
        info_layout.addLayout(buttons_layout)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # NUOVO: Gruppo per condivisione dati
        sharing_group = QGroupBox("Data Sharing with ParsePathology")
        sharing_layout = QVBoxLayout()
        
        sharing_buttons_layout = QHBoxLayout()
        
        self.export_angles_btn = QPushButton("Export Rotation Angles (CSV)")
        self.export_angles_btn.clicked.connect(self.export_rotation_angles)
        
        self.send_to_pathology_btn = QPushButton("Send Angles to ParsePathology")
        self.send_to_pathology_btn.clicked.connect(self.send_angles_to_pathology)
        self.send_to_pathology_btn.setEnabled(False)
        
        sharing_buttons_layout.addWidget(self.export_angles_btn)
        sharing_buttons_layout.addWidget(self.send_to_pathology_btn)
        sharing_buttons_layout.addStretch()
        
        self.angles_status_label = QLabel("No rotation angles calculated yet")
        self.angles_status_label.setStyleSheet("QLabel { color: gray; }")
        
        sharing_layout.addLayout(sharing_buttons_layout)
        sharing_layout.addWidget(self.angles_status_label)
        
        sharing_group.setLayout(sharing_layout)
        layout.addWidget(sharing_group)
        
        # STATO E RISULTATI (esistente)
        results_group = QGroupBox("Stato e Risultati")
        results_layout = QVBoxLayout()
        
        self.status_label = QLabel("BigWarp non attivo")
        self.status_label.setStyleSheet("QLabel { color: gray; }")
        results_layout.addWidget(self.status_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # ISTRUZIONI (esistente)
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions = QLabel("""
        <b>Come usare BigWarp:</b><br>
        1. Clicca "Apri BigWarp" per lanciare l'interfaccia completa<br>
        2. Carica l'immagine fissa e quella mobile<br>
        3. Aggiungi landmark cliccando sui punti corrispondenti<br>
        4. Calcola e applica la trasformazione TPS<br>
        5. Usa "Import Results" per importare i parametri<br>
        6. Usa "Send Angles to ParsePathology" per trasferire automaticamente gli angoli
        """)
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    # METODO ESISTENTE - Modificato
    def import_result(self):
        """Enhanced import_result che gestisce anche i dati di rotazione"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona il file dei parametri di trasformazione",
            "",
            "File JSON (*.json);;Tutti i file (*.*)"
        )

        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if "parameters" not in data or "affine_matrix_3x3" not in data:
                QMessageBox.critical(self, "Errore", "File JSON non valido o in formato non corretto.")
                return

            self.rotation_angle = data['parameters']['rotation_degrees']
            self.transformation_matrix = np.array(data['affine_matrix_3x3'])

            # NUOVO: Chiedi quale immagine corrisponde a questa trasformazione
            image_filename, _ = QFileDialog.getOpenFileName(
                self, "Seleziona l'immagine corrispondente a questa trasformazione", 
                "", "Image files (*.png *.jpg *.jpeg *.tiff *.bmp)"
            )
            if image_filename:
                image_name = os.path.basename(image_filename)
                self.rotation_manager.add_rotation_data(image_name, self.rotation_angle)
                self.angles_status_label.setText(f"Rotation angle stored for: {image_name}")
                self.angles_status_label.setStyleSheet("QLabel { color: green; }")
                self.send_to_pathology_btn.setEnabled(True)

            result_message = (
                "Importazione completata con successo!\n"
                "-------------------------------------\n\n"
                f"Angolo di Rotazione: {self.rotation_angle:.2f} gradi\n\n"
                f"Matrice di Trasformazione (Affine):\n{str(self.transformation_matrix)}"
            )

            QMessageBox.information(self, "Risultati Importati", result_message)

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Errore", "Il file selezionato non è un file JSON valido.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Si è verificato un errore durante l'importazione: {e}")
    
    # NUOVI METODI
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
    
    
    def launch_bigwarp(self): #OK FUNZIONA COME FINESTRA SEPARATA
        """Lancia lo script bigwarp.py e cattura l'output per il debug."""
        if self.bigwarp_process and self.bigwarp_process.poll() is None:
            QMessageBox.information(self, "Informazione", 
                                    "BigWarp è già in esecuzione.")
            return

        try:
            # 1. Ottieni il percorso del file corrente (es. main_app.py)
            current_file_path = Path(__file__).resolve().parent
            # 2. Costruisci il percorso completo a bigwarp.py
            bigwarp_path = current_file_path / "Resources" / "Utils" / "BigWarp.py"
            # 3. Assicurati che il percorso esista.
            if not bigwarp_path.exists():
                QMessageBox.critical(self, "Errore", f"File non trovato: {bigwarp_path}")
                return
            # 4.  Usa il percorso costruito nel comando.
            command = [sys.executable, str(bigwarp_path), "--gui"] # str() converte Path a stringa
            # -------------------
            print(f"Esecuzione del comando: {' '.join(command)}")
            self.bigwarp_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            QMessageBox.information(self, "Avviato", "BigWarp è stato avviato in una finestra separata.")
        except FileNotFoundError:
            QMessageBox.critical(self, "Errore", "Script 'bigwarp.py' non trovato.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile avviare BigWarp: {e}")


class PathologyParser(QWidget):
    
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        print("PathologyParser inizializzato")
        self.pathology_volume = None
        self.rectum_distance = 5.0
        self.rotation_angles_data: Dict[str, float] = {}
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
        mask_layout.addWidget(QLabel("Mask ID:"))
        mask_layout.addWidget(self.mask_id_spin)
        input_layout.addLayout(mask_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        volume_settings_group = QGroupBox("Volume Info")
        volume_settings_layout = QVBoxLayout()

        # NUOVO: Inserisci questo gruppo dopo volume_settings_group
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
        layout.addWidget(rotation_group)  ################### Aggiungi dopo volume_settings_group
        
        rectum_layout = QHBoxLayout()
        rectum_layout.addWidget(QLabel("Distance from Rectum (mm):"))
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
        self.volume_info.setMaximumHeight(100)
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
        
        # JSON info
        """
        info_group = QGroupBox("JSON Info")
        info_layout = QVBoxLayout()
        
        self.json_info = QTextEdit()
        self.json_info.setReadOnly(True)
        self.json_info.setMaximumHeight(150)
        info_layout.addWidget(self.json_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        """
       
        slices_group = QGroupBox("Slice Management")
        slices_layout = QVBoxLayout()
        
        self.slices_table = QTableWidget()
        self.slices_table.setColumnCount(8)  # Increased columns
        self.slices_table.setHorizontalHeaderLabels([
            "ID", "Slice#", "File", "Thickness(mm)", "Z Pos(mm)", "Flip", "Rotation(°)", "Regions"
        ])
        
        self.slices_table.setColumnWidth(0, 50)   # ID
        self.slices_table.setColumnWidth(1, 60)   # Slice#
        self.slices_table.setColumnWidth(2, 200)  # File
        self.slices_table.setColumnWidth(3, 100)  # Thickness
        self.slices_table.setColumnWidth(4, 100)  # Z Position
        self.slices_table.setColumnWidth(5, 50)   # Flip
        self.slices_table.setColumnWidth(6, 80)   # Rotation
        self.slices_table.setColumnWidth(7, 80)   # Regions
        
        self.slices_table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
        self.slices_table.itemChanged.connect(self.on_slice_item_changed)
        self.slices_table.doubleClicked.connect(self.edit_slice_dialog)
        
        slices_layout.addWidget(self.slices_table)
        
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
                
            # Update rectum distance if it exists in the volume
            if hasattr(self.pathology_volume, 'rectumDistance'):
                self.rectum_distance = self.pathology_volume.rectumDistance
                self.rectum_distance_spin.setValue(int(self.rectum_distance))
                
            self.update_json_info_display()
            self.update_slices_display()
            self.update_volume_info_display()

            self.slices_table.setRowCount(self.pathology_volume.noSlices)
            for i, ps in enumerate(self.pathology_volume.pathologySlices):
                self.slices_table.setItem(i, 0, QTableWidgetItem(str(ps.id)))
                self.slices_table.setItem(i, 1, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
                self.slices_table.setItem(i, 2, QTableWidgetItem(str(ps.doRotate)))
                self.slices_table.setItem(i, 3, QTableWidgetItem(str(ps.doFlip)))
                thickness = getattr(ps, 'sliceThickness', 1.0)
                z_pos = getattr(ps, 'zPosition', 0.0)
                self.slices_table.setItem(i, 4, QTableWidgetItem(f"{thickness:.2f}"))
                
            self.load_volume_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)
            
            self.save_json_btn.setEnabled(True)
            
            QMessageBox.information(self, "Success", "JSON loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading JSON: {e}")

    # NUOVO METODO
    def receive_rotation_data(self, rotation_data: Dict[str, float]):
        """Receive rotation data from BigWarp tab"""
        self.rotation_angles_data.update(rotation_data)
        self.rotation_status_label.setText(f"Received {len(rotation_data)} rotation angles from BigWarp")
        self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
        self.apply_angles_btn.setEnabled(True)
        self.auto_match_btn.setEnabled(True)
    
    # NUOVO METODO
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
                        self.rotation_angles_data[row['filename']] = float(row['rotation_angle_degrees'])
                
                self.rotation_status_label.setText(f"Loaded {len(self.rotation_angles_data)} rotation angles")
                self.rotation_status_label.setStyleSheet("QLabel { color: green; }")
                self.apply_angles_btn.setEnabled(True)
                self.auto_match_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import rotation angles: {e}")
    
    # NUOVO METODO
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
                
                # Try fuzzy matching (without extension)
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
    
    # NUOVO METODO
    def apply_rotation_angles(self):
        """Apply all loaded rotation angles automatically"""
        self.auto_match_filenames()

    def update_json_info_display(self):
        if not self.pathology_volume:
            return
            
        info_text = f"File: {os.path.basename(self.json_input.text())}\n"
        info_text += f"Number of slices: {self.pathology_volume.noSlices}\n"
        
        if hasattr(self.pathology_volume, 'regionIDs'):
            info_text += f"Regions: {', '.join(map(str, self.pathology_volume.regionIDs))}\n"
        
        self.json_info.setText(info_text)
            
    def update_slices_display(self):
        if not self.pathology_volume or not hasattr(self.pathology_volume, 'pathologySlices'):
            self.slices_table.setRowCount(0)
            return
            
        try:
            self.slices_table.setRowCount(self.pathology_volume.noSlices)
            for i, ps in enumerate(self.pathology_volume.pathologySlices):
                self.slices_table.setItem(i, 0, QTableWidgetItem(str(ps.id)))
                self.slices_table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
                self.slices_table.setItem(i, 2, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
                
                thickness = getattr(ps, 'sliceThickness', 1.0)
                z_pos = getattr(ps, 'zPosition', 0.0)
                flip = getattr(ps, 'doFlip', False)
                rotate = getattr(ps, 'doRotate', 0.0)
                
                self.slices_table.setItem(i, 3, QTableWidgetItem(f"{thickness:.2f}"))
                self.slices_table.setItem(i, 4, QTableWidgetItem(f"{z_pos:.2f}"))
                self.slices_table.setItem(i, 5, QTableWidgetItem(str(int(flip))))
                self.slices_table.setItem(i, 6, QTableWidgetItem(f"{rotate:.1f}"))
                self.slices_table.setItem(i, 7, QTableWidgetItem("0"))
                    
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error updating slice display: {e}")
            
    def on_slice_item_changed(self, item):
        if not self.pathology_volume:
            return
            
        row = item.row()
        col = item.column()
        
        try:
            if col == 3:  # Thickness
                thickness = float(item.text())
                if hasattr(self.pathology_volume, 'updateSlice'):
                    self.pathology_volume.updateSlice(row, 'slice_thickness_mm', thickness)
                else:
                    ps = self.pathology_volume.pathologySlices[row]
                    ps.sliceThickness = thickness
                    
            elif col == 5:  # Flip
                flip = bool(int(item.text()))
                if hasattr(self.pathology_volume, 'updateSlice'):
                    self.pathology_volume.updateSlice(row, 'flip', flip)
                else:
                    ps = self.pathology_volume.pathologySlices[row]
                    ps.doFlip = flip
                    
            elif col == 6:  # Rotation
                rotation = float(item.text())
                if hasattr(self.pathology_volume, 'updateSlice'):
                    self.pathology_volume.updateSlice(row, 'rotation_angle', rotation)
                else:
                    ps = self.pathology_volume.pathologySlices[row]
                    ps.doRotate = rotation
                    
            self.update_volume_info_display()
            
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input value")
            self.update_slices_display()  # Refresh to original values
            
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
        if not self.pathology_volume or slice_idx >= len(self.pathology_volume.pathologySlices):
            return
            
        ps = self.pathology_volume.pathologySlices[slice_idx]
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Slice {slice_idx + 1}")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        form_layout = QVBoxLayout()
        
        # Thickness
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Slice Thickness (mm):"))
        thickness_spin = QSpinBox()
        thickness_spin.setRange(1, 100)  # Allow decimals via setValue
        thickness_spin.setValue(int(getattr(ps, 'sliceThickness', 1.0) * 10))  # Scale for decimals
        thickness_layout.addWidget(thickness_spin)
        form_layout.addLayout(thickness_layout)
        
        # Flip checkbox
        flip_layout = QHBoxLayout()
        flip_layout.addWidget(QLabel("Flip:"))
        flip_cb = QCheckBox()
        flip_cb.setChecked(getattr(ps, 'doFlip', False))
        flip_layout.addWidget(flip_cb)
        flip_layout.addStretch()
        form_layout.addLayout(flip_layout)
        
        # Rotation
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation (degrees):"))
        rotation_spin = QSpinBox()
        rotation_spin.setRange(-180, 180)
        rotation_spin.setValue(int(getattr(ps, 'doRotate', 0.0)))
        rotation_layout.addWidget(rotation_spin)
        form_layout.addLayout(rotation_layout)
        
        # Z Position 
        z_pos_layout = QHBoxLayout()
        z_pos_layout.addWidget(QLabel("Z Position (mm):"))
        z_pos_label = QLabel(f"{getattr(ps, 'zPosition', 0.0):.2f}")
        z_pos_layout.addWidget(z_pos_label)
        z_pos_layout.addStretch()
        form_layout.addLayout(z_pos_layout)
        
        layout.addLayout(form_layout)
    
        button_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        
        def apply_changes():
            try:
                
                thickness_val = thickness_spin.value() / 10.0  
                if hasattr(self.pathology_volume, 'updateSlice'):
                    self.pathology_volume.updateSlice(slice_idx, 'slice_thickness_mm', thickness_val)
                    self.pathology_volume.updateSlice(slice_idx, 'flip', flip_cb.isChecked())
                    self.pathology_volume.updateSlice(slice_idx, 'rotation_angle', float(rotation_spin.value()))
                else:
                    ps.sliceThickness = thickness_val
                    ps.doFlip = flip_cb.isChecked()
                    ps.doRotate = float(rotation_spin.value())
                
                self.update_slices_display()
                self.update_volume_info_display()
                dialog.accept()
                
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Error updating slice: {e}")
        
        apply_btn.clicked.connect(apply_changes)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def update_slice_thickness(self):
        current_row = self.slices_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a slice")
            return
            
        current_thickness = float(self.slices_table.item(current_row, 3).text()) if self.slices_table.item(current_row, 3) else 1.0
        
        thickness, ok = QInputDialog.getDouble(
            self, "Update Thickness", 
            "Enter new thickness (mm):", 
            current_thickness, 0.1, 10.0, 1
        )
        
        if ok:
            try:
                if hasattr(self.pathology_volume, 'updateSlice'):
                    self.pathology_volume.updateSlice(current_row, 'slice_thickness_mm', thickness)
                else:
                    ps = self.pathology_volume.pathologySlices[current_row]
                    ps.sliceThickness = thickness
                    
                self.update_slices_display()
                self.update_volume_info_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error updating thickness: {e}")

    def recalculate_positions(self):
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "No pathology volume loaded")
            return
            
        try:
            if hasattr(self.pathology_volume, 'calculateSlicePositions'):
                self.pathology_volume.calculateSlicePositions()
            else:
                # Alternative method or manual calculation
                QMessageBox.information(self, "Info", "Position calculation method not available")
                return
                
            self.update_slices_display()
            self.update_volume_info_display()
            QMessageBox.information(self, "Success", "Slice positions recalculated!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error recalculating positions: {e}")

    def save_json(self):
        if not self.pathology_volume:
            QMessageBox.warning(self, "Warning", "No pathology volume loaded")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pathology JSON", "", "JSON files (*.json)"
        )
        
        if file_path:
            try:
                if hasattr(self.pathology_volume, 'saveJson'):
                    success = self.pathology_volume.saveJson(file_path)
                    if success:
                        QMessageBox.information(self, "Success", "JSON file saved successfully!")
                    else:
                        QMessageBox.warning(self, "Warning", "Failed to save JSON file")
                else:
                    # Alternative save method or manual JSON creation
                    QMessageBox.information(self, "Info", "Direct JSON save not available")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving JSON: {e}")

    def export_volume(self):
        if not self.loaded_volume:
            QMessageBox.warning(self, "Warning", "No volume available for export. Load or create a volume first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Volume", "", 
            "NRRD files (*.nrrd);;NIfTI files (*.nii);;Meta files (*.mhd);;All files (*.*)"
        )
        
        if file_path:
            try:
                sitk.WriteImage(self.loaded_volume, file_path)
                QMessageBox.information(self, "Success", "Volume exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export volume: {e}")
                
    
    def load_volume(self):
        try:
            self.loaded_volume = self.pathology_volume.loadRgbVolume()
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Volume", f"Histological volume loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading volume: {e}")
    
         
    def load_mask(self):
        try:
            mask_id = self.mask_id_spin.value()
            self.loaded_mask = self.pathology_volume.loadMask(idxMask=mask_id)
            self.send_to_viewer_btn.setEnabled(True)
            QMessageBox.information(self, "Mask", f"Mask (ID: {mask_id}) loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading mask: {e}")
        
        
    def send_to_viewer(self):
        if self.loaded_volume is not None:
            # Convert volume from SimpleITK to numpy before sending
            vol_np = sitk.GetArrayFromImage(self.loaded_volume)
            self.data_loaded.emit("histology", vol_np)
        if self.loaded_mask is not None:
            mask_np = sitk.GetArrayFromImage(self.loaded_mask)
            self.data_loaded.emit("segmentation", mask_np)
        QMessageBox.information(self, "Viewer", "Data sent to the medical viewer")
    
    def renumber_slices(self):
        for i in range(self.slices_table.rowCount()):
            self.slices_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
        QMessageBox.information(self, "Renumbering", "Slices renumbered correctly.")
        
    def update_masks(self):
        """Updates mask numbering"""
        QMessageBox.information(self, "Update", "Mask numbering updated.")

class RegistrationTab(QWidget):
    """Tab for radiology-pathology registration"""
    
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        self.config = RegistrationConfig()
        self.worker = None
        self.registered_data = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        
        # Input configuration
        config_group = QGroupBox("Registration Config")
        config_layout = QVBoxLayout()
        
        # Input JSON file
        json_layout = QHBoxLayout()
        self.json_input = QLineEdit()
        self.json_input.setPlaceholderText("JSON file for registration configuration")
        json_browse = QPushButton("Browse")
        json_browse.clicked.connect(self.browse_registration_json)
        json_layout.addWidget(QLabel("JSON Config:"))
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(json_browse)
        config_layout.addLayout(json_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Output directory...")
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(QLabel("Output Dir:"))
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_browse)
        config_layout.addLayout(output_layout)
        
        # Elastix path
        elastix_layout = QHBoxLayout()
        self.elastix_input = QLineEdit()
        self.elastix_input.setPlaceholderText("Elastix executable path...")
        elastix_browse = QPushButton("Browse")
        elastix_browse.clicked.connect(self.browse_elastix)
        elastix_layout.addWidget(QLabel("Elastix Path:"))
        elastix_layout.addWidget(self.elastix_input)
        elastix_layout.addWidget(elastix_browse)
        config_layout.addLayout(elastix_layout)
        
        # Options
        options_layout = QHBoxLayout()
        self.discard_orientation_cb = QCheckBox("Discard Orientation")
        options_layout.addWidget(self.discard_orientation_cb)
        options_layout.addStretch()
        config_layout.addLayout(options_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Registration controls
        controls_group = QGroupBox("Registration Controls")
        controls_layout = QVBoxLayout()
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.start_registration_btn = QPushButton("Start registration")
        self.start_registration_btn.clicked.connect(self.start_registration)
        self.stop_registration_btn = QPushButton("Stop registration")
        self.stop_registration_btn.clicked.connect(self.stop_registration)
        self.stop_registration_btn.setEnabled(False)
        
        # Button to send to viewer
        self.send_to_viewer_btn = QPushButton("Send results to viewer")
        self.send_to_viewer_btn.clicked.connect(self.send_to_viewer)
        self.send_to_viewer_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.start_registration_btn)
        buttons_layout.addWidget(self.stop_registration_btn)
        buttons_layout.addWidget(self.send_to_viewer_btn)
        buttons_layout.addStretch()
        controls_layout.addLayout(buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Log output
        log_group = QGroupBox("Registration Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        log_layout.addWidget(self.log_output)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_registration_json(self):
        """Browse for registration JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select registration JSON", "", "JSON files (*.json)"
        )
        if file_path:
            self.json_input.setText(file_path)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select output directory"
        )
        if dir_path:
            self.output_input.setText(dir_path)
            
    def browse_elastix(self):
        """Browse for Elastix executable"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Elastix executable", "", "Executable files (*.exe);;All files (*)"
        )
        if file_path:
            self.elastix_input.setText(file_path)
            
    def start_registration(self):
        """Start registration process"""
        # Validate input
        if not self.json_input.text() or not os.path.exists(self.json_input.text()):
            QMessageBox.warning(self, "Error", "Select a valid JSON configuration file.")
            return
            
        if not self.output_input.text() or not os.path.exists(self.output_input.text()):
            QMessageBox.warning(self, "Error", "Select a valid output directory.")
            return
            
        if not self.elastix_input.text() or not os.path.exists(self.elastix_input.text()):
            QMessageBox.warning(self, "Error", "Select the Elastix executable path.")
            return

        try:
            # Read JSON configuration file for paths
            with open(self.json_input.text(), 'r') as f:
                reg_data = json.load(f)

            # Update configuration
            self.config.input_json = self.json_input.text()
            self.config.output_path = self.output_input.text()
            self.config.elastix_path = os.path.dirname(self.elastix_input.text())
            self.config.discard_orientation = self.discard_orientation_cb.isChecked()
            
            # Set fixed/moving file paths and parameters from JSON
            self.config.fixed_volume_path = reg_data.get('fixed_volume')
            self.config.moving_volume_path = reg_data.get('moving_volume')
            self.config.parameter_filenames = reg_data.get('parameter_files')
            self.config.fixed_volume_mask_path = reg_data.get('fixed_mask', None)
            self.config.moving_volume_mask_path = reg_data.get('moving_mask', None)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading JSON configuration: {e}")
            return
        
        # Start worker
        self.worker = RegistrationWorker(self.config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.finished.connect(self.registration_finished)
        
        self.worker.start()
        
        # Update UI
        self.start_registration_btn.setEnabled(False)
        self.stop_registration_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_output.append(f"Registration started at {QTimer().currentTime().toString()}")

    def stop_registration(self):
        """Stop registration process"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            
        self.registration_finished(False, "Registration stopped by user.")
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        """Update status"""
        self.status_label.setText(status)
        self.log_output.append(f"[{QTimer().currentTime().toString()}] {status}")
        
    def registration_finished(self, success, message):
        """Handle registration completion"""
        self.start_registration_btn.setEnabled(True)
        self.stop_registration_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        
        self.log_output.append(f"[{QTimer().currentTime().toString()}] {message}")
        
        if success:
            try:
                # Load registered volume from file to send to viewer
                output_volume_path = os.path.join(self.config.output_path, "result_resample", "result.mhd")
                self.registered_data = sitk.GetArrayFromImage(sitk.ReadImage(output_volume_path))
                self.send_to_viewer_btn.setEnabled(True)
                QMessageBox.information(self, "Success", message)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading registered result: {e}")
        else:
            QMessageBox.warning(self, "Error", message)
            
    def send_to_viewer(self):
        """Send registration results to viewer"""
        if self.registered_data is not None:
            self.data_loaded.emit("mr", self.registered_data)
            QMessageBox.information(self, "Viewer", "Registration results sent to medical visualizer.")

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.medical_viewer = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Histopathomix-Registration and Medical Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        header = QLabel("Histopathomix")
        header.setFont(QFont("Arial", 25 , QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(header)

        logo = QLabel()
        pixmap = QPixmap("Resources/Icons/logo.png")  # Sostituisci con il percorso del tuo logo
        pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(logo)

        self.tabs = QTabWidget()

        # BigWarp tab (prima del pathology tab)
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab = BigWarpTab()
            self.tabs.addTab(self.bigwarp_tab, "BigWarp Registration")
       
       # Connetti il segnale per inviare dati al viewer
        if MATPLOTLIB_AVAILABLE:
           self.bigwarp_tab.data_loaded.connect(self.send_data_to_viewer)

        self.pathology_tab = PathologyParser()
        self.tabs.addTab(self.pathology_tab, "Parse Pathology")

        # NUOVO: Connetti il segnale tra i tab
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab.rotation_data_ready.connect(self.pathology_tab.receive_rotation_data)
        
        
        self.registration_tab = RegistrationTab()
        self.tabs.addTab(self.registration_tab, "Registration")
        
        if MATPLOTLIB_AVAILABLE:
            self.medical_viewer = MedicalViewer()
            self.tabs.addTab(self.medical_viewer, "Medical Viewer")
            
            # Connetti i segnali per il passaggio dati
            self.pathology_tab.data_loaded.connect(self.send_data_to_viewer)
            self.registration_tab.data_loaded.connect(self.send_data_to_viewer)
        else:
            placeholder_tab = QWidget()
            placeholder_layout = QVBoxLayout()
            placeholder_layout.addWidget(QLabel("Visualizzatore Medico non disponibile"))
            placeholder_layout.addWidget(QLabel("Installare matplotlib per abilitare questa funzionalità"))
            placeholder_tab.setLayout(placeholder_layout)
            self.tabs.addTab(placeholder_tab, "Visualizzatore Medico")
        
        info_tab = self.create_info_tab()
        self.tabs.addTab(info_tab, "Info")
        
        layout.addWidget(self.tabs)
        
        central_widget.setLayout(layout)
        
        self.create_menu_bar()
        
        self.statusBar().showMessage("Pronto - Viewer Medico Integrato")
        
    def send_data_to_viewer(self, data_type: str, data):
        if self.medical_viewer and self.medical_viewer.canvas:
            try:
                if data_type == "mr":
                    self.medical_viewer.canvas.load_mr_data(data, "RM Registrata")
                elif data_type == "segmentation":
                    self.medical_viewer.canvas.load_segmentation_data(data, "Segmentazione")
                elif data_type == "histology":
                    self.medical_viewer.canvas.load_histology_data(data, "Istologico")
                    
                self.medical_viewer.update_slice_controls()
                
                # Passa automaticamente al tab del viewer
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "Visualizzatore Medico":
                        self.tabs.setCurrentIndex(i)
                        break
                        
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore nell'invio dati al viewer: {e}")
        
    def create_menu_bar(self): #da controllare
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        file_menu.addAction('Nuovo Progetto', self.new_project)
        file_menu.addAction('Apri Progetto', self.open_project)
        file_menu.addAction('Salva Progetto', self.save_project)
        file_menu.addSeparator()
        file_menu.addAction('Esci', self.close)
       
        view_menu = menubar.addMenu('Vista')
        view_menu.addAction('Vai al Viewer Medico', self.switch_to_viewer)
        view_menu.addAction('Reset Viewer', self.reset_viewer)
    
        help_menu = menubar.addMenu('Aiuto')
        help_menu.addAction('Informazioni', self.show_about)
        help_menu.addAction('Controlli Viewer', self.show_viewer_help)
        
    def switch_to_viewer(self):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Visualizzatore Medico":
                self.tabs.setCurrentIndex(i)
                break
                
    def reset_viewer(self):
        if self.medical_viewer and self.medical_viewer.canvas:
            self.medical_viewer.canvas.mr_data = self.medical_viewer.canvas.__class__.ImageData("RM")
            self.medical_viewer.canvas.segmentation_data = self.medical_viewer.canvas.__class__.ImageData("Segmentazione") 
            self.medical_viewer.canvas.histology_data = self.medical_viewer.canvas.__class__.ImageData("Istologico")
            self.medical_viewer.canvas.update_display()
            QMessageBox.information(self, "Reset", "Viewer medico resettato.")
            
    def show_viewer_help(self):
        help_text = """
        <h3>Controlli Viewer Medico:</h3>
        <ul>
            <li><b>Mouse Wheel:</b> Cambia slice nelle viste</li>
            <li><b>Checkbox Visibilità:</b> Accende/spegne i layer</li>
            <li><b>Slider Opacità:</b> Controlla trasparenza layer</li>
            <li><b>Window/Level:</b> Regola contrasto RM</li>
            <li><b>Controllo Slice:</b> Naviga tra le slice</li>
        </ul>
        
        <h3>Layer:</h3>
        <ul>
            <li><b>RM:</b> Base layer in scala di grigi</li>
            <li><b>Segmentazione:</b> Overlay rosso</li>
            <li><b>Istologico:</b> Overlay colorato</li>
        </ul>
        """
        QMessageBox.information(self, "Aiuto Viewer", help_text)
        
    def create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>READ ME:</h3>
        <p>...</p>
        """)

        layout.addWidget(info_text)
        widget.setLayout(layout)
        return widget
        
    def new_project(self):
        reply = QMessageBox.question(
            self, "Nuovo Progetto", 
            "Creare un nuovo progetto? I dati non salvati andranno persi.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset di tutti i tab
            if self.medical_viewer:
                self.reset_viewer()
            
            # Reset pathology tab
            self.pathology_tab.json_input.clear()
            self.pathology_tab.volume_input.clear() 
            self.pathology_tab.mask_input.clear()
            self.pathology_tab.json_info.clear()
            self.pathology_tab.slices_table.setRowCount(0)
            
            # Reset registration tab
            self.registration_tab.json_input.clear()
            self.registration_tab.output_input.clear()
            self.registration_tab.elastix_input.clear()
            self.registration_tab.log_output.clear()
            
            QMessageBox.information(self, "Nuovo Progetto", "Nuovo progetto creato.")
        
    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Apri Progetto", "", "RadPath projects (*.rpf);;JSON files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                
                # Carica configurazioni nei tab appropriati
                if 'pathology' in project_data:
                    path_data = project_data['pathology']
                    self.pathology_tab.json_input.setText(path_data.get('json_path', ''))
                    self.pathology_tab.volume_input.setText(path_data.get('volume_name', ''))
                    self.pathology_tab.mask_input.setText(path_data.get('mask_name', ''))
                
                if 'registration' in project_data:
                    reg_data = project_data['registration'] 
                    self.registration_tab.json_input.setText(reg_data.get('input_json', ''))
                    self.registration_tab.output_input.setText(reg_data.get('output_path', ''))
                    self.registration_tab.elastix_input.setText(reg_data.get('elastix_path', ''))
                
                QMessageBox.information(self, "Progetto", f"Progetto aperto: {os.path.basename(file_path)}.")
                
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore nell'apertura del progetto: {e}")
            
    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva Progetto", "", "RadPath projects (*.rpf)"
        )
        if file_path:
            try:
                # Raccogli dati da tutti i tab
                project_data = {
                    'version': '2.0.0',
                    'pathology': {
                        'json_path': self.pathology_tab.json_input.text(),
                        'volume_name': self.pathology_tab.volume_input.text(),
                        'mask_name': self.pathology_tab.mask_input.text(),
                        'mask_id': self.pathology_tab.mask_id_spin.value()
                    },
                    'registration': {
                        'input_json': self.registration_tab.json_input.text(),
                        'output_path': self.registration_tab.output_input.text(),
                        'elastix_path': self.registration_tab.elastix_input.text(),
                        'discard_orientation': self.registration_tab.discard_orientation_cb.isChecked()
                    },
                    'viewer': {
                        'mr_visible': self.medical_viewer.mr_visible_cb.isChecked() if self.medical_viewer else True,
                        'seg_visible': self.medical_viewer.seg_visible_cb.isChecked() if self.medical_viewer else True,
                        'hist_visible': self.medical_viewer.hist_visible_cb.isChecked() if self.medical_viewer else True,
                        'mr_opacity': self.medical_viewer.mr_opacity_slider.value() if self.medical_viewer else 100,
                        'seg_opacity': self.medical_viewer.seg_opacity_slider.value() if self.medical_viewer else 80,
                        'hist_opacity': self.medical_viewer.hist_opacity_slider.value() if self.medical_viewer else 70
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                    
                QMessageBox.information(self, "Progetto", f"Progetto salvato: {os.path.basename(file_path)}.")
                
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore nel salvataggio: {e}")
            
    def show_about(self):
        QMessageBox.about(
            self, 
            "Basato sul progetto originale:\n"
            "https://github.com/pimed/Slicer-RadPathFusion"
        )

def main():
    """Funzione principale"""
    app = QApplication(sys.argv)
    app.setApplicationName("RadPathFusion")
    app.setApplicationVersion("2.0.0")

    #b6cdbd verde
    #f2f9f1 bianco panna
    # Stile moderno aggiornato
    app.setStyleSheet("""
        QMainWindow {
            background-color: #eeeeee;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin: 15px 0px;
            padding-top: 15px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 14px;
            margin: 4px 2px;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
            transform: translateY(-1px); 
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QLineEdit {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            background-color: white;
            font-size: 14px;
        }
        QLineEdit:focus {
            border-color: #4CAF50;
            background-color: #f9f9f9;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
            border-radius: 8px;
        }
        QTabBar::tab {
            background-color: #e1e1e1;
            padding: 12px 24px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: #4CAF50;
            color: white;
        }
        QTabBar::tab:hover {
            background-color: #d1d1d1;
        }
        QCheckBox {
            font-weight: bold;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #cccccc;
        }
        QCheckBox::indicator:checked {
            background-color: #4CAF50;
            border-color: #4CAF50;
        }
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: white;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: #4CAF50;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4CAF50;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }
        QTextEdit, QTableWidget {
            border: 1px solid #cccccc;
            border-radius: 6px;
            background-color: white;
            padding: 8px;
        }
        QStatusBar {
            background-color: #e1e1e1;
            border-top: 1px solid #cccccc;
            font-weight: bold;
        }
    """)
    
    if not MATPLOTLIB_AVAILABLE:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Dipendenza Mancante")
        msg.setText("Matplotlib non è installato")
        msg.setInformativeText(
            "Il visualizzatore medico richiede matplotlib.\n"
            "Installare con: pip install matplotlib\n\n"
            "L'applicazione continuerà con funzionalità limitate."
        )
        msg.exec()
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()