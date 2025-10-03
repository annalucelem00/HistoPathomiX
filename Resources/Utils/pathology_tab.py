import os
import csv
from pathlib import Path
from typing import Dict

import SimpleITK as sitk
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QTextEdit, QFileDialog, 
                             QMessageBox, QSpinBox, QCheckBox, QTableWidget, 
                             QTableWidgetItem, QScrollArea, QDialog, QInputDialog)
from PyQt6.QtCore import pyqtSignal, Qt

# Importa le tue utility. Assicurati che i percorsi siano corretti.
from .ImageStack_3 import PathologyVolume

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


class PathologyTab(QWidget):
    """
    Tab per il parsing dei dati di patologia, la gestione delle slice,
    e l'applicazione delle trasformazioni da BigWarp.
    """
    data_loaded = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.pathology_volume: PathologyVolume = None
        self.rectum_distance = 5.0
        self.rotation_angles_data: Dict[str, float] = {}
        self.flip_data: Dict[str, Dict] = {}
        self.loaded_volume = None
        self.loaded_mask = None
        self.init_ui()

    def init_ui(self):
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        
        # --- Gruppo Input ---
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)
        json_layout = QHBoxLayout()
        self.json_input = QLineEdit()
        json_browse = QPushButton("Browse")
        json_browse.clicked.connect(self.browse_json)
        json_layout.addWidget(QLabel("JSON file:"))
        json_layout.addWidget(self.json_input)
        json_layout.addWidget(json_browse)
        input_layout.addLayout(json_layout)
        layout.addWidget(input_group)

        # --- Gruppo Azioni ---
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        primary_actions = QHBoxLayout()
        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.clicked.connect(self.load_json)
        self.load_volume_btn = QPushButton("Load Volume")
        self.load_volume_btn.clicked.connect(self.load_volume)
        self.load_volume_btn.setEnabled(False)
        self.export_volume_btn = QPushButton("Export Volume")
        self.export_volume_btn.clicked.connect(self.export_volume)
        self.export_volume_btn.setEnabled(False)
        primary_actions.addWidget(self.load_json_btn)
        primary_actions.addWidget(self.load_volume_btn)
        primary_actions.addWidget(self.export_volume_btn)
        actions_layout.addLayout(primary_actions)
        layout.addWidget(actions_group)
        
        # --- Gruppo Dati da BigWarp ---
        external_data_group = QGroupBox("Data from BigWarp")
        external_data_layout = QVBoxLayout(external_data_group)
        self.import_angles_btn = QPushButton("Importa Angoli (CSV)")
        self.import_angles_btn.clicked.connect(self.import_rotation_angles)
        self.apply_angles_btn = QPushButton("Applica Angoli")
        self.apply_angles_btn.clicked.connect(self.apply_rotation_angles)
        self.apply_angles_btn.setEnabled(False)
        self.rotation_status_label = QLabel("Nessun angolo di rotazione caricato.")
        external_data_layout.addWidget(self.import_angles_btn)
        external_data_layout.addWidget(self.apply_angles_btn)
        external_data_layout.addWidget(self.rotation_status_label)
        layout.addWidget(external_data_group)

        # --- Gruppo Gestione Slice ---
        slices_group = QGroupBox("Slice Management")
        slices_layout = QVBoxLayout(slices_group)
        self.slices_table = QTableWidget()
        self.slices_table.setColumnCount(5)
        self.slices_table.setHorizontalHeaderLabels(["ID", "File", "Spessore(mm)", "Flip", "Rotazione(°)"])
        slices_layout.addWidget(self.slices_table)
        layout.addWidget(slices_group)

        scroll_area = QScrollArea()
        scroll_area.setWidget(container_widget)
        scroll_area.setWidgetResizable(True)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
    
    def browse_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON file", "", "JSON files (*.json)")
        if file_path:
            self.json_input.setText(file_path)

    def load_json(self):
        json_path = self.json_input.text()
        if not json_path or not os.path.exists(json_path):
            QMessageBox.warning(self, "Errore", "Seleziona un file JSON valido.")
            return

        try:
            self.pathology_volume = PathologyVolume()
            self.pathology_volume.setPath(json_path)
            if not self.pathology_volume.initComponents():
                raise Exception("Errore di inizializzazione di PathologyVolume")
            
            for ps in self.pathology_volume.pathologySlices:
                ps.doHorizontalFlip = getattr(ps, 'doFlip', 0) == 1
                ps.doVerticalFlip = False # Inizializza a False
            
            self.update_slice_display()
            self.load_volume_btn.setEnabled(True)
            self.export_volume_btn.setEnabled(True)
            QMessageBox.information(self, "Successo", "JSON caricato con successo.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore caricamento JSON: {e}\nControlla la console per dettagli.")
            print(e)
            
    def receive_rotation_data(self, rotation_data: Dict[str, float]):
        self.rotation_angles_data.update(rotation_data)
        self.rotation_status_label.setText(f"Ricevuti dati di rotazione da BigWarp per {list(rotation_data.keys())[0]}")
        self.rotation_status_label.setStyleSheet("color: blue;")
        self.apply_angles_btn.setEnabled(True)

    def import_rotation_angles(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Importa Angoli di Rotazione", "", "CSV (*.csv)")
        if filepath:
            try:
                self.rotation_angles_data.clear()
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.rotation_angles_data[row['filename']] = float(row['rotation_angle_degrees'])
                self.rotation_status_label.setText(f"Caricati {len(self.rotation_angles_data)} angoli dal file CSV.")
                self.rotation_status_label.setStyleSheet("color: green;")
                self.apply_angles_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Importazione fallita: {e}")

    def apply_rotation_angles(self):
        if not self.pathology_volume or not self.rotation_angles_data:
            QMessageBox.warning(self, "Attenzione", "Caricare prima i dati di patologia e gli angoli di rotazione.")
            return
        matches = 0
        for ps in self.pathology_volume.pathologySlices:
            slice_filename = os.path.basename(ps.rgbImageFn)
            if slice_filename in self.rotation_angles_data:
                ps.doRotate = self.rotation_angles_data[slice_filename]
                matches += 1
        self.update_slice_display()
        QMessageBox.information(self, "Risultati", f"Angoli di rotazione applicati a {matches} slice.")

    def load_volume(self):
        if not self.pathology_volume: return
        try:
            self.loaded_volume = self.pathology_volume.loadRgbVolume()
            QMessageBox.information(self, "Volume", "Volume istologico caricato con successo.")
            vol_np = sitk.GetArrayFromImage(self.loaded_volume)
            self.data_loaded.emit("histology", vol_np)
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore caricamento volume: {e}")

    def export_volume(self):
        if not self.loaded_volume:
            QMessageBox.warning(self, "Attenzione", "Nessun volume caricato da esportare.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Esporta Volume", "", "NRRD (*.nrrd);;NIfTI (*.nii)")
        if file_path:
            try:
                sitk.WriteImage(self.loaded_volume, file_path)
                QMessageBox.information(self, "Successo", "Volume esportato con successo!")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Esportazione fallita: {e}")

    def update_slice_display(self):
        if not self.pathology_volume: return
        self.slices_table.setRowCount(self.pathology_volume.noSlices)
        for i, ps in enumerate(self.pathology_volume.pathologySlices):
            self.slices_table.setItem(i, 0, QTableWidgetItem(str(ps.id)))
            self.slices_table.setItem(i, 1, QTableWidgetItem(os.path.basename(ps.rgbImageFn)))
            self.slices_table.setItem(i, 2, QTableWidgetItem(f"{getattr(ps, 'sliceThickness', 1.0):.2f}"))
            self.slices_table.setItem(i, 3, QTableWidgetItem(str(int(getattr(ps, 'doHorizontalFlip', False)))))
            self.slices_table.setItem(i, 4, QTableWidgetItem(f"{getattr(ps, 'doRotate', 0.0):.1f}"))

    def reset_state(self):
        self.pathology_volume = None
        self.loaded_volume = None
        self.loaded_mask = None
        self.rotation_angles_data.clear()
        self.flip_data.clear()
        self.json_input.clear()
        self.slices_table.setRowCount(0)
        self.rotation_status_label.setText("Nessun angolo di rotazione caricato.")
        self.rotation_status_label.setStyleSheet("color: gray;")
        self.load_volume_btn.setEnabled(False)
        self.export_volume_btn.setEnabled(False)
        self.apply_angles_btn.setEnabled(False)