import sys
import json
import os
import csv
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QFileDialog, QMessageBox, QLabel)
from PyQt6.QtCore import pyqtSignal

class FlipDataManager:
    """Gestisce i dati di flip (verticale/orizzontale) per le immagini."""
    def __init__(self):
        self.flip_data: Dict[str, Dict[str, bool]] = {}

    def add_flip_data(self, filename: str, vertical_flip: bool, horizontal_flip: bool):
        self.flip_data[filename] = {'vertical': vertical_flip, 'horizontal': horizontal_flip}

    def export_to_csv(self, filepath: str):
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'vertical_flip', 'horizontal_flip'])
            writer.writeheader()
            for filename, flips in self.flip_data.items():
                writer.writerow({'filename': filename, 'vertical_flip': flips['vertical'], 'horizontal_flip': flips['horizontal']})

class RotationDataManager:
    """Gestisce i dati di rotazione tra BigWarp e il parser di patologia."""
    def __init__(self):
        self.rotation_data: Dict[str, float] = {}

    def add_rotation_data(self, filename: str, rotation_angle: float):
        self.rotation_data[filename] = rotation_angle

    def export_to_csv(self, filepath: str):
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'rotation_angle_degrees'])
            writer.writeheader()
            for filename, angle in self.rotation_data.items():
                writer.writerow({'filename': filename, 'rotation_angle_degrees': angle})


class BigWarpTab(QWidget):
    """Tab per l'interfaccia di BigWarp e la gestione dei dati di trasformazione."""
    data_loaded = pyqtSignal(str, object)
    rotation_data_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.bigwarp_process = None
        self.rotation_manager = RotationDataManager()
        self.flip_manager = FlipDataManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        info_group = QGroupBox("BigWarp - Registrazione Deformabile")
        info_layout = QVBoxLayout(info_group)
        buttons_layout = QHBoxLayout()
        self.launch_bigwarp_btn = QPushButton("Apri BigWarp")
        self.launch_bigwarp_btn.clicked.connect(self.launch_bigwarp)
        self.import_result_btn = QPushButton("Importa Risultati")
        self.import_result_btn.clicked.connect(self.import_result)
        buttons_layout.addWidget(self.launch_bigwarp_btn)
        buttons_layout.addWidget(self.import_result_btn)
        info_layout.addLayout(buttons_layout)
        layout.addWidget(info_group)

        sharing_group = QGroupBox("Condivisione Dati con ParsePathology")
        sharing_layout = QVBoxLayout(sharing_group)
        sharing_buttons_layout = QHBoxLayout()
        self.export_angles_btn = QPushButton("Esporta Angoli di Rotazione (CSV)")
        self.export_angles_btn.clicked.connect(self.export_rotation_angles)
        self.export_flips_btn = QPushButton("Esporta Stati di Flip (CSV)")
        self.export_flips_btn.clicked.connect(self.export_flip_states)
        sharing_buttons_layout.addWidget(self.export_angles_btn)
        sharing_buttons_layout.addWidget(self.export_flips_btn)
        self.angles_status_label = QLabel("Nessun angolo di rotazione calcolato.")
        self.angles_status_label.setStyleSheet("color: gray;")
        sharing_layout.addLayout(sharing_buttons_layout)
        sharing_layout.addWidget(self.angles_status_label)
        layout.addWidget(sharing_group)

        instructions_group = QGroupBox("Istruzioni")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions = QLabel(
            """
            <b>Come usare BigWarp:</b><br><br>
            1. Clicca “Apri BigWarp” per avviare l'interfaccia esterna.<br>
            2. Carica l'immagine fissa e quella/e mobile/i.<br>
            3. Aggiungi landmark cliccando sui punti corrispondenti.<br>
            4. Calcola e applica la trasformazione TPS.<br>
            5. Esporta la trasformazione come file JSON.<br>
            6. Usa "Importa Risultati" per caricare il JSON e associarlo a un'immagine.<br>
            7. Una volta importati tutti i risultati, usa "Esporta Angoli" per creare un CSV.
            """
        )
        instructions.setWordWrap(True)
        font = instructions.font()
        font.setPointSize(14)
        instructions.setFont(font)
        instructions_layout.addWidget(instructions)
        layout.addWidget(instructions_group)

        layout.addStretch()

    def import_result(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleziona file parametri", "", "File JSON (*.json)")
        if not filename: return

        try:
            with open(filename, 'r') as f: data = json.load(f)

            if "parameters" not in data or "affine_matrix_3x3" not in data:
                raise ValueError("File JSON non valido o formato non corretto.")

            rotation_angle = data['parameters']['rotation_degrees']
            
            image_filename, _ = QFileDialog.getOpenFileName(self, "Seleziona l'immagine corrispondente a questa trasformazione", "", "Immagini (*.png *.jpg *.jpeg *.tiff *.bmp)")
            if image_filename:
                image_name = os.path.basename(image_filename)
                self.rotation_manager.add_rotation_data(image_name, rotation_angle)
                self.angles_status_label.setText(f"Angolo di rotazione salvato per: {image_name}")
                self.angles_status_label.setStyleSheet("color: green;")
                self.rotation_data_ready.emit({image_name: rotation_angle})
                QMessageBox.information(self, "Risultato Importato", f"Importato angolo di {rotation_angle:.2f}° per {image_name}.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore durante l'importazione: {e}")

    def export_rotation_angles(self):
        if not self.rotation_manager.rotation_data:
            QMessageBox.warning(self, "Attenzione", "Nessun angolo di rotazione da esportare.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Esporta Angoli di Rotazione", "", "CSV (*.csv)")
        if filepath:
            try:
                self.rotation_manager.export_to_csv(filepath)
                QMessageBox.information(self, "Successo", f"Angoli esportati con successo in {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Esportazione fallita: {e}")

    def export_flip_states(self):
        QMessageBox.information(self, "Info", "Questa funzione è un placeholder. I dati di flip devono essere generati esternamente.")
    
    def launch_bigwarp(self):
        if self.bigwarp_process and self.bigwarp_process.poll() is None:
            QMessageBox.information(self, "Informazione", "BigWarp è già in esecuzione.")
            return
        try:
            script_dir = Path(__file__).resolve().parent
            bigwarp_script_path = script_dir / "BigWarp.py"
            if not bigwarp_script_path.exists():
                QMessageBox.critical(self, "Errore", f"Script 'BigWarp.py' non trovato nel percorso: {bigwarp_script_path}")
                return

            command = [sys.executable, str(bigwarp_script_path), "--gui"]
            self.bigwarp_process = subprocess.Popen(command)
            QMessageBox.information(self, "Avviato", "BigWarp è stato avviato in una finestra separata.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile avviare BigWarp: {e}")
            
    def reset_state(self):
        """Resetta lo stato del tab a quello iniziale."""
        self.rotation_manager = RotationDataManager()
        self.flip_manager = FlipDataManager()
        self.angles_status_label.setText("Nessun angolo di rotazione calcolato.")
        self.angles_status_label.setStyleSheet("color: gray;")