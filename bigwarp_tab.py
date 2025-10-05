# File: bigwarp_tab.py

import sys
import json
import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QLabel
)
from PyQt6.QtCore import pyqtSignal

class FlipDataManager:
    """Class to manage flip data for images"""
    
    def __init__(self):
        self.flip_data: Dict[str, Dict[str, bool]] = {}  # {filename: {'vertical': bool, 'horizontal': bool}}
    
    def add_flip_data(self, filename: str, vertical_flip: bool, horizontal_flip: bool):
        """Add flip data for a specific image"""
        self.flip_data[filename] = {
            'vertical': vertical_flip,
            'horizontal': horizontal_flip
        }
    
    def get_flip_data(self, filename: str) -> Optional[Dict[str, bool]]:
        """Get flip data for a specific image"""
        return self.flip_data.get(filename)
    
    def export_to_csv(self, filepath: str):
        """Export flip data to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'vertical_flip', 'horizontal_flip'])
            writer.writeheader()
            for filename, flips in self.flip_data.items():
                writer.writerow({
                    'filename': filename, 
                    'vertical_flip': flips['vertical'],
                    'horizontal_flip': flips['horizontal']
                })

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


class BigWarpTab(QWidget):
    """Tab wrapper per BigWarp che integra la GUI Tkinter esistente"""
    
    data_loaded = pyqtSignal(str, object)
    rotation_data_ready = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.bigwarp_app = None
        self.bigwarp_process = None
        self.rotation_manager = RotationDataManager()
        self.flip_manager= FlipDataManager()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        info_group = QGroupBox("BigWarp - Registrazione Deformabile")
        info_layout = QVBoxLayout()

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
        
        sharing_group = QGroupBox("Data Sharing with ParsePathology")
        sharing_layout = QVBoxLayout()
        
        sharing_buttons_layout = QHBoxLayout()
        
        self.export_angles_btn = QPushButton("Export Rotation Angles (CSV)")
        self.export_angles_btn.clicked.connect(self.export_rotation_angles)
        self.export_flips_btn = QPushButton("Export Flip States (CSV)")
        self.export_flips_btn.clicked.connect(self.export_flip_states)
        
        sharing_buttons_layout.addWidget(self.export_angles_btn)
        sharing_buttons_layout.addWidget(self.export_flips_btn)
        sharing_buttons_layout.addStretch()
        
        self.angles_status_label = QLabel("No rotation angles calculated yet")
        self.angles_status_label.setStyleSheet("QLabel { color: gray; }")
        
        sharing_layout.addLayout(sharing_buttons_layout)
        sharing_layout.addWidget(self.angles_status_label)
        
        sharing_group.setLayout(sharing_layout)
        layout.addWidget(sharing_group)
        
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions = QLabel("""
        <b>How to use BigWarp:</b><br><br>
                              
        1. Click “Open BigWarp” to launch the full interface<br><br>
        2. Upload the fixed and moving image(s)<br><br>
        3. Add landmarks by clicking on the corresponding points on the fixed and the moving<br><br>
        4. Calculate and apply the TPS transformation<br><br>
        5. Do the point 3 and 4 for all the moving images loaded<br><br>
        6. Export all the angles<br>
        """)
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)

        font = instructions.font()
        font.setPointSize(16)
        instructions.setFont(font)
        
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
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

            image_filename, _ = QFileDialog.getOpenFileName(
                self, "Seleziona l'immagine corrispondente a questa trasformazione", 
                "", "Image files (*.png *.jpg *.jpeg *.tiff *.bmp)"
            )
            if image_filename:
                image_name = os.path.basename(image_filename)
                self.rotation_manager.add_rotation_data(image_name, self.rotation_angle)
                self.angles_status_label.setText(f"Rotation angle stored for: {image_name}")
                self.angles_status_label.setStyleSheet("QLabel { color: green; }")
                
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
    
    def export_flip_states(self):
        """Export flips to CSV file"""
        if not self.flip_manager.flip_data:
            QMessageBox.warning(self, "Warning", "No flips to export.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Flips", "", "CSV files (*.csv)"
        )
        if filepath:
            try:
                self.flip_manager.export_to_csv(filepath)
                QMessageBox.information(self, "Success", f"Flips exported to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def launch_bigwarp(self):
        """Lancia lo script bigwarp.py e cattura l'output per il debug."""
        if self.bigwarp_process and self.bigwarp_process.poll() is None:
            QMessageBox.information(self, "Informazione", "BigWarp è già in esecuzione.")
            return

        try:
            current_file_path = Path(__file__).resolve().parent
            bigwarp_path = current_file_path / "Resources" / "Utils" / "BigWarp.py"
            if not bigwarp_path.exists():
                QMessageBox.critical(self, "Errore", f"File non trovato: {bigwarp_path}")
                return
            command = [sys.executable, str(bigwarp_path), "--gui"]
            print(f"Esecuzione del comando: {' '.join(command)}")
            self.bigwarp_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            QMessageBox.information(self, "Avviato", "BigWarp è stato avviato in una finestra separata.")
        except FileNotFoundError:
            QMessageBox.critical(self, "Errore", "Script 'bigwarp.py' non trovato.")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile avviare BigWarp: {e}")