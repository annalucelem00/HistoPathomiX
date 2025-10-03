import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import SimpleITK as sitk
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QTextEdit, QProgressBar,
                             QFileDialog, QMessageBox, QComboBox, QSizePolicy,
                             QFrame, QDialog)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer

# Assumendo che RegisterVolumesElastix sia nello stesso percorso
from .RegisterVolumesElastix import RegisterVolumesElastix

@dataclass
class RegistrationConfig:
    input_json: str = ""
    output_path: str = ""
    elastix_path: str = ""
    fixed_volume_path: str = ""
    moving_volume_path: str = ""
    parameter_filenames: List[str] = None
    fixed_volume_mask_path: Optional[str] = None
    moving_volume_mask_path: Optional[str] = None

class RegistrationWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config: RegistrationConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.status_updated.emit("Avvio di Elastix...")
            reg_elastix = RegisterVolumesElastix()
            reg_elastix.set_elastix_bin_dir(self.config.elastix_path)
            
            params_dir = os.path.dirname(self.config.input_json)
            full_param_paths = [os.path.join(params_dir, f) for f in self.config.parameter_filenames]

            self.status_updated.emit("Avvio registrazione volumi...")
            reg_elastix.register_volumes(
                fixed_volume_path=self.config.fixed_volume_path,
                moving_volume_path=self.config.moving_volume_path,
                parameter_filenames=full_param_paths,
                output_dir=self.config.output_path,
                fixed_volume_mask_path=self.config.fixed_volume_mask_path
            )
            self.status_updated.emit("Registrazione completata.")
            self.finished.emit(True, "Registrazione completata con successo.")
        except Exception as e:
            self.finished.emit(False, f"Errore durante la registrazione: {e}")

class RegistrationTab(QWidget):
    data_loaded = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.config = RegistrationConfig()
        self.worker = None
        self.registered_data = None
        self.json_full_path = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        config_group = QGroupBox("Registration Configuration")
        config_layout = QVBoxLayout(config_group)

        # Volumi e Maschere
        self.fixed_volume_combo = self.create_file_combo("Fixed Volume:", self.browse_fixed_volume, config_layout)
        self.fixed_mask_combo = self.create_file_combo("Fixed Mask (Opt):", self.browse_fixed_mask, config_layout)
        
        # Il volume mobile è ancora un LineEdit perché di solito viene da un file (es. istologia esportata)
        moving_volume_layout = QHBoxLayout()
        self.moving_volume_input = QLineEdit()
        moving_browse_btn = QPushButton("Browse")
        moving_browse_btn.clicked.connect(self.browse_moving_volume)
        moving_volume_layout.addWidget(QLabel("Moving Volume:"))
        moving_volume_layout.addWidget(self.moving_volume_input)
        moving_volume_layout.addWidget(moving_browse_btn)
        config_layout.addLayout(moving_volume_layout)
        
        # Parametri e percorsi
        self.create_path_input("Parameters JSON:", self.browse_registration_json, config_layout)
        self.output_input = self.create_path_input("Output Dir:", self.browse_output_dir, config_layout)
        self.elastix_input = self.create_path_input("Elastix Path:", self.browse_elastix, config_layout)
        
        layout.addWidget(config_group)

        # Controlli e Log
        controls_group = QGroupBox("Registration Controls & Log")
        controls_layout = QVBoxLayout(controls_group)
        buttons_layout = QHBoxLayout()
        self.start_registration_btn = QPushButton("Start Registration")
        self.start_registration_btn.clicked.connect(self.start_registration)
        self.send_to_viewer_btn = QPushButton("Send Result to Viewer")
        self.send_to_viewer_btn.clicked.connect(self.send_to_viewer)
        self.send_to_viewer_btn.setEnabled(False)
        buttons_layout.addWidget(self.start_registration_btn)
        buttons_layout.addWidget(self.send_to_viewer_btn)
        controls_layout.addLayout(buttons_layout)
        self.progress_bar = QProgressBar()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.log_output)
        layout.addWidget(controls_group)
        
        self.load_default_config_on_startup()

    def create_file_combo(self, label, browse_func, parent_layout):
        layout = QHBoxLayout()
        combo = QComboBox()
        combo.addItem("Carica dal Viewer o Sfoglia...")
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(browse_func)
        layout.addWidget(QLabel(label))
        layout.addWidget(combo)
        layout.addWidget(browse_btn)
        parent_layout.addLayout(layout)
        return combo

    def create_path_input(self, label_text, browse_func, parent_layout):
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_general(line_edit, label_text))
        layout.addWidget(QLabel(label_text))
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        parent_layout.addLayout(layout)
        return line_edit

    def browse_general(self, line_edit, title):
        if "Dir" in title or "Path" in title:
            path = QFileDialog.getExistingDirectory(self, f"Select {title}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", "All Files (*)")
        if path:
            line_edit.setText(path)

    def browse_fixed_volume(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Fixed Volume", "", "Images (*.nii *.mhd)")
        if path: self.add_fixed_volume_option(os.path.basename(path), path)
            
    def browse_fixed_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Fixed Mask", "", "Images (*.nii *.mhd)")
        if path: self.add_fixed_mask_option(os.path.basename(path), path)

    def browse_moving_volume(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Moving Volume", "", "Images (*.nii *.nrrd)")
        if path: self.moving_volume_input.setText(path)

    def browse_registration_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Parameters JSON", "", "JSON files (*.json)")
        if path:
            self.json_full_path = path
            # Assumendo che il LineEdit per il JSON sia il primo creato
            self.findChild(QLineEdit).setText(path)

    def add_fixed_volume_option(self, name: str, path: str):
        self.fixed_volume_combo.addItem(name, userData=path)
        self.fixed_volume_combo.setCurrentIndex(self.fixed_volume_combo.count() - 1)

    def add_fixed_mask_option(self, name: str, path: str):
        self.fixed_mask_combo.addItem(name, userData=path)
        self.fixed_mask_combo.setCurrentIndex(self.fixed_mask_combo.count() - 1)

    def start_registration(self):
        try:
            # Validazione input
            self.config.fixed_volume_path = self.fixed_volume_combo.currentData()
            self.config.moving_volume_path = self.moving_volume_input.text()
            self.config.elastix_path = self.elastix_input.text()
            self.config.output_path = self.output_input.text()
            self.config.input_json = self.json_full_path

            if not all([self.config.fixed_volume_path, self.config.moving_volume_path, self.config.elastix_path, self.config.output_path, self.config.input_json]):
                raise ValueError("Per favore, compila tutti i campi richiesti.")

            with open(self.config.input_json, 'r') as f:
                reg_data = json.load(f)
            self.config.parameter_filenames = reg_data.get('parameter_files')
            self.config.fixed_volume_mask_path = self.fixed_mask_combo.currentData()

            self.worker = RegistrationWorker(self.config)
            self.worker.status_updated.connect(self.log_output.append)
            self.worker.finished.connect(self.registration_finished)
            self.worker.start()
            self.start_registration_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Errore di configurazione", str(e))

    def registration_finished(self, success, message):
        self.log_output.append(f"--- {message} ---")
        self.start_registration_btn.setEnabled(True)
        if success:
            try:
                result_path = Path(self.config.output_path) / "result.0.mhd"
                if not result_path.exists():
                    result_path = Path(self.config.output_path) / "result.mhd" # fallback
                
                self.registered_data = sitk.GetArrayFromImage(sitk.ReadImage(str(result_path)))
                self.send_to_viewer_btn.setEnabled(True)
                QMessageBox.information(self, "Successo", message)
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Impossibile caricare il risultato della registrazione: {e}")
        else:
            QMessageBox.warning(self, "Registrazione Fallita", message)

    def send_to_viewer(self):
        if self.registered_data is not None:
            self.data_loaded.emit("mr", self.registered_data)

    def load_default_config_on_startup(self):
        try:
            script_dir = Path(__file__).resolve().parent.parent.parent # Va su fino alla root del progetto
            default_config = script_dir / "Resources" / "RegistrationParameters" / "registration_config.json"
            if default_config.exists():
                self.json_full_path = str(default_config)
                self.findChild(QLineEdit).setText(str(default_config))
        except Exception as e:
            print(f"Nessun file di configurazione di default trovato: {e}")
            
    def reset_state(self):
        self.config = RegistrationConfig()
        self.fixed_volume_combo.clear()
        self.fixed_volume_combo.addItem("Carica dal Viewer o Sfoglia...")
        self.fixed_mask_combo.clear()
        self.fixed_mask_combo.addItem("Carica dal Viewer o Sfoglia...")
        self.moving_volume_input.clear()
        self.output_input.clear()
        self.elastix_input.clear()
        self.log_output.clear()
        self.send_to_viewer_btn.setEnabled(False)
        self.load_default_config_on_startup()