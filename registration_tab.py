# File: registration_tab.py

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import SimpleITK as sitk
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox, QSizePolicy,
    QPushButton, QLabel, QLineEdit, QFrame, QDialog, QTextEdit,
    QProgressBar, QMessageBox, QFileDialog, QCheckBox
)

# Importa le utility necessarie dalla cartella Resources
from Resources.Utils.RegisterVolumesElastix import RegisterVolumesElastix

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
    fixed_volume_mask_path: Optional[str] = None
    moving_volume_mask_path: Optional[str] = None

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
            
            reg_elastix.register_volumes(
                fixed_volume_path=self.config.fixed_volume_path,
                moving_volume_path=self.config.moving_volume_path,
                parameter_filenames=self.config.parameter_filenames,
                output_dir=self.config.output_path,
                fixed_volume_mask_path=self.config.fixed_volume_mask_path,
                moving_volume_mask_path=self.config.moving_volume_mask_path
            )

            self.status_updated.emit("Registration completed")
            self.progress_updated.emit(75)
            self.status_updated.emit("Saving results")
            self.progress_updated.emit(100)
            
            self.finished.emit(True, "Registration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            self.finished.emit(False, f"Error: {str(e)}")


class RegistrationTab(QWidget):
    """Tab for radiology-pathology registration"""
    
    data_loaded = pyqtSignal(str, object)
    
    def __init__(self):
        super().__init__()
        self.config = RegistrationConfig()
        self.worker = None
        self.registered_data = None
        self.json_full_path = ""
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        config_group = QGroupBox("Registration Configuration")
        config_layout = QVBoxLayout(config_group)

        fixed_volume_layout = QHBoxLayout()
        self.fixed_volume_combo = QComboBox()
        self.fixed_volume_combo.addItem("Load from Viewer or Browse...")
        self.fixed_volume_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fixed_browse_btn = QPushButton("Browse")
        fixed_browse_btn.clicked.connect(self.browse_fixed_volume)
        fixed_volume_layout.addWidget(QLabel("Fixed Volume:"))
        fixed_volume_layout.addWidget(self.fixed_volume_combo)
        fixed_volume_layout.addWidget(fixed_browse_btn)
        config_layout.addLayout(fixed_volume_layout)

        fixed_mask_layout = QHBoxLayout()
        self.fixed_mask_combo = QComboBox()
        self.fixed_mask_combo.addItem("Load from Viewer or Browse...")
        self.fixed_mask_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fixed_mask_browse_btn = QPushButton("Browse")
        fixed_mask_browse_btn.clicked.connect(self.browse_fixed_mask)
        fixed_mask_layout.addWidget(QLabel("Fixed Mask:"))
        fixed_mask_layout.addWidget(self.fixed_mask_combo)
        fixed_mask_layout.addWidget(fixed_mask_browse_btn)
        config_layout.addLayout(fixed_mask_layout)
        
        """moving_volume_layout = QHBoxLayout()
        self.moving_volume_input = QLineEdit()
        self.moving_volume_input.setPlaceholderText("Path to Moving Volume (e.g., Histology)")
        moving_browse_btn = QPushButton("Browse")
        moving_browse_btn.clicked.connect(self.browse_moving_volume)
        moving_volume_layout.addWidget(QLabel("Moving Volume:"))
        moving_volume_layout.addWidget(self.moving_volume_input)
        moving_volume_layout.addWidget(moving_browse_btn)
        config_layout.addLayout(moving_volume_layout)"""

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        config_layout.addWidget(separator)

        json_layout = QHBoxLayout()
        self.json_display = QLineEdit()
        self.json_display.setReadOnly(True)
        self.json_display.setPlaceholderText("No configuration file loaded")
        json_browse = QPushButton("Browse")
        json_browse.clicked.connect(self.browse_registration_json)
        self.json_view_btn = QPushButton("View")
        self.json_view_btn.clicked.connect(self.view_json_content)
        self.json_view_btn.setEnabled(False)
        json_layout.addWidget(QLabel("Parameters JSON:"))
        json_layout.addWidget(self.json_display)
        json_layout.addWidget(json_browse)
        json_layout.addWidget(self.json_view_btn)
        config_layout.addLayout(json_layout)
            
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select an output directory")
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(QLabel("Output Dir:"))
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_browse)
        config_layout.addLayout(output_layout)

        elastix_layout = QHBoxLayout()
        self.elastix_input = QLineEdit()
        self.elastix_input.setPlaceholderText("Path to Elastix bin directory")
        elastix_browse = QPushButton("Browse")
        elastix_browse.clicked.connect(self.browse_elastix)
        elastix_layout.addWidget(QLabel("Elastix Path:"))
        elastix_layout.addWidget(self.elastix_input)
        elastix_layout.addWidget(elastix_browse)
        config_layout.addLayout(elastix_layout)
        
        # Add checkbox for discard orientation
        self.discard_orientation_cb = QCheckBox("Discard Orientation")
        config_layout.addWidget(self.discard_orientation_cb)

        layout.addWidget(config_group)

        controls_group = QGroupBox("Registration Controls & Log")
        controls_layout = QVBoxLayout(controls_group)
        buttons_layout = QHBoxLayout()
        self.start_registration_btn = QPushButton("Start Registration")
        self.start_registration_btn.clicked.connect(self.start_registration)
        self.stop_registration_btn = QPushButton("Stop Registration")
        self.stop_registration_btn.clicked.connect(self.stop_registration)
        self.stop_registration_btn.setEnabled(False)
        self.send_to_viewer_btn = QPushButton("Send Result to Viewer")
        self.send_to_viewer_btn.clicked.connect(self.send_to_viewer)
        self.send_to_viewer_btn.setEnabled(False)
        buttons_layout.addWidget(self.start_registration_btn)
        buttons_layout.addWidget(self.stop_registration_btn)
        buttons_layout.addWidget(self.send_to_viewer_btn)
        buttons_layout.addStretch()
        controls_layout.addLayout(buttons_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Registration logs will appear here...")
        controls_layout.addWidget(self.log_output)
        layout.addWidget(controls_group)
        layout.addStretch()

        self.load_default_config_on_startup()

    def add_fixed_volume_option(self, name: str, path: str):
        self.fixed_volume_combo.addItem(name, userData=path)
        self.fixed_volume_combo.setCurrentIndex(self.fixed_volume_combo.count() - 1)
        self.log_output.append(f"Fixed Volume added from viewer: {name}")
    
    def add_fixed_mask_option(self, name: str, path: str):
        self.fixed_mask_combo.addItem(name, userData=path)
        self.fixed_mask_combo.setCurrentIndex(self.fixed_mask_combo.count() - 1)
        self.log_output.append(f"Fixed Mask added from viewer: {name}")

    def browse_fixed_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Fixed Volume", "", "Image Files (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path:
            self.add_fixed_volume_option(os.path.basename(file_path), file_path)

    def browse_fixed_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Fixed Mask", "", "Image Files (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path:
            self.add_fixed_mask_option(os.path.basename(file_path), file_path)

    def browse_moving_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Moving Volume", "", "Image Files (*.nii *.nii.gz *.mhd *.nrrd)")
        if file_path:
            self.moving_volume_input.setText(file_path)

    def browse_registration_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Parameters JSON", "", "JSON files (*.json)")
        if file_path:
            self.json_full_path = file_path
            self.json_display.setText(os.path.basename(file_path))
            self.json_view_btn.setEnabled(True)

    def view_json_content(self):
        if not self.json_full_path or not os.path.exists(self.json_full_path):
            QMessageBox.warning(self, "Error", "No valid JSON file loaded")
            return
        try:
            with open(self.json_full_path, 'r') as f:
                json_content = json.load(f)
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"View JSON - {os.path.basename(self.json_full_path)}")
            dialog.resize(600, 400)
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(json.dumps(json_content, indent=2))
            layout.addWidget(text_edit)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.setLayout(layout)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read JSON file: {e}")

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_input.setText(dir_path)
            
    def browse_elastix(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Elastix Bin Directory")
        if dir_path:
            self.elastix_input.setText(dir_path)
            
    def start_registration(self):
        fixed_volume_path = self.fixed_volume_combo.currentData()
        fixed_mask_path = self.fixed_mask_combo.currentData()
        moving_volume_path = self.moving_volume_input.text()

        if not all([fixed_volume_path, moving_volume_path, self.json_full_path, self.output_input.text(), self.elastix_input.text()]):
            QMessageBox.warning(self, "Input Error", "Please provide Fixed Volume, Moving Volume, and all configuration paths.")
            return

        try:
            with open(self.json_full_path, 'r') as f:
                reg_data = json.load(f)

            self.config.input_json = self.json_full_path
            self.config.output_path = self.output_input.text()
            self.config.elastix_path = self.elastix_input.text()
            self.config.fixed_volume_path = fixed_volume_path
            self.config.moving_volume_path = moving_volume_path
            self.config.fixed_volume_mask_path = fixed_mask_path if fixed_mask_path else None
            self.config.parameter_filenames = reg_data.get('parameter_files')
            self.config.moving_volume_mask_path = reg_data.get('moving_mask', None)
            self.config.discard_orientation = self.discard_orientation_cb.isChecked()

        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to read configuration: {e}")
            return
        
        self.worker = RegistrationWorker(self.config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.finished.connect(self.registration_finished)
        self.worker.start()
        
        self.start_registration_btn.setEnabled(False)
        self.stop_registration_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.append("Registration started...")
    
    def stop_registration(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.registration_finished(False, "Registration stopped by user.")

    def load_default_config_on_startup(self):
        try:
            script_dir = Path(__file__).resolve().parent
            default_config_path = script_dir / "Resources" / "RegistrationParameters" / "registration_config.json"
            
            if default_config_path.exists():
                self.json_full_path = str(default_config_path)
                self.json_display.setText(default_config_path.name)
                self.json_view_btn.setEnabled(True)
                self.log_output.append(f"Default registration config loaded: {default_config_path.name}")
            else:
                self.log_output.append("Default registration config not found.")
                
        except Exception as e:
            self.log_output.append(f"Error loading default config: {e}")
            
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        self.status_label.setText(status)
        self.log_output.append(f"[{QTimer.currentTime().toString('hh:mm:ss')}] {status}")
        
    def registration_finished(self, success, message):
        self.start_registration_btn.setEnabled(True)
        self.stop_registration_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.log_output.append(f"--- {message} ---")
        
        if success:
            try:
                # Search for result file, elastix might name it result.0.mhd, result.1.mhd etc.
                result_file = "result.mhd"
                potential_files = [f for f in os.listdir(self.config.output_path) if f.startswith('result.') and f.endswith('.mhd')]
                if potential_files:
                    # Sort them to get the last one (e.g., result.1.mhd after result.0.mhd)
                    potential_files.sort()
                    result_file = potential_files[-1]

                output_volume_path = os.path.join(self.config.output_path, result_file)
                
                if not os.path.exists(output_volume_path):
                     raise FileNotFoundError(f"Registration result file not found at {output_volume_path}")

                self.registered_data = sitk.GetArrayFromImage(sitk.ReadImage(output_volume_path))
                self.send_to_viewer_btn.setEnabled(True)
                QMessageBox.information(self, "Success", message)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load registration result: {e}")
        else:
            QMessageBox.warning(self, "Registration Failed", message)
            
    def send_to_viewer(self):
        if self.registered_data is not None:
            self.data_loaded.emit("mr", self.registered_data)
            QMessageBox.information(self, "Viewer", "Registration result sent to medical viewer.")