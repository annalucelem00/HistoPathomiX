# File: main.py
#in tesi_7_finale manipolo le cartelle!!! ATTENZIONE
import sys
import json
import os
import logging
from pathlib import Path


# Importazioni di base di PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QLabel, QTextEdit, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor, QPalette

# Importa le classi dei tab dai loro file dedicati
from pathology_parser_tab import PathologyParser
from registration_tab import RegistrationTab

# Importa le utility e le classi opzionali
from registration_volume_viewer_tab import RegistrationResultsViewer

# Gestisci importazioni opzionali per i tab
try:
    from bigwarp_tab import BigWarpTab
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
        
        layout = QVBoxLayout(central_widget)

        logo = QLabel()
        try:
            script_dir = Path(__file__).resolve().parent
            logo_path = script_dir / "Resources" / "Icons" / "logo.PNG"
            if logo_path.exists():
                pixmap = QPixmap(str(logo_path))
                pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo.setPixmap(pixmap)
            else:
                logo.setText("Logo non trovato")
        except Exception as e:
            logo.setText(f"Logo non caricato: {e}")
        logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(logo)

        self.tabs = QTabWidget()

        # Istanzia i tab importati
        self.pathology_tab = PathologyParser()
        self.registration_tab = RegistrationTab()
        info_tab = self.create_info_tab()


        # Gestisci i tab condizionali
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab = BigWarpTab()
            self.tabs.addTab(self.bigwarp_tab, "1. BigWarp Registration")
        
        self.tabs.addTab(self.pathology_tab, "2. Parse Pathology")
        
        if MATPLOTLIB_AVAILABLE:
            self.medical_viewer = MedicalViewer()
            self.tabs.addTab(self.medical_viewer, "3. Medical Viewer")
        else:
            placeholder_tab = QWidget()
            placeholder_layout = QVBoxLayout(placeholder_tab)
            placeholder_layout.addWidget(QLabel("Medical Viewer not available."))
            placeholder_layout.addWidget(QLabel("Please install matplotlib to enable this feature."))
            self.tabs.addTab(placeholder_tab, "Medical Viewer")

        self.medical_viewer.mr_file_loaded.connect(
        lambda name, path: self.registration_tab.update_from_viewer('mr', name, path))
        self.medical_viewer.seg_file_loaded.connect(
        lambda name, path: self.registration_tab.update_from_viewer('segmentation', name, path))
            
        self.tabs.addTab(self.registration_tab, "4. Registration")
        self.pathology_tab.pathology_volume_updated.connect(self.registration_tab.receive_pathology_volume)

        self.registration_results_tab = RegistrationResultsViewer()
        self.tabs.addTab(self.registration_results_tab, "5. Registration Results")
        self.pathology_tab.load_json_btn.clicked.connect(self.link_pathology_to_results_viewer)
        
        # Collega il riferimento a PathologyVolume
        self.pathology_tab.load_json_btn.clicked.connect(
            lambda: setattr(self.registration_results_tab, 'pathology_volume', self.pathology_tab.pathology_volume)
        )

        self.tabs.addTab(info_tab, "Info")

        # Connessioni dei segnali
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab.rotation_data_ready.connect(self.pathology_tab.receive_rotation_data)
            if MATPLOTLIB_AVAILABLE:
                self.bigwarp_tab.data_loaded.connect(self.send_data_to_viewer)
        
        if MATPLOTLIB_AVAILABLE:
            self.pathology_tab.data_loaded.connect(self.send_data_to_viewer)
            #self.registration_tab.data_loaded.connect(self.send_data_to_viewer)
            self.medical_viewer.mr_file_loaded.connect(
                lambda name, path: self.registration_tab.update_from_viewer('mr', name, path))
            self.medical_viewer.seg_file_loaded.connect(
                lambda name, path: self.registration_tab.update_from_viewer('segmentation', name, path))

            # <<< AGGIUNTA: Collega il parser al registration tab per i volumi MOVING >>>
            self.pathology_tab.moving_volume_generated.connect(
                lambda name, path: self.registration_tab.update_from_viewer('moving_volume', name, path))
            self.pathology_tab.moving_mask_generated.connect(
                lambda name, path: self.registration_tab.update_from_viewer('moving_mask', name, path))
            self.medical_viewer.status_message_changed.connect(self.statusBar().showMessage)

            self.registration_tab.registration_succeeded.connect(self.handle_registration_success)

        layout.addWidget(self.tabs)
        self.create_menu_bar()
        self.statusBar().showMessage("Ready")


        
    def send_data_to_viewer(self, data_type: str, data):
        if self.medical_viewer and self.medical_viewer.canvas:
            try:
                if data_type == "mr":
                    self.medical_viewer.canvas.load_mr_data(data, "Registered MR")
                elif data_type == "segmentation":
                    self.medical_viewer.canvas.load_segmentation_data(data, "Segmentation")
                elif data_type == "histology":
                    self.medical_viewer.canvas.load_histology_data(data, "Histology")
                self.medical_viewer.update_slice_controls()
                
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "3. Medical Viewer":
                        self.tabs.setCurrentIndex(i)
                        break
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error sending data to viewer: {e}")
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction('New Project', self.new_project)
        file_menu.addAction('Open Project', self.open_project)
        file_menu.addAction('Save Project', self.save_project)
        file_menu.addSeparator()
        file_menu.addAction('Exit', self.close)
       
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Go to Medical Viewer', self.switch_to_viewer)
        view_menu.addAction('Reset Viewer', self.reset_viewer)
    
        help_menu = menubar.addMenu('Help')
        help_menu.addAction('About', self.show_about)
        help_menu.addAction('Viewer Controls', self.show_viewer_help)
        
    def switch_to_viewer(self):
        for i in range(self.tabs.count()):
            if "Medical Viewer" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                break
                
    def reset_viewer(self):
        if self.medical_viewer and hasattr(self.medical_viewer, 'canvas') and self.medical_viewer.canvas:
            self.medical_viewer.canvas.mr_data = self.medical_viewer.canvas.__class__.ImageData("RM")
            self.medical_viewer.canvas.segmentation_data = self.medical_viewer.canvas.__class__.ImageData("Segmentazione") 
            self.medical_viewer.canvas.histology_data = self.medical_viewer.canvas.__class__.ImageData("Istologico")
            self.medical_viewer.canvas.update_display()
            QMessageBox.information(self, "Reset", "Medical viewer has been reset.")
            
    def show_viewer_help(self):
        help_text = """
        <h3>Medical Viewer Controls:</h3>
        <ul>
            <li><b>Mouse Wheel:</b> Change slice in views</li>
            <li><b>Visibility Checkbox:</b> Turn layers on/off</li>
            <li><b>Opacity Slider:</b> Control layer transparency</li>
            <li><b>Window/Level:</b> Adjust MR contrast</li>
            <li><b>Slice Control:</b> Navigate through slices</li>
        </ul>
        <h3>Layers:</h3>
        <ul>
            <li><b>MR:</b> Base grayscale layer</li>
            <li><b>Segmentation:</b> Red overlay</li>
            <li><b>Histology:</b> Colored overlay</li>
        </ul>
        """
        QMessageBox.information(self, "Viewer Help", help_text)

    def link_pathology_to_results_viewer(self):
        """
        This slot is called after the pathology JSON is loaded.
        It passes the pathology_volume object to the results viewer tab
        and enables the high-resolution button.
        """
        if self.registration_results_tab and self.pathology_tab.pathology_volume:
            self.registration_results_tab.pathology_volume = self.pathology_tab.pathology_volume
            # Abilita il pulsante high-res solo se anche un volume di istologia low-res è già stato caricato
            if self.registration_results_tab.canvas and self.registration_results_tab.canvas.histology_data is not None:
                self.registration_results_tab.load_highres_btn.setEnabled(True)
            print("PathologyVolume linked to Registration Results Viewer.")
   
    def create_info_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("<h3>READ ME:</h3><p>Application for Radiology-Pathology Fusion.</p>")
        layout.addWidget(info_text)
        return widget
        
    def new_project(self):
        reply = QMessageBox.question(
            self, "New Project", 
            "Create a new project? Unsaved data will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.medical_viewer:
                self.reset_viewer()
            
            # Pulisce il tab di Patologia
            self.pathology_tab.json_input.clear()
            self.pathology_tab.volume_input.clear() 
            self.pathology_tab.mask_input.clear()
            self.pathology_tab.json_info.clear()
            self.pathology_tab.slices_table.setRowCount(0)

            # <<< MODIFICA: Pulisce correttamente il tab di Registrazione >>>
            # Svuota i QComboBox e ripristina l'opzione di default
            for combo in [self.registration_tab.fixed_volume_combo,
                          self.registration_tab.fixed_mask_combo,
                          self.registration_tab.moving_volume_combo,
                          self.registration_tab.moving_mask_combo]:
                combo.clear()
                combo.addItem("Load from Viewer or Browse...")

            self.registration_tab.output_input.clear()
            self.registration_tab.elastix_input.clear()
            self.registration_tab.log_output.clear()
            QMessageBox.information(self, "New Project", "New project created.")

        
    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "RadPath projects (*.rpf);;JSON files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                if 'pathology' in project_data:
                    path_data = project_data['pathology']
                    self.pathology_tab.json_input.setText(path_data.get('json_path', ''))
                    self.pathology_tab.volume_input.setText(path_data.get('volume_name', ''))
                    self.pathology_tab.mask_input.setText(path_data.get('mask_name', ''))
                if 'registration' in project_data:
                    reg_data = project_data['registration'] 
                    self.registration_tab.json_display.setText(os.path.basename(reg_data.get('input_json', '')))
                    self.registration_tab.json_full_path = reg_data.get('input_json', '')
                    self.registration_tab.output_input.setText(reg_data.get('output_path', ''))
                    self.registration_tab.elastix_input.setText(reg_data.get('elastix_path', ''))
                QMessageBox.information(self, "Project", f"Project opened: {os.path.basename(file_path)}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error opening project: {e}")
            
    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "RadPath projects (*.rpf)"
        )
        if file_path:
            try:
                project_data = {
                    'version': '2.0.0',
                    'pathology': {
                        'json_path': self.pathology_tab.json_input.text(),
                        'volume_name': self.pathology_tab.volume_input.text(),
                        'mask_name': self.pathology_tab.mask_input.text(),
                        'mask_id': self.pathology_tab.mask_id_spin.value()
                    },
                    'registration': {
                        'input_json': self.registration_tab.json_full_path,
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
                QMessageBox.information(self, "Project", f"Project saved: {os.path.basename(file_path)}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving project: {e}")
            
    def show_about(self):
        QMessageBox.about(
            self, 
            "About RadPathFusion",
            "Based on the original project:\n"
            "https://github.com/pimed/Slicer-RadPathFusion"
        )

    @pyqtSlot(dict)
    def handle_registration_success(self, results_dict):
        """
        Questo slot viene chiamato quando la registrazione ha successo.
        1. Chiama lo slot del viewer per caricare i dati.
        2. Passa alla scheda del viewer per mostrare i risultati.
        """
        logger.info("Main window received registration_succeeded signal.")
        
        # Passa il dizionario dei risultati allo slot del viewer dei risultati
        if self.registration_results_tab:
            # Inside your main window class, in the handle_registration_success method
            self.registration_results_tab.load_data_from_registration(results_dict)
            
            # Cambia automaticamente il focus sulla scheda del viewer dei risultati
            self.tabs.setCurrentWidget(self.registration_results_tab)
        else:
            QMessageBox.warning(self, "Error", "Registration results viewer tab not found.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RadPathFusion")
    app.setApplicationVersion("2.0.0")

    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(238, 238, 238))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    app.setStyleSheet("""
        QMainWindow { background-color: #eeeeee; color: #ececec; }
        QWidget { background-color: #ffffff; color: #000000; }
        QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 8px; margin: 15px 0px; padding-top: 15px; background-color: #ffffff; color: #000000; }
        QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px 0 8px; background-color: #f5f5f5; color: #000000; }
        QPushButton { background-color: #4CAF50; border: none; color: #ffffff; padding: 10px 20px; text-align: center; font-size: 14px; margin: 4px 2px; border-radius: 6px; font-weight: bold; }
        QPushButton:hover { background-color: #45a049; }
        QPushButton:pressed { background-color: #3d8b40; }
        QPushButton:disabled { background-color: #cccccc; color: #666666; }
        QLineEdit { padding: 10px; border: 2px solid #ddd; border-radius: 6px; background-color: #ffffff; color: #000000; font-size: 14px; }
        QLineEdit:focus { border-color: #4CAF50; background-color: #f9f9f9; }
        QTabWidget::pane { border: 1px solid #cccccc; background-color: #ffffff; border-radius: 8px; }
        QTabBar::tab { background-color: #e1e1e1; color: #000000; padding: 12px 24px; margin-right: 2px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; }
        QTabBar::tab:selected { background-color: #4CAF50; color: #ffffff; }
        QTabBar::tab:hover { background-color: #d1d1d1; }
        QCheckBox { font-weight: bold; spacing: 8px; color: #000000; background-color: #ffffff; }
        QCheckBox::indicator { width: 20px; height: 20px; border-radius: 4px; border: 2px solid #cccccc; background-color: #ffffff; }
        QCheckBox::indicator:checked { background-color: #4CAF50; border-color: #4CAF50; }
        QLabel { color: #000000; background-color: transparent; }
        QSlider::groove:horizontal { border: 1px solid #bbb; background: #ffffff; height: 10px; border-radius: 4px; }
        QSlider::sub-page:horizontal { background: #4CAF50; border: 1px solid #777; height: 10px; border-radius: 4px; }
        QSlider::handle:horizontal { background: #4CAF50; border: 1px solid #5c5c5c; width: 18px; margin: -2px 0; border-radius: 3px; }
        QTextEdit, QTableWidget { border: 1px solid #cccccc; border-radius: 6px; background-color: #ffffff; color: #000000; padding: 8px; }
        QTableWidget::item { background-color: #ffffff; color: #000000; border: none; }
        QTableWidget::item:selected { background-color: #4CAF50; color: #ffffff; }
        QHeaderView::section { background-color: #f0f0f0; color: #000000; border: 1px solid #cccccc; padding: 8px; font-weight: bold; }
        QStatusBar { background-color: #e1e1e1; color: #000000; border-top: 1px solid #cccccc; font-weight: bold; }
        QScrollBar:vertical { background-color: #f0f0f0; width: 16px; border-radius: 8px; }
        QScrollBar::handle:vertical { background-color: #cccccc; border-radius: 8px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background-color: #999999; }
        QSpinBox { background-color: #ffffff; color: #000000; border: 2px solid #ddd; border-radius: 6px; padding: 8px; }
        QComboBox { background-color: #ffffff; color: #000000; border: 2px solid #ddd; border-radius: 6px; padding: 8px; }
        QComboBox::drop-down { border: none; background-color: #f0f0f0; }
        QComboBox QAbstractItemView { background-color: #ffffff; color: #000000; selection-background-color: #4CAF50; selection-color: #ffffff; }
        QProgressBar { background-color: #e0e0e0; border: 2px solid #cccccc; border-radius: 8px; text-align: center; color: #000000; }
        QProgressBar::chunk { background-color: #4CAF50; border-radius: 6px; }
        QMenuBar { background-color: #f0f0f0; color: #000000; border-bottom: 1px solid #cccccc; }
        QMenuBar::item { background-color: transparent; padding: 8px 12px; }
        QMenuBar::item:selected { background-color: #4CAF50; color: #ffffff; }
        QMenu { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; }
        QMenu::item { padding: 8px 20px; }
        QMenu::item:selected { background-color: #4CAF50; color: #ffffff; }
    """)
    
    if not MATPLOTLIB_AVAILABLE:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Missing Dependency")
        msg.setText("Matplotlib is not installed")
        msg.setInformativeText(
            "The medical viewer requires matplotlib.\n"
            "Install with: pip install matplotlib\n\n"
            "The application will continue with limited functionality."
        )
        msg.exec()
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()