#!/usr/bin/env python3
"""
File principale dell'applicazione Histopathomix-Registration.
Questo script assembla i vari moduli (TAB) in un'unica finestra e gestisce
le interazioni tra di loro.
"""

import sys
import os
import json
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QTabWidget, QLabel, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor

# --- Importazione dei TAB dai loro moduli separati ---
try:
    from Resources.Utils.pathology_tab import PathologyTab
    from Resources.Utils.registration_tab import RegistrationTab
    from Resources.Utils.info_tab import create_info_tab
except ImportError as e:
    msg = (f"Errore: Impossibile importare i moduli dei tab. Assicurati che la "
           f"struttura delle cartelle sia corretta e che lo script sia eseguito "
           f"dalla cartella principale del progetto.\n\nDettagli: {e}")
    QMessageBox.critical(None, "Errore di Importazione", msg)
    sys.exit(1)

# --- Gestione delle dipendenze opzionali ---
try:
    from Resources.Utils.bigwarp_tab import BigWarpTab
    BIGWARP_AVAILABLE = True
except ImportError:
    BIGWARP_AVAILABLE = False
    print("Avviso: BigWarp non trovato. Il tab di registrazione deformabile non sarà disponibile.")

try:
    from medical_viewer_tab import MedicalViewer
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Avviso: Matplotlib non trovato. Il visualizzatore medico non sarà disponibile.")


class MainWindow(QMainWindow):
    """
    La finestra principale dell'applicazione.
    Crea e gestisce i tab, la menubar e le connessioni tra i componenti.
    """
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

        # --- Logo ---
        logo = QLabel()
        try:
            script_dir = Path(__file__).resolve().parent
            logo_path = script_dir / "Resources" / "Icons" / "logo.PNG"
            if logo_path.exists():
                pixmap = QPixmap(str(logo_path))
                logo.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                logo.setText("Logo non trovato")
        except Exception:
            logo.setText("Logo non caricato")
        logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(logo)

        # --- Creazione e assemblaggio dei Tab ---
        self.tabs = QTabWidget()

        self.pathology_tab = PathologyTab()
        self.registration_tab = RegistrationTab()
        info_tab = create_info_tab()

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
            placeholder_layout.addWidget(QLabel("Visualizzatore Medico non disponibile."))
            placeholder_layout.addWidget(QLabel("Installare matplotlib per abilitare questa funzionalità."))
            self.tabs.addTab(placeholder_tab, "Visualizzatore Medico")

        self.tabs.addTab(self.registration_tab, "4. Registration")
        self.tabs.addTab(info_tab, "Info")

        self.connect_signals()
        layout.addWidget(self.tabs)
        self.create_menu_bar()
        self.statusBar().showMessage("Ready")

    def connect_signals(self):
        """Centralizza tutte le connessioni segnale-slot tra i tab."""
        if BIGWARP_AVAILABLE:
            self.bigwarp_tab.rotation_data_ready.connect(self.pathology_tab.receive_rotation_data)
            if MATPLOTLIB_AVAILABLE:
                self.bigwarp_tab.data_loaded.connect(self.send_data_to_viewer)

        if MATPLOTLIB_AVAILABLE:
            self.pathology_tab.data_loaded.connect(self.send_data_to_viewer)
            self.registration_tab.data_loaded.connect(self.send_data_to_viewer)
            self.medical_viewer.mr_file_loaded.connect(self.registration_tab.add_fixed_volume_option)
            self.medical_viewer.seg_file_loaded.connect(self.registration_tab.add_fixed_mask_option)

    def send_data_to_viewer(self, data_type: str, data):
        """Invia dati al Medical Viewer e passa a quel tab."""
        if not self.medical_viewer or not self.medical_viewer.canvas:
            QMessageBox.warning(self, "Attenzione", "Il Medical Viewer non è disponibile.")
            return

        try:
            if data_type == "mr":
                self.medical_viewer.canvas.load_mr_data(data, "RM Registrata")
            elif data_type == "segmentation":
                self.medical_viewer.canvas.load_segmentation_data(data, "Segmentazione")
            elif data_type == "histology":
                self.medical_viewer.canvas.load_histology_data(data, "Istologico")
            self.medical_viewer.update_slice_controls()
            self.switch_to_viewer()
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore nell'invio dati al viewer: {e}")

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Nuovo Progetto', self.new_project)
        file_menu.addAction('Esci', self.close)

        view_menu = menubar.addMenu('Vista')
        view_menu.addAction('Vai al Viewer Medico', self.switch_to_viewer)
        view_menu.addAction('Reset Viewer', self.reset_viewer)

        help_menu = menubar.addMenu('Aiuto')
        help_menu.addAction('Informazioni', self.show_about)
        help_menu.addAction('Controlli Viewer', self.show_viewer_help)

    def switch_to_viewer(self):
        for i in range(self.tabs.count()):
            if "Medical Viewer" in self.tabs.tabText(i) or "Visualizzatore Medico" in self.tabs.tabText(i):
                self.tabs.setCurrentIndex(i)
                return
        QMessageBox.information(self, "Info", "Il tab del visualizzatore medico non è stato trovato.")

    def reset_viewer(self):
        if self.medical_viewer and self.medical_viewer.canvas:
            self.medical_viewer.canvas.reset_all()
            QMessageBox.information(self, "Reset", "Viewer medico resettato.")

    def show_viewer_help(self):
        help_text = """
        <h3>Controlli Viewer Medico:</h3>
        <ul>
            <li><b>Mouse Wheel:</b> Cambia slice nelle viste</li>
            <li><b>Checkbox Visibilità:</b> Accende/spegne i layer</li>
            <li><b>Slider Opacità:</b> Controlla trasparenza layer</li>
            <li><b>Window/Level:</b> Regola contrasto RM</li>
        </ul>
        """
        QMessageBox.information(self, "Aiuto Viewer", help_text)

    def new_project(self):
        reply = QMessageBox.question(self, "Nuovo Progetto",
                                     "Creare un nuovo progetto? I dati non salvati andranno persi.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.medical_viewer: self.reset_viewer()
            self.pathology_tab.reset_state()
            self.registration_tab.reset_state()
            if BIGWARP_AVAILABLE: self.bigwarp_tab.reset_state()
            self.statusBar().showMessage("Nuovo progetto creato.")
            QMessageBox.information(self, "Nuovo Progetto", "Nuovo progetto creato.")

    def show_about(self):
        QMessageBox.about(self, "About Histopathomix-Registration",
                          "Basato sul progetto originale:\n"
                          "https://github.com/pimed/Slicer-RadPathFusion")

def main():
    """Funzione principale: inizializza e avvia l'applicazione."""
    app = QApplication(sys.argv)
    app.setApplicationName("RadPathFusion")
    app.setApplicationVersion("2.1.0 Modular")

    style_path = Path(__file__).parent / "style.qss"
    if not style_path.exists():
        with open(style_path, "w") as f:
            # Scrive un CSS di default se non esiste
            f.write("""
                QMainWindow { background-color: #eeeeee; }
                QWidget { background-color: #ffffff; color: #000000; }
                QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 8px; margin-top: 15px; padding-top: 15px; }
                QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; background-color: #f5f5f5; }
                QPushButton { background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; font-size: 14px; margin: 4px 2px; border-radius: 6px; font-weight: bold; }
                QPushButton:hover { background-color: #45a049; }
                QPushButton:disabled { background-color: #cccccc; color: #666666; }
                QLineEdit, QTextEdit, QTableWidget, QComboBox, QSpinBox { padding: 8px; border: 2px solid #ddd; border-radius: 6px; }
                QTabBar::tab { background-color: #e1e1e1; color: #000000; padding: 12px 24px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; }
                QTabBar::tab:selected { background-color: #4CAF50; color: white; }
                QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
                QProgressBar::chunk { background-color: #4CAF50; width: 20px; }
            """)
    
    with open(style_path, "r") as f:
        app.setStyleSheet(f.read())

    if not MATPLOTLIB_AVAILABLE:
        QMessageBox.warning(None, "Dipendenza Mancante",
                            "Matplotlib non è installato. Il visualizzatore medico non sarà disponibile.\n"
                            "Installare con: pip install matplotlib")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()