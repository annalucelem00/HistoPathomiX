from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

def create_info_tab() -> QWidget:
    """Crea e restituisce il widget per il tab 'Info'."""
    widget = QWidget()
    layout = QVBoxLayout(widget)

    info_text = QTextEdit()
    info_text.setReadOnly(True)
    info_text.setHtml(
        """
        <h3>README: Histopathomix-Registration</h3>
        <p>Questa applicazione facilita il processo di fusione di immagini radiologiche e istologiche attraverso una pipeline modulare.</p>
        <h4>Flusso di Lavoro Consigliato:</h4>
        <ol>
            <li>
                <b>BigWarp Registration (Tab 1):</b> 
                <ul>
                    <li>Avvia l'interfaccia esterna di BigWarp per eseguire una registrazione deformabile 2D tra le slice di patologia e le immagini di riferimento.</li>
                    <li>Importa i file JSON di trasformazione e associali alle immagini corrette.</li>
                    <li>Esporta un singolo file CSV contenente tutti gli angoli di rotazione.</li>
                </ul>
            </li>
            <li>
                <b>Parse Pathology (Tab 2):</b> 
                <ul>
                    <li>Carica il file JSON che descrive lo stack di patologia.</li>
                    <li>Importa il file CSV degli angoli generato al passo precedente.</li>
                    <li>Applica le rotazioni alle slice.</li>
                    <li>Carica il volume 3D, che verrà automaticamente inviato al viewer.</li>
                </ul>
            </li>
             <li>
                <b>Medical Viewer (Tab 3):</b> 
                <ul>
                    <li>Visualizza il volume istologico. Carica anche l'immagine RM e la sua segmentazione.</li>
                    <li>I volumi caricati qui popoleranno automaticamente i menu a tendina nel tab di registrazione.</li>
                </ul>
            </li>
            <li>
                <b>Registration (Tab 4):</b> 
                <ul>
                    <li>Configura i percorsi per la registrazione 3D con Elastix.</li>
                    <li>Seleziona i volumi (fixed, moving) dai menu a tendina.</li>
                    <li>Avvia la registrazione e, al termine, invia il risultato al viewer per la valutazione.</li>
                </ul>
            </li>
        </ol>
        """
    )

    layout.addWidget(info_text)
    return widget