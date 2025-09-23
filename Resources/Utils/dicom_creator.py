# dicom_creator.py
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, generate_uid
import datetime
import os

def create_dicom(
    slice_data_list,
    output_dicom_path,
    patient_name="Histology^Patient",
    patient_id="HISTO001",
    study_id="1001",
    series_number="1",
    series_description="Histological Volume",
    modality="XC",
    target_slice_thickness_mm=1.0
):
    """
    Crea un file DICOM multi-frame (Enhanced DICOM) da una lista di fette.

    Args:
        slice_data_list (list): Lista di tuple (numpy_array, z_position_mm, thickness_mm, instance_number).
        output_dicom_path (str): Percorso del file DICOM da salvare.
        ... altri metadati DICOM ...
    
    Returns:
        bool: True se la creazione ha avuto successo, altrimenti False.
    """
    if not slice_data_list or not slice_data_list[0][0] is not None:
        print("Errore: la lista delle fette o i dati delle immagini sono vuoti.")
        return False

    try:
        # Ordina le fette in base alla posizione Z o all'indice
        slice_data_list.sort(key=lambda x: x[1])

        # Ottieni i metadati di base dalla prima fetta
        first_slice_data = slice_data_list[0][0]
        rows, cols = first_slice_data.shape[:2]
        
        # Inizializza il dataset DICOM
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()

        ds = Dataset()
        ds.file_meta = file_meta
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # SC Image Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = modality
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyID = study_id
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_description
        ds.Manufacturer = "YourAppName"
        ds.InstanceCreationDate = datetime.date.today().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S.%f')

        # Dati immagine
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = 'RGB'
        
        # Dati per DICOM multi-frame
        ds.NumberOfFrames = len(slice_data_list)
        ds.FrameIncrementPointer = (0x0028, 0x9132) # Points to PerFrameFunctionalGroupsSequence

        # Aggiungi i dati delle fette
        pixel_array_list = []
        for slice_data, _, _, _ in slice_data_list:
            if slice_data.ndim == 3:
                # Se è un'immagine RGB (H, W, 3), converti in (3, H, W)
                slice_data = np.transpose(slice_data, (2, 0, 1))
            pixel_array_list.append(slice_data)

        # Crea un array 4D: (Fette, Canali, Altezza, Larghezza)
        pixel_data = np.stack(pixel_array_list, axis=0)
        ds.PixelData = pixel_data.tobytes()

        # Aggiungi i metadati per-frame
        ds.PerFrameFunctionalGroupsSequence = pydicom.sequence.Sequence()
        for i, (_, z_pos, thickness, instance_number) in enumerate(slice_data_list):
            item = Dataset()

            # Plane Position (Z)
            plane_position = Dataset()
            plane_position.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
            item.PlanePositionSequence = pydicom.sequence.Sequence([plane_position])

            # Pixel Measures (Spessore, Spaziatura)
            pixel_measures = Dataset()
            pixel_measures.SliceThickness = thickness
            # Placeholder per Pixel Spacing. Potrebbe essere necessario recuperarlo
            # da un'immagine DICOM di riferimento.
            pixel_measures.PixelSpacing = [1.0, 1.0] # Valore di default
            item.PixelMeasuresSequence = pydicom.sequence.Sequence([pixel_measures])
            
            # Frame Content
            frame_content = Dataset()
            frame_content.FrameAcquisitionNumber = instance_number + 1
            frame_content.FrameReferenceDateTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            item.FrameContentSequence = pydicom.sequence.Sequence([frame_content])

            ds.PerFrameFunctionalGroupsSequence.append(item)

        # Salva il file
        ds.save_as(output_dicom_path, write_like_as=False)
        return True

    except Exception as e:
        print(f"Errore durante la creazione del file DICOM: {e}")
        return False

if __name__ == '__main__':
    # Esempio di utilizzo
    print("Esempio di utilizzo non implementato in questo file.")
    print("Importare e chiamare la funzione 'create_dicom' dal proprio script principale.")