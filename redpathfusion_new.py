#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script autonomo per la fusione Radiologia-Patologia tramite registrazione di immagini con Elastix.

Questo script unifica la logica di diversi moduli di 3D Slicer, rimuovendo tutte le
dipendenze da Slicer per consentire l'esecuzione da riga di comando.

Funzionalità principali:
- Esegue la registrazione di un volume mobile (es. patologia) su un volume fisso (es. risonanza magnetica).
- Utilizza gli eseguibili command-line di Elastix e Transformix.
- Pre-elabora automaticamente le immagini a colori (RGB/RGBA) convertendole in scala di grigi,
  un passaggio necessario per molte immagini istologiche.
- Gestisce la creazione e la pulizia di file e directory temporanee.
- Fornisce un'interfaccia a riga di comando per una facile integrazione in pipeline.

Prerequisiti:
1. Python 3.x
2. Libreria SimpleITK: pip install SimpleITK
3. Installazione di Elastix: scaricare dal sito ufficiale e annotare il percorso della directory 'bin'.

Esempio di utilizzo da riga di comando:
python rad_path_fusion.py \
    --elastix-path /percorso/a/elastix-5.0.1-linux/bin \
    --params-path /percorso/a/RadPathFusion/Resources/RegistrationParameters \
    --fixed /dati/mri.nii.gz \
    --moving /dati/pathology.tiff \
    --output /risultati/fused_pathology.nii.gz \
    --output-transform /risultati/transform.mha
"""
from __future__ import print_function
import os
import sys
import shutil
import subprocess
import argparse
from time import sleep

try:
    import SimpleITK as sitk
    import numpy as np
except ImportError:
    print("Errore: La libreria SimpleITK non è installata.")
    print("Per favore, installala eseguendo: pip install SimpleITK")
    sys.exit(1)


class RadPathFusion:
    """
    Classe principale che orchestra il processo di registrazione di immagini 
    utilizzando Elastix e Transformix.
    """
    def __init__(self, elastix_path, params_path, verbose=True):
        self.verbose = verbose
        self.elastix_bin_dir = self._validate_elastix_path(elastix_path)
        self.params_dir = self._validate_params_path(params_path)
        
        # Nomi degli eseguibili con estensione per Windows
        executable_ext = '.exe' if sys.platform == 'win32' else ''
        self.elastix_executable = os.path.join(self.elastix_bin_dir, 'elastix' + executable_ext)
        self.transformix_executable = os.path.join(self.elastix_bin_dir, 'transformix' + executable_ext)

        if not os.path.exists(self.elastix_executable):
            raise FileNotFoundError(f"Eseguibile 'elastix' non trovato in: {self.elastix_bin_dir}")
        if not os.path.exists(self.transformix_executable):
            raise FileNotFoundError(f"Eseguibile 'transformix' non trovato in: {self.elastix_bin_dir}")

        self.parameter_files = [
            "QuickCenteringRigid.txt", 
            "Similarity.txt", 
            "Affine.txt", 
            "Deformable.txt"
        ]

    def _validate_elastix_path(self, path):
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"Il percorso di Elastix specificato non è una directory valida: {path}")
        return path

    def _validate_params_path(self, path):
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"Il percorso dei file di parametri non è una directory valida: {path}")
        return path
        
    def _create_temp_directory(self):
        """Crea una directory temporanea per contenere input, output e trasformate."""
        import tempfile
        base_temp_dir = tempfile.gettempdir()
        temp_dir_name = f"RadPathFusion_{os.getpid()}"
        full_temp_path = os.path.join(base_temp_dir, temp_dir_name)
        
        # Pulisce una eventuale directory residua da esecuzioni precedenti
        if os.path.exists(full_temp_path):
            shutil.rmtree(full_temp_path)
            
        # Crea le sotto-directory necessarie
        input_dir = os.path.join(full_temp_path, 'input')
        result_transform_dir = os.path.join(full_temp_path, 'result-transform')
        result_resample_dir = os.path.join(full_temp_path, 'result-resample')
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(result_transform_dir, exist_ok=True)
        os.makedirs(result_resample_dir, exist_ok=True)
        
        if self.verbose:
            print(f"Directory temporanea creata in: {full_temp_path}")
            
        return full_temp_path, input_dir, result_transform_dir, result_resample_dir

    def _prepare_input_image(self, image_path, output_dir, output_filename):
        """
        Carica un'immagine, la converte in scala di grigi se è RGB/RGBA,
        e la salva nel formato MHD per Elastix.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File immagine non trovato: {image_path}")
            
        try:
            sitk_image = sitk.ReadImage(image_path)
        except Exception as e:
            raise IOError(f"Impossibile leggere il file immagine {image_path} con SimpleITK. Errore: {e}")

        # Controlla se l'immagine è a colori (multi-componente)
        if sitk_image.GetNumberOfComponentsPerPixel() > 1:
            if self.verbose:
                print(f"L'immagine '{os.path.basename(image_path)}' è a colori. Conversione in scala di grigi...")
            
            # Converte l'immagine in un array NumPy
            arr = sitk.GetArrayFromImage(sitk_image)
            
            # Calcola la media lungo l'asse dei colori (ultimo asse per SimpleITK/NumPy)
            # Aggiunge un controllo per formati diversi (es. (z,y,x,c) o (y,x,c))
            if arr.ndim == 4 and arr.shape[3] in [3, 4]: # 3D a colori
                arr_out = np.mean(arr, axis=3).astype(arr.dtype)
            elif arr.ndim == 3 and arr.shape[2] in [3, 4]: # 2D a colori
                arr_out = np.mean(arr, axis=2).astype(arr.dtype)
            else:
                raise ValueError(f"Formato array immagine a colori non supportato: {arr.shape}")
                
            # Crea una nuova immagine SimpleITK in scala di grigi
            grayscale_image = sitk.GetImageFromArray(arr_out)
            grayscale_image.CopyInformation(sitk_image) # Copia metadati (spacing, origin, direction)
            sitk_image = grayscale_image

        output_path = os.path.join(output_dir, output_filename)
        sitk.WriteImage(sitk_image, output_path)
        return output_path

    def _get_elastix_env(self):
        """Crea un ambiente per elastix in cui gli eseguibili sono aggiunti al PATH."""
        elastix_env = os.environ.copy()
        
        # Aggiunge la directory bin al PATH
        if elastix_env.get("PATH"):
            elastix_env["PATH"] = self.elastix_bin_dir + os.pathsep + elastix_env["PATH"] 
        else:
            elastix_env["PATH"] = self.elastix_bin_dir

        # Su Linux/macOS, è cruciale impostare LD_LIBRARY_PATH
        if sys.platform.startswith('linux') or sys.platform == 'darwin':
            elastix_lib_dir = os.path.abspath(os.path.join(self.elastix_bin_dir, '../lib'))
            if os.path.isdir(elastix_lib_dir):
                # Sovrascrive per evitare conflitti con le librerie di sistema/altre applicazioni
                elastix_env["LD_LIBRARY_PATH"] = elastix_lib_dir
                if self.verbose:
                    print(f"Impostato LD_LIBRARY_PATH a: {elastix_lib_dir}")
        return elastix_env

    def _run_and_log_process(self, command_args):
        """Esegue un processo e stampa il suo output in tempo reale."""
        if self.verbose:
            print("\n" + "="*80)
            print(f"Esecuzione comando: {' '.join(command_args)}")
            print("="*80)
        
        env = self._get_elastix_env()
        
        # Su Windows, nasconde la finestra della console
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(command_args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, # Reindirizza stderr a stdout
                                   universal_newlines=True,
                                   env=env,
                                   startupinfo=startupinfo)

        output_log = ""
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_log += line
            if self.verbose:
                print(line.strip())
            sys.stdout.flush()

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print("\n" + "!"*80)
            print(f"ERRORE: Il processo è terminato con codice di uscita {return_code}.")
            print("Log completo dell'output:")
            print(output_log)
            print("!"*80)
            raise subprocess.CalledProcessError(return_code, command_args)

    def run(self, fixed_volume_path, moving_volume_path, output_volume_path, output_transform_path,
            fixed_mask_path=None, moving_mask_path=None):
        """
        Esegue l'intero workflow di registrazione.
        """
        temp_dir, input_dir, result_transform_dir, result_resample_dir = self._create_temp_directory()
        
        try:
            # --- 1. Preparazione degli input ---
            if self.verbose:
                print("\n--- Fase 1: Preparazione delle immagini di input ---")
            
            processed_fixed_path = self._prepare_input_image(fixed_volume_path, input_dir, "fixed.mhd")
            processed_moving_path = self._prepare_input_image(moving_volume_path, input_dir, "moving.mhd")
            
            elastix_cmd = [self.elastix_executable,
                           '-f', processed_fixed_path,
                           '-m', processed_moving_path,
                           '-out', result_transform_dir]

            if fixed_mask_path:
                processed_fmask_path = self._prepare_input_image(fixed_mask_path, input_dir, "fixedMask.mhd")
                elastix_cmd.extend(['-fMask', processed_fmask_path])
            
            if moving_mask_path:
                processed_mmask_path = self._prepare_input_image(moving_mask_path, input_dir, "movingMask.mhd")
                elastix_cmd.extend(['-mMask', processed_mmask_path])

            for param_file in self.parameter_files:
                param_file_path = os.path.join(self.params_dir, param_file)
                if not os.path.exists(param_file_path):
                    raise FileNotFoundError(f"File di parametri non trovato: {param_file_path}")
                elastix_cmd.extend(['-p', param_file_path])

            # --- 2. Esecuzione di Elastix ---
            if self.verbose:
                print("\n--- Fase 2: Esecuzione di Elastix per la registrazione ---")
            self._run_and_log_process(elastix_cmd)
            
            # --- 3. Esecuzione di Transformix ---
            if self.verbose:
                print("\n--- Fase 3: Esecuzione di Transformix per applicare la trasformata ---")
            
            final_transform_file = os.path.join(result_transform_dir, f'TransformParameters.{len(self.parameter_files)-1}.txt')
            if not os.path.exists(final_transform_file):
                raise FileNotFoundError(f"File di trasformazione finale non trovato: {final_transform_file}")
            
            transformix_cmd = [self.transformix_executable,
                               '-in', processed_moving_path,
                               '-out', result_resample_dir,
                               '-tp', final_transform_file]
            
            # Richiede a transformix di generare il campo di deformazione se richiesto un output
            if output_transform_path:
                transformix_cmd.extend(['-def', 'all'])
            
            self._run_and_log_process(transformix_cmd)
            
            # --- 4. Copia dei risultati ---
            if self.verbose:
                print("\n--- Fase 4: Finalizzazione e copia dei risultati ---")
            
            # Copia il volume risultante
            result_volume_temp_path = os.path.join(result_resample_dir, "result.mhd")
            shutil.copy(result_volume_temp_path, output_volume_path)
            # Copia anche il file raw associato
            shutil.copy(os.path.join(result_resample_dir, "result.raw"), 
                        os.path.join(os.path.dirname(output_volume_path), "result.raw"))
            if self.verbose:
                print(f"Volume registrato salvato in: {output_volume_path}")
                
            # Copia la trasformata risultante (campo di deformazione)
            if output_transform_path:
                result_transform_temp_path = os.path.join(result_resample_dir, "deformationField.mhd")
                if os.path.exists(result_transform_temp_path):
                    shutil.copy(result_transform_temp_path, output_transform_path)
                    shutil.copy(os.path.join(result_resample_dir, "deformationField.raw"),
                                os.path.join(os.path.dirname(output_transform_path), "deformationField.raw"))
                    if self.verbose:
                        print(f"Trasformata salvata in: {output_transform_path}")
                else:
                    print(f"ATTENZIONE: Il campo di deformazione non è stato generato in {result_transform_temp_path}")

        finally:
            # --- 5. Pulizia ---
            if self.verbose:
                print("\n--- Fase 5: Pulizia dei file temporanei ---")
            shutil.rmtree(temp_dir)
            if self.verbose:
                print(f"Directory temporanea {temp_dir} eliminata.")
        
        print("\nRegistrazione completata con successo!")


# =============================================================================
# Classe alternativa per la registrazione basata puramente su SimpleITK.
# Non è utilizzata nel workflow principale di Elastix, ma è inclusa per completezza.
# =============================================================================
class RegisterImages:
    """
    Esegue la registrazione di immagini 2D/3D usando solo SimpleITK.
    Questa classe è un'alternativa a Elastix per registrazioni più semplici.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def register_affine(self, fixed_img, moving_img, n_iterations=250):
        """Esegue una registrazione affine."""
        fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        moving = sitk.Cast(moving_img, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(fixed.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=n_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        registration_method.SetInitialTransform(initial_transform)

        if self.verbose:
            registration_method.AddCommand(sitk.sitkIterationEvent, 
                                           lambda: print(f"Iter: {registration_method.GetOptimizerIteration()} "
                                                         f"Metric: {registration_method.GetMetricValue():.4f}"))

        final_transform = registration_method.Execute(fixed, moving)
        
        if self.verbose:
            print(f"Condizione di stop: {registration_method.GetOptimizerStopConditionDescription()}")
            
        return final_transform

# =============================================================================
# Main execution block
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegue la registrazione di immagini Radiologia-Patologia usando Elastix.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Esempio:
  python rad_path_fusion.py \\
    --elastix-path "C:\\elastix-5.0.1-win64\\bin" \\
    --params-path ".\\Resources\\RegistrationParameters" \\
    --fixed "data\\mri_slice.nii.gz" \\
    --moving "data\\pathology_slide.tiff" \\
    --output "results\\registered_pathology.mhd" \\
    --output-transform "results\\final_transform.mhd"
"""
    )
    
    parser.add_argument("--elastix-path", type=str, required=True,
                        help="Percorso alla directory 'bin' della tua installazione di Elastix.")
    parser.add_argument("--params-path", type=str, required=True,
                        help="Percorso alla directory contenente i file di parametri di registrazione (es. Affine.txt).")
    parser.add_argument("--fixed", type=str, required=True,
                        help="Percorso al volume fisso (es. immagine RM).")
    parser.add_argument("--moving", type=str, required=True,
                        help="Percorso al volume mobile (es. immagine istologica).")
    parser.add_argument("--output", type=str, required=True,
                        help="Percorso del file di output per il volume mobile registrato.")
    parser.add_argument("--output-transform", type=str,
                        help="Percorso opzionale per salvare la trasformata finale (campo di deformazione).")
    parser.add_argument("--fixed-mask", type=str,
                        help="Percorso opzionale a una maschera per il volume fisso.")
    parser.add_argument("--moving-mask", type=str,
                        help="Percorso opzionale a una maschera per il volume mobile.")
    parser.add_argument("--verbose", action='store_true',
                        help="Abilita output dettagliato durante l'esecuzione.")

    args = parser.parse_args()

    try:
        # Crea le directory di output se non esistono
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.output_transform:
            os.makedirs(os.path.dirname(args.output_transform), exist_ok=True)

        # Istanzia e avvia il processo di fusione
        fusion_process = RadPathFusion(
            elastix_path=args.elastix_path,
            params_path=args.params_path,
            verbose=args.verbose
        )
        
        fusion_process.run(
            fixed_volume_path=args.fixed,
            moving_volume_path=args.moving,
            output_volume_path=args.output,
            output_transform_path=args.output_transform,
            fixed_mask_path=args.fixed_mask,
            moving_mask_path=args.moving_mask
        )

    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, IOError) as e:
        print(f"\nERRORE CRITICO: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nSi è verificato un errore inaspettato: {e}", file=sys.stderr)
        sys.exit(1)