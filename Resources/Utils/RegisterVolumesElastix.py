import os
import platform
import subprocess
import shutil
import datetime

class RegisterVolumesElastix:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.elastix_bin_dir = None
        self.registration_parameter_files_dir = None

        executable_ext = '.exe' if platform.system() == 'Windows' else ''
        self.elastix_filename = f'elastix{executable_ext}'
        self.transformix_filename = f'transformix{executable_ext}'

        self.delete_temporary_files = True
        self.abort_requested = False

    def set_elastix_bin_dir(self, path):
        """Sets the directory containing the Elastix and Transformix executables."""
        self.elastix_bin_dir = path

    def set_registration_parameter_files_dir(self, path):
        """Sets the directory containing the registration parameter files."""
        self.registration_parameter_files_dir = path

    def register_volumes(self,
                         fixed_volume_path,
                         moving_volume_path,
                         parameter_filenames=None,
                         output_dir=None,
                         fixed_volume_mask_path=None,
                         moving_volume_mask_path=None):
        """
        Performs volume registration using Elastix and Transformix.

        Args:
            fixed_volume_path (str): Path to the fixed volume file.
            moving_volume_path (str): Path to the moving volume file.
            parameter_filenames (list of str, optional): List of parameter files to use.
            output_dir (str, optional): Directory to save the output files. If None, a temporary directory is used.
            fixed_volume_mask_path (str, optional): Path to the fixed volume mask file.
            moving_volume_mask_path (str, optional): Path to the moving volume mask file.

        Returns:
            dict: A dictionary containing the paths to the output files.
        """
        if self.verbose:
            print("Starting volume registration process...")

        # Create temporary working directory
        if output_dir:
            temp_dir = output_dir
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            self.delete_temporary_files = False
        else:
            temp_dir = self._create_temp_directory()

        input_params_elastix, input_params_transformix = self._prepare_elastix_parameters(
            temp_dir,
            fixed_volume_path,
            moving_volume_path,
            parameter_filenames,
            fixed_volume_mask_path,
            moving_volume_mask_path
        )

        # Run registration and transformation
        self._run_process(self.elastix_filename, input_params_elastix)
        self._run_process(self.transformix_filename, input_params_transformix)

        # Determine output paths
        output_paths = {
            "output_volume": os.path.join(temp_dir, "result_resample", "result.mhd"),
            "output_transform": os.path.join(temp_dir, "result_transform", "TransformParameters.0.txt")
        }

        if self.delete_temporary_files:
            self._clean_up_temp_files(temp_dir)
        else:
            if self.verbose:
                print(f"Output files are located in: {temp_dir}")
        
        return output_paths

    def _prepare_elastix_parameters(self, temp_dir, fixed_path, moving_path, param_files, fixed_mask_path, moving_mask_path):
        """Prepares the command-line arguments for Elastix and Transformix."""
        input_dir = os.path.join(temp_dir, 'input')
        result_transform_dir = os.path.join(temp_dir, 'result_transform')
        result_resample_dir = os.path.join(temp_dir, 'result_resample')
        
        # Create subdirectories
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(result_transform_dir, exist_ok=True)
        os.makedirs(result_resample_dir, exist_ok=True)

        # Copy input files to the temporary input directory
        input_files = [
            (fixed_path, 'fixed.mha', '-f'),
            (moving_path, 'moving.mha', '-m')
        ]
        if fixed_mask_path:
            input_files.append((fixed_mask_path, 'fixedMask.mha', '-fMask'))
        if moving_mask_path:
            input_files.append((moving_mask_path, 'movingMask.mha', '-mMask'))
        
        elastix_params = []
        for original_path, filename, param_name in input_files:
            temp_path = os.path.join(input_dir, filename)
            shutil.copy(original_path, temp_path)
            elastix_params.extend([param_name, temp_path])

        # Add output and parameter files
        elastix_params.extend(['-out', result_transform_dir])
        if param_files:
            for param_file in param_files:
                param_path = os.path.join(self.registration_parameter_files_dir, param_file)
                if not os.path.exists(param_path):
                    raise FileNotFoundError(f"Parameter file not found: {param_path}")
                elastix_params.extend(['-p', param_path])
        else:
             raise ValueError("Parameter files must be provided.")

        # Prepare transformix parameters for resampling
        transformix_params = [
            '-in', os.path.join(input_dir, 'moving.mha'),
            '-out', result_resample_dir,
            '-tp', os.path.join(result_transform_dir, f'TransformParameters.{len(param_files) - 1}.txt')
        ]

        return elastix_params, transformix_params
    
    def _run_process(self, executable, cmd_line_arguments):
        """Runs a command-line executable and logs its output."""
        executable_path = os.path.join(self.elastix_bin_dir, executable)
        
        if self.verbose:
            print(f"\nRunning {executable_path}: {repr(cmd_line_arguments)}")

        startupinfo = self._get_startup_info()
        
        try:
            process = subprocess.Popen(
                [executable_path] + cmd_line_arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                startupinfo=startupinfo
            )
            stdout, stderr = process.communicate()
            return_code = process.wait()

            if self.verbose:
                print("--- Process STDOUT ---")
                print(stdout)
                print("--- Process STDERR ---")
                print(stderr)

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, executable, output=stdout, stderr=stderr)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Executable not found: {executable_path}. "
                                    "Please ensure `elastix_bin_dir` is set correctly.")
        except subprocess.CalledProcessError as e:
            if self.abort_requested:
                raise RuntimeError("User requested cancel.")
            else:
                raise RuntimeError(f"Elastix process failed with return code {e.returncode}. "
                                   f"Output: {e.output}\nError: {e.stderr}") from e

    def _create_temp_directory(self):
        """Creates a unique temporary directory for the process."""
        base_dir = os.path.join(os.path.expanduser('~'), 'ElastixTemp')
        os.makedirs(base_dir, exist_ok=True)
        
        temp_dir_name = datetime.datetime.now().strftime("Elastix_%Y%m%d_%H%M%S_%f")
        temp_dir_path = os.path.join(base_dir, temp_dir_name)
        os.makedirs(temp_dir_path)
        return temp_dir_path

    def _clean_up_temp_files(self, path):
        """Removes the temporary directory."""
        if os.path.exists(path):
            shutil.rmtree(path)
            if self.verbose:
                print(f"Cleaned up temporary directory: {path}")

    def _get_startup_info(self):
        """Creates a startup info object to hide the console window on Windows."""
        if platform.system() != 'Windows':
            return None
        info = subprocess.STARTUPINFO()
        info.dwFlags = subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = subprocess.SW_HIDE
        return info
