"""
Register Volumes using Elastix 5 - Standalone Version
Based on SlicerElastix implementation
"""
from __future__ import print_function
import os
import sys
import platform
import subprocess
import tempfile
import shutil
import SimpleITK as sitk


class RegisterVolumesElastix:
    """Handle volume registration using Elastix 5"""
    
    def __init__(self):
        self.verbose = True
        self.elastix_bin_dir = None
        self.elastix_lib_dir = None
        self.registration_parameter_files_dir = None
        
        # Set executable names based on platform
        executable_ext = '.exe' if platform.system() == 'Windows' else ''
        self.elastix_filename = 'elastix' + executable_ext
        self.transformix_filename = 'transformix' + executable_ext
        
        self.delete_temporary_files = True
        self.abort_requested = False
    
    def setElastixBinDir(self, path):
        """Set Elastix binary directory"""
        self.elastix_bin_dir = path
        if self.verbose:
            print(f"Elastix binary directory set to: {path}")
    
    def setRegistrationParameterFilesDir(self, path):
        """Set directory containing registration parameter files"""
        self.registration_parameter_files_dir = path
        if self.verbose:
            print(f"Registration parameters directory set to: {path}")
    
    def getElastixBinDir(self):
        """Get Elastix binary directory"""
        return self.elastix_bin_dir
    
    def getElastixEnv(self):
        """Create environment for elastix with proper library paths"""
        elastix_bin_dir = self.getElastixBinDir()
        elastix_env = os.environ.copy()
        
        # Add binaries to PATH
        if elastix_env.get("PATH"):
            elastix_env["PATH"] = elastix_bin_dir + os.pathsep + elastix_env["PATH"]
        else:
            elastix_env["PATH"] = elastix_bin_dir
        
        # Set library path for Linux/Mac
        if platform.system() != 'Windows':
            elastix_lib_dir = os.path.abspath(os.path.join(elastix_bin_dir, '../lib'))
            elastix_env["LD_LIBRARY_PATH"] = elastix_lib_dir
        
        return elastix_env
    
    def getStartupInfo(self):
        """Get startup info for subprocess (Windows only)"""
        if platform.system() != 'Windows':
            return None
        
        # Hide console window on Windows
        info = subprocess.STARTUPINFO()
        info.dwFlags = 1
        info.wShowWindow = 0
        return info
    
    def createTempDirectory(self):
        """Create temporary directory structure"""
        # Create main temp directory
        temp_dir = tempfile.mkdtemp(prefix='RadPathFusion_')
        
        # Create subdirectories
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir)
        
        result_transform_dir = os.path.join(temp_dir, 'result-transform')
        os.makedirs(result_transform_dir)
        
        result_resample_dir = os.path.join(temp_dir, 'result-resample')
        os.makedirs(result_resample_dir)
        
        if self.verbose:
            print(f"Created temporary directory: {temp_dir}")
        
        return temp_dir, input_dir, result_transform_dir, result_resample_dir
    
    def cleanUpTempFiles(self, path):
        """Clean up temporary files"""
        if self.delete_temporary_files and os.path.exists(path):
            shutil.rmtree(path)
            if self.verbose:
                print(f"Cleaned up temporary directory: {path}")
    
    def getInputParameters(self, fixed_volume, moving_volume,
                          parameter_filenames=None,
                          fixed_volume_mask=None,
                          moving_volume_mask=None):
        """
        Prepare input parameters for Elastix
        
        Args:
            fixed_volume: Fixed image (SimpleITK Image or path)
            moving_volume: Moving image (SimpleITK Image or path)
            parameter_filenames: List of parameter file names
            fixed_volume_mask: Fixed mask (optional)
            moving_volume_mask: Moving mask (optional)
            
        Returns:
            Tuple of (elastix_params, transformix_params, temp_dir, result_dir)
        """
        if self.verbose:
            print("Preparing registration parameters...")
        
        # Create temporary directories
        temp_dir, input_dir, result_transform_dir, result_resample_dir = \
            self.createTempDirectory()
        
        # Initialize parameter lists
        input_params_elastix = []
        
        # Save input volumes
        input_volumes = [
            (fixed_volume, 'fixed.mha', '-f'),
            (moving_volume, 'moving.mha', '-m'),
            (fixed_volume_mask, 'fixedMask.mha', '-fMask'),
            (moving_volume_mask, 'movingMask.mha', '-mMask')
        ]
        
        for volume, filename, param_name in input_volumes:
            if volume is None:
                continue
            
            file_path = os.path.join(input_dir, filename)
            
            # Save volume
            if isinstance(volume, str):
                # If path provided, copy file
                shutil.copy(volume, file_path)
            else:
                # If SimpleITK image, write it
                sitk.WriteImage(volume, file_path, True)  # useCompression=True
            
            input_params_elastix.append(param_name)
            input_params_elastix.append(file_path)
        
        # Specify output location
        input_params_elastix += ['-out', result_transform_dir]
        
        # Add parameter files
        if parameter_filenames is None:
            parameter_filenames = ['Parameters_Rigid.txt']
        
        for parameter_filename in parameter_filenames:
            input_params_elastix.append('-p')
            parameter_file_path = os.path.abspath(
                os.path.join(self.registration_parameter_files_dir, parameter_filename)
            )
            input_params_elastix.append(parameter_file_path)
        
        # Prepare transformix parameters
        input_params_transformix = [
            '-in', os.path.join(input_dir, 'moving.mha'),
            '-out', result_resample_dir,
            '-tp', os.path.join(
                result_transform_dir,
                f'TransformParameters.{len(parameter_filenames)-1}.txt'
            )
        ]
        
        return (input_params_elastix, input_params_transformix,
                temp_dir, result_resample_dir)
    
    def startElastix(self, cmd_line_arguments):
        """Start Elastix process"""
        executable_file_path = os.path.join(
            self.getElastixBinDir(),
            self.elastix_filename
        )
        
        if self.verbose:
            print("Register volumes...")
            print(f"Register volumes using: {executable_file_path}: {cmd_line_arguments}")
        
        if platform.system() == 'Windows':
            return subprocess.Popen(
                [executable_file_path] + cmd_line_arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                startupinfo=self.getStartupInfo()
            )
        else:
            return subprocess.Popen(
                [executable_file_path] + cmd_line_arguments,
                env=self.getElastixEnv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
    
    def startTransformix(self, cmd_line_arguments):
        """Start Transformix process"""
        executable_file_path = os.path.join(
            self.getElastixBinDir(),
            self.transformix_filename
        )
        
        if self.verbose:
            print("Generate output...")
            print(f"Generate output using: {executable_file_path}: {cmd_line_arguments}")
        
        if platform.system() == 'Windows':
            return subprocess.Popen(
                [executable_file_path] + cmd_line_arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                startupinfo=self.getStartupInfo()
            )
        else:
            return subprocess.Popen(
                [executable_file_path] + cmd_line_arguments,
                env=self.getElastixEnv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
    
    def logProcessOutput(self, process, total_steps=1, start_step=0):
        """
        Log process output and track progress
        
        Args:
            process: subprocess.Popen object
            total_steps: Total number of registration steps
            start_step: Starting step number
        """
        step_size = 1.0 / total_steps
        progress = start_step * step_size
        output_log = ""
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            output_log += line.rstrip() + "\n"
            
            # Check for progress indicators
            running_with_new_file1 = line[:35] == "Running elastix with parameter file"
            running_with_new_file2 = line[len(line)-14:] == "has finished.\n"
            
            if running_with_new_file1 and running_with_new_file2:
                progress += step_size
                if self.verbose:
                    print(f"Progress: {progress*100:.1f}%")
            
            # Check for abort
            if self.abort_requested:
                process.kill()
                raise ValueError("User requested cancel.")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code:
            if self.abort_requested:
                raise ValueError("User requested cancel.")
            else:
                print(output_log)
                print(f'Return code: {return_code}')
                raise subprocess.CalledProcessError(return_code, "elastix")
        
        return output_log
    
    def registerVolumes(self, fixed_volume, moving_volume,
                       parameter_filenames=None,
                       output_volume_path=None,
                       output_transform_path=None,
                       fixed_volume_mask=None,
                       moving_volume_mask=None):
        """
        Register two volumes using Elastix
        
        Args:
            fixed_volume: Fixed image (SimpleITK Image or path)
            moving_volume: Moving image (SimpleITK Image or path)
            parameter_filenames: List of parameter file names
            output_volume_path: Path to save registered volume
            output_transform_path: Path to save transform
            fixed_volume_mask: Fixed mask (optional)
            moving_volume_mask: Moving mask (optional)
        """
        # Get input parameters
        output = self.getInputParameters(
            fixed_volume,
            moving_volume,
            parameter_filenames,
            fixed_volume_mask,
            moving_volume_mask
        )
        
        input_params_elastix, input_params_transformix, temp_dir, result_resample_dir = output
        
        # Run Elastix registration
        ep = self.startElastix(input_params_elastix)
        self.logProcessOutput(ep, len(parameter_filenames) + 1, 0)
        
        # Run Transformix resampling
        tp = self.startTransformix(input_params_transformix)
        self.logProcessOutput(tp, len(parameter_filenames) + 1, len(parameter_filenames))
        
        # Load results
        if output_volume_path:
            output_volume_file = os.path.join(result_resample_dir, "result.mhd")
            if os.path.exists(output_volume_file):
                result_volume = sitk.ReadImage(output_volume_file)
                sitk.WriteImage(result_volume, output_volume_path)
                if self.verbose:
                    print(f"Registered volume saved to: {output_volume_path}")
            else:
                print(f"Warning: Result volume not found at {output_volume_file}")
        
        if output_transform_path:
            output_transform_file = os.path.join(result_resample_dir, "deformationField.mhd")
            if os.path.exists(output_transform_file):
                shutil.copy(output_transform_file, output_transform_path)
                # Also copy raw file
                raw_file = output_transform_file.replace('.mhd', '.raw')
                if os.path.exists(raw_file):
                    shutil.copy(raw_file, output_transform_path.replace('.mhd', '.raw'))
                if self.verbose:
                    print(f"Transform saved to: {output_transform_path}")
        
        # Clean up
        self.cleanUpTempFiles(temp_dir)