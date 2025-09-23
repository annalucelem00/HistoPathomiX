from __future__ import print_function
import os
import subprocess
import sys
import shutil
from Resources.Utils.RegisterVolumesElastix import RegisterVolumesElastix

# To handle images, we'll use SimpleITK as it was already partially used
# in the original code and is a standard for medical image processing.
try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK is not installed. Please install it with 'pip install SimpleITK'.")
    sys.exit(1)

#
# Rad-path fusion logic
#
class RadPathFusionLogic:
    """
    A standalone class for performing radiology-pathology fusion using Elastix.
    """

    def __init__(self, elastix_path=None):
        self.elastix_path = elastix_path
        self.registration_logic = None
        self.verbose = True
        self.abort = False

        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.registration_parameter_files_dir = os.path.abspath(os.path.join(self.script_path,
                                                                             'Resources',
                                                                             'RegistrationParameters'))

    def set_elastix_path(self, path):
        """Sets the path to the Elastix executables."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Elastix executable directory not found at: {path}")
        self.elastix_path = path

    def _setup_registration_logic(self):
        # This part of the code assumes you have a a `RegisterVolumesElastix` module
        # that handles the command-line calls to Elastix.
        try:
            # Assumes RegisterVolumesElastix is in Resources/Utils relative to the script
            sys.path.append(os.path.join(self.script_path, "Resources", "Utils"))
            import Resources.Utils.RegisterVolumesElastix as rve
            self.registration_logic = rve.RegisterVolumesElastix()
            self.registration_logic.setElastixBinDir(self.elastix_path)
            self.registration_logic.setRegistrationParameterFilesDir(self.registration_parameter_files_dir)
        except ImportError:
            raise ImportError("Could not load RegisterVolumesElastix module. Make sure it's in the correct path.")

    def run(self, fixed_volume_path, moving_volume_path, output_volume_path, output_transform_path):
        """
        Performs the radiology-pathology fusion.

        Args:
            fixed_volume_path (str): Path to the fixed volume file.
            moving_volume_path (str): Path to the moving volume file.
            output_volume_path (str): Path to save the transformed output volume.
            output_transform_path (str): Path to save the resulting transform.
        """
        if self.verbose:
            print("Starting the fusion process...")

        self._setup_registration_logic()
        
        # Load SimpleITK images from file paths
        fixed_volume = sitk.ReadImage(fixed_volume_path)
        moving_volume = sitk.ReadImage(moving_volume_path)

        # The original code had a check for RGB images. We can keep that logic here.
        if moving_volume.GetNumberOfComponentsPerPixel() == 3:
            if self.verbose:
                print("Converting moving volume from RGB to grayscale.")
            moving_volume = sitk.GetImageFromArray(sitk.GetArrayFromImage(moving_volume).mean(axis=-1))

        parameter_filenames = ["QuickCenteringRigid.txt",
                               "Similarity.txt",
                               "Affine.txt",
                               "Deformable.txt"]

        # The original code's logic is encapsulated here.
        try:
            self.registration_logic.registerVolumes(
                fixed_volume,
                moving_volume,
                parameter_filenames=parameter_filenames,
                outputVolumeNode=output_volume_path,
                outputTransformNode=output_transform_path,
            )
            print("Fusion process completed successfully.")
        except Exception as e:
            print(f"An error occurred during fusion: {e}")
            if self.abort:
                print("Process was aborted by the user.")

    def test_elastix_logic(self, fixed_volume_path, moving_volume_path, output_volume_path):
        """
        A standalone test function for the registration logic.
        """
        print("Starting the standalone test.")
        self._setup_registration_logic()

        fixed_volume = sitk.ReadImage(fixed_volume_path)
        moving_volume = sitk.ReadImage(moving_volume_path)

        parameter_filenames = ["QuickCenteringRigid.txt"]
        
        self.registration_logic.registerVolumes(
            fixed_volume,
            moving_volume,
            parameter_filenames=parameter_filenames,
            outputVolumeNode=output_volume_path
        )
        print("Test completed successfully.")


#
# Example usage of the refactored logic (replaces the Slicer UI part)
#
def main():
    """
    A main function to demonstrate the usage of the refactored logic.
    This simulates the user interactions from the Slicer widget.
    """
    # 1. Configuration: Elastix executable path
    # Replace this with the actual path to your Elastix binaries
    elastix_bin_dir = "C:/Programs/elastix-4.9.0-win64/"

    try:
        fusion_logic = RadPathFusionLogic(elastix_path=elastix_bin_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Input and Output paths
    # Replace these with your actual file paths for testing
    fixed_vol_path = "path/to/fixed_volume.mha"
    moving_vol_path = "path/to/moving_volume.mha"
    output_vol_path = "path/to/output_volume.mha"
    output_transform_path = "path/to/output_transform.tfm"

    # Example 1: Run the main fusion process
    print("--- Running main fusion process ---")
    try:
        fusion_logic.run(fixed_vol_path, moving_vol_path, output_vol_path, output_transform_path)
    except Exception as e:
        print(f"Failed to run fusion process: {e}")

    # Example 2: Run the test logic
    print("\n--- Running test logic ---")
    try:
        fusion_logic.test_elastix_logic(fixed_vol_path, moving_vol_path, "path/to/test_output.mha")
    except Exception as e:
        print(f"Failed to run test logic: {e}")

if __name__ == "__main__":
    main()
