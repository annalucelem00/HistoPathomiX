"""
Author: Mirabela Rusu
Refactored from original code created: 2018 Aug 16
"""

from __future__ import print_function
from Resources.Utils.ImageStack_3 import PathologyVolume

class ParsePathJsonUtils:
    def __init__(self, path=None, verbose=False):
        """
        Initializes the ParsePathJsonUtils instance.

        Args:
            path (str, optional): The initial path to the JSON file. Defaults to None.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        self.verbose = verbose
        self.path = path
        self.pathology_volume = PathologyVolume()
        self.initialized_successfully = None

    def set_path(self, path):
        """
        Sets the file path for the JSON data.
        
        This method allows setting the path after the object has been created.
        
        Args:
            path (str): The file path to the JSON data.
        """
        self.path = path

    def initialize_components(self):
        """
        Initializes the pathology volume from the stored path.
        
        This method validates the path and attempts to initialize the
        underlying PathologyVolume object.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if self.verbose:
            print("Attempting to initialize components...")

        if not self.path:
            if self.verbose:
                print("Error: No path has been set. Initialization failed.")
            self.initialized_successfully = False
            return False

        self.pathology_volume.setPath(self.path)
        
        # The return value from initComponents in the original code seems to be
        # a success code, but the comment suggests it's 0 on success.
        # We'll assume a boolean return for clarity.
        success = self.pathology_volume.initComponents()
        
        self.initialized_successfully = bool(success)
        
        if self.verbose:
            status = "succeeded" if self.initialized_successfully else "failed"
            print(f"Initialization {status}.")

        return self.initialized_successfully
