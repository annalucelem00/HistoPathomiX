from __future__ import print_function
import os
import json
from time import sleep
from Resources.Utils.ParsePathJsonUtils import ParsePathJsonUtils

class ParsePathJson:
    def __init__(self, parent=None):
        self.title = "1. Parse Pathology"
        self.categories = ["Radiology-Pathology Fusion"]
        self.dependencies = []
        self.contributors = ["Mirabela Rusu (Stanford)"]
        self.helpText = """
        This module provides a basic functionality to parse and create a json file that will be used as an interface for radiology-pathology fusion.
        <br /><br />
        For detailed information about a specific model, please consult the piMed website.
        """
        self.acknowledgementText = """
        The developers would like to thank the support of the PiMed and Stanford University.
        """
        self.parent = parent


class ParsePathJsonWidget:
    """
    This class is a placeholder for the original Slicer GUI.
    The methods here would be called by a UI, but they now directly
    call the logic functions without a Slicer-specific GUI.
    """
    def __init__(self, parent=None):
        self.logic = ParsePathJsonLogic()
        self.verbose = True
        self.idxMask = None
        self.advancedOptions = None
        # Simulating widget values for a console-based approach
        self.inputJsonFn = ""
        self.outputVolumeNode = "mock_output_volume"
        self.outputMaskVolumeNode = "mock_output_mask_volume"
        self.inputVolumeSelector = "mock_input_volume"
        self.inputMaskSelector = "mock_input_mask_volume"

    def onLoadJson(self, json_path):
        if self.verbose:
            print("onLoadJson")
        self.inputJsonFn = json_path
        json_info = self.logic.getJsonInfo4UI(self.inputJsonFn)
        # In a real application, you would populate a UI with this data.
        print("JSON info loaded:", json_info)

    def onSaveJson(self, path_out_json):
        if self.verbose:
            print("onSaveJson")
        try:
            self.logic.saveJson(path_out_json)
        except Exception as e:
            print(f"Couldn't save: {e}")

    def onRefineVolume(self):
        if self.verbose:
            print("onRefineVolume")
        self.logic.refineVolume(self.inputJsonFn,
                                outputVolumeNode=self.outputVolumeNode)

    def onRegisterVolume(self):
        if self.verbose:
            print("onRegisterVolume")
        self.logic.registerVolume(self.inputJsonFn,
                                  self.inputVolumeSelector,
                                  self.inputMaskSelector,
                                  outputVolumeNode=self.outputVolumeNode)

    def onLoadVolume(self):
        if self.verbose:
            print("onLoadVolume")
        self.logic.loadRgbVolume(self.inputJsonFn,
                                 outputVolumeNode=self.outputVolumeNode)

    def onLoadMaskVolume(self, idx_mask):
        if self.verbose:
            print("onMaskLoadVolume")
        self.logic.loadMask(self.inputJsonFn, idx_mask,
                            outputMaskVolumeNode=self.outputMaskVolumeNode)

    def onMaskIDSelect(self, selector_index):
        if selector_index < 0:
            return
        self.idxMask = selector_index
        print("Selected Mask", self.idxMask)

    def onClearAll(self):
        print("Clear ALL not implemented in this version.")


class ParsePathJsonLogic:
    def __init__(self):
        self.verbose = True
        self.scriptPath = os.path.dirname(os.path.abspath(__file__))
        self.logic = None

    def yieldPythonGIL(self, seconds=0):
        sleep(seconds)

    def _initialize_logic(self, json_path):
        """Initializes the ParsePathJsonUtils logic component."""
        if not self.logic or str(self.logic.path) != str(json_path):
            import sys
            # Assuming ParsePathJsonUtils exists at this path
            sys.path.append(os.path.join(self.scriptPath, "Resources", "Utils"))
            import Resources.Utils.ParsePathJsonUtils as ppju
            self.logic = ppju.ParsePathJsonUtils()
            self.logic.setPath(json_path)
            success = self.logic.initComponents()
            if not success:
                raise RuntimeError("Failure to load json. Check path!")
        return True

    def loadRgbVolume(self, json_path, outputVolumeNode=None):
        if self.verbose:
            print("Loading RGB Volume from:", json_path)
        try:
            self._initialize_logic(json_path)
            if outputVolumeNode:
                outputVolume = self.logic.pathologyVolume.loadRgbVolume()
                # Instead of sitkUtils, you'd handle the volume data directly.
                print(f"RGB volume loaded. Would push to '{outputVolumeNode}'.")
                # Example: save to a file or return the data
                # self.logic.pathologyVolume.saveVolume(outputVolumeNode)
            else:
                raise ValueError("Output Volume was not set!")
        except Exception as e:
            print(f"Error in loadRgbVolume: {e}")

    def refineVolume(self, json_path, outputVolumeNode=None):
        if self.verbose:
            print("Refining Volume from:", json_path)
        try:
            self._initialize_logic(json_path)
            if outputVolumeNode:
                self.logic.pathologyVolume.registerSlices(False)
                outputVolume = self.logic.pathologyVolume.loadRgbVolume()
                print(f"Refined volume loaded. Would push to '{outputVolumeNode}'.")
            else:
                raise ValueError("Output Volume was not set!")
        except Exception as e:
            print(f"Error in refineVolume: {e}")

    def registerVolume(self, json_path, fixedNode, fixedMaskNode, outputVolumeNode=None):
        if self.verbose:
            print("Registering Volume from:", json_path)
        try:
            self._initialize_logic(json_path)
            if outputVolumeNode:
                # Replace slicer-specific data pull with your own data loading logic
                # self.logic.pathologyVolume.imagingContraint = load_volume_data(fixedNode)
                # self.logic.pathologyVolume.imagingContraintMask = load_volume_data(fixedMaskNode)
                self.logic.pathologyVolume.registerSlices(True)
                outputVolume = self.logic.pathologyVolume.loadRgbVolume()
                print(f"Registered volume loaded. Would push to '{outputVolumeNode}'.")
            else:
                raise ValueError("Output Volume was not set!")
        except Exception as e:
            print(f"Error in registerVolume: {e}")

    def loadMask(self, json_path, idxMask=0, outputMaskVolumeNode=None):
        if self.verbose:
            print("Loading Mask from:", json_path)
        try:
            self._initialize_logic(json_path)
            if idxMask >= 0 and outputMaskVolumeNode:
                outputVolume = self.logic.pathologyVolume.loadMask(idxMask)
                print(f"Mask loaded for ID {idxMask}. Would push to '{outputMaskVolumeNode}'.")
            else:
                raise ValueError("Output Mask was not set!")
        except Exception as e:
            print(f"Error in loadMask: {e}")

    def getJsonInfo4UI(self, json_path):
        if self.verbose:
            print("Reading json", json_path)
        try:
            self._initialize_logic(json_path)
            data = self.logic.pathologyVolume.getInfo4UI()
            return data
        except Exception as e:
            print(f"Error reading JSON info: {e}")
            return None

    def setIdxToSlice(self, idx, newSliceIdx):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSlice(idx, 'slice_number', int(newSliceIdx) - 1)

    def setRgbPathToSlice(self, idx, newPath):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSlice(idx, 'filename', newPath)

    def setFlipToSlice(self, idx, newFlip):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSlice(idx, 'flip', int(newFlip))

    def setRotateToSlice(self, idx, newRotate):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSlice(idx, 'rotation_angle', int(newRotate))

    def setMaskIdx(self, idxSlice, idxMask, newIdx):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSliceMask(idxSlice, idxMask, 'key', newIdx)

    def setMaskFilename(self, idxSlice, idxMask, value):
        if not self.logic:
            print("Logic doesn't exist")
            return
        self.logic.pathologyVolume.updateSliceMask(idxSlice, idxMask, 'filename', value)

    def saveJson(self, path):
        if not self.logic:
            print("Can't save, logic doesn't exist")
            return
        self.logic.pathologyVolume.saveJson(path)

    def deleteData(self):
        print("Deleting data from logic")
        if self.logic and hasattr(self.logic, 'PathologyVolume'):
            self.logic.PathologyVolume.deleteData()
        self.logic = None

    def test(self):
        print("Starting the test")
        # Test implementation would go here, e.g.,
        # self.loadRgbVolume("path/to/test.json", "test_output")
