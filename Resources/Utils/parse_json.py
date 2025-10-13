"""
Parse Pathology JSON Logic - Standalone Version
Handles loading and processing of pathology JSON files
"""
from __future__ import print_function
import os
import sys
import json
import SimpleITK as sitk
from image_stack import PathologyVolume


class ParsePathJsonLogic:
    """Logic for parsing and processing pathology JSON files"""
    
    def __init__(self):
        self.verbose = True
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.logic = None
        self.pathology_volume = None
        self.successful_initialization = False
        
    def loadRgbVolume(self, json_path, output_volume_path):
        """
        Load RGB volume from pathology JSON
        
        Args:
            json_path: Path to JSON file
            output_volume_path: Path to save output volume
        """
        if self.verbose:
            print("Loading RGB volume...")
        
        if not self.logic:
            self.logic = PathologyVolume()
            self.logic.verbose = self.verbose
            self.logic.setPath(json_path)
        
        if not str(self.logic.path) == str(json_path):
            self.logic.setPath(json_path)
        
        if not self.logic.successful_initialization:
            success = self.logic.initComponents()
            if not success:
                raise RuntimeError("Failure to load json. Check path!")
        
        # Load RGB volume
        output_volume = self.logic.loadRgbVolume()
        
        # Save volume
        sitk.WriteImage(output_volume, output_volume_path)
        
        if self.verbose:
            print(f"RGB volume saved to: {output_volume_path}")
    
    def refineVolume(self, json_path, output_volume_path):
        """
        Refine pathology volume (register slices internally)
        
        Args:
            json_path: Path to JSON file
            output_volume_path: Path to save refined volume
        """
        if self.verbose:
            print("Refining volume...")
        
        if not self.logic:
            self.logic = PathologyVolume()
            self.logic.verbose = self.verbose
            self.logic.setPath(json_path)
        
        if not str(self.logic.path) == str(json_path):
            self.logic.setPath(json_path)
        
        if not self.logic.successful_initialization:
            success = self.logic.initComponents()
            if not success:
                raise RuntimeError("Failure to load json. Check path!")
        
        # Set registration options
        self.logic.do_reconstruct = True
        self.logic.do_affine = True
        self.logic.do_deformable = False
        
        # Register slices internally
        self.logic.registerSlices(use_imaging_constraint=False)
        
        # Load refined RGB volume
        output_volume = self.logic.loadRgbVolume()
        
        # Save volume
        sitk.WriteImage(output_volume, output_volume_path)
        
        if self.verbose:
            print(f"Refined volume saved to: {output_volume_path}")
    
    def registerVolume(self, json_path, fixed_volume_path, fixed_mask_path,
                      output_volume_path, output_transform_path=None,
                      elastix_path=None):
        """
        Register pathology volume to radiology volume
        
        Args:
            json_path: Path to pathology JSON
            fixed_volume_path: Path to fixed (radiology) volume
            fixed_mask_path: Path to fixed volume mask
            output_volume_path: Path to save registered volume
            output_transform_path: Path to save transform (optional)
            elastix_path: Path to Elastix binaries
        """
        if self.verbose:
            print("Registering volumes...")
            print(f"Fixed volume: {fixed_volume_path}")
            print(f"Fixed mask: {fixed_mask_path}")
        
        if not self.logic:
            self.logic = PathologyVolume()
            self.logic.verbose = self.verbose
            self.logic.setPath(json_path)
        
        if not str(self.logic.path) == str(json_path):
            self.logic.setPath(json_path)
        
        if not self.logic.successful_initialization:
            success = self.logic.initComponents()
            if not success:
                raise RuntimeError("Failure to load json. Check path!")
        
        # Load fixed volume and mask
        fixed_volume = sitk.ReadImage(fixed_volume_path, sitk.sitkFloat32)
        fixed_mask = sitk.ReadImage(fixed_mask_path)
        
        # Set imaging constraints
        self.logic.imaging_constraint = fixed_volume
        self.logic.imaging_constraint_mask = fixed_mask
        
        # Set registration options
        self.logic.do_affine = True
        self.logic.do_deformable = False
        self.logic.fast_execution = False
        
        # Register slices to imaging constraint
        self.logic.registerSlices(use_imaging_constraint=True)
        
        # Load registered volume
        output_volume = self.logic.loadRgbVolume()
        
        # Save volume
        sitk.WriteImage(output_volume, output_volume_path)
        
        if self.verbose:
            print(f"Registered volume saved to: {output_volume_path}")
    
    def loadMask(self, json_path, idx_mask, output_mask_path):
        """
        Load specific mask from pathology data
        
        Args:
            json_path: Path to JSON file
            idx_mask: Mask index to load
            output_mask_path: Path to save mask
        """
        if self.verbose:
            print(f"Loading mask {idx_mask}...")
        
        if not self.logic:
            self.logic = PathologyVolume()
            self.logic.verbose = self.verbose
            self.logic.setPath(json_path)
        
        if not str(self.logic.path) == str(json_path):
            self.logic.setPath(json_path)
        
        if not self.logic.successful_initialization:
            success = self.logic.initComponents()
            if not success:
                raise RuntimeError("Failure to load json. Check path!")
        
        # Load mask
        if idx_mask >= 0:
            output_mask = self.logic.loadMask(idx_mask)
            
            # Save mask
            sitk.WriteImage(output_mask, output_mask_path)
            
            if self.verbose:
                print(f"Mask saved to: {output_mask_path}")
        else:
            raise ValueError(f"Invalid mask index: {idx_mask}")
    
    def getJsonInfo4UI(self, json_path):
        """
        Get JSON information for UI display
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            List of slice information
        """
        if self.verbose:
            print(f"Reading json: {json_path}")
        
        if not self.logic:
            self.logic = PathologyVolume()
            self.logic.verbose = self.verbose
        
        self.logic.setPath(json_path)
        self.logic.initComponents()
        
        data = self.logic.getInfo4UI()
        
        return data
    
    def setIdxToSlice(self, idx, new_slice_idx):
        """Update slice index in pathology volume"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        # Internally idx starts at 0, but in UI it starts at 1
        self.logic.updateSlice(idx, 'slice_number', int(new_slice_idx) - 1)
    
    def setRgbPathToSlice(self, idx, new_path):
        """Update RGB image path for slice"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        self.logic.updateSlice(idx, 'filename', new_path)
    
    def setFlipToSlice(self, idx, new_flip):
        """Update flip parameter for slice"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        self.logic.updateSlice(idx, 'flip', int(new_flip))
    
    def setRotateToSlice(self, idx, new_rotate):
        """Update rotation angle for slice"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        self.logic.updateSlice(idx, 'rotation_angle', int(new_rotate))
    
    def setMaskIdx(self, idx_slice, idx_mask, new_idx):
        """Update mask index"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        self.logic.updateSliceMask(idx_slice, idx_mask, 'key', new_idx)
    
    def setMaskFilename(self, idx_slice, idx_mask, value):
        """Update mask filename"""
        if not self.logic:
            print("Logic doesn't exist")
            return
        
        self.logic.updateSliceMask(idx_slice, idx_mask, 'filename', value)
    
    def saveJson(self, path):
        """Save modified JSON to file"""
        if not self.logic:
            print("Can't save - Logic doesn't exist")
            return
        
        self.logic.saveJson(path)
        
        if self.verbose:
            print(f"JSON saved to: {path}")
    
    def deleteData(self):
        """Clean up and delete data"""
        if self.verbose:
            print("Deleting volume from logic")
        
        if self.logic:
            self.logic.deleteData()
            self.logic = None