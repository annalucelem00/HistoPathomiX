"""
Image Stack - Pathology Volume and Slice Management
Handles 3D reconstruction from 2D histology slices
"""
from __future__ import print_function
import os
import json
import numpy as np
import SimpleITK as sitk
from image_registration import RegisterImages


class PathologyVolume:
    """Manage pathology volume from multiple 2D slices"""
    
    def __init__(self, parent=None):
        self.verbose = False
        self.path = None
        
        self.no_regions = 0
        self.region_ids = None
        self.no_slices = 0
        
        # Pixel size in micrometers
        self.pix_size_x = 0
        self.pix_size_y = 0
        
        # Max image size
        self.max_slice_size = [0, 0]
        self.volume_size = [0, 0, 0]
        self.rgb_volume = None
        self.store_volume = False
        
        self.in_plane_scaling = 1.2
        
        self.pathology_slices = None
        self.json_dict = None
        
        # Imaging constraint files
        self.imaging_constraint_filename = None
        self.imaging_constraint_mask_filename = None
        
        # Imaging constraint images
        self.imaging_constraint = None
        self.imaging_constraint_mask = None
        
        # Volume properties
        self.volume_origin = None
        self.volume_direction = None
        self.volume_spacing = None
        
        # Reference volumes
        self.ref_wo_constraints = None
        self.ref_w_constraints = None
        self.msk_ref_wo_constraints = None
        self.msk_ref_w_constraints = None
        
        # Registration options
        self.do_affine = True
        self.do_deformable = None
        self.do_reconstruct = None
        self.fast_execution = None
        self.discard_orientation = None
        
        self.successful_initialization = False
    
    def initComponents(self):
        """
        Initialize components by reading JSON and determining volume size
        Does NOT read actual images or create the volume
        """
        if self.verbose:
            print("PathologyVolume: Initialize components")
        
        if not self.path:
            print("The path was not set")
            self.successful_initialization = False
            return 0
        
        if self.verbose:
            print(f"PathologyVolume: Loading from {self.path}")
        
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(e)
            self.successful_initialization = False
            return 0
        
        self.json_dict = data
        self.pix_size_x = 0
        self.pix_size_y = 0
        
        self.pathology_slices = []
        self.region_ids = []
        
        # Process each slice in JSON
        print(np.sort(list(self.json_dict)))
        for key in np.sort(list(self.json_dict)):
            ps = PathologySlice()
            ps.json_key = key
            ps.rgb_image_fn = data[key]['filename']
            ps.mask_dict = data[key]['regions']
            ps.id = data[key]['id']
            
            ps.do_flip = None
            ps.do_rotate = None
            
            # Handle transform information
            if 'transform' in data[key]:  # New format
                ps.transform_dict = data[key]['transform']
                if 'flip' in ps.transform_dict:
                    ps.do_flip = ps.transform_dict['flip']
                if 'rotation_angle' in ps.transform_dict:
                    ps.do_rotate = ps.transform_dict['rotation_angle']
            else:  # Old format
                if 'flip' in data[key]:
                    ps.do_flip = int(data[key]['flip'])
                if 'rotate' in data[key]:
                    ps.do_rotate = data[key].get('rotate', None)
            
            # Set defaults if not found
            if ps.do_flip is None:
                print("Setting default parameters")
                ps.do_flip = 0
                ps.do_rotate = 0
            
            # Load image size
            ps.loadImageSize()
            size = ps.rgb_image_size
            
            # Update max slice size
            for dim in range(ps.dimension):
                if self.max_slice_size[dim] < size[dim]:
                    self.max_slice_size[dim] = size[dim]
            
            # Get slice index
            idx = data[key].get('slice_number', None)
            if idx:
                # Assumes numbering in JSON starts from 1, but Python starts at 0
                ps.ref_slice_idx = int(idx) - 1
            else:
                ps.ref_slice_idx = len(self.pathology_slices)
            
            # Collect region IDs
            for r in list(data[key]['regions']):
                if r not in self.region_ids:
                    self.region_ids.append(r)
            
            self.no_regions = len(self.region_ids)
            ps.region_ids = self.region_ids
            
            self.pathology_slices.append(ps)
            
            # Get pixel resolution
            xml_res_x = None
            if 'resolution_x_um' in data[key]:
                xml_res_x = float(data[key]['resolution_x_um'])
            if 'resolution_x' in data[key]:
                xml_res_x = float(data[key]['resolution_x'])
            if xml_res_x is None:
                xml_res_x = 0
            
            xml_res_y = None
            if 'resolution_y_um' in data[key]:
                xml_res_y = float(data[key]['resolution_y_um'])
            if 'resolution_y' in data[key]:
                xml_res_y = float(data[key]['resolution_y'])
            if xml_res_y is None:
                xml_res_y = 0
            
            # Update minimum pixel size
            if self.pix_size_x == 0 and xml_res_x > 0:
                self.pix_size_x = xml_res_x
            if self.pix_size_y == 0 and xml_res_y > 0:
                self.pix_size_y = xml_res_y
            
            if self.pix_size_x > xml_res_x > 0:
                self.pix_size_x = xml_res_x
            if self.pix_size_y > xml_res_y > 0:
                self.pix_size_y = xml_res_y
        
        self.no_slices = len(list(data))
        
        # Set volume size
        self.volume_size = [
            int(self.max_slice_size[0] * self.in_plane_scaling),
            int(self.max_slice_size[1] * self.in_plane_scaling),
            self.no_slices
        ]
        
        if self.verbose:
            print(f"PathologyVolume: Found {self.no_slices} slices @ max size {self.max_slice_size}")
            print(f"PathologyVolume: Create volume at {self.volume_size}")
        
        self.successful_initialization = True
        return 1
    
    def printTransform(self, ref=None):
        """Print transforms for all slices"""
        for i, ps in enumerate(self.pathology_slices):
            print(i, ps.transform)
    
    def setPath(self, path):
        """Set path to JSON file"""
        self.path = path
    
    def loadRgbVolume(self):
        """Load RGB volume from all slices"""
        if self.verbose:
            print("Loading RGB")
        
        # Create new volume with white background
        vol = sitk.Image(self.volume_size, sitk.sitkVectorUInt8, 3)
        if self.volume_origin:
            vol.SetOrigin(self.volume_origin)
        if self.volume_direction:
            vol.SetDirection(self.volume_direction)
        
        is_spacing_set = False
        if self.volume_spacing:
            vol.SetSpacing(self.volume_spacing)
            is_spacing_set = True
        
        # Fill the volume
        for i, ps in enumerate(self.pathology_slices):
            if self.verbose:
                print(f"Loading slice {i}")
            
            if not is_spacing_set:
                ps.fast_execution = self.fast_execution
                im = ps.loadRgbImage()
                
                if not im:
                    continue
                
                # Set spacing based on first image
                im_sp = im.GetSpacing()
                vol_sp = [s for s in im_sp]
                vol_sp.append(1.0)
                vol.SetSpacing(vol_sp)
                is_spacing_set = True
            
            if not ps.ref_size:
                ps.setReference(vol)
            
            relative_idx = int(i > 0)
            ps.fast_execution = self.fast_execution
            vol = ps.setTransformedRgb(vol, relative_idx)
        
        if self.store_volume:
            self.rgb_volume = vol
            return self.rgb_volume
        else:
            return vol
    
    def loadMask(self, idx_mask=0):
        """Load all masks from a specific region"""
        # Create new volume
        vol = sitk.Image(self.volume_size, sitk.sitkUInt8)
        if self.volume_origin:
            vol.SetOrigin(self.volume_origin)
        if self.volume_direction:
            vol.SetDirection(self.volume_direction)
        
        is_spacing_set = False
        if self.volume_spacing:
            vol.SetSpacing(self.volume_spacing)
            is_spacing_set = True
        
        # Fill the volume
        for i, ps in enumerate(self.pathology_slices):
            if not is_spacing_set:
                ps.fast_execution = self.fast_execution
                im = ps.loadMask(idx_mask)
                
                if not im:
                    continue
                
                # Set spacing based on first image
                im_sp = im.GetSpacing()
                vol_sp = [s for s in im_sp]
                vol_sp.append(1.0)
                vol.SetSpacing(vol_sp)
                is_spacing_set = True
            
            if not ps.ref_size:
                ps.setReference(vol)
            
            relative_idx = int(i > 0)
            vol = ps.setTransformedMask(vol, idx_mask, relative_idx)
        
        return vol
    
    def getInfo4UI(self):
        """Get information for UI display"""
        data = []
        
        for idx, ps in enumerate(self.pathology_slices):
            masks = []
            for mask_key in list(ps.mask_dict):
                fn = ps.mask_dict[mask_key]['filename']
                try:
                    read_idx_mask = int(mask_key[6:])
                except:
                    read_idx_mask = 1
                masks.append([read_idx_mask, fn])
            
            el = [
                idx,
                ps.ref_slice_idx + 1,  # Start count from 1 in UI
                ps.rgb_image_fn,
                masks,
                ps.do_flip,
                ps.do_rotate
            ]
            data.append(el)
        
        return data
    
    def updateSlice(self, idx, param, value):
        """Update slice parameters"""
        if len(self.pathology_slices) > idx:
            # Transform needs to be updated
            self.pathology_slices[idx].transform = None
            json_key = False
            
            if param == 'slice_number':
                self.pathology_slices[idx].ref_slice_idx = value
                json_key = True
                json_value = value + 1
            
            elif param == 'filename':
                self.pathology_slices[idx].rgb_image_fn = value
                json_key = True
                json_value = str(value)
            
            elif param == 'flip':
                self.pathology_slices[idx].do_flip = value
                json_key = True
                json_value = value
            
            elif param == 'rotation_angle':
                self.pathology_slices[idx].do_rotate = value
                json_key = True
                print(f'Rotating {idx}, {self.pathology_slices[idx].do_rotate}')
                json_value = value
            
            if not json_key:
                print(f"Adding new key {param}")
            
            # Update JSON dict
            if param in ['flip', 'rotation_angle']:
                if 'transform' not in self.json_dict[self.pathology_slices[idx].json_key]:
                    self.json_dict[self.pathology_slices[idx].json_key]['transform'] = {}
                self.json_dict[self.pathology_slices[idx].json_key]['transform'][param] = json_value
            else:
                self.json_dict[self.pathology_slices[idx].json_key][param] = json_value
    
    def updateSliceMask(self, idx_slice, idx_mask, param, value):
        """Update mask parameters"""
        if len(self.pathology_slices) > idx_slice:
            if param == 'key':
                old_key = f'region{idx_mask}'
                new_key = f'region{int(value)}'
                self.pathology_slices[idx_slice].mask_dict[new_key] = \
                    self.pathology_slices[idx_slice].mask_dict[old_key]
                del self.pathology_slices[idx_slice].mask_dict[old_key]
            
            elif param == 'filename':
                self.pathology_slices[idx_slice].mask_dict[f'region{idx_mask}']['filename'] = value
    
    def saveJson(self, path_out_json):
        """Save JSON file"""
        if self.verbose:
            print("Saving Json File")
        
        with open(path_out_json, 'w') as outfile:
            json.dump(self.json_dict, outfile, indent=4, sort_keys=True)


    def getConstraint(self):
        """Get imaging constraint and prepare it for registration"""
        # Load constraint if filename provided
        if not self.imaging_constraint and self.imaging_constraint_filename:
            try:
                print("Reading the fixed image")
                self.imaging_constraint = sitk.ReadImage(
                    self.imaging_constraint_filename, sitk.sitkFloat32
                )
                if self.discard_orientation:
                    if self.verbose:
                        print("Discarding Orientation")
                    tr = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                    self.imaging_constraint.SetDirection(tr)
            except Exception as e:
                print(e)
        
        # Load mask if filename provided
        if not self.imaging_constraint_mask and self.imaging_constraint_mask_filename:
            try:
                self.imaging_constraint_mask = sitk.ReadImage(
                    self.imaging_constraint_mask_filename
                )
                if self.discard_orientation:
                    tr = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                    self.imaging_constraint_mask.SetDirection(tr)
            except Exception as e:
                print(e)
        
        # Resample mask to constraint
        self.imaging_constraint_mask = sitk.Cast(
            sitk.Resample(
                self.imaging_constraint_mask > 0,
                self.imaging_constraint,
                sitk.Transform(),
                sitk.sitkNearestNeighbor
            ),
            sitk.sitkUInt16
        )
        
        # Get bounding box
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(self.imaging_constraint, self.imaging_constraint_mask)
        box = label_stats.GetBoundingBox(1)
        
        # Pad the box
        pad = 15
        min_x = max(box[0] - pad, 0)
        max_x = min(box[1] + pad, self.imaging_constraint.GetSize()[0])
        min_y = max(box[2] - pad, 0)
        max_y = min(box[3] + pad, self.imaging_constraint.GetSize()[1])
        
        crop_im_c = self.imaging_constraint[min_x:max_x, min_y:max_y, box[4]:box[5]+1]
        box_size = crop_im_c.GetSize()
        sp = self.imaging_constraint.GetSpacing()
        
        # Create image with same pixel size as pathology volume
        # Convert mm (from constraint) to um (from histology)
        roi_size = [
            int(box_size[0] * 1000 * sp[0] / self.pix_size_x),
            int(box_size[1] * 1000 * sp[1] / self.pix_size_y),
            int(box_size[2])
        ]
        roi_sp = [self.pix_size_x / 1000.0, self.pix_size_y / 1000.0, sp[2]]
        constraint = sitk.Image(roi_size, sitk.sitkUInt16)
        constraint.SetSpacing(roi_sp)
        constraint.SetOrigin(crop_im_c.GetOrigin())
        constraint.SetDirection(crop_im_c.GetDirection())
        
        # Resample to new size
        im_cm = sitk.Cast(
            sitk.Resample(
                self.imaging_constraint_mask, constraint,
                sitk.Transform(), sitk.sitkNearestNeighbor
            ),
            sitk.sitkUInt16
        )
        
        im_c = sitk.Cast(
            sitk.Resample(
                self.imaging_constraint, constraint,
                sitk.Transform(), sitk.sitkLinear
            ),
            sitk.sitkUInt16
        )
        
        im_c = im_c * im_cm
        constraint_range = [i for i in range(im_c.GetSize()[2])]
        
        return im_c, constraint_range

    def registerSlices(self, use_imaging_constraint=False):
        """Register slices internally or to imaging constraint"""
        print("Register Slices")
        
        if not use_imaging_constraint:
            if not self.do_affine and not self.do_deformable and not self.do_reconstruct:
                print("Nothing to be done - no reconstruction, affine, or deformable selected")
                return
        
        print("Reconstruct?", self.do_reconstruct)
        self.store_volume = True
        
        # With imaging constraints
        if use_imaging_constraint:
            print("Reading Input data...")
            ref = self.loadRgbVolume()
            ref_mask = self.loadMask(0)
            
            import time
            start_input_time = time.time()
            
            if ((self.imaging_constraint_filename and self.imaging_constraint_mask_filename) or
                (self.imaging_constraint and self.imaging_constraint_mask)):
                im_c, constraint_range = self.getConstraint()
                self.volume_size = im_c.GetSize()
                self.volume_spacing = im_c.GetSpacing()
                self.volume_origin = im_c.GetOrigin()
                self.volume_direction = im_c.GetDirection()
            else:
                print("Using Imaging Constraints was set, but no filenames were provided")
                print("Set imagingContraintFilename and imagingContraintMaskFilename")
                print("No constraints will be used")
                use_imaging_constraint = False
                return
            
            self.ref_w_constraints = self.loadRgbVolume()
            self.msk_ref_w_constraints = self.loadMask(0)
            
            end_input_time = time.time()
            print(f"Done in {(end_input_time - start_input_time) / 60} min")
            
            if self.verbose:
                sitk.WriteImage(im_c, "fixed3D.nii.gz")
                sitk.WriteImage(ref, "reference3D.nii.gz")
            
            # Register each slice to constraint
            for imov in range(self.no_slices):
                if imov >= len(constraint_range):
                    break
                
                ifix = constraint_range[imov]
                start_reg_time = time.time()
                print(f"----Refine slice to imaging constraint {imov} {ifix} ------")
                
                mov_ps = self.pathology_slices[imov]
                mov_ps.do_affine = self.do_affine
                mov_ps.do_deformable = self.do_deformable
                mov_ps.fast_execution = self.fast_execution
                
                if self.ref_wo_constraints is None or self.msk_ref_wo_constraints is None:
                    mov_ps.registerToConstraint(
                        im_c[:, :, ifix],
                        self.ref_w_constraints,
                        self.msk_ref_w_constraints,
                        ref, ref_mask, ifix
                    )
                else:
                    mov_ps.registerToConstraint(
                        im_c[:, :, ifix],
                        self.ref_wo_constraints,
                        self.msk_ref_wo_constraints,
                        ref, ref_mask, ifix
                    )
                
                end_reg_time = time.time()
                print(f"Done registration in {(end_reg_time - start_reg_time) / 60} min")
        
        else:
            # Without constraints - internal reconstruction
            if self.fast_execution:
                print("Fast execution: Pathology reconstruction is not performed")
                return
            
            print("Doing Reconstruction?")
            if self.do_reconstruct:
                print("Doing Reconstruction")
                ref = self.loadRgbVolume()
                ref_mask = self.loadMask(0)
                
                self.ref_wo_constraints = ref
                self.msk_ref_wo_constraints = self.loadMask(0)
                
                # Register consecutive slices
                length = len(self.pathology_slices)
                middle_idx = int(length / 2) + 1
                idx_fixed = []
                idx_moving = []
                
                for i in range(middle_idx - 1, length - 1):
                    idx_fixed.append(i)
                    idx_moving.append(i + 1)
                
                for i in range(middle_idx - 1, 0, -1):
                    idx_fixed.append(i)
                    idx_moving.append(i - 1)
                
                # Register consecutive histology slices
                for ifix, imov in zip(idx_fixed, idx_moving):
                    print(f"----Registering slices {ifix} {imov} ------")
                    fix_ps = self.pathology_slices[ifix]
                    mov_ps = self.pathology_slices[imov]
                    mov_ps.do_affine = self.do_affine
                    mov_ps.registerTo(fix_ps, ref, ref_mask, True, 10 + imov)

    def deleteData(self):
        """Delete volume data"""
        print("Deleting Volume")
        for ps in self.pathology_slices:
            ps.deleteData()
        self.__init__()

    # Add these methods to PathologyVolume class
    #PathologyVolume.getConstraint = getConstraint
    #PathologyVolume.registerSlices = registerSlices
    #PathologyVolume.deleteData = deleteData


class PathologySlice:
    """Single pathology slice management"""
    
    def __init__(self):
        self.verbose = False
        self.id = None
        self.json_key = None
        self.rgb_image_fn = None
        self.mask_dict = None
        self.do_flip = None
        self.do_rotate = None
        
        self.rgb_image_size = None
        self.rgb_pixel_type = None
        self.dimension = None
        self.rgb_image = None
        self.store_image = False
        
        # Reference model information
        self.transform = None
        self.ref_size = None
        self.ref_spacing = None
        self.ref_origin = None
        self.ref_direction = None
        self.ref_slice_idx = None
        
        self.unit_mode = 0  # 0=microns, 1=millimeters
        
        self.region_ids = None
        self.do_affine = True
        self.do_deformable = None
        self.fast_execution = None
        self.run_longer = False
        self.transform_dict = None
    
    def loadImageSize(self):
        """Load image size without reading full image"""
        if not self.rgb_image_fn:
            print("The path to the rgb images was not set")
            return None
        
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.rgb_image_fn))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        self.rgb_image_size = reader.GetSize()
        self.rgb_pixel_type = sitk.GetPixelIDValueAsString(reader.GetPixelID())
        self.dimension = reader.GetDimension()
        
        if self.verbose:
            print(f"PathologySlice: Reading from '{self.rgb_image_fn}'")
            print(f"PathologySlice: Image Size: {self.rgb_image_size}")
    
    def loadRgbImage(self):
        """Load RGB image"""
        if not self.rgb_image_fn:
            print("The path to the rgb images was not set")
            return None
        
        try:
            rgb_image = sitk.ReadImage(str(self.rgb_image_fn))
        except Exception as e:
            print(e)
            print(f"Couldn't read {self.rgb_image_fn}")
            return None
        
        if self.verbose:
            print(f"PathologySlice: Reading {self.ref_slice_idx} "
                  f"({self.do_flip},{self.do_rotate}) from '{self.rgb_image_fn}'")
        
        # Apply flip if needed
        if self.do_flip is not None and self.do_flip == 1:
            arr = sitk.GetArrayFromImage(rgb_image)
            arr = arr[:, arr.shape[1]:0:-1, :]
            rgb_image2 = sitk.GetImageFromArray(arr, isVector=True)
            rgb_image2.SetSpacing(rgb_image.GetSpacing())
            rgb_image2.SetOrigin(rgb_image.GetOrigin())
            rgb_image2.SetDirection(rgb_image.GetDirection())
            rgb_image = rgb_image2
        
        if self.store_image:
            self.rgb_image = rgb_image
            return self.rgb_image
        else:
            return rgb_image
    
    def getGrayFromRGB(self, im, invert=True):
        """Convert RGB image to grayscale"""
        select = sitk.VectorIndexSelectionCastImageFilter()
        select.SetIndex(0)
        im_gray = select.Execute(im) / 3
        
        select.SetIndex(1)
        im_gray += select.Execute(im) / 3
        
        select.SetIndex(2)
        im_gray += select.Execute(im) / 3
        
        if invert:
            im_gray = 255 - im_gray
        
        return im_gray
    
    def loadMask(self, idx_mask):
        """Load mask for this slice"""
        if not self.mask_dict:
            print("No mask information was provided")
            return None
        
        mask_fn = None
        for mask_key in list(self.mask_dict):
            fn = self.mask_dict[mask_key]['filename']
            for idx_region, r in enumerate(self.region_ids):
                if mask_key == r:
                    read_idx_mask = idx_region
            
            if self.verbose:
                print(f"PathologySlice: Mask: {idx_mask} {read_idx_mask} {fn}")
            
            if read_idx_mask == idx_mask:
                mask_fn = fn
        
        if self.verbose and not mask_fn:
            print(f"PathologySlice: Mask {idx_mask} not found for slice {self.ref_slice_idx}")
        
        if not mask_fn:
            return None
        
        try:
            im = sitk.ReadImage(str(mask_fn))
        except Exception as e:
            print(e)
            print(f"Couldn't read {mask_fn}")
            return None
        
        # Handle multi-channel masks (e.g., RGB from GIMP)
        if im.GetNumberOfComponentsPerPixel() > 1:
            select = sitk.VectorIndexSelectionCastImageFilter()
            select.SetIndex(0)  # Select first channel
            im = select.Execute(im)
            im = sitk.Cast(im, sitk.sitkUInt8)
        
        # Apply flip if needed
        if self.do_flip is not None and self.do_flip == 1:
            arr = sitk.GetArrayFromImage(im)
            arr = arr[:, arr.shape[1]:0:-1]
            im2 = sitk.GetImageFromArray(arr)
            im2.SetSpacing(im.GetSpacing())
            im2.SetOrigin(im.GetOrigin())
            im2.SetDirection(im.GetDirection())
            im = im2
        
        if self.verbose:
            print(f"PathologySlice: Reading {self.ref_slice_idx} from '{mask_fn}'")
        
        return im

    def setReference(self, vol):
        """Set reference volume characteristics"""
        self.ref_size = vol.GetSize()
        self.ref_spacing = vol.GetSpacing()
        self.ref_origin = vol.GetOrigin()
        self.ref_direction = vol.GetDirection()
        
        # Transform needs to be recomputed when reference changes
        self.transform = None

    def computeCenterTransform(self, im, ref, relative_idx=0, mode=0, 
                            do_rotate=None, transform_type=0):
        """
        Compute center-of-mass transform
        
        Args:
            im: Input 2D image
            ref: Reference 3D volume
            relative_idx: 0 if ref same index as self, 1 if ref is one before
            mode: 0=rgb, 1=grayscale
            do_rotate: Rotation angle in degrees
            transform_type: 0=Affine, 1=Euler
        """
        # Get first channel for centered transform
        if not mode:
            select = sitk.VectorIndexSelectionCastImageFilter()
            select.SetIndex(0)
            im0 = select.Execute(im) / 3
            select.SetIndex(1)
            im0 += select.Execute(im) / 3
            select.SetIndex(2)
            im0 += select.Execute(im) / 3
            
            # Invert intensities for RGB
            im0 = 255 - im0
            
            # Get reference slice
            try:
                select.SetIndex(0)
                ref0 = select.Execute(ref[:, :, self.ref_slice_idx - relative_idx]) / 3
                select.SetIndex(1)
                ref0 += select.Execute(ref[:, :, self.ref_slice_idx - relative_idx]) / 3
                select.SetIndex(2)
                ref0 += select.Execute(ref[:, :, self.ref_slice_idx - relative_idx]) / 3
            except Exception as e:
                print(e)
                select.SetIndex(0)
                ref0 = select.Execute(ref[:, :, self.ref_slice_idx - 1]) / 3
                select.SetIndex(1)
                ref0 += select.Execute(ref[:, :, self.ref_slice_idx - 1]) / 3
                select.SetIndex(2)
                ref0 += select.Execute(ref[:, :, self.ref_slice_idx - 1]) / 3
            
            ref0 = 255 - ref0
        else:
            im0 = im
            try:
                ref0 = ref[:, :, self.ref_slice_idx - relative_idx]
            except Exception as e:
                print(e)
                ref0 = ref[:, :, self.ref_slice_idx - 1]
        
        if self.verbose:
            print(f"Computing Center of mass {self.ref_slice_idx} "
                f"{np.max(sitk.GetArrayFromImage(im0))} "
                f"{np.min(sitk.GetArrayFromImage(ref0))} "
                f"{np.max(sitk.GetArrayFromImage(ref0))} {do_rotate}")
        
        # Apply rotation first if needed
        if do_rotate:
            center = ref0.TransformContinuousIndexToPhysicalPoint(
                np.array(ref0.GetSize()) / 2.0
            )
            
            if transform_type == 0:
                rotation = sitk.AffineTransform(im0.GetDimension())
                rotation.Rotate(0, 1, np.radians(do_rotate))
            else:
                rotation = sitk.Euler2DTransform()
                rotation.SetAngle(np.radians(do_rotate))
            
            rotation.SetCenter(center)
            self.transform = sitk.Transform(rotation)
            
            # Apply rotation
            im0 = sitk.Resample(im0, ref0, self.transform)
        else:
            self.transform = None
        
        # Recompute centering
        transform = None
        try:
            tr = sitk.CenteredTransformInitializer(
                ref0, im0,
                sitk.AffineTransform(im.GetDimension()),
                sitk.CenteredTransformInitializerFilter.MOMENTS
            )
            transform = sitk.AffineTransform(tr)
            if self.verbose:
                print("Using COM")
        except:
            tr = sitk.CenteredTransformInitializer(
                ref0, im0,
                sitk.AffineTransform(im.GetDimension()),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            if self.verbose:
                print("Using Geometric")
            transform = sitk.AffineTransform(tr)
        
        if self.transform:
            self.transform = sitk.CompositeTransform([self.transform, transform])
        else:
            self.transform = sitk.Transform(transform)

    def setTransformedRgb(self, ref, relative_idx):
        """Set transformed RGB image into reference volume"""
        im = self.loadRgbImage()
        
        # Nothing was read
        if not im:
            return ref
        
        if not self.transform:
            self.computeCenterTransform(im, ref, relative_idx, 0, self.do_rotate)
        
        try:
            im_tr = sitk.Resample(
                im, ref[:, :, self.ref_slice_idx], self.transform,
                sitk.sitkNearestNeighbor, 255
            )
            ref_tr = sitk.JoinSeries(im_tr)
            ref = sitk.Paste(
                ref, ref_tr, ref_tr.GetSize(),
                destinationIndex=[0, 0, self.ref_slice_idx]
            )
        except Exception as e:
            print(e)
            im_tr = sitk.Resample(
                im, ref[:, :, self.ref_slice_idx - 1], self.transform,
                sitk.sitkNearestNeighbor, 255
            )
            ref_tr = sitk.JoinSeries(im_tr)
            ref = sitk.Paste(
                ref, ref_tr, ref_tr.GetSize(),
                destinationIndex=[0, 0, self.ref_slice_idx]
            )
        
        return ref

    def setTransformedMask(self, ref, idx_mask, relative_idx):
        """Set transformed mask into reference volume"""
        im = self.loadMask(idx_mask)
        
        # Nothing was read
        if not im:
            return ref
        
        if not self.transform:
            self.computeCenterTransform(im, ref, relative_idx, 1, self.do_rotate)
        
        try:
            im_tr = sitk.Resample(
                im, ref[:, :, self.ref_slice_idx],
                self.transform,
                sitk.sitkNearestNeighbor
            )
            ref_tr = sitk.JoinSeries(im_tr)
            ref = sitk.Paste(
                ref, ref_tr, ref_tr.GetSize(),
                destinationIndex=[0, 0, self.ref_slice_idx]
            )
        except Exception as e:
            print(e)
            print(f"The index doesn't exist {self.ref_slice_idx}")
            im_tr = sitk.Resample(
                im, ref[:, :, self.ref_slice_idx - 1],
                self.transform,
                sitk.sitkNearestNeighbor
            )
            ref_tr = sitk.JoinSeries(im_tr)
            ref = sitk.Paste(
                ref, ref_tr, ref_tr.GetSize(),
                destinationIndex=[0, 0, self.ref_slice_idx]
            )
        
        return ref

    def registerTo(self, ref_ps, ref, ref_mask, apply_transf_2_ref=True, idx=0):
        """Register this slice to reference slice"""
        if apply_transf_2_ref:
            old = ref_ps.ref_slice_idx
            ref_ps.ref_slice_idx = self.ref_slice_idx
            fixed_image = ref_ps.setTransformedRgb(ref, 1)[:, :, self.ref_slice_idx]
            ref_ps.ref_slice_idx = old
        else:
            fixed_image = ref_ps.loadRgbImage()
        
        fixed_image = self.getGrayFromRGB(fixed_image)
        moving_image = self.loadRgbImage()
        moving_image = self.getGrayFromRGB(moving_image)
        
        # Apply mask to fixed image
        try:
            if apply_transf_2_ref:
                fixed_mask = ref_ps.setTransformedMask(ref_mask, 0, 1)[:, :, ref_ps.ref_slice_idx]
            else:
                fixed_mask = ref_ps.loadMask(0)
        except Exception as e:
            print("No mask 0 was found")
            fixed_mask = None
        
        if fixed_mask:
            fixed_mask = sitk.Cast(
                sitk.Resample(
                    fixed_mask, fixed_image, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0.0, fixed_image.GetPixelID()
                ) > 0,
                fixed_image.GetPixelID()
            )
            fixed_image = fixed_image * fixed_mask
        
        # Apply mask to moving image
        try:
            moving_mask = self.loadMask(0)
        except Exception as e:
            print("No mask 0 was found")
            moving_mask = None
        
        if moving_mask:
            moving_mask = sitk.Cast(
                sitk.Resample(
                    moving_mask, moving_image, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID()
                ) > 0,
                moving_image.GetPixelID()
            )
            moving_image = moving_image * moving_mask
        
        if self.verbose:
            print(f"PathologySlice: Do no constraints affine: {self.do_affine}")
            print(f"PathologySlice: Do no constraints deformable: {self.do_deformable}")
        
        n_iter = 500 if self.run_longer else 250
        
        # Perform registration
        reg = RegisterImages()
        self.transform = reg.RegisterAffine(
            fixed_image, moving_image, self.transform, n_iter, idx, 1
    )



    def registerToConstraint(self, fixed_image, ref_mov, ref_mov_mask, ref, ref_mask, idx, apply_transf=True):
        """Register slice to imaging constraint"""
        if apply_transf:
            moving_image = self.setTransformedRgb(ref, 0)[:, :, self.ref_slice_idx]
        else:
            moving_image = self.loadRgbImage()
        
        moving_image = self.getGrayFromRGB(moving_image)
        
        # Load mask if available
        try:
            if apply_transf:
                mask = self.setTransformedMask(ref_mask, 0, 0)[:, :, self.ref_slice_idx]
            else:
                mask = self.loadMask(0)
        except Exception as e:
            print("No mask 0 was found")
            mask = None
        
        if mask:
            mask = sitk.Cast(
                sitk.Resample(
                    mask, moving_image, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID()
                ) > 0,
                moving_image.GetPixelID()
            )
            moving_image = moving_image * mask
        
        if self.verbose:
            sitk.WriteImage(fixed_image, f"{idx:02d}_Center_Fixed.nii.gz")
            sitk.WriteImage(moving_image, f"{idx:02d}_Center_Moving.nii.gz")
        
        # Compute center of mass alignment
        try:
            transform = sitk.CenteredTransformInitializer(
                sitk.Cast(fixed_image > 0, sitk.sitkFloat32),
                sitk.Cast(moving_image > 0, sitk.sitkFloat32),
                sitk.AffineTransform(moving_image.GetDimension()),
                sitk.CenteredTransformInitializerFilter.MOMENTS
            )
        except:
            transform = sitk.CenteredTransformInitializer(
                sitk.Cast(fixed_image, sitk.sitkFloat32),
                sitk.Cast(moving_image, sitk.sitkFloat32),
                sitk.AffineTransform(moving_image.GetDimension()),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        
        # Build composite transform
        all_transf = []
        try:
            n = self.transform.GetNumberOfTransforms()
            for i in range(n):
                tr = self.transform.GetNthTransform(i)
                all_transf.append(tr)
        except Exception:
            all_transf.append(self.transform)
        
        all_transf.append(transform)
        self.transform = sitk.CompositeTransform(all_transf)
        
        if self.verbose:
            print(f"PathologySlice: Do constraints affine: {self.do_affine}")
            print(f"PathologySlice: Do constraints deformable: {self.do_deformable}")
            print(f"PathologySlice: Fast execution: {self.fast_execution}")
        
        # Reload moving image without transform
        reg = RegisterImages()
        moving_image = self.loadRgbImage()
        moving_image = self.getGrayFromRGB(moving_image)
        mask = self.loadMask(0)
        
        if mask:
            mask = sitk.Cast(
                sitk.Resample(
                    mask, moving_image, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID()
                ) > 0,
                moving_image.GetPixelID()
            )
            moving_image = moving_image * mask
        
        if self.verbose:
            moved_image = sitk.Resample(
                moving_image, fixed_image,
                self.transform, sitk.sitkLinear, 0.0, fixed_image.GetPixelID()
            )
            sitk.WriteImage(moved_image, f"{idx:02d}_Center_Moved.nii.gz")
        
        # Downsample for fast execution
        if self.fast_execution:
            r = 4
            new_size = [
                int(fixed_image.GetSize()[0] / r),
                int(fixed_image.GetSize()[1] / r)
            ]
            new_sp = [
                fixed_image.GetSpacing()[0] * r,
                fixed_image.GetSpacing()[0] * r
            ]
            ref_reg_img = sitk.Image(new_size, sitk.sitkFloat32)
            ref_reg_img.SetSpacing(new_sp)
            ref_reg_img.SetDirection(fixed_image.GetDirection())
            ref_reg_img.SetOrigin(fixed_image.GetOrigin())
            fixed_image = sitk.Resample(fixed_image, ref_reg_img, sitk.Transform())
        
        n_iter = 500 if self.run_longer else 250
        
        # Perform affine registration
        if self.do_affine:
            fixed_image_input = sitk.Cast(fixed_image > 0, sitk.sitkFloat32) * 255
            moving_image_input = sitk.Cast(moving_image > 0, sitk.sitkFloat32) * 255
            
            # Rigid registration (2 passes)
            self.transform = reg.RegisterAffine(
                fixed_image_input, moving_image_input, self.transform, n_iter, idx, 1, 0, True, False
            )
            self.transform = reg.RegisterAffine(
                fixed_image_input, moving_image_input, self.transform, n_iter, idx, 1, 0, True, False
            )
            
            # Affine registration
            self.transform = reg.RegisterAffine(
                fixed_image_input, moving_image_input, self.transform, n_iter, idx, 0, 0, True, False
            )
        
        n_iter = 50 if self.run_longer else 10
        
        # Perform deformable registration
        if self.do_deformable:
            transform_def = reg.RegisterDeformable(
                fixed_image, moving_image, self.transform, 10, n_iter, idx
            )
            self.transform.AddTransform(transform_def)

    def deleteData(self):
        """Delete slice data"""
        print("Deleting Slice")
        self.__init__()


    # Add these methods to PathologySlice class
    """PathologySlice.loadMask = loadMask
    PathologySlice.setReference = setReference
    PathologySlice.computeCenterTransform = computeCenterTransform
    PathologySlice.setTransformedRgb = setTransformedRgb
    PathologySlice.setTransformedMask = setTransformedMask
    PathologySlice.registerTo = registerTo
    PathologySlice.registerToConstraint = registerToConstraint
    PathologySlice.deleteData = deleteData"""