import os
import json
import numpy as np
import SimpleITK as sitk
from Resources.Utils.image_registration import RegisterImages

#questo file è giusto


class PathologyVolume:
    
    def __init__(self, parent=None):
        self.verbose = False
        self.path = None

        self.noRegions = 0
        self.regionIDs = None
        self.noSlices = 0
        self.pix_size_x = 0
        self.pix_size_y = 0

        self.maxSliceSize = [0, 0]
        self.volumeSize = [0, 0, 0]
        self.rgbVolume = None
        self.storeVolume = False

        self.inPlaneScaling = 1.2
        self.pathologySlices = None
        self.jsonDict = None
        
        # Distance and thickness parameters
        self.rectumDistance = 0.0
        self.totalVolumeThickness = 0.0
        self.slicePositions = []
        
        # files 
        self.imagingContraint = None
        self.imagingContraintMask = None
        
        # CRITICAL: Physical space parameters
        self.volumeOrigin = None
        self.volumeDirection = None
        self.volumeSpacing = None
        
        # Physical space alignment from MRI constraint
        self.constraintRegion = None
        self.targetPhysicalSize = None
        
        # NEW: Labelmap association
        self.mri_labelmap = None  # The MRI segmentation as labelmap
        self.slice_to_label_map = {}  # Maps histology slice index to MRI label value
        self.label_to_slice_map = {}  # Maps MRI label value to histology slice index
        
        # Reference volumes
        self.refWoContraints = None
        self.refWContraints = None
        self.mskRefWoContraints = None
        self.mskRefWContraints = None
        
        self.doAffine = True
        self.doDeformable = None
        self.doReconstruct = None
        self.fastExecution = None
        self.discardOrientation = None

        self.successfulInitialization = False

    def set_verbose(self, verbose):
        """Set verbose mode"""
        self.verbose = verbose
    
    def setPath(self, path):
        """Set the path to the JSON file"""
        self.path = path
        if self.verbose:
            print(f"Pathology JSON path set to: {path}")
    
    def setRectumDistance(self, distance_mm):
        """Set the distance from rectum where histological slices will be placed"""
        self.rectumDistance = float(distance_mm)
        if self.verbose:
            print(f"Rectum distance set to: {self.rectumDistance} mm")
    
    def calculateSlicePositions(self):
        """Calculate Z positions for each slice based on thickness and rectum distance"""
        if not self.pathologySlices:
            return
            
        self.slicePositions = []
        current_z = self.rectumDistance
        
        for i, ps in enumerate(self.pathologySlices):
            self.slicePositions.append(current_z)
            ps.zPosition = current_z
            
            if self.verbose:
                print(f"Slice {i} positioned at Z = {current_z} mm (thickness: {ps.sliceThickness} mm)")
            
            current_z += ps.sliceThickness
            
        self.totalVolumeThickness = current_z - self.rectumDistance
        
        if self.verbose:
            print(f"Total volume thickness: {self.totalVolumeThickness} mm")
    
    def extractConstraintRegionWithLabels(self, mri_volume, mri_labelmap):
        """
        Extract constraint region from MRI PRESERVING LABELMAP STRUCTURE
        This is critical for slice-to-slice association
        
        Args:
            mri_volume: MRI T2 volume
            mri_labelmap: MRI segmentation as labelmap (with distinct label values per slice)
        
        Returns:
            Tuple of (constrained_volume, constrained_labelmap, bbox, label_to_z_map)
        """
        if self.verbose:
            print("Extracting constraint region WITH LABELMAP preservation...")
        
        # Store the original labelmap
        self.mri_labelmap = mri_labelmap
        
        # Get bounding box from labelmap
        stats_filter = sitk.LabelShapeStatisticsImageFilter()
        
        # Create binary mask from all labels
        binary_mask = mri_labelmap != 0
        connected = sitk.ConnectedComponent(binary_mask)
        stats_filter.Execute(connected)
        
        labels = stats_filter.GetLabels()
        if not labels:
            raise ValueError("Labelmap is empty")
        
        label = labels[0]
        bounding_box = stats_filter.GetBoundingBox(label)
        
        # Extract ROI from both volume and labelmap
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize([bounding_box[3], bounding_box[4], bounding_box[5]])
        roi_filter.SetIndex([bounding_box[0], bounding_box[1], bounding_box[2]])
        
        constrained_volume = roi_filter.Execute(mri_volume)
        constrained_labelmap = roi_filter.Execute(mri_labelmap)
        
        # Analyze labelmap structure to map Z-slices to labels
        label_to_z_map = self.analyzeLabelmap(constrained_labelmap, bounding_box)
        
        self.constraintRegion = bounding_box
        
        if self.verbose:
            print(f"Constraint region extracted:")
            print(f"  Bounding box: {bounding_box}")
            print(f"  Physical size: {constrained_volume.GetSize()}")
            print(f"  Number of labeled slices: {len(label_to_z_map)}")
            print(f"  Labels found: {sorted(label_to_z_map.keys())}")
        
        return constrained_volume, constrained_labelmap, bounding_box, label_to_z_map
    
    def analyzeLabelmap(self, labelmap, bounding_box):
        """
        Analyze labelmap to determine which labels are present in which Z-slices
        
        Args:
            labelmap: The constrained labelmap volume
            bounding_box: Bounding box coordinates
        
        Returns:
            Dictionary mapping label_value -> list of z_indices where that label appears
        """
        if self.verbose:
            print("\nAnalyzing labelmap structure...")
        
        label_to_z_map = {}
        labelmap_array = sitk.GetArrayFromImage(labelmap)  # Shape: [Z, Y, X]
        
        # For each Z slice
        for z_idx in range(labelmap_array.shape[0]):
            slice_2d = labelmap_array[z_idx, :, :]
            unique_labels = np.unique(slice_2d)
            
            # Remove background (0)
            unique_labels = unique_labels[unique_labels != 0]
            
            for label_value in unique_labels:
                if label_value not in label_to_z_map:
                    label_to_z_map[label_value] = []
                label_to_z_map[label_value].append(z_idx)
        
        if self.verbose:
            print(f"  Found {len(label_to_z_map)} distinct labels")
            for label_val, z_indices in sorted(label_to_z_map.items()):
                print(f"    Label {int(label_val)}: appears in {len(z_indices)} slices "
                      f"(Z indices: {min(z_indices)}-{max(z_indices)})")
        
        return label_to_z_map
    
    def createSliceToLabelMapping(self, label_to_z_map, num_histology_slices):
        """
        Create mapping between histology slices and MRI labels
        
        Strategy:
        - If MRI has one label per slice: 1-to-1 mapping
        - If MRI has fewer labels than histology slices: distribute histology across labels
        - If MRI has more labels: map multiple labels to one histology slice
        
        Args:
            label_to_z_map: Dictionary from analyzeLabelmap
            num_histology_slices: Number of histology slices to register
        """
        if self.verbose:
            print(f"\nCreating slice-to-label mapping...")
            print(f"  Histology slices: {num_histology_slices}")
            print(f"  MRI labels: {len(label_to_z_map)}")
        
        # Get sorted list of labels
        sorted_labels = sorted(label_to_z_map.keys())
        
        if len(sorted_labels) == num_histology_slices:
            # Perfect 1-to-1 mapping
            for i, label_val in enumerate(sorted_labels):
                self.slice_to_label_map[i] = int(label_val)
                self.label_to_slice_map[int(label_val)] = i
            
            if self.verbose:
                print("  Using 1-to-1 mapping (equal number of slices and labels)")
        
        elif len(sorted_labels) > num_histology_slices:
            # More labels than histology slices - group labels
            # Distribute labels evenly across histology slices
            labels_per_slice = len(sorted_labels) / num_histology_slices
            
            for hist_idx in range(num_histology_slices):
                start_idx = int(hist_idx * labels_per_slice)
                end_idx = int((hist_idx + 1) * labels_per_slice)
                
                # Assign the middle label from this range
                mid_idx = (start_idx + end_idx) // 2
                label_val = sorted_labels[mid_idx]
                
                self.slice_to_label_map[hist_idx] = int(label_val)
                self.label_to_slice_map[int(label_val)] = hist_idx
            
            if self.verbose:
                print(f"  Using grouped mapping (more labels than slices)")
                print(f"  Labels per histology slice: ~{labels_per_slice:.1f}")
        
        else:
            # Fewer labels than histology slices - interpolate
            # Distribute histology slices across available labels
            for hist_idx in range(num_histology_slices):
                # Map to nearest label
                label_idx = int(hist_idx * len(sorted_labels) / num_histology_slices)
                label_idx = min(label_idx, len(sorted_labels) - 1)
                
                label_val = sorted_labels[label_idx]
                self.slice_to_label_map[hist_idx] = int(label_val)
                
                # Multiple histology slices may map to same label
                if int(label_val) not in self.label_to_slice_map:
                    self.label_to_slice_map[int(label_val)] = []
                if isinstance(self.label_to_slice_map[int(label_val)], list):
                    self.label_to_slice_map[int(label_val)].append(hist_idx)
                else:
                    self.label_to_slice_map[int(label_val)] = [
                        self.label_to_slice_map[int(label_val)], hist_idx
                    ]
            
            if self.verbose:
                print(f"  Using interpolated mapping (fewer labels than slices)")
        
        if self.verbose:
            print("\n  Final mapping:")
            for hist_idx in range(min(num_histology_slices, 10)):  # Show first 10
                label_val = self.slice_to_label_map.get(hist_idx, "N/A")
                print(f"    Histology slice {hist_idx} → MRI label {label_val}")
            if num_histology_slices > 10:
                print(f"    ... ({num_histology_slices - 10} more)")
    
    def alignToMRISpace(self, mri_volume, mri_labelmap):
        """
        Aligns the histology volume's physical space to the MRI constraint region
        WITH LABELMAP ASSOCIATION
        """
        if self.verbose:
            print("\n" + "="*70)
            print("ALIGNING HISTOLOGY PHYSICAL SPACE TO MRI WITH LABELMAP")
            print("="*70)

        if not mri_volume or not mri_labelmap:
            raise ValueError("MRI volume and labelmap are required for alignment.")

        # Extract constraint region PRESERVING LABELMAP
        constrained_mri, constrained_labelmap, bbox, label_to_z_map = \
            self.extractConstraintRegionWithLabels(mri_volume, mri_labelmap)
        
        # Create slice-to-label mapping
        self.createSliceToLabelMapping(label_to_z_map, self.noSlices)
        
        # Update physical properties to match MRI
        self.volumeOrigin = constrained_mri.GetOrigin()
        self.volumeSpacing = constrained_mri.GetSpacing()
        self.volumeDirection = constrained_mri.GetDirection()
        self.volumeSize = list(constrained_mri.GetSize())
        
        # Z-size matches number of histology slices
        self.volumeSize[2] = self.noSlices
        
        # Store target physical size
        mri_phys_size = [
            constrained_mri.GetSize()[i] * constrained_mri.GetSpacing()[i] 
            for i in range(3)
        ]
        self.targetPhysicalSize = mri_phys_size

        if self.verbose:
            print("Histology physical space aligned to MRI:")
            print(f"  New Origin:    {self.volumeOrigin}")
            print(f"  New Spacing:   {self.volumeSpacing}")
            print(f"  New Size:      {self.volumeSize}")
            print(f"  Slice-to-label mapping created: {len(self.slice_to_label_map)} associations")
            print("="*70 + "\n")
    
    def createInitialHistologyVolume(self):
        """Create initial histology volume in its OWN coordinate system"""
        if self.verbose:
            print("\nCreating initial histology volume in histology space...")
        
        # Calculate histology spacing
        hist_spacing_x = self.pix_size_x / 1000.0 if self.pix_size_x > 0 else 1.0
        hist_spacing_y = self.pix_size_y / 1000.0 if self.pix_size_y > 0 else 1.0
        
        if len(self.slicePositions) > 1:
            hist_spacing_z = abs(self.slicePositions[1] - self.slicePositions[0])
        elif self.pathologySlices and len(self.pathologySlices) > 0:
            hist_spacing_z = self.pathologySlices[0].sliceThickness
        else:
            hist_spacing_z = 1.0
        
        if hist_spacing_z <= 1e-6:
            hist_spacing_z = 1.0
        
        # Calculate volume size
        hist_size_x = int(self.maxSliceSize[0] * self.inPlaneScaling)
        hist_size_y = int(self.maxSliceSize[1] * self.inPlaneScaling)
        hist_size_z = self.noSlices
        
        # Create volume
        self.volumeSpacing = [hist_spacing_x, hist_spacing_y, hist_spacing_z]
        self.volumeOrigin = [0.0, 0.0, self.rectumDistance]
        self.volumeDirection = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.volumeSize = [hist_size_x, hist_size_y, hist_size_z]
        
        if self.verbose:
            print(f"Initial histology volume:")
            print(f"  Size: {self.volumeSize}")
            print(f"  Spacing: {self.volumeSpacing} mm")
    
    def resampleHistologyToMRISpace(self, mri_constraint_volume, hist_rgb_volume, 
                                    hist_mask_volume=None):
        """Resample histology volume to MRI physical space"""
        if self.verbose:
            print("\n" + "="*70)
            print("RESAMPLING HISTOLOGY TO MRI PHYSICAL SPACE")
            print("="*70)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_constraint_volume)
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(255)
        
        resampled_rgb = resampler.Execute(hist_rgb_volume)
        
        resampled_mask = None
        if hist_mask_volume is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampled_mask = resampler.Execute(hist_mask_volume)
        
        if self.verbose:
            print(f"✓ Resampling complete")
            print(f"  Output size: {resampled_rgb.GetSize()}")
            print("="*70 + "\n")
        
        return resampled_rgb, resampled_mask
    
    def getTargetLabelForSlice(self, histology_slice_index):
        """
        Get the MRI label value that should be associated with this histology slice
        
        Args:
            histology_slice_index: Index of histology slice
            
        Returns:
            MRI label value (integer)
        """
        return self.slice_to_label_map.get(histology_slice_index, 0)
    
    def initComponents(self):
        """Initialize components from JSON"""
        if self.verbose:
            print("PathologyVolume: Initialize components") 

        if not self.path:
            print("ERROR: The path was not set")
            self.successfulInitialization = False
            return False

        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load JSON file: {e}")
            self.successfulInitialization = False
            return False
            
        self.jsonDict = data
        
        # Read rectum distance from JSON if available
        if 'volume_settings' in data and 'rectum_distance_mm' in data['volume_settings']:
            self.rectumDistance = float(data['volume_settings']['rectum_distance_mm'])

        self.pix_size_x = 0
        self.pix_size_y = 0
        self.pathologySlices = []
        self.regionIDs = []
        
        slice_keys = [k for k in self.jsonDict.keys() if k != 'volume_settings']
        
        for key in np.sort(slice_keys):
            try:
                ps = PathologySlice()
                ps.jsonKey = key
                ps.rgbImageFn = data[key]['filename']
                ps.maskDict = data[key]['regions']
                ps.id = data[key]['id']
                ps.sliceThickness = float(data[key].get('slice_thickness_mm', 1.0))
                
                # Handle transforms
                ps.doFlip = None
                ps.doRotate = None
                
                if 'transform' in data[key]:
                    ps.transformDict = data[key]['transform']
                    ps.doFlip = ps.transformDict.get('flip')
                    ps.doRotate = ps.transformDict.get('rotation_angle')
                else:
                    ps.doFlip = data[key].get('flip', 0)
                    ps.doRotate = data[key].get('rotate', 0)
                
                if ps.doFlip is None: 
                    ps.doFlip = 0
                if ps.doRotate is None:
                    ps.doRotate = 0
                    
                if not ps.loadImageSize():
                    print(f"WARNING: Failed to load image size for slice {key}")
                    continue
                    
                size = ps.rgbImageSize

                for dim in range(min(ps.dimension, 2)):
                    if len(self.maxSliceSize) > dim and len(size) > dim:
                        if self.maxSliceSize[dim] < size[dim]:
                            self.maxSliceSize[dim] = size[dim]

                idx = data[key].get('slice_number')
                ps.refSliceIdx = int(idx) - 1 if idx else len(self.pathologySlices)
                
                for r in list(data[key]['regions']):
                    if r not in self.regionIDs:
                        self.regionIDs.append(r)
                self.noRegions = len(self.regionIDs)
                
                ps.regionIDs = self.regionIDs
                self.pathologySlices.append(ps)

                # Process resolution
                xml_res_x = data[key].get('resolution_x_um', data[key].get('resolution_x', 0))
                xml_res_y = data[key].get('resolution_y_um', data[key].get('resolution_y', 0))
                
                if self.pix_size_x == 0 and xml_res_x > 0:
                    self.pix_size_x = xml_res_x
                if self.pix_size_y == 0 and xml_res_y > 0:
                    self.pix_size_y = xml_res_y
                    
            except Exception as e:
                print(f"ERROR: Failed to process slice {key}: {e}")
                continue

        self.noSlices = len(self.pathologySlices)
        
        if self.noSlices == 0:
            print("ERROR: No valid slices were processed")
            self.successfulInitialization = False
            return False
        
        # Calculate slice positions
        self.calculateSlicePositions()
        
        # Create initial volume parameters
        self.createInitialHistologyVolume()

        if self.verbose:
            print(f"Found {self.noSlices} slices @ max size {self.maxSliceSize}")
            print(f"Initial volume size: {self.volumeSize}")
        
        self.successfulInitialization = True
        return True

    def loadRgbVolume(self):
        """Load the RGB volume from pathology slices"""
        if not self.pathologySlices:
            raise RuntimeError("No pathology slices available")
        
        vol = sitk.Image(self.volumeSize, sitk.sitkVectorUInt8, 3)
        vol.SetOrigin(self.volumeOrigin)
        vol.SetDirection(self.volumeDirection)
        vol.SetSpacing(self.volumeSpacing)

        for i, ps in enumerate(self.pathologySlices):
            try:
                ps.fastExecution = self.fastExecution
                im = ps.loadRgbImage()
                
                if not im:
                    continue
                
                if not ps.refSize:
                    ps.setReference(vol) 
                
                relativeIdx = int(i > 0)
                vol = ps.setTransformedRgb(vol, relativeIdx)
                
            except Exception as e:
                print(f"ERROR: Failed to process slice {i}: {e}")
                continue

        if self.storeVolume:
            self.rgbVolume = vol
        
        return vol

    def loadMaskWithLabels(self, idxMask=0):
        """
        Load mask AS LABELMAP with proper label values from MRI
        Each histology slice gets its corresponding MRI label value
        """
        if not self.pathologySlices:
            raise RuntimeError("No pathology slices available")
            
        # Create volume
        vol = sitk.Image(self.volumeSize, sitk.sitkUInt16)  # Use UInt16 for labels
        vol.SetOrigin(self.volumeOrigin)
        vol.SetDirection(self.volumeDirection)
        vol.SetSpacing(self.volumeSpacing)
        
        if self.verbose:
            print(f"\nLoading mask {idxMask} AS LABELMAP...")
            
        # Fill the volume with proper label values
        for i, ps in enumerate(self.pathologySlices):
            try:
                # Get the MRI label value for this slice
                target_label = self.getTargetLabelForSlice(i)
                
                if self.verbose and i < 3:
                    print(f"  Slice {i} → Label {target_label}")
                
                ps.fastExecution = self.fastExecution
                im = ps.loadMask(idxMask)
                
                if not im:
                    continue
          
                if not ps.refSize:
                    ps.setReference(vol)
                
                # Set mask with correct label value
                relativeIdx = int(i > 0)
                vol = ps.setTransformedMaskWithLabel(vol, idxMask, relativeIdx, target_label)
                
            except Exception as e:
                print(f"ERROR: Failed to process mask for slice {i}: {e}")
                continue

        return vol

    def registerSlices(self, useImagingConstraint=True):
        """Register slices with LABELMAP association"""
        print("="*70)
        print("REGISTER SLICES WITH LABELMAP ASSOCIATION")
        print("="*70)
        
        if not useImagingConstraint:
            print("ERROR: Registration without imaging constraint not supported")
            return
        
        if not self.imagingContraint or not self.imagingContraintMask:
            print("ERROR: MRI volume and labelmap required")
            return
        
        # STEP 1: Extract constraint WITH LABELMAP
        print("\nStep 1: Extracting MRI constraint region WITH LABELMAP...")
        constrained_mri, constrained_labelmap, bbox, label_to_z_map = \
            self.extractConstraintRegionWithLabels(
                self.imagingContraint,
                self.imagingContraintMask
            )
        
        # STEP 2: Create slice-to-label mapping
        print("\nStep 2: Creating slice-to-label mapping...")
        self.createSliceToLabelMapping(label_to_z_map, self.noSlices)
        
        # STEP 3: Load histology volumes
        print("\nStep 3: Loading histology volumes...")
        self.storeVolume = True
        hist_rgb_native = self.loadRgbVolume()
        hist_mask_native = self.loadMask(0)
        
        # STEP 4: Resample to MRI space
        print("\nStep 4: Resampling histology to MRI space...")
        hist_rgb_mri_space, hist_mask_mri_space = self.resampleHistologyToMRISpace(
            constrained_mri,
            hist_rgb_native,
            hist_mask_native
        )
        
        self.refWContraints = hist_rgb_mri_space
        self.mskRefWContraints = hist_mask_mri_space
        
        # STEP 5: Register each slice to its corresponding label
        print(f"\nStep 5: Registering slices to corresponding MRI labels...")
        
        for hist_idx in range(self.noSlices):
            target_label = self.getTargetLabelForSlice(hist_idx)
            
            # Find which Z-slice(s) contain this label
            z_indices_for_label = label_to_z_map.get(target_label, [])
            
            if not z_indices_for_label:
                print(f"  ⚠️  Slice {hist_idx}: No Z-index found for label {target_label}")
                continue
            
            # Use the middle Z-index if label spans multiple slices
            target_z = z_indices_for_label[len(z_indices_for_label) // 2]
            
            print(f"\n  Registering histology slice {hist_idx+1} → MRI label {target_label} (Z={target_z})...")
            
            mov_ps = self.pathologySlices[hist_idx]
            mov_ps.doAffine = self.doAffine
            mov_ps.doDeformable = self.doDeformable
            mov_ps.fastExecution = self.fastExecution
            
            # Extract the specific label as mask for registration
            label_mask_3d = constrained_labelmap == target_label
            label_mask_2d_slice = label_mask_3d[:, :, target_z]
            
            # Register
            mov_ps.registerToConstraintWithLabel(
                constrained_mri[:, :, target_z],
                label_mask_2d_slice,
                hist_rgb_mri_space,
                hist_mask_mri_space,
                hist_rgb_mri_space,
                hist_mask_mri_space,
                target_z,
                target_label
            )
        
        print("\n" + "="*70)
        print("REGISTRATION WITH LABELMAP COMPLETE")
        print("="*70)


    # Rest of the methods remain the same
    def getInfo4UI(self):
        """Get information for UI display"""
        data = []
        if not self.pathologySlices:
            return data
        
        for idx, ps in enumerate(self.pathologySlices):
            masks = []
            if ps.maskDict:
                for mask_key in list(ps.maskDict):
                    fn = ps.maskDict[mask_key]['filename']
                    try:
                        readIdxMask = int(mask_key[6:])
                    except:
                        readIdxMask = 1
                    masks.append([readIdxMask, fn])
                    
            el = [idx,
                ps.refSliceIdx + 1,
                ps.rgbImageFn, 
                masks, 
                ps.doFlip, 
                ps.doRotate,
                getattr(ps, 'sliceThickness', 1.0),
                getattr(ps, 'zPosition', 0.0)]
            data.append(el)
        
        return data
    
    def updateSlice(self, idx, param, value):
        """Update slice parameters"""
        if not self.pathologySlices or len(self.pathologySlices) <= idx:
            return
            
        self.pathologySlices[idx].transform = None 
        jsonKey = False
        
        if param == 'slice_number':
            self.pathologySlices[idx].refSliceIdx = value 
            jsonKey = True
            jsonValue = value + 1
            
        elif param == 'filename':
            self.pathologySlices[idx].rgbImageFn = value
            jsonKey = True           
            jsonValue = str(value)
            
        elif param == 'flip':
            self.pathologySlices[idx].doFlip = value
            jsonKey = True   
            jsonValue = value
            
        elif param == 'rotation_angle':
            self.pathologySlices[idx].doRotate = value
            jsonKey = True        
            jsonValue = value
            
        elif param == 'slice_thickness_mm':
            self.pathologySlices[idx].sliceThickness = float(value)
            jsonKey = True
            jsonValue = float(value)
            self.calculateSlicePositions()
        
        # Update JSON
        if param in ['flip', 'rotation_angle']:
            if 'transform' not in self.jsonDict[self.pathologySlices[idx].jsonKey]:
                self.jsonDict[self.pathologySlices[idx].jsonKey]['transform'] = {}
            self.jsonDict[self.pathologySlices[idx].jsonKey]['transform'][param] = jsonValue
        else:
            self.jsonDict[self.pathologySlices[idx].jsonKey][param] = jsonValue
    
    def saveJson(self, path_out_json):
        """Save JSON configuration"""
        if self.verbose: 
            print("Saving Json File")
        
        try:
            with open(path_out_json, 'w') as outfile:
                json.dump(self.jsonDict, outfile, indent=4, sort_keys=True)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save JSON: {e}")
            return False
    
    def deleteData(self):
        """Clean up data"""
        print("Deleting Volume")
        if self.pathologySlices:
            for ps in self.pathologySlices:
                ps.deleteData()
        self.__init__()


class PathologySlice:
    """PathologySlice class with label association support"""
    
    def __init__(self):
        self.verbose = False
        self.id = None
        self.rgbImageFn = None
        self.maskDict = None
        self.doFlip = None
        self.doRotate = None
        
        self.sliceThickness = 1.0
        self.zPosition = 0.0
        
        # NEW: Label association
        self.target_mri_label = None  # The MRI label this slice should match

        self.rgbImageSize = None
        self.rgbPixelType = None
        self.dimension = None
        self.rgbImage = None
        self.storeImage = False

        self.transform = None
        self.refSize = None
        self.refSpacing = None
        self.refOrigin = None
        self.refDirection = None
        self.refSliceIdx = None

        self.unitMode = 0
        self.regionIDs = None
        self.doAffine = True
        self.doDeformable = None
        self.fastExecution = None
        self.runLonger = False
        self.jsonKey = None
        self.transformDict = None

    def loadImageSize(self):
        """Load image size information"""
        if not self.rgbImageFn:
            print("ERROR: The path to the rgb images was not set")
            return False
    
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(self.rgbImageFn))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            self.rgbImageSize = reader.GetSize()
            self.rgbPixelType = sitk.GetPixelIDValueAsString(reader.GetPixelID())
            self.dimension = reader.GetDimension()

            if self.verbose:
                print(f"PathologySlice: Reading from '{self.rgbImageFn}'")
                print(f"PathologySlice: Image Size: {self.rgbImageSize}")
                
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load image size: {e}")
            return False
    
    def loadRgbImage(self):
        """Load RGB image"""
        if not self.rgbImageFn:
            return None

        try:
            rgbImage = sitk.ReadImage(str(self.rgbImageFn))
        except Exception as e:
            print(f"ERROR: Couldn't read {self.rgbImageFn}: {e}")
            return None

        # Apply flip if needed
        if (self.doFlip is not None) and self.doFlip == 1:
            try:
                arr = sitk.GetArrayFromImage(rgbImage)
                arr = arr[:, arr.shape[1]:0:-1, :]
                rgbImage2 = sitk.GetImageFromArray(arr, isVector=True)
                rgbImage2.SetSpacing(rgbImage.GetSpacing()) 
                rgbImage2.SetOrigin(rgbImage.GetOrigin()) 
                rgbImage2.SetDirection(rgbImage.GetDirection()) 
                rgbImage = rgbImage2 
            except Exception as e:
                print(f"ERROR: Failed to flip image: {e}")

        if self.storeImage:
            self.rgbImage = rgbImage
            return self.rgbImage
        else:
            return rgbImage
    
    def getGrayFromRGB(self, im, invert=True):
        """Convert RGB to grayscale"""
        try:
            select = sitk.VectorIndexSelectionCastImageFilter()
            
            select.SetIndex(0)
            im_gray = sitk.Cast(select.Execute(im), sitk.sitkFloat32) / 3
            
            select.SetIndex(1)
            im_gray += sitk.Cast(select.Execute(im), sitk.sitkFloat32) / 3
            
            select.SetIndex(2)
            im_gray += sitk.Cast(select.Execute(im), sitk.sitkFloat32) / 3
           
            if invert:
                im_gray = 255 - im_gray
                
            return im_gray
            
        except Exception as e:
            print(f"ERROR: Failed to convert RGB to gray: {e}")
            return None
    
    def loadMask(self, idxMask):
        """Load mask"""
        if not self.maskDict:
            return None

        maskFn = None
        for mask_key in list(self.maskDict):
            fn = self.maskDict[mask_key]['filename']
            readIdxMask = 0
            
            for idxRegion, r in enumerate(self.regionIDs or []):
                if mask_key == r:
                    readIdxMask = idxRegion
                    break

            if readIdxMask == idxMask:
                maskFn = fn
                break

        if not maskFn:
            return None

        try:
            im = sitk.ReadImage(str(maskFn))
        except Exception as e:
            print(f"ERROR: Couldn't read mask {maskFn}: {e}")
            return None

        # Handle multi-channel masks
        if im.GetNumberOfComponentsPerPixel() > 1:
            try:
                select = sitk.VectorIndexSelectionCastImageFilter()
                select.SetIndex(0)
                im = select.Execute(im)
                im = sitk.Cast(im, sitk.sitkUInt8)
            except Exception as e:
                print(f"ERROR: Failed to process multi-channel mask: {e}")
                return None
                
        # Apply flip if needed
        if (self.doFlip is not None) and self.doFlip == 1:
            try:
                arr = sitk.GetArrayFromImage(im)
                arr = arr[:, arr.shape[1]:0:-1]
                im2 = sitk.GetImageFromArray(arr)
                im2.SetSpacing(im.GetSpacing()) 
                im2.SetOrigin(im.GetOrigin()) 
                im2.SetDirection(im.GetDirection()) 
                im = im2
            except Exception as e:
                print(f"ERROR: Failed to flip mask: {e}")

        return im
    
    def setReference(self, vol):
        """Set reference volume"""
        self.refSize = vol.GetSize()
        self.refSpacing = vol.GetSpacing()
        self.refOrigin = vol.GetOrigin()
        self.refDirection = vol.GetDirection()
        self.transform = None
    
    def computeCenterTransform(self, im, ref, relativeIdx=0, mode=0, doRotate=None, transform_type=0):
        """Compute center transform"""
        try:
            if not mode:
                im0 = self.getGrayFromRGB(im, invert=True)
                if im0 is None:
                    return
                    
                try: 
                    ref0 = self.getGrayFromRGB(ref[:, :, self.refSliceIdx - relativeIdx], invert=True)
                    if ref0 is None:
                        ref0 = self.getGrayFromRGB(ref[:, :, max(0, self.refSliceIdx - 1)], invert=True)
                except Exception as e:
                    ref0 = self.getGrayFromRGB(ref[:, :, max(0, self.refSliceIdx - 1)], invert=True)
                    if ref0 is None:
                        return
            else:
                im0 = im
                try:
                    ref0 = ref[:, :, self.refSliceIdx - relativeIdx]
                except Exception as e:
                    ref0 = ref[:, :, max(0, self.refSliceIdx - 1)]
            
            if doRotate:
                try:
                    center = ref0.TransformContinuousIndexToPhysicalPoint(
                        np.array(ref0.GetSize()) / 2.0)
                        
                    if transform_type == 0:
                        rotation = sitk.AffineTransform(im0.GetDimension())
                        rotation.Rotate(0, 1, np.radians(doRotate))
                    else:
                        rotation = sitk.Euler2DTransform()
                        rotation.SetAngle(np.radians(doRotate))
                        
                    rotation.SetCenter(center)
                    self.transform = sitk.Transform(rotation)
                    im0 = sitk.Resample(im0, ref0, self.transform)
                except Exception as e:
                    print(f"ERROR: Failed to apply rotation: {e}")
                    self.transform = None
            else:
                self.transform = None
                     
            try:
                tr = sitk.CenteredTransformInitializer(
                    ref0, im0, 
                    sitk.AffineTransform(im.GetDimension()), 
                    sitk.CenteredTransformInitializerFilter.MOMENTS)
                transform = sitk.AffineTransform(tr)
            except:
                try:
                    tr = sitk.CenteredTransformInitializer(
                        ref0, im0, 
                        sitk.AffineTransform(im.GetDimension()), 
                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
                    transform = sitk.AffineTransform(tr)
                except Exception as e:
                    print(f"ERROR: Failed to compute center transform: {e}")
                    return

            if self.transform:
                self.transform = sitk.CompositeTransform([self.transform, transform])
            else:
                self.transform = sitk.Transform(transform)
                
        except Exception as e:
            print(f"ERROR: Failed to compute center transform: {e}")
            self.transform = None
    
    def setTransformedRgb(self, ref, relativeIdx):
        """Set transformed RGB"""
        try:
            im = self.loadRgbImage()
            if not im:
                return ref
                
            if not self.transform:
                self.computeCenterTransform(im, ref, relativeIdx, 0, self.doRotate)
                
            if not self.transform:
                return ref

            try:    
                im_tr = sitk.Resample(im, ref[:, :, self.refSliceIdx], self.transform,
                    sitk.sitkNearestNeighbor, 255)
                ref_tr = sitk.JoinSeries(im_tr)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, self.refSliceIdx])    
            except Exception as e:
                im_tr = sitk.Resample(im, ref[:, :, max(0, self.refSliceIdx - 1)], self.transform,
                    sitk.sitkNearestNeighbor, 255)
                ref_tr = sitk.JoinSeries(im_tr)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, max(0, self.refSliceIdx - 1)])

            return ref
            
        except Exception as e:
            print(f"ERROR: Failed to set transformed RGB: {e}")
            return ref
    
    def setTransformedMask(self, ref, idxMask, relativeIdx):
        """Set transformed mask with binary values (0 or 1)"""
        try:
            im = self.loadMask(idxMask)
            if not im:
                return ref

            if not self.transform:
                self.computeCenterTransform(im, ref, relativeIdx, 1, self.doRotate)
                
            if not self.transform:
                return ref

            try:    
                im_tr = sitk.Resample(im, ref[:, :, self.refSliceIdx], 
                        self.transform, 
                        sitk.sitkNearestNeighbor)
                ref_tr = sitk.JoinSeries(im_tr)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, self.refSliceIdx])
            except Exception as e:
                im_tr = sitk.Resample(im, ref[:, :, max(0, self.refSliceIdx - 1)], 
                        self.transform, 
                        sitk.sitkNearestNeighbor)
                ref_tr = sitk.JoinSeries(im_tr)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, max(0, self.refSliceIdx - 1)])
                    
            return ref
            
        except Exception as e:
            print(f"ERROR: Failed to set transformed mask: {e}")
            return ref
    
    def setTransformedMaskWithLabel(self, ref, idxMask, relativeIdx, label_value):
        """
        Set transformed mask WITH SPECIFIC LABEL VALUE
        This creates a proper labelmap where each slice has its MRI label value
        
        Args:
            ref: Reference volume (labelmap)
            idxMask: Mask index to load
            relativeIdx: Relative index
            label_value: The MRI label value to assign to this slice
        """
        try:
            im = self.loadMask(idxMask)
            if not im:
                return ref

            if not self.transform:
                self.computeCenterTransform(im, ref, relativeIdx, 1, self.doRotate)
                
            if not self.transform:
                return ref

            # Transform the mask
            try:    
                im_tr = sitk.Resample(im, ref[:, :, self.refSliceIdx], 
                        self.transform, 
                        sitk.sitkNearestNeighbor)
                
                # Convert binary mask to specific label value
                im_tr_array = sitk.GetArrayFromImage(im_tr)
                im_tr_labeled = np.where(im_tr_array > 0, label_value, 0).astype(np.uint16)
                im_tr_labeled_sitk = sitk.GetImageFromArray(im_tr_labeled)
                im_tr_labeled_sitk.CopyInformation(im_tr)
                
                ref_tr = sitk.JoinSeries(im_tr_labeled_sitk)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, self.refSliceIdx])
                    
            except Exception as e:
                im_tr = sitk.Resample(im, ref[:, :, max(0, self.refSliceIdx - 1)], 
                        self.transform, 
                        sitk.sitkNearestNeighbor)
                
                # Convert binary mask to specific label value
                im_tr_array = sitk.GetArrayFromImage(im_tr)
                im_tr_labeled = np.where(im_tr_array > 0, label_value, 0).astype(np.uint16)
                im_tr_labeled_sitk = sitk.GetImageFromArray(im_tr_labeled)
                im_tr_labeled_sitk.CopyInformation(im_tr)
                
                ref_tr = sitk.JoinSeries(im_tr_labeled_sitk)
                ref = sitk.Paste(ref, ref_tr, ref_tr.GetSize(), 
                    destinationIndex=[0, 0, max(0, self.refSliceIdx - 1)])
                    
            return ref
            
        except Exception as e:
            print(f"ERROR: Failed to set transformed mask with label: {e}")
            return ref

    def registerToConstraintWithLabel(self, fixed_image, fixed_label_mask,
                                  refMov, refMovMask, ref, refMask,
                                  idx, target_label, applyTransf=True):
        """
        Register histology slice to MRI slice using specific label mask.
        Uses a robust multi-stage approach with proper physical space handling.
        
        Args:
            fixed_image: 2D MRI slice (fixed).
            fixed_label_mask: 2D binary mask for target label.
            refMov: Reference histology RGB volume.
            refMovMask: Reference histology mask volume.
            ref: Reference volume.
            refMask: Reference mask.
            idx: Z-index of target MRI slice.
            target_label: MRI label value this slice should match.
            applyTransf: Whether to apply transform.
        """
        if self.verbose:
            print(f"\n  -> Registering to MRI label {target_label} (slice Z={idx})")

        self.target_mri_label = target_label

        try:
            # --- PHASE 0: EXTRACT AND VALIDATE IMAGES ---
            
            # Fixed image (MRI) - ensure Float32 and valid spacing
            fixed_gray = sitk.Cast(fixed_image, sitk.sitkFloat32)
            fixed_mask_2d = sitk.Cast(fixed_label_mask, sitk.sitkUInt8)
            
            # Validate fixed image spacing
            fixed_spacing = fixed_gray.GetSpacing()
            if any(s <= 0 for s in fixed_spacing):
                if self.verbose:
                    print(f"     - Warning: Invalid fixed spacing {fixed_spacing}, using [1.0, 1.0]")
                fixed_gray.SetSpacing([1.0, 1.0])
                fixed_mask_2d.SetSpacing([1.0, 1.0])
            
            # Extract moving slice (histology)
            try:
                moving_rgb_slice = refMov[:, :, self.refSliceIdx]
            except:
                moving_rgb_slice = refMov[:, :, max(0, self.refSliceIdx - 1)]
            
            moving_gray_original = self.getGrayFromRGB(moving_rgb_slice, invert=True)
            if moving_gray_original is None:
                raise RuntimeError("Failed to convert histology to grayscale")
            
            moving_gray_original = sitk.Cast(moving_gray_original, sitk.sitkFloat32)
            
            # Validate and fix moving image spacing
            moving_spacing = moving_gray_original.GetSpacing()
            if any(s <= 0 for s in moving_spacing):
                if self.verbose:
                    print(f"     - Warning: Invalid moving spacing {moving_spacing}, using [1.0, 1.0]")
                moving_gray_original.SetSpacing([1.0, 1.0])
            
            # Extract moving mask if available
            moving_mask_original = None
            if refMovMask is not None:
                try:
                    moving_mask_slice = refMovMask[:, :, self.refSliceIdx]
                    moving_mask_original = sitk.Cast(moving_mask_slice, sitk.sitkUInt8)
                    
                    # Validate mask spacing
                    mask_spacing = moving_mask_original.GetSpacing()
                    if any(s <= 0 for s in mask_spacing):
                        moving_mask_original.SetSpacing([1.0, 1.0])
                except Exception as e:
                    if self.verbose:
                        print(f"     - Warning: Could not extract moving mask: {e}")
                    moving_mask_original = None
            
            # --- PHASE 1: ROBUST INITIALIZATION ---
            if self.verbose:
                print("     - Phase 1: Computing initial alignment")
            
            # Strategy: Use GEOMETRY mode for initial alignment
            # This aligns image centers and handles spacing differences
            centering_transform = None
            
            try:
                # Try GEOMETRY mode (aligns centers)
                centering_transform = sitk.CenteredTransformInitializer(
                    fixed_gray,
                    moving_gray_original,
                    sitk.Euler2DTransform(),  # Use simpler transform for initialization
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                if self.verbose:
                    print("     - ✓ GEOMETRY initialization succeeded")
            except Exception as e1:
                if self.verbose:
                    print(f"     - GEOMETRY mode failed: {e1}")
                
                try:
                    # Fallback: MOMENTS mode (aligns intensity centroids)
                    centering_transform = sitk.CenteredTransformInitializer(
                        fixed_gray,
                        moving_gray_original,
                        sitk.Euler2DTransform(),
                        sitk.CenteredTransformInitializerFilter.MOMENTS
                    )
                    if self.verbose:
                        print("     - ✓ MOMENTS initialization succeeded")
                except Exception as e2:
                    if self.verbose:
                        print(f"     - MOMENTS mode also failed: {e2}")
                    
                    # Last resort: Manual center alignment
                    centering_transform = sitk.Euler2DTransform()
                    centering_transform.SetIdentity()
                    
                    # Calculate translation to align centers
                    fixed_center = fixed_gray.TransformContinuousIndexToPhysicalPoint(
                        [s/2.0 for s in fixed_gray.GetSize()]
                    )
                    moving_center = moving_gray_original.TransformContinuousIndexToPhysicalPoint(
                        [s/2.0 for s in moving_gray_original.GetSize()]
                    )
                    
                    translation = [fixed_center[i] - moving_center[i] for i in range(2)]
                    centering_transform.SetTranslation(translation)
                    
                    if self.verbose:
                        print(f"     - ✓ Manual center alignment: translation={translation}")
            
            # --- PHASE 2: APPLY CENTERING AND VALIDATE OVERLAP ---
            if self.verbose:
                print("     - Phase 2: Applying initial alignment")
            
            # Apply centering to moving image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_gray)
            resampler.SetTransform(centering_transform)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            
            moving_gray_centered = resampler.Execute(moving_gray_original)
            
            # Apply to mask if available
            moving_mask_centered = None
            if moving_mask_original:
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                moving_mask_centered = resampler.Execute(moving_mask_original)
            
            # Validate overlap
            moving_stats = sitk.StatisticsImageFilter()
            moving_stats.Execute(moving_gray_centered)
            
            if moving_stats.GetSum() == 0:
                if self.verbose:
                    print("     - Warning: No overlap after centering, using identity transform")
                self.transform = sitk.Transform(2, sitk.sitkIdentity)
                return
            
            # --- PHASE 3: AFFINE REGISTRATION ---
            if self.verbose:
                print("     - Phase 3: Affine registration")
            
            # Check if masks have sufficient content
            use_masks = True
            if fixed_mask_2d is not None:
                mask_stats = sitk.StatisticsImageFilter()
                mask_stats.Execute(fixed_mask_2d)
                if mask_stats.GetSum() < 10:  # Need at least 10 pixels
                    use_masks = False
                    if self.verbose:
                        print("     - Warning: Fixed mask too small, registering without masks")
            
            if moving_mask_centered is not None:
                mask_stats = sitk.StatisticsImageFilter()
                mask_stats.Execute(moving_mask_centered)
                if mask_stats.GetSum() < 10:
                    use_masks = False
                    if self.verbose:
                        print("     - Warning: Moving mask too small, registering without masks")
            
            # Setup registration
            reg = RegisterImages()
            reg.verbose = False  # Reduce noise
            n_iterations_affine = 50 if self.fastExecution else 200
            
            # Use identity as initial transform for refinement
            identity_transform = sitk.AffineTransform(2)
            identity_transform.SetIdentity()
            
            try:
                if use_masks and fixed_mask_2d is not None and moving_mask_centered is not None:
                    affine_refinement = reg.RegisterAffineWithMasks(
                        fixed_img=fixed_gray,
                        moving_img=moving_gray_centered,
                        fixed_mask=fixed_mask_2d,
                        moving_mask=moving_mask_centered,
                        initial_transf=identity_transform,
                        n_iterations=n_iterations_affine,
                        mode=0,
                        mode_score=1
                    )
                else:
                    # Register without masks
                    affine_refinement = reg.RegisterAffineWithMasks(
                        fixed_img=fixed_gray,
                        moving_img=moving_gray_centered,
                        fixed_mask=None,
                        moving_mask=None,
                        initial_transf=identity_transform,
                        n_iterations=n_iterations_affine,
                        mode=0,
                        mode_score=1
                    )
                
                # Compose transforms
                final_affine = sitk.CompositeTransform([centering_transform, affine_refinement])
                
                if self.verbose:
                    print("     - ✓ Affine registration succeeded")
            
            except Exception as e:
                if self.verbose:
                    print(f"     - Affine registration failed: {e}")
                # Use only centering transform
                final_affine = centering_transform
            
            final_transform = final_affine
            
            # --- PHASE 4: DEFORMABLE REGISTRATION (if requested) ---
            if self.doDeformable:
                if self.verbose:
                    print("     - Phase 4: Deformable registration")
                
                try:
                    n_iterations_deformable = 5 if self.fastExecution else 15
                    grid_spacing = 25 if self.fastExecution else 15
                    
                    deformable_transform = reg.RegisterDeformable(
                        fixed_img=fixed_gray,
                        moving_img=moving_gray_original,
                        initial_transf=final_affine,
                        dist_between_grid_points=grid_spacing,
                        n_iterations=n_iterations_deformable
                    )
                    
                    final_transform = sitk.CompositeTransform([final_affine, deformable_transform])
                    
                    if self.verbose:
                        print("     - ✓ Deformable registration succeeded")
                
                except Exception as e:
                    if self.verbose:
                        print(f"     - Deformable registration failed: {e}")
                    # Keep affine-only result
            
            # --- STORE RESULT ---
            self.transform = final_transform
            
            if self.verbose:
                print(f"  -> ✓ Registration completed for label {target_label}")
        
        except Exception as e:
            print(f"  -> ❌ ERROR during registration for slice {self.refSliceIdx} "
                f"(label {target_label}): {e}")
            # Use identity transform to avoid blocking process
            self.transform = sitk.Transform(2, sitk.sitkIdentity)
            
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def deleteData(self):
        """Clean up slice data"""
        if self.verbose:
            print("Deleting Slice data")
        self.__init__()