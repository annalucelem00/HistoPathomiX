"""
Script diagnostico per identificare problemi nella registrazione e visualizzazione
Salva questo file come 'diagnose_registration.py' nella cartella del progetto
"""

import SimpleITK as sitk
import numpy as np
import os
import json

def diagnose_registration_results(output_dir):
    """
    Analizza i risultati della registrazione per identificare problemi
    
    Args:
        output_dir: Directory contenente i risultati (es. /Volumes/Recupero/XAUSA/output)
    """
    print("="*70)
    print("DIAGNOSTIC REPORT - REGISTRATION ANALYSIS")
    print("="*70)
    
    # File da analizzare
    mr_path = None
    hist_rgb_path = os.path.join(output_dir, "registered_histology_rgb.nrrd")
    hist_mask_path = os.path.join(output_dir, "registered_histology_labelmap.nrrd")
    mapping_path = os.path.join(output_dir, "slice_to_label_mapping.json")
    
    # 1. Trova il volume MRI originale
    parent_dir = os.path.dirname(output_dir)
    for file in os.listdir(parent_dir):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            if 't2' in file.lower() or 'mri' in file.lower():
                mr_path = os.path.join(parent_dir, file)
                break
    
    if not mr_path:
        print("\n⚠️ WARNING: Could not find MRI volume. Please provide path manually.")
        return
    
    print(f"\n📁 Files to analyze:")
    print(f"  MRI: {os.path.basename(mr_path)}")
    print(f"  Histology RGB: {os.path.basename(hist_rgb_path)}")
    print(f"  Histology Mask: {os.path.basename(hist_mask_path)}")
    
    # 2. Carica le immagini
    print("\n" + "="*70)
    print("LOADING IMAGES")
    print("="*70)
    
    try:
        mr_img = sitk.ReadImage(mr_path)
        print(f"\n✓ MRI loaded successfully")
    except Exception as e:
        print(f"\n❌ Failed to load MRI: {e}")
        return
    
    try:
        hist_rgb_img = sitk.ReadImage(hist_rgb_path)
        print(f"✓ Histology RGB loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load histology RGB: {e}")
        return
    
    try:
        hist_mask_img = sitk.ReadImage(hist_mask_path)
        print(f"✓ Histology mask loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load histology mask: {e}")
        return
    
    # 3. Analizza le proprietà fisiche
    print("\n" + "="*70)
    print("PHYSICAL PROPERTIES ANALYSIS")
    print("="*70)
    
    def print_image_info(name, img):
        print(f"\n{name}:")
        print(f"  Size:      {img.GetSize()}")
        print(f"  Spacing:   {img.GetSpacing()}")
        print(f"  Origin:    {img.GetOrigin()}")
        print(f"  Direction: {img.GetDirection()[:3]} (first row)")
        print(f"             {img.GetDirection()[3:6]} (second row)")
        print(f"             {img.GetDirection()[6:9]} (third row)")
    
    print_image_info("MRI Volume", mr_img)
    print_image_info("Histology RGB", hist_rgb_img)
    print_image_info("Histology Mask", hist_mask_img)
    
    # 4. Verifica allineamento spaziale
    print("\n" + "="*70)
    print("SPATIAL ALIGNMENT CHECK")
    print("="*70)
    
    # Check if spacing matches
    mr_spacing = np.array(mr_img.GetSpacing())
    hist_spacing = np.array(hist_rgb_img.GetSpacing())
    spacing_diff = np.abs(mr_spacing[:2] - hist_spacing[:2])
    
    print(f"\nIn-plane spacing comparison (X, Y):")
    print(f"  MRI:       {mr_spacing[:2]}")
    print(f"  Histology: {hist_spacing[:2]}")
    print(f"  Difference: {spacing_diff}")
    
    if np.any(spacing_diff > 0.001):
        print("  ⚠️ WARNING: Spacing mismatch detected!")
    else:
        print("  ✓ Spacing matches correctly")
    
    # Check if origins are compatible
    mr_origin = np.array(mr_img.GetOrigin())
    hist_origin = np.array(hist_rgb_img.GetOrigin())
    
    print(f"\nOrigin comparison:")
    print(f"  MRI:       {mr_origin}")
    print(f"  Histology: {hist_origin}")
    print(f"  Difference: {np.abs(mr_origin - hist_origin)}")
    
    # Check if directions match
    mr_dir = np.array(mr_img.GetDirection())
    hist_dir = np.array(hist_rgb_img.GetDirection())
    dir_match = np.allclose(mr_dir, hist_dir, atol=1e-6)
    
    print(f"\nDirection matrix comparison:")
    if dir_match:
        print("  ✓ Direction matrices match")
    else:
        print("  ⚠️ WARNING: Direction matrices differ!")
        print(f"  Max difference: {np.max(np.abs(mr_dir - hist_dir))}")
    
    # 5. Analizza il contenuto delle immagini
    print("\n" + "="*70)
    print("IMAGE CONTENT ANALYSIS")
    print("="*70)
    
    # Check MRI
    mr_array = sitk.GetArrayFromImage(mr_img)
    print(f"\nMRI statistics:")
    print(f"  Min:  {np.min(mr_array):.2f}")
    print(f"  Max:  {np.max(mr_array):.2f}")
    print(f"  Mean: {np.mean(mr_array):.2f}")
    print(f"  Non-zero voxels: {np.count_nonzero(mr_array)} / {mr_array.size}")
    
    # Check histology RGB
    hist_rgb_array = sitk.GetArrayFromImage(hist_rgb_img)
    print(f"\nHistology RGB statistics:")
    print(f"  Min:  {np.min(hist_rgb_array):.2f}")
    print(f"  Max:  {np.max(hist_rgb_array):.2f}")
    print(f"  Mean: {np.mean(hist_rgb_array):.2f}")
    print(f"  Non-zero voxels: {np.count_nonzero(hist_rgb_array)} / {hist_rgb_array.size}")
    
    # Check which slices have histology data
    print(f"\nHistology content per slice:")
    for z in range(hist_rgb_array.shape[0]):
        slice_data = hist_rgb_array[z, :, :]
        non_zero = np.count_nonzero(slice_data)
        total = slice_data.size
        percentage = (non_zero / total) * 100
        print(f"  Slice {z}: {non_zero}/{total} ({percentage:.1f}%) non-zero")
    
    # Check mask
    hist_mask_array = sitk.GetArrayFromImage(hist_mask_img)
    unique_labels = np.unique(hist_mask_array)
    print(f"\nHistology mask labels: {sorted(unique_labels)}")
    
    for label in unique_labels:
        if label == 0:
            continue
        count = np.count_nonzero(hist_mask_array == label)
        print(f"  Label {label}: {count} voxels")
    
    # 6. Verifica slice-to-label mapping
    print("\n" + "="*70)
    print("SLICE-TO-LABEL MAPPING CHECK")
    print("="*70)
    
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        print(f"\nMapping information:")
        print(f"  Histology slices: {mapping_data.get('num_histology_slices', 'N/A')}")
        print(f"  MRI labels: {mapping_data.get('num_mri_labels', 'N/A')}")
        
        slice_to_label = mapping_data.get('slice_to_label', {})
        print(f"\nSlice to label associations:")
        for slice_idx, label_val in slice_to_label.items():
            print(f"  Histology slice {slice_idx} → MRI label {label_val}")
    else:
        print("\n⚠️ Mapping file not found")
    
    # 7. Test di resampling su una singola slice
    print("\n" + "="*70)
    print("RESAMPLING TEST")
    print("="*70)
    
    try:
        # Estrai slice centrale MRI
        mid_z = mr_img.GetSize()[2] // 2
        
        mr_size = list(mr_img.GetSize())
        mr_size[2] = 0
        mr_index = [0, 0, mid_z]
        
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(mr_size)
        extractor.SetIndex(mr_index)
        mr_slice = extractor.Execute(mr_img)
        
        print(f"\nExtracted MRI slice {mid_z}:")
        print(f"  Size: {mr_slice.GetSize()}")
        print(f"  Spacing: {mr_slice.GetSpacing()}")
        
        # Verifica se esiste istologia per questa slice
        if mid_z < hist_rgb_img.GetSize()[2]:
            hist_size = list(hist_rgb_img.GetSize())
            hist_size[2] = 0
            hist_index = [0, 0, mid_z]
            
            extractor.SetSize(hist_size)
            extractor.SetIndex(hist_index)
            hist_slice = extractor.Execute(hist_rgb_img)
            
            # Resample histology to MR slice space
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mr_slice)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)
            resampler.SetTransform(sitk.Transform())
            
            hist_resampled = resampler.Execute(hist_slice)
            
            print(f"\nResampled histology to MR space:")
            print(f"  Size: {hist_resampled.GetSize()}")
            print(f"  Spacing: {hist_resampled.GetSpacing()}")
            
            # Check content
            resampled_array = sitk.GetArrayFromImage(hist_resampled)
            non_zero = np.count_nonzero(resampled_array)
            print(f"  Non-zero voxels: {non_zero} / {resampled_array.size}")
            
            if non_zero == 0:
                print("  ⚠️ WARNING: Resampled slice is empty!")
            else:
                print("  ✓ Resampling produced non-empty result")
        else:
            print(f"\n⚠️ No histology slice available for MR slice {mid_z}")
    
    except Exception as e:
        print(f"\n❌ Resampling test failed: {e}")
    
    # 8. Raccomandazioni
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    if not dir_match:
        recommendations.append(
            "⚠️ Direction matrices don't match - this could cause misalignment.\n"
            "   Consider reorienting images to standard space (e.g., LPS or RAS)"
        )
    
    if np.any(spacing_diff > 0.001):
        recommendations.append(
            "⚠️ Spacing mismatch detected - verify registration parameters"
        )
    
    if hist_rgb_img.GetSize()[2] < mr_img.GetSize()[2]:
        recommendations.append(
            f"ℹ️ Histology has fewer slices ({hist_rgb_img.GetSize()[2]}) than MRI ({mr_img.GetSize()[2]}).\n"
            "   This is expected if histology only covers part of the MRI volume"
        )
    
    if len(recommendations) == 0:
        print("\n✓ No critical issues detected!")
        print("\nIf visualization still looks wrong, the issue may be in:")
        print("  1. The viewer's display orientation logic (flipud)")
        print("  2. The overlay blending parameters")
        print("  3. The window/level settings for MRI")
    else:
        print("\nIssues found:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_registration.py <output_directory>")
        print("\nExample:")
        print("  python diagnose_registration.py /Volumes/Recupero/XAUSA/output")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    diagnose_registration_results(output_dir)

    #cd "/Volumes/Recupero/XAUSA/Resources/Utils"
    #python3 diagnose_registration.py /Volumes/Recupero/XAUSA/output