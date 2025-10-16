import SimpleITK as sitk
import numpy as np

def convert_binary_mask_to_labelmap(input_mask_path, output_labelmap_path):
    """
    Converts a binary mask (all non-zero pixels are 1) to a slice-wise labelmap.
    Each Z-slice with a mask will be assigned a unique, incrementing label.
    """
    # Read the binary mask
    mask_image = sitk.ReadImage(input_mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image) # Shape: (Z, Y, X)

    # Create a new array for the labelmap, initialized to zeros
    labelmap_array = np.zeros_like(mask_array, dtype=np.uint16)

    current_label = 1
    # Iterate over each Z-slice
    for z in range(mask_array.shape[0]):
        slice_mask = mask_array[z, :, :]
        # If there are any non-zero pixels on this slice...
        if np.any(slice_mask > 0):
            # ...assign the current label to those pixels in the output array.
            labelmap_array[z, :, :][slice_mask > 0] = current_label
            print(f"Slice {z}: Assigned label {current_label}")
            current_label += 1

    if current_label == 1:
        print("Warning: No segmented regions found in the input mask.")
        return

    # Create a new SimpleITK image from the numpy array
    labelmap_image = sitk.GetImageFromArray(labelmap_array)
    # Important: Copy the physical information (spacing, origin, direction)
    labelmap_image.CopyInformation(mask_image)

    # Save the new labelmap
    sitk.WriteImage(labelmap_image, output_labelmap_path)
    print(f"\nSuccessfully created labelmap with {current_label - 1} labels.")
    print(f"Saved to: {output_labelmap_path}")


# --- USAGE ---
input_file = "/path/to/your/Segmentation_xausa.nii"
output_file = "/path/to/your/Segmentation_xausa_labelmap.nii"

convert_binary_mask_to_labelmap(input_file, output_file)