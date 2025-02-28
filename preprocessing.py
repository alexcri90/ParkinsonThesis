#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing utilities for DATSCAN images.
Includes DICOM loading, volume processing, and brain masking functions.
"""

import os
import logging
import numpy as np
import pydicom
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, ball

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dicom(file_path):
    """
    Loads and processes a DICOM file:
    - Reads the file using pydicom.
    - Converts the pixel array to float32.
    - Applies RescaleSlope and RescaleIntercept if available.

    Args:
        file_path: Path to the DICOM file.
    
    Returns:
        Tuple (processed_pixel_array, dicom_metadata)
    """
    try:
        ds = pydicom.dcmread(file_path)
    except Exception as e:
        raise IOError(f"Error reading DICOM file {file_path}: {e}")

    # Extract pixel array and convert to float32
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescaling if attributes are present
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = ds.RescaleSlope
        intercept = ds.RescaleIntercept
        pixel_array = pixel_array * slope + intercept

    return pixel_array, ds


def resize_volume(volume, target_shape=(64, 128, 128)):
    """
    Resizes the volume to the target shape using zero-padding or center cropping.

    Args:
        volume: Input 3D volume as numpy array with shape (d, h, w)
        target_shape: Desired output shape as tuple (d_new, h_new, w_new)

    Returns:
        Resized volume with shape target_shape
    """
    def get_pad_amounts(current_size, target_size):
        """Helper to calculate padding amounts"""
        if current_size >= target_size:
            return 0, 0
        diff = target_size - current_size
        pad_before = diff // 2
        pad_after = diff - pad_before
        return pad_before, pad_after

    current_shape = volume.shape
    resized = volume.copy()

    # Calculate padding/cropping for each dimension
    pads = [get_pad_amounts(current_shape[i], target_shape[i]) for i in range(3)]

    # Apply padding if needed
    if any(sum(p) > 0 for p in pads):
        resized = np.pad(
            resized,
            pad_width=pads,
            mode="constant",
            constant_values=0
        )

    # Apply cropping if needed
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            # Calculate slicing indices
            start = (current_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            # Apply slice
            if i == 0:
                resized = resized[start:end, :, :]
            elif i == 1:
                resized = resized[:, start:end, :]
            else:
                resized = resized[:, :, start:end]

    return resized


def process_volume(volume, target_shape=(64, 128, 128)):
    """
    Process a 3D volume by:
    1. Normalizing intensity (truncating negatives and min-max scaling)
    2. Resizing to target_shape
    3. Generating a brain mask via Otsu thresholding and morphological closing

    Args:
        volume: Input 3D volume
        target_shape: Desired output shape (depth, height, width)

    Returns:
        norm_vol: Normalized and resized volume
        mask: Brain mask
        masked_vol: Masked volume
    """
    # 1. Resize the normalized volume
    norm_vol = resize_volume(volume - volume.min(), target_shape=target_shape)
    
    # Apply pre-defined mask (as in the notebook)
    mask = np.zeros(target_shape, dtype=bool)
    mask[20:40, 82:103, 43:82] = 1
    
    # Normalize by mean intensity in the mask region
    norm_vol /= np.mean(norm_vol[mask])

    # 2. Compute brain mask using Otsu thresholding
    thresh = threshold_otsu(norm_vol)
    mask = norm_vol > thresh
    mask = binary_closing(mask, footprint=ball(2))
    
    # 3. Apply mask to volume
    masked_vol = norm_vol * mask

    return norm_vol, mask, masked_vol


def preprocess_dicom_file(file_path, target_shape=(64, 128, 128)):
    """
    Preprocess a single DICOM file by loading and processing the volume.
    
    Args:
        file_path: Path to the DICOM file
        target_shape: Target shape for the processed volume
    
    Returns:
        norm_vol: Normalized and processed volume
    """
    try:
        # Load the DICOM file
        original_volume, _ = load_dicom(file_path)
        
        # Extract relevant slices as done in the notebook
        original_volume = original_volume[9:73, :, :]
        
        # Process the volume
        norm_vol, _, _ = process_volume(original_volume, target_shape=target_shape)
        
        return norm_vol
    
    except Exception as e:
        logger.error(f"Error preprocessing file {file_path}: {str(e)}")
        raise


# For testing purposes
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    
    # Load the first file from the validated file paths
    df = pd.read_csv("validated_file_paths.csv")
    sample_file = df.iloc[0]["file_path"]
    
    print(f"Processing sample file: {sample_file}")
    
    # Process the volume
    original_volume, _ = load_dicom(sample_file)
    original_volume = original_volume[9:73, :, :]
    
    norm_vol, mask, masked_vol = process_volume(original_volume)
    
    # Display middle slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(original_volume[32, :, :], cmap='gray')
    axes[0, 0].set_title('Original Axial Slice')
    axes[0, 1].imshow(original_volume[:, 64, :], cmap='gray')
    axes[0, 1].set_title('Original Coronal Slice')
    axes[0, 2].imshow(original_volume[:, :, 64], cmap='gray')
    axes[0, 2].set_title('Original Sagittal Slice')
    
    axes[1, 0].imshow(norm_vol[32, :, :], cmap='gray')
    axes[1, 0].set_title('Processed Axial Slice')
    axes[1, 1].imshow(norm_vol[:, 64, :], cmap='gray')
    axes[1, 1].set_title('Processed Coronal Slice')
    axes[1, 2].imshow(norm_vol[:, :, 64], cmap='gray')
    axes[1, 2].set_title('Processed Sagittal Slice')
    
    plt.tight_layout()
    plt.savefig('preprocessing_test.png')
    print("Preprocessing test complete. Visualization saved to preprocessing_test.png")