import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy import ndimage
from skimage.transform import resize
import pydicom
from typing import Tuple, Optional, Dict
import torch.utils.data

class DATSCANPreprocessor:
    def __init__(self, 
                 target_shape: Tuple[int, int, int] = (128, 128, 128),
                 normalize_method: str = 'minmax',
                 apply_brain_mask: bool = True,
                 augment: bool = False):
        self.target_shape = target_shape
        self.normalize_method = normalize_method
        self.apply_brain_mask = apply_brain_mask
        self.augment = augment

    def normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image intensities with expert-recommended approach:
        - Set minimum to 0
        - Truncate negative values
        - No contrast enhancement
        - Optional very light clipping at 99.99th percentile
        """
        # Truncate negative values
        img = np.maximum(img, 0)
        
        if self.normalize_method == 'minmax':
            # Optional very light clipping (comment out if not needed)
            # max_val = np.percentile(img, 99.99)
            # img = np.clip(img, 0, max_val)
            
            # Min-max normalization
            if img.max() > 0:  # Avoid division by zero
                img = (img - img.min()) / (img.max() - img.min())
        
        return img

    def create_brain_mask(self, img: np.ndarray, threshold_factor: float = 0.1) -> np.ndarray:
        """Create a brain mask using basic thresholding"""
        threshold = img.mean() * threshold_factor
        mask = img > threshold
        # Apply morphological operations to clean up the mask
        mask = ndimage.binary_fill_holes(mask)
        return mask

    def pad_volume(self, img: np.ndarray) -> np.ndarray:
        """
        Pad the volume to target shape while maintaining aspect ratio.
        Pads with zeros to reach target dimensions.
        """
        target_shape = self.target_shape
        current_shape = img.shape
        
        # Calculate padding for each dimension
        pad_width = []
        for target_dim, current_dim in zip(target_shape, current_shape):
            if current_dim > target_dim:
                # If current dimension is larger, we need to crop
                start = (current_dim - target_dim) // 2
                end = start + target_dim
                # Store negative values to indicate cropping
                pad_width.append((-start, -(current_dim - end)))
            else:
                # If current dimension is smaller, we need to pad
                pad_before = (target_dim - current_dim) // 2
                pad_after = target_dim - current_dim - pad_before
                pad_width.append((pad_before, pad_after))
        
        # Apply padding/cropping
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            if pad_before >= 0 and pad_after >= 0:
                # Padding case
                padding = [(0, 0)] * img.ndim
                padding[axis] = (pad_before, pad_after)
                img = np.pad(img, padding, mode='constant', constant_values=0)
            else:
                # Cropping case
                slicing = [slice(None)] * img.ndim
                slicing[axis] = slice(-pad_before, pad_after)
                img = img[tuple(slicing)]
        
        return img

    def resize_volume(self, img: np.ndarray) -> np.ndarray:
        """
        Instead of resizing/stretching, we now use padding to maintain aspect ratio
        """
        # First, determine if we need to do any initial scaling
        current_max_dim = max(img.shape)
        target_max_dim = max(self.target_shape)
        
        if current_max_dim > target_max_dim:
            # Scale down proportionally if the image is too large
            scale_factor = target_max_dim / current_max_dim
            new_shape = tuple(int(dim * scale_factor) for dim in img.shape)
            img = resize(img, new_shape, mode='reflect', anti_aliasing=True)
        
        # Now pad to target shape
        return self.pad_volume(img)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Process the image:
        1. Normalize intensities
        2. Apply brain mask if requested
        3. Resize to target shape
        """
        # Normalize intensities
        img = self.normalize_intensity(img)
        
        # Apply brain mask if requested
        if self.apply_brain_mask:
            mask = self.create_brain_mask(img)
            img = img * mask
        
        # Resize to target shape
        if self.target_shape is not None and img.shape != self.target_shape:
            img = self.resize_volume(img)
        
        return img
    
    

class DATSCANDataset(Dataset):
    def __init__(self, 
                 file_paths: list,
                 preprocessor: DATSCANPreprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        img = pydicom.dcmread(file_path).pixel_array.astype(np.float32)
        img = self.preprocessor(img)
        return torch.from_numpy(img).float().unsqueeze(0)

def create_dataloaders(df: pd.DataFrame,
                      batch_size: int = 32,
                      target_shape: Tuple[int, int, int] = (128, 128, 128),
                      normalize_method: str = 'minmax',
                      apply_brain_mask: bool = True,
                      augment: bool = False,
                      device: torch.device = torch.device('cpu'),
                      num_workers: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """Create DataLoaders for each group"""
    preprocessor = DATSCANPreprocessor(
        target_shape=target_shape,
        normalize_method=normalize_method,
        apply_brain_mask=apply_brain_mask,
        augment=augment
    )
    
    dataloaders = {}
    for group in ['PD', 'SWEDD', 'Control']:
        group_files = df[df['label'] == group]['file_path'].tolist()
        dataset = DATSCANDataset(
            file_paths=group_files,
            preprocessor=preprocessor
        )
        
        dataloaders[group] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    return dataloaders

def find_striatum_center(self, img: np.ndarray) -> int:
    """
    Find the approximate center of the striatum in the z-axis.
    Uses the fact that striatal slices typically have higher intensity and more focused signal.
    """
    # Calculate variance for each axial slice
    variances = np.var(img, axis=(1, 2))
    
    # Apply slight smoothing to avoid noise-based peaks
    from scipy.ndimage import gaussian_filter1d
    smoothed_variances = gaussian_filter1d(variances, sigma=2.0)
    
    # Find the region of highest variance (likely to be the striatum)
    center_idx = np.argmax(smoothed_variances)
    
    return center_idx

def extract_striatum_region(self, img: np.ndarray, slice_thickness: int = 64) -> np.ndarray:
    """
    Extract a thick slice centered on the striatum.
    
    Args:
        img: Input 3D image
        slice_thickness: Number of slices to extract (default 64)
    """
    # Find center of striatum
    center = self.find_striatum_center(img)
    
    # Calculate start and end slices
    half_thickness = slice_thickness // 2
    start_slice = max(0, center - half_thickness)
    end_slice = min(img.shape[0], center + half_thickness)
    
    # Extract the region
    striatum_region = img[start_slice:end_slice]
    
    # If we didn't get enough slices (edge cases), pad with zeros
    if striatum_region.shape[0] < slice_thickness:
        padding_needed = slice_thickness - striatum_region.shape[0]
        pad_before = padding_needed // 2
        pad_after = padding_needed - pad_before
        striatum_region = np.pad(
            striatum_region,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )
    
    return striatum_region

def resize_volume(self, img: np.ndarray) -> np.ndarray:
    """
    Modified to focus on striatal region and maintain aspect ratio
    """
    # First, extract the striatal region (64 slices)
    img = self.extract_striatum_region(img, slice_thickness=64)
    
    # Now handle the x and y dimensions while maintaining aspect ratio
    current_shape = img.shape[1:]  # Only consider x,y dimensions
    target_shape = self.target_shape[1:]  # Only consider x,y dimensions
    
    # Scale if necessary
    current_max_dim = max(current_shape)
    target_max_dim = max(target_shape)
    
    if current_max_dim > target_max_dim:
        # Scale down proportionally if the image is too large
        scale_factor = target_max_dim / current_max_dim
        new_shape = (img.shape[0],) + tuple(int(dim * scale_factor) for dim in current_shape)
        img = resize(img, new_shape, mode='reflect', anti_aliasing=True)
    
    # Pad to target shape
    return self.pad_volume(img)

def __call__(self, img: np.ndarray) -> np.ndarray:
    """
    Process the image:
    1. Normalize intensities
    2. Apply brain mask if requested
    3. Extract striatal region and resize/pad
    """
    # Normalize intensities
    img = self.normalize_intensity(img)
    
    # Apply brain mask if requested
    if self.apply_brain_mask:
        mask = self.create_brain_mask(img)
        img = img * mask
    
    # Extract striatal region and resize to target shape
    if self.target_shape is not None:
        img = self.resize_volume(img)
    
    return img
