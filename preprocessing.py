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

    def resize_volume(self, img: np.ndarray) -> np.ndarray:
        """Resize the volume to target shape"""
        return resize(img, self.target_shape, mode='reflect', anti_aliasing=True)

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