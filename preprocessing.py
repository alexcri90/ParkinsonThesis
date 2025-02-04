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
        # Truncate negative values
        img = np.maximum(img, 0)
        if self.normalize_method == 'minmax':
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())
        return img

    def create_brain_mask(self, img: np.ndarray, threshold_factor: float = 0.1) -> np.ndarray:
        threshold = img.mean() * threshold_factor
        mask = img > threshold
        mask = ndimage.binary_fill_holes(mask)
        return mask

    def pad_volume(self, img: np.ndarray) -> np.ndarray:
        target_shape = self.target_shape
        current_shape = img.shape
        pad_width = []
        for target_dim, current_dim in zip(target_shape, current_shape):
            if current_dim > target_dim:
                start = (current_dim - target_dim) // 2
                end = start + target_dim
                pad_width.append((-start, -(current_dim - end)))
            else:
                pad_before = (target_dim - current_dim) // 2
                pad_after = target_dim - current_dim - pad_before
                pad_width.append((pad_before, pad_after))
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            if pad_before >= 0 and pad_after >= 0:
                padding = [(0, 0)] * img.ndim
                padding[axis] = (pad_before, pad_after)
                img = np.pad(img, padding, mode='constant', constant_values=0)
            else:
                slicing = [slice(None)] * img.ndim
                slicing[axis] = slice(-pad_before, img.shape[axis] + pad_after)
                img = img[tuple(slicing)]
        return img

    def resize_volume(self, img: np.ndarray) -> np.ndarray:
        current_shape = img.shape
        target_shape = self.target_shape
        current_max_dim = max(current_shape)
        target_max_dim = max(target_shape)
        if current_max_dim > target_max_dim:
            scale_factor = target_max_dim / current_max_dim
            new_shape = tuple(int(dim * scale_factor) for dim in current_shape)
            img = resize(img, new_shape, mode='reflect', anti_aliasing=True)
        return self.pad_volume(img)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self.normalize_intensity(img)
        if self.apply_brain_mask:
            mask = self.create_brain_mask(img)
            img = img * mask
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
        # Read the DICOM file and convert pixel data to float32
        img = pydicom.dcmread(file_path).pixel_array.astype(np.float32)
        
        # Ensure 3D shape before preprocessing
        if len(img.shape) == 2:
            img = np.stack([img] * self.preprocessor.target_shape[0], axis=0)
        
        # Preprocess the image
        img = self.preprocessor(img)
        
        # Ensure the image has the correct shape
        if img.shape != self.preprocessor.target_shape:
            raise ValueError(f"Image shape {img.shape} does not match target shape {self.preprocessor.target_shape}")
        
        # Add channel dimension and return
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

def load_and_preprocess_data(file_path, target_shape=(128, 128, 128)):
    """
    Load and preprocess DICOM data with consistent dimensions
    """
    try:
        # Load the DICOM image
        ds = pydicom.dcmread(file_path)
        image = ds.pixel_array.astype(np.float32)
        
        # Normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Ensure 3D shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
            
        # Resize to target shape
        current_shape = image.shape
        scale_factors = (target_shape[0]/current_shape[0],
                        target_shape[1]/current_shape[1],
                        target_shape[2]/current_shape[2] if len(current_shape) > 2 else 1)
        
        image = zoom(image, scale_factors, order=1)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        return image
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None