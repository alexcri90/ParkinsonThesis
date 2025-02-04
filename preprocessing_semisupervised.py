import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy import ndimage
from skimage.transform import resize
import pydicom
from typing import Tuple, Optional, Dict, List
import torch.utils.data

class DATSCANSemiSupervisedPreprocessor:
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

class DATSCANSemiSupervisedDataset(Dataset):
    def __init__(self, 
                 file_paths: list,
                 labels: List[str],
                 preprocessor: DATSCANSemiSupervisedPreprocessor,
                 metadata: Optional[pd.DataFrame] = None):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        
        # Convert string labels to numeric
        self.label_map = {'PD': 0, 'SWEDD': 1, 'Control': 2}
        self.labels = [self.label_map[label] for label in labels]
        
        # Store metadata if provided
        self.metadata = metadata
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Read and preprocess the image
        img = pydicom.dcmread(file_path).pixel_array.astype(np.float32)
        img = self.preprocessor(img)
        
        if img.ndim == 2:
            desired_depth = self.preprocessor.target_shape[0]
            img = np.repeat(np.expand_dims(img, axis=0), repeats=desired_depth, axis=0)
        
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Create label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # If metadata is available, include it
        if self.metadata is not None:
            metadata_row = self.metadata.loc[self.metadata['file_path'] == file_path]
            metadata_tensor = torch.tensor(metadata_row.values[0], dtype=torch.float)
            return img_tensor, label_tensor, metadata_tensor
        
        return img_tensor, label_tensor

def create_semisupervised_dataloaders(df: pd.DataFrame,
                                    batch_size: int = 32,
                                    target_shape: Tuple[int, int, int] = (128, 128, 128),
                                    normalize_method: str = 'minmax',
                                    apply_brain_mask: bool = True,
                                    augment: bool = False,
                                    device: torch.device = torch.device('cuda'),
                                    num_workers: int = 4,
                                    metadata_cols: Optional[List[str]] = None) -> Dict[str, torch.utils.data.DataLoader]:
    """Create DataLoaders for semi-supervised learning"""
    preprocessor = DATSCANSemiSupervisedPreprocessor(
        target_shape=target_shape,
        normalize_method=normalize_method,
        apply_brain_mask=apply_brain_mask,
        augment=augment
    )
    
    dataloaders = {}
    
    # Prepare metadata if specified
    metadata = None
    if metadata_cols is not None:
        metadata = df[metadata_cols]
    
    for group in ['PD', 'SWEDD', 'Control']:
        group_data = df[df['label'] == group]
        dataset = DATSCANSemiSupervisedDataset(
            file_paths=group_data['file_path'].tolist(),
            labels=group_data['label'].tolist(),
            preprocessor=preprocessor,
            metadata=metadata
        )
        
        dataloaders[group] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    return dataloaders