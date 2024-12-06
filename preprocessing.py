import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy import ndimage
from skimage import filters
from skimage.transform import resize
import albumentations as A
from typing import Tuple, Optional, Dict
from pydicom import dcmread
import torch.utils.data

def load_dicom_image(file_path, target_shape=None):
    """
    Load a DICOM file and return the image data as a NumPy array.
    Optionally resizes the image to the target_shape.
    """
    ds = dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept if present
    if hasattr(ds, 'RescaleSlope'):
        img *= float(ds.RescaleSlope)
    if hasattr(ds, 'RescaleIntercept'):
        img += float(ds.RescaleIntercept)

    # Normalize the image intensities
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Resize image if target_shape is specified
    if target_shape and img.shape != target_shape:
        img = resize(img, target_shape, mode='reflect', anti_aliasing=True)

    return img

class DATSCANPreprocessor:
    """
    Preprocessor class for DATSCAN images.
    """
    def __init__(self, 
                 target_shape: Tuple[int, int, int] = (128, 128, 128),
                 normalize_method: str = 'minmax',
                 apply_brain_mask: bool = True,
                 augment: bool = False):
        self.target_shape = target_shape
        self.normalize_method = normalize_method
        self.apply_brain_mask = apply_brain_mask
        self.augment = augment
        
        # Define augmentation pipeline if enabled
        if self.augment:
            self.aug_pipeline = A.Compose([
                A.RandomRotate90(p=0.5),
                A.GaussNoise(var_limit=(0, 0.01), p=0.3),
                A.RandomBrightnessContrast(p=0.3),
            ])
    
    def create_brain_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a brain mask using Otsu thresholding and morphological operations.
        Ensures proper handling of 3D images.
        """
        # Apply Otsu thresholding
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        
        # Create a 3D spherical structuring element
        radius = 2
        struct = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
        x, y, z = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        mask = x*x + y*y + z*z <= radius*radius
        struct[mask] = 1
        
        try:
            # Apply morphological operations to clean up the mask
            binary = ndimage.binary_closing(binary, structure=struct, iterations=2)
            binary = ndimage.binary_fill_holes(binary)
            
            # Keep only the largest connected component
            labeled, num_features = ndimage.label(binary, structure=struct)
            if num_features > 1:
                sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
                mask = binary.copy()
                mask[labeled != np.argmax(sizes) + 1] = 0
                binary = mask
        except Exception as e:
            print(f"Warning: Morphological operations failed, returning simple threshold mask. Error: {str(e)}")
            return (image > thresh).astype(float)
        
        return binary.astype(float)
    
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity using specified method.
        """
        if self.normalize_method == 'minmax':
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val != 0:
                return (image - min_val) / (max_val - min_val)
            return image - min_val
        
        elif self.normalize_method == 'zscore':
            mean_val = np.mean(image)
            std_val = np.std(image)
            if std_val != 0:
                return (image - mean_val) / std_val
            return image - mean_val
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
    
    def resize_volume(self, image: np.ndarray) -> np.ndarray:
        """
        Resize volume to target shape.
        """
        if image.shape != self.target_shape:
            return resize(image, 
                        self.target_shape, 
                        mode='reflect', 
                        anti_aliasing=True)
        return image
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to the image if enabled.
        """
        if self.augment:
            # Convert to format expected by albumentations
            augmented = self.aug_pipeline(image=image)
            return augmented['image']
        return image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        """
        # Initial normalization
        image = self.normalize_intensity(image)
        
        # Create and apply brain mask if requested
        if self.apply_brain_mask:
            mask = self.create_brain_mask(image)
            image = image * mask
        
        # Resize to target shape
        image = self.resize_volume(image)
        
        # Apply augmentation if enabled
        image = self.apply_augmentation(image)
        
        return image

class DATSCANDataset(Dataset):
    """
    PyTorch Dataset for DATSCAN images.
    """
    def __init__(self, 
                 file_paths: list,
                 preprocessor: DATSCANPreprocessor,
                 device: torch.device = torch.device('cpu')):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        self.device = device
        # Keep load_dicom_image as an instance method
        self.load_dicom_image = load_dicom_image
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image using the instance method
        img = self.load_dicom_image(self.file_paths[idx])
        
        # Apply preprocessing
        img = self.preprocessor(img)
        
        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        return img_tensor  # Return CPU tensor

def create_dataloaders(df: pd.DataFrame,
                      batch_size: int = 32,
                      target_shape: Tuple[int, int, int] = (128, 128, 128),
                      normalize_method: str = 'minmax',
                      apply_brain_mask: bool = True,
                      augment: bool = False,
                      device: torch.device = torch.device('cpu'),
                      num_workers: int = 4) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for each group (PD, SWEDD, Control).
    """
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
            preprocessor=preprocessor,
            device=device
        )
        
        dataloaders[group] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    return dataloaders