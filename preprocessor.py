"""
Preprocessing module for DATSCAN images.
This module provides functionality to preprocess DATSCAN images
including normalization, voxel standardization, and brain masking.
"""

import logging
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# Import from our data_loader module
from data_loader import DATSCANMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingParams:
    """Parameters for preprocessing pipeline."""
    target_shape: Tuple[int, ...] = (91, 109, 91)  # Standard shape for brain images
    normalization_method: str = 'minmax'  # Options: 'minmax', 'zscore', 'percentile'
    mask_threshold: float = 0.1  # Threshold for brain masking
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DATSCANPreprocessor:
    """Main class for preprocessing DATSCAN images."""
    
    def __init__(self, params: Optional[PreprocessingParams] = None):
        """
        Initialize preprocessor with given parameters.
        
        Args:
            params: Preprocessing parameters. If None, uses defaults.
        """
        self.params = params or PreprocessingParams()
        self.device = torch.device(self.params.device)
        
    def normalize_intensity(self, image: Tensor, method: str = None) -> Tensor:
        """
        Normalize image intensity using specified method.
        
        Args:
            image: Input image tensor
            method: Normalization method ('minmax', 'zscore', or 'percentile')
            
        Returns:
            Normalized image tensor
        """
        method = method or self.params.normalization_method
        image = image.to(self.device)
        
        if method == 'minmax':
            min_val = image.min()
            max_val = image.max()
            normalized = (image - min_val) / (max_val - min_val + 1e-8)
            
        elif method == 'zscore':
            mean = image.mean()
            std = image.std()
            normalized = (image - mean) / (std + 1e-8)
            
        elif method == 'percentile':
            p_low = torch.quantile(image, 0.01)
            p_high = torch.quantile(image, 0.99)
            normalized = torch.clamp((image - p_low) / (p_high - p_low + 1e-8), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return normalized
        
    def standardize_voxel_size(self, 
                              image: Tensor, 
                              original_spacing: Tuple[float, ...],
                              target_spacing: Tuple[float, ...]) -> Tensor:
        """
        Resample image to standardized voxel size.
        
        Args:
            image: Input image tensor
            original_spacing: Original voxel spacing
            target_spacing: Target voxel spacing
            
        Returns:
            Resampled image tensor
        """
        image = image.to(self.device)
        
        # Calculate scaling factors
        scale_factors = torch.tensor(original_spacing) / torch.tensor(target_spacing)
        
        # Add batch and channel dimensions for interpolation
        image = image.unsqueeze(0).unsqueeze(0)
        
        # Calculate new size
        new_size = torch.round(torch.tensor(image.shape[2:]) * scale_factors).int()
        
        # Resample using trilinear interpolation
        resampled = F.interpolate(image, size=tuple(new_size), mode='trilinear', align_corners=False)
        
        return resampled.squeeze(0).squeeze(0)
        
    def create_brain_mask(self, image: Tensor) -> Tensor:
        """
        Create binary mask for brain region.
        
        Args:
            image: Input image tensor
            
        Returns:
            Binary mask tensor
        """
        image = image.to(self.device)
        
        # Simple threshold-based masking
        threshold = self.params.mask_threshold * image.max()
        mask = (image > threshold).float()
        
        # Optional: clean up mask using morphological operations
        # This would require additional dependencies like torch-morphology
        
        return mask
        
    def apply_brain_mask(self, image: Tensor, mask: Tensor) -> Tensor:
        """
        Apply brain mask to image.
        
        Args:
            image: Input image tensor
            mask: Binary mask tensor
            
        Returns:
            Masked image tensor
        """
        return image * mask
        
    def preprocess_single(self, 
                         image: Union[np.ndarray, Tensor], 
                         metadata: DATSCANMetadata) -> Tensor:
        """
        Apply full preprocessing pipeline to single image.
        
        Args:
            image: Input image
            metadata: Associated metadata
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to tensor if necessary
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            
        image = image.to(self.device)
        
        # Normalize intensity
        image = self.normalize_intensity(image)
        
        # Create and apply brain mask
        mask = self.create_brain_mask(image)
        image = self.apply_brain_mask(image, mask)
        
        # Standardize voxel size if spacing information is available
        if metadata.pixel_spacing:
            target_spacing = (1.0, 1.0, 1.0)  # Example target spacing
            image = self.standardize_voxel_size(image, metadata.pixel_spacing, target_spacing)
            
        return image
        
    def preprocess_batch(self, 
                        images: List[Union[np.ndarray, Tensor]], 
                        metadata_list: List[DATSCANMetadata]) -> List[Tensor]:
        """
        Apply preprocessing to a batch of images.
        
        Args:
            images: List of input images
            metadata_list: List of associated metadata
            
        Returns:
            List of preprocessed image tensors
        """
        return [self.preprocess_single(img, meta) for img, meta in zip(images, metadata_list)]

class PreprocessingTransform:
    """PyTorch-style transform for preprocessing DATSCAN images."""
    
    def __init__(self, params: Optional[PreprocessingParams] = None):
        """
        Initialize transform.
        
        Args:
            params: Preprocessing parameters
        """
        self.preprocessor = DATSCANPreprocessor(params)
        
    def __call__(self, image: Tensor, metadata: DATSCANMetadata) -> Tuple[Tensor, DATSCANMetadata]:
        """
        Apply preprocessing transform.
        
        Args:
            image: Input image tensor
            metadata: Associated metadata
            
        Returns:
            Tuple of (preprocessed image tensor, metadata)
        """
        processed_image = self.preprocessor.preprocess_single(image, metadata)
        return processed_image, metadata

def main():
    """Example usage of the preprocessor."""
    # Example preprocessing pipeline
    params = PreprocessingParams(
        target_shape=(91, 109, 91),
        normalization_method='zscore',
        mask_threshold=0.1
    )
    
    preprocessor = DATSCANPreprocessor(params)
    
    # Example preprocessing transform for use with PyTorch Dataset
    transform = PreprocessingTransform(params)
    
    logger.info("Preprocessor and transform initialized successfully")
    
if __name__ == "__main__":
    main()