"""
Preprocessing module for DATSCAN images.
Implements various preprocessing steps for DATSCAN image analysis.
"""

import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import nibabel as nib
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters."""
    target_size: Tuple[int, ...] = (91, 109, 91)  # Standard size for brain images
    normalization: str = 'minmax'  # Options: 'minmax', 'zscore'
    apply_brain_mask: bool = True
    intensity_clipping: bool = True
    clip_percentile: float = 99.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DATSCANPreprocessor:
    """Class to handle preprocessing of DATSCAN images."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor with given configuration.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config or PreprocessingConfig()
        self.device = torch.device(self.config.device)
        
    def normalize_intensity(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image intensity using specified method.
        
        Args:
            image: Input image tensor
            
        Returns:
            Normalized image tensor
        """
        if self.config.intensity_clipping:
            max_val = torch.quantile(image, self.config.clip_percentile/100)
            image = torch.clamp(image, min=0, max=max_val)
        
        if self.config.normalization == 'minmax':
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        elif self.config.normalization == 'zscore':
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization}")
            
        return image
        
    def ensure_consistent_size(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize image to target size using trilinear interpolation.
        
        Args:
            image: Input image tensor
            
        Returns:
            Resized image tensor
        """
        if image.shape != self.config.target_size:
            # Add batch and channel dimensions for interpolation
            image = image.unsqueeze(0).unsqueeze(0)
            image = F.interpolate(
                image,
                size=self.config.target_size,
                mode='trilinear',
                align_corners=True
            )
            # Remove batch and channel dimensions
            image = image.squeeze(0).squeeze(0)
        return image
        
    def apply_brain_masking(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply brain masking to remove non-brain regions.
        Uses Otsu thresholding for basic skull stripping.
        
        Args:
            image: Input image tensor
            
        Returns:
            Masked image tensor
        """
        # Convert to numpy for processing
        image_np = image.cpu().numpy()
        
        # Otsu thresholding for basic skull stripping
        thresh = ndimage.gaussian_filter(image_np, sigma=2)
        mask = thresh > thresh.mean()
        
        # Clean up the mask using morphological operations
        mask = ndimage.binary_closing(mask)
        mask = ndimage.binary_fill_holes(mask)
        
        # Apply the mask
        masked_image = torch.from_numpy(image_np * mask).to(self.device)
        
        return masked_image
        
    def preprocess_single_image(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Apply full preprocessing pipeline to a single image.
        
        Args:
            image: Input image (numpy array or torch tensor)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to torch tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            
        # Move to appropriate device
        image = image.to(self.device)
        
        # Apply preprocessing steps
        image = self.ensure_consistent_size(image)
        
        if self.config.apply_brain_mask:
            image = self.apply_brain_masking(image)
            
        image = self.normalize_intensity(image)
        
        return image
        
class DATSCANPreprocessingTransform:
    """PyTorch transform class for DATSCAN preprocessing."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the transform.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.preprocessor = DATSCANPreprocessor(config)
        
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Apply preprocessing transform to an image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        return self.preprocessor.preprocess_single_image(image)


def create_preprocessing_pipeline(config: Optional[PreprocessingConfig] = None) -> transforms.Compose:
    """
    Create a complete preprocessing pipeline including data augmentation if needed.
    
    Args:
        config: Configuration object with preprocessing parameters
        
    Returns:
        Composed transform pipeline
    """
    transform_list = [
        DATSCANPreprocessingTransform(config),
        # Add any additional transforms here (e.g., data augmentation)
    ]
    
    return transforms.Compose(transform_list)


# Example usage with our existing dataset
class PreprocessedDATSCANDataset(Dataset):
    """Dataset class with preprocessing pipeline."""
    
    def __init__(self, base_path: str, preprocessing_config: Optional[PreprocessingConfig] = None):
        """
        Initialize the dataset with preprocessing.
        
        Args:
            base_path: Root directory containing the DATSCAN images
            preprocessing_config: Configuration for preprocessing
        """
        self.dataset = DATSCANDataset(base_path)
        self.transform = create_preprocessing_pipeline(preprocessing_config)
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, DATSCANMetadata]:
        """
        Get a preprocessed item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (preprocessed_image, metadata)
        """
        image, metadata = self.dataset[idx]
        preprocessed_image = self.transform(image)
        return preprocessed_image, metadata


def main():
    """Example usage of the preprocessing pipeline."""
    # Example configuration
    config = PreprocessingConfig(
        target_size=(91, 109, 91),
        normalization='minmax',
        apply_brain_mask=True,
        intensity_clipping=True,
        clip_percentile=99.0
    )
    
    # Create preprocessor
    preprocessor = DATSCANPreprocessor(config)
    
    # Load and preprocess a sample image
    dataset = PreprocessedDATSCANDataset("path/to/data", config)
    
    # Get a sample
    image, metadata = dataset[0]
    
    logger.info(f"Preprocessed image shape: {image.shape}")
    logger.info(f"Preprocessing complete!")

if __name__ == "__main__":
    main()