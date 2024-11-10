"""
Data loader module for DATSCAN DICOM images.
This module provides functionality to load and preprocess DATSCAN images
for the Parkinson's Disease analysis project.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DATSCANMetadata:
    """Data class to store relevant DICOM metadata."""
    patient_id: str
    exam_date: str
    exam_id: str
    image_shape: Tuple[int, ...]
    pixel_spacing: Optional[Tuple[float, ...]] = None
    
class DATSCANDataLoader:
    """Class to handle loading and preprocessing of DATSCAN DICOM images."""
    
    def __init__(self, base_path: Union[str, Path], use_gpu: bool = True):
        """
        Initialize the data loader.
        
        Args:
            base_path: Root directory containing the DATSCAN images
            use_gpu: Whether to use GPU for processing (if available)
        """
        self.base_path = Path(base_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.dicom_paths: List[Path] = []
        self.metadata: Dict[str, DATSCANMetadata] = {}
        
    def find_dicom_files(self) -> None:
        """
        Recursively traverse directory structure to find all DICOM files.
        Updates self.dicom_paths with found files.
        """
        logger.info(f"Searching for DICOM files in {self.base_path}")
        try:
            for root, _, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith('.dcm'):
                        path = Path(root) / file
                        self.dicom_paths.append(path)
            
            logger.info(f"Found {len(self.dicom_paths)} DICOM files")
        except Exception as e:
            logger.error(f"Error while searching for DICOM files: {str(e)}")
            raise
            
    def load_single_dicom(self, path: Path) -> Tuple[np.ndarray, DATSCANMetadata]:
        """
        Load a single DICOM file and extract relevant metadata.
        
        Args:
            path: Path to the DICOM file
            
        Returns:
            Tuple containing:
                - np.ndarray: The image data
                - DATSCANMetadata: Associated metadata
        """
        try:
            dcm = pydicom.dcmread(path)
            
            # Extract image data
            image_data = dcm.pixel_array.astype(float)
            
            # Extract metadata
            patient_id = str(path).split(os.sep)[3]  # Based on folder structure
            exam_date = str(path).split(os.sep)[5]   # Based on folder structure
            exam_id = str(path).split(os.sep)[6]     # Based on folder structure
            
            metadata = DATSCANMetadata(
                patient_id=patient_id,
                exam_date=exam_date,
                exam_id=exam_id,
                image_shape=image_data.shape,
                pixel_spacing=tuple(float(x) for x in dcm.PixelSpacing) if hasattr(dcm, 'PixelSpacing') else None
            )
            
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {path}: {str(e)}")
            raise
            
    def load_all_data(self) -> Tuple[List[np.ndarray], Dict[str, DATSCANMetadata]]:
        """
        Load all DICOM files found in the directory structure.
        
        Returns:
            Tuple containing:
                - List of image arrays
                - Dictionary mapping file paths to metadata
        """
        if not self.dicom_paths:
            self.find_dicom_files()
            
        images = []
        metadata_dict = {}
        
        for path in self.dicom_paths:
            try:
                image, metadata = self.load_single_dicom(path)
                images.append(image)
                metadata_dict[str(path)] = metadata
            except Exception as e:
                logger.warning(f"Skipping file {path} due to error: {str(e)}")
                continue
                
        return images, metadata_dict

class DATSCANDataset(Dataset):
    """PyTorch Dataset class for DATSCAN images."""
    
    def __init__(self, base_path: Union[str, Path], transform=None):
        """
        Initialize the dataset.
        
        Args:
            base_path: Root directory containing the DATSCAN images
            transform: Optional transform to be applied to the images
        """
        self.loader = DATSCANDataLoader(base_path)
        self.transform = transform
        self.images, self.metadata = self.loader.load_all_data()
        
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, DATSCANMetadata]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple containing:
                - torch.Tensor: The image data
                - DATSCANMetadata: Associated metadata
        """
        image = self.images[idx]
        metadata = list(self.metadata.values())[idx]
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float()
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, metadata

def main():
    """Example usage of the data loader."""
    # Example path - replace with actual path
    base_path = Path("Images_Test/dicom")
    
    # Initialize data loader
    loader = DATSCANDataLoader(base_path)
    
    # Find all DICOM files
    loader.find_dicom_files()
    
    # Load all data
    images, metadata = loader.load_all_data()
    
    # Print summary
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"First image shape: {images[0].shape}")
    
if __name__ == "__main__":
    main()