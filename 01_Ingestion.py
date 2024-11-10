"""
DaTSCAN Data Ingestion Module for Parkinson's Disease Analysis.

This module provides functionality to load and process DaTSCAN DICOM images
from the PPMI dataset. It includes utilities for traversing the nested folder
structure, reading DICOM files, and converting them to PyTorch tensors.

Author: [Your Name]
Date: November 2024
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime

import numpy as np
import torch
import pydicom
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DaTScanImage:
    """Data class to store DaTSCAN image information and metadata."""
    patient_id: str
    exam_date: datetime
    exam_id: str
    pixel_array: np.ndarray
    metadata: Dict
    file_path: Path
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the pixel array."""
        return self.pixel_array.shape
    
    def to_tensor(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
        """Convert pixel array to PyTorch tensor and move to specified device."""
        return torch.from_numpy(self.pixel_array).to(device)

class DaTScanLoader:
    """Class for loading and managing DaTSCAN DICOM images."""
    
    def __init__(self, root_dir: Union[str, Path], recursive: bool = True):
        """
        Initialize the DaTScan loader.
        
        Args:
            root_dir: Root directory containing the DICOM files
            recursive: Whether to search recursively through subdirectories
        """
        self.root_dir = Path(root_dir)
        self.recursive = recursive
        self.image_paths = []
        self._scan_directory()
    
    def _scan_directory(self) -> None:
        """Scan the directory structure and identify all DICOM files."""
        logger.info(f"Scanning directory: {self.root_dir}")
        
        pattern = '**/*.dcm' if self.recursive else '*.dcm'
        self.image_paths = list(self.root_dir.glob(pattern))
        
        logger.info(f"Found {len(self.image_paths)} DICOM files")
    
    def _parse_path_info(self, file_path: Path) -> Tuple[str, str, str]:
        """
        Extract patient ID, exam date, and exam ID from file path.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            Tuple containing patient_id, exam_date_str, and exam_id
        """
        # Extract patient ID from the path (assuming it's the first numeric folder)
        patient_id = None
        for part in file_path.parts:
            if part.isdigit():
                patient_id = part
                break
        
        # Extract exam date from the path (format: YYYY-MM-DD_HH_MM_SS.V)
        exam_date_str = None
        for part in file_path.parts:
            if '_' in part and part[0].isdigit():
                exam_date_str = part
                break
        
        # Extract exam ID (format: IXXX)
        exam_id = None
        for part in file_path.parts:
            if part.startswith('I') and part[1:].isdigit():
                exam_id = part
                break
        
        return patient_id, exam_date_str, exam_id

    def load_image(self, file_path: Path) -> Optional[DaTScanImage]:
        """
        Load a single DICOM image and return a DaTScanImage object.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            DaTScanImage object or None if loading fails
        """
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(file_path)
            
            # Extract path information
            patient_id, exam_date_str, exam_id = self._parse_path_info(file_path)
            
            # Parse exam date
            exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d_%H_%M_%S.%f')
            
            # Create DaTScanImage object
            image = DaTScanImage(
                patient_id=patient_id,
                exam_date=exam_date,
                exam_id=exam_id,
                pixel_array=dcm.pixel_array,
                metadata={
                    'manufacturer': getattr(dcm, 'Manufacturer', None),
                    'model': getattr(dcm, 'ManufacturerModelName', None),
                    'study_description': getattr(dcm, 'StudyDescription', None),
                    'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
                    'slice_thickness': getattr(dcm, 'SliceThickness', None)
                },
                file_path=file_path
            )
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {str(e)}")
            return None

    def load_all_images(self, max_images: Optional[int] = None) -> List[DaTScanImage]:
        """
        Load all DICOM images found in the directory.
        
        Args:
            max_images: Maximum number of images to load (for testing)
            
        Returns:
            List of DaTScanImage objects
        """
        images = []
        paths_to_process = self.image_paths[:max_images] if max_images else self.image_paths
        
        for file_path in tqdm(paths_to_process, desc="Loading DICOM files"):
            image = self.load_image(file_path)
            if image is not None:
                images.append(image)
        
        logger.info(f"Successfully loaded {len(images)} images")
        return images

def load_dataset(root_dir: Union[str, Path], 
                max_images: Optional[int] = None) -> List[DaTScanImage]:
    """
    Convenience function to load DaTSCAN dataset.
    
    Args:
        root_dir: Root directory containing the DICOM files
        max_images: Maximum number of images to load (for testing)
        
    Returns:
        List of DaTScanImage objects
    """
    loader = DaTScanLoader(root_dir)
    return loader.load_all_images(max_images)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load DaTSCAN DICOM images')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing DICOM files')
    parser.add_argument('--max_images', type=int, default=None,
                      help='Maximum number of images to load (for testing)')
    
    args = parser.parse_args()
    
    # Load images
    images = load_dataset(args.root_dir, args.max_images)
    
    # Print summary
    print(f"\nLoaded {len(images)} images")
    if images:
        print(f"Sample image shape: {images[0].shape}")
        print(f"First image metadata: {images[0].metadata}")