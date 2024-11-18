PromptingText

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from rich.console import Console
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DICOMMetadata:
    """Data class to store relevant DICOM metadata."""
    patient_id: str
    study_date: str
    modality: str
    manufacturer: str
    pixel_spacing: Tuple[float, float]
    image_shape: Tuple[int, ...]

class DATSCANIngestion:
    """Class for handling DATSCAN DICOM data ingestion.
    
    This class provides functionality to:
    1. Recursively find DICOM files in a directory structure
    2. Load and validate DICOM files
    3. Convert DICOM data to numpy arrays
    4. Optionally transfer data to GPU using PyTorch
    
    Attributes:
        root_dir (Path): Root directory containing DICOM files
        device (torch.device): Device to use for PyTorch operations
        console (Console): Rich console for enhanced output
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        use_gpu: bool = True,
        file_pattern: str = "*.dcm"
    ):
        """Initialize the DATSCAN ingestion handler.
        
        Args:
            root_dir: Path to the root directory containing DICOM files
            use_gpu: Whether to use GPU for processing (if available)
            file_pattern: Pattern to match DICOM files
        """
        self.root_dir = Path(root_dir)
        self.file_pattern = file_pattern
        self.console = Console()
        
        # Set up device for PyTorch operations
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Validate root directory
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        logger.info(f"Initialized DATSCANIngestion with root directory: {self.root_dir}")

    def find_dicom_files(self) -> List[Path]:
        """Recursively find all DICOM files in the root directory.
        
        Returns:
            List of Path objects for all DICOM files found
        """
        try:
            dicom_files = list(self.root_dir.rglob(self.file_pattern))
            logger.info(f"Found {len(dicom_files)} DICOM files")
            return dicom_files
        except Exception as e:
            logger.error(f"Error finding DICOM files: {e}")
            raise

    def validate_dicom(self, ds: pydicom.dataset.FileDataset) -> bool:
        """Validate DICOM metadata for DATSCAN compatibility.
        
        Args:
            ds: PyDICOM dataset to validate
            
        Returns:
            bool: Whether the DICOM file is valid for our purposes
        """
        required_attributes = [
            'PatientID',
            'StudyDate',
            'Modality',
            'Manufacturer',
            'PixelSpacing'
        ]
        
        return all(hasattr(ds, attr) for attr in required_attributes)

    def extract_metadata(self, ds: pydicom.dataset.FileDataset) -> DICOMMetadata:
        """Extract relevant metadata from DICOM dataset.
        
        Args:
            ds: PyDICOM dataset
            
        Returns:
            DICOMMetadata object containing relevant metadata
        """
        return DICOMMetadata(
            patient_id=ds.PatientID,
            study_date=ds.StudyDate,
            modality=ds.Modality,
            manufacturer=ds.Manufacturer,
            pixel_spacing=tuple(ds.PixelSpacing),
            image_shape=ds.pixel_array.shape
        )

    def load_dicom(self, file_path: Path) -> Tuple[np.ndarray, DICOMMetadata]:
        """Load a single DICOM file and return its pixel array and metadata.
        
        Args:
            file_path: Path to the DICOM file
            
        Returns:
            Tuple containing:
            - np.ndarray: The pixel array data
            - DICOMMetadata: Metadata extracted from the DICOM file
        """
        try:
            ds = pydicom.dcmread(file_path)
            
            if not self.validate_dicom(ds):
                raise ValueError(f"Invalid DICOM file: {file_path}")
            
            # Extract pixel array and convert to float32
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Extract metadata
            metadata = self.extract_metadata(ds)
            
            logger.debug(f"Successfully loaded DICOM file: {file_path}")
            return pixel_array, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {e}")
            raise

    def load_batch(
        self,
        max_files: Optional[int] = None,
        return_torch: bool = True,
        expected_shape: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], List[DICOMMetadata]]:
        """Load multiple DICOM files and return as a batch.
        
        Args:
            max_files: Maximum number of files to load (None for all files)
            return_torch: If True, return torch.Tensor instead of np.ndarray
            expected_shape: Expected shape of the DICOM images
            
        Returns:
            Tuple containing:
            - Union[np.ndarray, torch.Tensor]: Batch of images
            - List[DICOMMetadata]: List of metadata for each image
        """
        # Find all DICOM files
        dicom_files = self.find_dicom_files()
        
        if max_files is not None:
            dicom_files = dicom_files[:max_files]
        
        # Lists to store results
        images = []
        metadata_list = []
        
        # Load files with progress bar
        self.console.print(f"Loading {len(dicom_files)} DICOM files...")
        for file_path in tqdm(dicom_files, desc="Loading DICOM files"):
            try:
                image, metadata = self.load_dicom(file_path)
                
                if expected_shape and image.shape != expected_shape:
                    logger.warning(f"Skipping file {file_path} due to unexpected shape: {image.shape}")
                    continue
                
                images.append(image)
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue
        
        # Stack images into a single array
        image_batch = np.stack(images)
        
        # Convert to torch tensor if requested
        if return_torch:
            image_batch = torch.from_numpy(image_batch).to(self.device)
            logger.info(f"Converted data to torch tensor on {self.device}")
        
        logger.info(f"Successfully loaded {len(images)} DICOM files")
        return image_batch, metadata_list

print("DATSCAN Ingestion module loaded successfully!")

# Test the updated loader
if __name__ == "__main__":
    data_loader = DATSCANIngestion(
        root_dir="Images/PPMI_Images",
        use_gpu=True
    )
    
    try:
        # Load only DICOM files with shape (91, 109, 91)
        images, metadata = data_loader.load_batch(
            max_files=None,  # Load all files
            expected_shape=(91, 109, 91)
        )
        
        print(f"\nLoaded data summary:")
        print(f"Number of images: {len(metadata)}")
        print(f"Image batch shape: {images.shape}")
        print(f"Data type: {images.dtype}")
        print(f"Device: {images.device}")
        
    except Exception as e:
        print(f"Error during data loading: {e}")

import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

def analyze_data_distribution(data_loader):
    """
    Analyze the loaded data distribution and quality.
    """
    # Configure logging to show only critical errors
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Create a progress bar wrapper
    print("\nLoading DICOM files...")
    
    # Load a batch of data
    volumes, metadata = data_loader.load_batch(
        max_files=None,
        expected_shape=(91, 109, 91)
    )
    
    print(f"\nSuccessfully loaded {len(volumes)} volumes")
    
    # Convert to numpy for analysis
    if torch.is_tensor(volumes):
        volumes = volumes.cpu().numpy()
    
    # Basic statistics
    print("\nData Statistics:")
    print("-" * 50)
    print(f"Total volumes: {len(volumes)}")
    print(f"Data shape: {volumes.shape}")
    print(f"Value range: [{volumes.min():.3f}, {volumes.max():.3f}]")
    print(f"Mean value: {volumes.mean():.3f}")
    print(f"Std deviation: {volumes.std():.3f}")
    
    # Check for NaN or Inf values
    print(f"\nData Quality Checks:")
    print("-" * 50)
    print(f"Contains NaN: {np.isnan(volumes).any()}")
    print(f"Contains Inf: {np.isinf(volumes).any()}")
    
    # Plot histogram of values
    plt.figure(figsize=(10, 5))
    plt.hist(volumes.ravel(), bins=100)
    plt.title('Distribution of Voxel Values')
    plt.xlabel('Voxel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plot mean image across all volumes
    mean_volume = volumes.mean(axis=0)
    plt.figure(figsize=(15, 5))
    
    # Corrected plotting section with proper axes and labels
    plt.subplot(131)
    plt.imshow(np.flipud(mean_volume.mean(axis=0)), cmap='gray')  # Axial view
    plt.title('Mean Axial View')
    
    plt.subplot(132)
    plt.imshow(np.flipud(mean_volume.mean(axis=1)), cmap='gray')  # Coronal view
    plt.title('Mean Coronal View')
    
    plt.subplot(133)
    plt.imshow(np.flipud(mean_volume.mean(axis=2)), cmap='gray')  # Sagittal view
    plt.title('Mean Sagittal View')
    
    plt.show()

if __name__ == "__main__":
    data_loader = DATSCANIngestion(
        root_dir="Images/PPMI_Images",
        use_gpu=True
    )
    
    analyze_data_distribution(data_loader)