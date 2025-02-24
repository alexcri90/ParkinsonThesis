# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from pathlib import Path
import gc
import pydicom
from preprocessing import process_volume

class DATSCANDataset(Dataset):
    """Custom Dataset for DATSCAN images."""
    
    def __init__(self, file_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            file_paths (list): List of paths to DICOM files
            labels (list): List of labels corresponding to the files
            transform (callable, optional): Optional transform to be applied
        """
        self.file_paths = file_paths
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in labels]
        
        self.logger.info(f"Label mapping: {self.label_to_idx}")
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that all file paths exist."""
        invalid_paths = []
        for path in self.file_paths:
            if not Path(path).exists():
                invalid_paths.append(path)
        
        if invalid_paths:
            self.logger.warning(f"Found {len(invalid_paths)} invalid paths")
            valid_indices = [i for i, path in enumerate(self.file_paths) 
                           if Path(path).exists()]
            self.file_paths = [self.file_paths[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
    
    def __len__(self):
        return len(self.file_paths)
    
    def _load_dicom(self, file_path):
        """Load a DICOM file."""
        try:
            ds = pydicom.dcmread(file_path)
            if not hasattr(ds, 'pixel_array'):
                return None
            
            pixel_array = ds.pixel_array.astype(np.float32)
            
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            return torch.from_numpy(pixel_array)
            
        except Exception as e:
            self.logger.error(f"Error reading DICOM file {file_path}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        try:
            # Load DICOM file
            volume = self._load_dicom(self.file_paths[idx])
            
            if volume is None:
                raise ValueError(f"Failed to load DICOM file: {self.file_paths[idx]}")
            
            # Extract required slices [9:73]
            volume = volume[9:73]
            
            # Process volume
            processed_vol = process_volume(volume)
            
            # Apply any additional transforms
            if self.transform is not None:
                processed_vol = self.transform(processed_vol)
            
            # Get label (already converted to numerical index)
            label = self.labels[idx]
            
            # Convert to tensor with proper type
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return processed_vol.float(), label_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a default sample with the correct types
            return (torch.zeros((1, 64, 128, 128), dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long))
    
    def get_label_distribution(self):
        """Get the distribution of labels in the dataset."""
        from collections import Counter
        label_counts = Counter(self.labels)
        # Convert numerical indices back to original labels
        idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        return {idx_to_label[idx]: count for idx, count in label_counts.items()}
    
    def get_sample_shape(self):
        """Get the shape of a sample from the dataset."""
        sample, _ = self[0]
        return tuple(sample.shape)
    
    def get_num_classes(self):
        """Get the number of unique classes."""
        return len(self.label_to_idx)

# Test the dataset
if __name__ == "__main__":
    import pandas as pd
    
    # Load file paths
    df = pd.read_csv("validated_file_paths.csv")
    
    # Create dataset
    dataset = DATSCANDataset(df['file_path'].tolist(), df['label'].tolist())
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    print(f"Sample shape: {dataset.get_sample_shape()}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        sample, label = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Shape: {sample.shape}")
        print(f"  Label: {label}")
        print(f"  Value range: [{sample.min():.2f}, {sample.max():.2f}]")
        print(f"  Data type: {sample.dtype}")
        print(f"  Label type: {label.dtype}")