# dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pydicom
import logging
from tqdm import tqdm
import gc
from pathlib import Path

class DATSCANDataset(Dataset):
    """Custom Dataset class for DATSCAN images"""
    
    def __init__(self, file_paths, labels, transform=None, preload=False, cache_dir=None):
        """
        Initialize dataset with paths and labels
        
        Args:
            file_paths (list): List of file paths to DICOM images
            labels (list): List of labels corresponding to each file
            transform (callable, optional): Optional transform to apply to the data
            preload (bool): Whether to preload all data into memory
            cache_dir (str): Directory to cache processed volumes
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload
        self.cache_dir = cache_dir
        
        # Validate paths
        self._validate_paths()
        
        # Map string labels to integer indices
        self.label_map = {
            'Control': 0,
            'PD': 1,
            'SWEDD': 2
        }
        
        # Preload data if requested
        self.data_cache = {}
        if self.preload:
            self._preload_data()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _validate_paths(self):
        """Validate file existence"""
        valid_paths = []
        valid_labels = []
        
        for i, path in enumerate(self.file_paths):
            if os.path.exists(path):
                valid_paths.append(path)
                valid_labels.append(self.labels[i])
            else:
                logging.warning(f"File not found: {path}")
        
        self.file_paths = valid_paths
        self.labels = valid_labels
        
        if len(self.file_paths) == 0:
            raise ValueError("No valid files found in the dataset")
    
    def _preload_data(self):
        """Preload all data into memory"""
        logging.info("Preloading dataset...")
        
        for i in tqdm(range(len(self.file_paths)), desc="Preloading data"):
            path = self.file_paths[i]
            cache_path = None
            
            # Check if cache directory is specified
            if self.cache_dir:
                cache_path = Path(self.cache_dir) / f"{Path(path).stem}.pt"
            
            # Try to load from cache first
            if cache_path and cache_path.exists():
                try:
                    self.data_cache[i] = torch.load(cache_path)
                    continue
                except Exception as e:
                    logging.warning(f"Failed to load cached data: {e}")
            
            # Load and process DICOM
            try:
                volume = self._load_dicom(path)
                self.data_cache[i] = volume
                
                # Cache the processed volume if cache directory specified
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(volume, cache_path)
                
            except Exception as e:
                logging.error(f"Error loading file {path}: {e}")
                # Insert a placeholder tensor with zeros
                self.data_cache[i] = torch.zeros((1, 64, 128, 128), dtype=torch.float32)
        
        # Force garbage collection after preloading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_dicom(self, file_path):
        """Load and preprocess DICOM file"""
        try:
            # Load DICOM
            ds = pydicom.dcmread(file_path)
            
            # Convert to float32 array
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescaling if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            # Extract slices [9:73]
            volume = pixel_array[9:73]
            
            # Process volume
            from preprocessing import process_volume
            processed_volume = process_volume(volume, target_shape=(64, 128, 128))
            
            return processed_volume
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Handle out of bounds index
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.file_paths)}")
        
        # Get the file path and label
        file_path = self.file_paths[idx]
        label = self.label_map.get(self.labels[idx], -1)  # Default to -1 if label not found
        
        # Load volume from cache if preloaded
        if self.preload and idx in self.data_cache:
            volume = self.data_cache[idx]
        else:
            try:
                # Load and process the DICOM file
                volume = self._load_dicom(file_path)
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")
                # Return a placeholder tensor with zeros
                volume = torch.zeros((1, 64, 128, 128), dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            volume = self.transform(volume)
        
        return volume, label
    
    def get_label_distribution(self):
        """Analyze label statistics"""
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def get_sample_shape(self, idx=0):
        """Return data dimensions"""
        sample, _ = self[idx]
        return sample.shape
    
    def get_num_classes(self):
        """Return number of unique classes"""
        return len(set(self.labels))