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
from preprocessing import DATSCANPreprocessor  # Import from original preprocessing

class EnhancedDATSCANPreprocessor(DATSCANPreprocessor):
    """
    Enhanced preprocessor that inherits from the original one
    to maintain consistency while allowing for extensions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def load_dicom_metadata(file_path):
    """
    Load DICOM metadata without loading the full image
    """
    try:
        ds = dcmread(file_path, stop_before_pixels=True)
        return ds
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return None

class EnhancedDATSCANDataset(Dataset):
    """
    Enhanced Dataset class that includes metadata handling
    """
    def __init__(self, 
                file_paths: list,
                metadata_df: pd.DataFrame,
                preprocessor: EnhancedDATSCANPreprocessor,
                metadata_fields: list = ['group', 'PatientSex', 'StudyDescription', 
                                        'Manufacturer', 'ManufacturerModelName']):
        self.file_paths = file_paths
        self.metadata_df = metadata_df
        self.preprocessor = preprocessor
        self.metadata_fields = metadata_fields
        
        # Create encoders for categorical variables
        self.encoders = {}
        self.num_classes = {}
        for field in metadata_fields:
            # Get all values including NaN
            values = self.metadata_df[field].unique()
            # Add 'UNKNOWN' to valid categories
            valid_values = np.append(values[pd.notna(values)], 'UNKNOWN')
            # Create encoder dictionary
            self.encoders[field] = {val: i for i, val in enumerate(valid_values)}
            self.num_classes[field] = len(self.encoders[field])
            print(f"Field {field} has {self.num_classes[field]} unique values (including UNKNOWN)")
    
    def __len__(self):
        return len(self.file_paths)
    
    def encode_metadata(self, metadata_dict):
        """
        Encode metadata values to integers
        """
        encoded = {}
        for field in self.metadata_fields:
            value = metadata_dict.get(field)
            # Handle missing or NaN values
            if pd.isna(value) or value not in self.encoders[field]:
                value = 'UNKNOWN'
            encoded[field] = torch.tensor(self.encoders[field][value])
        return encoded
    
    def __getitem__(self, idx):
        # Load and preprocess image
        file_path = self.file_paths[idx]
        img = dcmread(file_path).pixel_array.astype(np.float32)
        img = self.preprocessor(img)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Get metadata
        metadata_dict = {}
        for field in self.metadata_fields:
            try:
                value = self.metadata_df.loc[self.metadata_df['file_path'] == file_path, field].iloc[0]
                metadata_dict[field] = value
            except:
                metadata_dict[field] = None
        
        metadata_encoded = self.encode_metadata(metadata_dict)
        
        return img_tensor, metadata_encoded

def create_enhanced_dataloaders(df: pd.DataFrame,
                              metadata_df: pd.DataFrame,
                              batch_size: int = 32,
                              target_shape: Tuple[int, int, int] = (128, 128, 128),
                              normalize_method: str = 'minmax',
                              apply_brain_mask: bool = True,
                              augment: bool = False,
                              device: torch.device = torch.device('cpu'),
                              num_workers: int = 4,
                              metadata_fields: list = ['PatientSex', 'StudyDescription', 
                                                     'Manufacturer', 'ManufacturerModelName']
                              ) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders with metadata handling
    """
    preprocessor = EnhancedDATSCANPreprocessor(
        target_shape=target_shape,
        normalize_method=normalize_method,
        apply_brain_mask=apply_brain_mask,
        augment=augment
    )
    
    dataloaders = {}
    for group in ['PD', 'SWEDD', 'Control']:
        group_files = df[df['label'] == group]['file_path'].tolist()
        dataset = EnhancedDATSCANDataset(
            file_paths=group_files,
            metadata_df=metadata_df,
            preprocessor=preprocessor,
            metadata_fields=metadata_fields
        )
        
        dataloaders[group] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    # Store number of classes for each metadata field
    metadata_dims = dataset.num_classes
    
    return dataloaders, metadata_dims

class CombinedEnhancedDataloader:
    """
    Utility class to combine multiple dataloaders
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.dataset = torch.utils.data.ConcatDataset([
            dl.dataset for dl in dataloaders.values()
        ])
        
        # Get metadata dimensions from the first dataloader
        first_dataset = next(iter(dataloaders.values())).dataset
        self.metadata_dims = first_dataset.num_classes
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=next(iter(dataloaders.values())).batch_size,
            shuffle=True,
            num_workers=next(iter(dataloaders.values())).num_workers,
            pin_memory=next(iter(dataloaders.values())).pin_memory
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)