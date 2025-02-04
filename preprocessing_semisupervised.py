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
    
    def __call__(self, img):
        """
        Ensure consistent output shape by explicitly reshaping
        """
        processed = super().__call__(img)
        target_shape = (128, 128, 128)  # Fixed shape for all images
        if processed.shape != target_shape:
            # Resize if shape doesn't match
            processed = resize(processed, target_shape, anti_aliasing=True, preserve_range=True)
        return processed
    
def normalize_intensity(self, img: np.ndarray) -> np.ndarray:
    img = np.maximum(img, 0)
    if self.normalize_method == 'minmax':
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min())
            # Scale to [-0.5, 0.5] range instead of [0, 1]
            img = img - 0.5
    return img

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
        
        # Ensure img has the correct shape before creating tensor
        if img.shape != (128, 128, 128):
            img = resize(img, (128, 128, 128), anti_aliasing=True, preserve_range=True)
        
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dimension
        
        # Verify tensor shape
        assert img_tensor.shape == (1, 128, 128, 128), f"Unexpected tensor shape: {img_tensor.shape}"
        
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
    Combines multiple dataloaders into a single loader for semi-supervised learning.
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        
        # Create concatenated dataset
        self.dataset = torch.utils.data.ConcatDataset([
            dl.dataset for dl in dataloaders.values()
        ])
        
        # Get metadata dimensions from first dataloader
        first_dataset = next(iter(dataloaders.values())).dataset
        self.metadata_dims = first_dataset.num_classes
        
        # Initialize iterators and track lengths
        self.iterators = {group: iter(loader) for group, loader in dataloaders.items()}
        self.lengths = {group: len(loader) for group, loader in dataloaders.items()}
        self.total_length = sum(self.lengths.values())
        
        # Create the combined loader
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=next(iter(dataloaders.values())).batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=self.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    @staticmethod
    def collate_fn(batch):
        """
        Improved collate function with shape verification
        """
        images = []
        metadata = []
        
        for item in batch:
            img, meta = item
            # Ensure consistent shape
            if img.shape != (1, 128, 128, 128):
                raise ValueError(f"Inconsistent tensor shape encountered: {img.shape}")
            images.append(img)
            metadata.append(meta)
        
        # Stack images and combine metadata
        images = torch.stack(images, dim=0)
        combined_metadata = {}
        for key in metadata[0].keys():
            combined_metadata[key] = torch.stack([d[key] for d in metadata])
        
        return images, combined_metadata