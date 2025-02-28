# dataset.py
import os
import torch
import numpy as np
import pydicom
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gc

class MemoryEfficientDATSCANDataset(Dataset):
    """Dataset for DATSCAN images that loads data on-demand rather than preloading everything."""
    
    def __init__(self, dataframe, transform=None, cache_dir=None, preprocess_fn=None):
        self.df = dataframe
        self.transform = transform
        self.cache_dir = cache_dir
        self.preprocess_fn = preprocess_fn
        
        # Create cache directory if specified
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]["file_path"]
        label = self.df.iloc[idx]["label"]
        
        # Check if processed volume exists in cache
        cache_path = None
        if self.cache_dir:
            # Create a unique identifier based on file path
            file_id = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
            cache_path = os.path.join(self.cache_dir, f"{file_id}.npy")
            
            if os.path.exists(cache_path):
                # Load from cache
                try:
                    volume = np.load(cache_path)
                except Exception as e:
                    print(f"Error loading from cache: {e}, reprocessing file {file_path}")
                    volume = self._load_and_process(file_path)
                    np.save(cache_path, volume)
            else:
                # Process and cache
                volume = self._load_and_process(file_path)
                np.save(cache_path, volume)
        else:
            # Process without caching
            volume = self._load_and_process(file_path)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            volume_tensor = self.transform(volume_tensor)
        
        return {
            "volume": volume_tensor,
            "label": label,
            "path": file_path
        }
    
    def _load_and_process(self, file_path):
        """Load DICOM file and apply preprocessing."""
        try:
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescaling if attributes are present
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = ds.RescaleSlope
                intercept = ds.RescaleIntercept
                pixel_array = pixel_array * slope + intercept
            
            # Extract slice range [9:73]
            volume = pixel_array[9:73, :, :]
            
            # Apply custom preprocessing if provided
            if self.preprocess_fn:
                volume = self.preprocess_fn(volume)
            else:
                # Default preprocessing: normalize and resize
                volume = self._default_preprocess(volume)
            
            return volume
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Return a zero volume of appropriate shape as fallback
            return np.zeros((64, 128, 128), dtype=np.float32)
    
    @staticmethod
    def _default_preprocess(volume, target_shape=(64, 128, 128)):
        """Default preprocessing function."""
        from skimage.transform import resize
        
        # Resize volume to target shape if needed
        if volume.shape != target_shape:
            volume_resized = resize(volume, target_shape, mode='constant', anti_aliasing=True, preserve_range=True)
        else:
            volume_resized = volume.copy()
        
        # Normalize
        volume_norm = volume_resized - volume_resized.min()
        
        # Create striatal mask
        striatal_mask = np.zeros(target_shape, dtype=bool)
        striatal_mask[20:40, 82:103, 43:82] = 1
        
        # Normalize based on mean intensity in striatal region (if non-zero)
        mean_intensity = np.mean(volume_norm[striatal_mask])
        if mean_intensity > 0:
            volume_norm /= mean_intensity
        
        return volume_norm.astype(np.float32)

def create_efficient_dataloaders(df, batch_size=8, train_ratio=0.8, num_workers=4, 
                      cache_dir=None, preprocess_fn=None, pin_memory=True):
    """Create train and validation dataloaders with stratified split."""
    
    # Stratified split
    train_df, val_df = train_test_split(
        df, 
        test_size=1-train_ratio,
        stratify=df['label'],
        random_state=42
    )
    
    print("Training set distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation set distribution:")
    print(val_df['label'].value_counts())
    
    # Create datasets
    train_dataset = MemoryEfficientDATSCANDataset(
        train_df, 
        cache_dir=cache_dir, 
        preprocess_fn=preprocess_fn
    )
    
    val_dataset = MemoryEfficientDATSCANDataset(
        val_df, 
        cache_dir=cache_dir, 
        preprocess_fn=preprocess_fn
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader