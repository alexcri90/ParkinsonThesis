# dataset_utils.py
import os
import gc
import logging
import torch
import numpy as np
import pandas as pd
import psutil
import pydicom
from pathlib import Path
from tqdm.notebook import tqdm  # <-- Added import

import torch.nn.functional as F

# Set up a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_dicom(file_path, device):
    """
    Loads and processes a DICOM file with GPU acceleration.
    """
    try:
        logger.info(f"Loading DICOM file: {file_path}")
        ds = pydicom.dcmread(file_path)
        if not hasattr(ds, 'pixel_array'):
            logger.warning(f"No pixel array in {file_path}")
            return None, None
        pixel_array = ds.pixel_array
        logger.info(f"Initial pixel array shape: {pixel_array.shape}")
        # Convert to float32
        pixel_array = pixel_array.astype(np.float32)
        # Apply rescaling if attributes are present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
            logger.info("Applied rescale slope and intercept")
        # Convert to PyTorch tensor and move to device
        tensor = torch.from_numpy(pixel_array).to(device)
        logger.info(f"Final tensor shape: {tensor.shape}")
        return tensor, ds
    except Exception as e:
        logger.error(f"Error reading DICOM file {file_path}: {str(e)}")
        return None, None

def process_volume(volume, target_shape=(64, 128, 128), device=None):
    """
    Process a 3D volume tensor:
      1. Extract axial slices [9:73]
      2. Adjust in-plane dimensions to target_shape[1:].
      3. Normalize intensities using a fixed mask.
    """
    try:
        if device is None:
            device = volume.device
        logger.info(f"Initial volume shape: {volume.shape}")
        if volume.ndim != 3:
            logger.error(f"Expected 3D volume, got shape {volume.shape}")
            return None
        if volume.shape[0] < 73:
            logger.error(f"Volume has insufficient slices: {volume.shape[0]} (need at least 73)")
            return None
        # Extract axial slices [9:73]
        volume = volume[9:73, :, :]
        logger.info(f"After slicing shape: {volume.shape}")
        current_h, current_w = volume.shape[1:]
        pad_h = max(0, (target_shape[1] - current_h))
        pad_w = max(0, (target_shape[2] - current_w))
        crop_h = max(0, (current_h - target_shape[1]))
        crop_w = max(0, (current_w - target_shape[2]))
        if pad_h > 0 or pad_w > 0:
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            padding = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half, 0, 0)
            volume = F.pad(volume, padding)
            logger.info(f"After padding shape: {volume.shape}")
        if crop_h > 0 or crop_w > 0:
            start_h = crop_h // 2
            start_w = crop_w // 2
            volume = volume[:, start_h:start_h + target_shape[1], start_w:start_w + target_shape[2]]
            logger.info(f"After cropping shape: {volume.shape}")
        volume = volume - volume.min()
        if volume.max() > 0:
            volume = volume / volume.max()
        # Create an example anatomical mask (adjust these indices as needed)
        mask = torch.zeros((target_shape[0], target_shape[1], target_shape[2]), dtype=torch.bool, device=device)
        mask[20:40,82:103,43:82] = True
        masked_vol = volume * mask.float()
        mask_mean = volume[mask].mean()
        if mask_mean > 0:
            volume = volume / mask_mean
        logger.info(f"Final volume shape: {volume.shape}")
        return volume
    except Exception as e:
        logger.error(f"Error processing volume: {str(e)}")
        return None

class LPRamDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset for DaTSCAN images with robust error handling.
    Loads and processes volumes on-demand.
    """
    def __init__(self, dataframe, device=None, cache_dir=None, retry_attempts=3):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.retry_attempts = retry_attempts
        logger.info(f"Initializing dataset with device: {self.device}")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory set to: {self.cache_dir}")
        self.valid_files = []
        self.failed_files = []
        self.total_files = len(dataframe)
        print(f"Validating {self.total_files} files...")
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            try:
                volume_tensor, _ = self._load_with_retry(row["file_path"])
                if volume_tensor is not None and volume_tensor.shape[0] >= 73:
                    self.valid_files.append(row)
                else:
                    self.failed_files.append((row["file_path"], "Invalid volume shape"))
                del volume_tensor
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"Error validating file {row['file_path']}: {str(e)}")
                self.failed_files.append((row["file_path"], str(e)))
                continue
        self.df = pd.DataFrame(self.valid_files)
        logger.info(f"\nDataset initialization complete:")
        logger.info(f"Total files: {self.total_files}")
        logger.info(f"Valid files: {len(self.valid_files)}")
        logger.info(f"Failed files: {len(self.failed_files)}")
        self._save_failed_files_log()
        if len(self.valid_files) == 0:
            raise RuntimeError("No valid volumes were found! Check the logs for details.")
        self._print_memory_stats()

    def _load_with_retry(self, file_path):
        for attempt in range(self.retry_attempts):
            try:
                if self.cache_dir:
                    cached_path = self.cache_dir / f"{Path(file_path).stem}.pt"
                    if cached_path.exists():
                        return torch.load(cached_path), None
                volume_tensor, metadata = load_dicom(file_path, 'cpu')
                if self.cache_dir and volume_tensor is not None:
                    torch.save(volume_tensor, cached_path)
                return volume_tensor, metadata
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry_attempts} failed for {file_path}: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    logger.error(f"All attempts failed for {file_path}")
                    return None, None
                torch.cuda.empty_cache()
                gc.collect()

    def _save_failed_files_log(self):
        with open(Path('logs') / 'failed_files.log', 'w') as f:
            for file_path, error in self.failed_files:
                f.write(f"{file_path}\t{error}\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            file_path = self.df.iloc[idx]["file_path"]
            volume_tensor, metadata = self._load_with_retry(file_path)
            if volume_tensor is None:
                raise RuntimeError(f"Failed to load volume after all retries: {file_path}")
            processed_vol = process_volume(volume_tensor, target_shape=(64, 128, 128), device='cpu')
            if processed_vol is None:
                raise RuntimeError(f"Failed to process volume: {file_path}")
            # Add channel dimension; DO NOT move to GPU here.
            processed_vol = processed_vol.unsqueeze(0)
            torch.cuda.empty_cache()
            return {
                "volume": processed_vol,  # Remains on CPU
                "label": self.df.iloc[idx]["label"],
                "path": file_path
            }
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            return {
                "volume": torch.zeros((1, 64, 128, 128)),  # Remains on CPU
                "label": self.df.iloc[idx]["label"],
                "path": file_path,
                "is_valid": False
            }

    def _print_memory_stats(self):
        if torch.cuda.is_available():
            print("\nGPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"CPU Memory Usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    valid_batch = [b for b in batch if b.get("is_valid", True)]
    if not valid_batch:
        return None
    # Stack the volumes; do not call .cuda() here
    volumes = torch.stack([b["volume"] for b in valid_batch])
    labels = [b["label"] for b in valid_batch]
    paths = [b["path"] for b in valid_batch]
    return {
        "volume": volumes,
        "label": labels,
        "path": paths
    }
