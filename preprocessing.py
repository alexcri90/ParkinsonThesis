# preprocessing.py
import torch
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, ball

def resize_volume(volume, target_shape=(64, 128, 128)):
    """
    Resizes the volume to the target shape using zero-padding or center cropping.
    """
    def get_pad_amounts(current_size, target_size):
        if current_size >= target_size:
            return 0, 0
        diff = target_size - current_size
        pad_before = diff // 2
        pad_after = diff - pad_before
        return pad_before, pad_after

    # Convert to numpy if needed
    is_tensor = isinstance(volume, torch.Tensor)
    if is_tensor:
        device = volume.device
        volume = volume.cpu().numpy()

    current_shape = volume.shape
    resized = volume.copy()

    # Calculate padding/cropping for each dimension
    pads = [get_pad_amounts(current_shape[i], target_shape[i]) for i in range(3)]

    # Apply padding if needed
    if any(sum(p) > 0 for p in pads):
        resized = np.pad(
            resized,
            pad_width=pads,
            mode="constant",
            constant_values=0
        )

    # Apply cropping if needed
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            start = (current_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            if i == 0:
                resized = resized[start:end, :, :]
            elif i == 1:
                resized = resized[:, start:end, :]
            else:
                resized = resized[:, :, start:end]

    # Convert back to tensor if input was tensor
    if is_tensor:
        resized = torch.from_numpy(resized).to(device)

    return resized

def normalize_intensity(volume):
    """
    Normalize intensity values:
    1. Set minimum to 0
    2. Scale to [0,1] range
    3. Normalize based on anatomical region
    """
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    
    # Ensure float32
    volume = volume.float()
    
    # Basic normalization
    volume = volume - volume.min()
    if volume.max() > 0:
        volume = volume / volume.max()
    
    # Create anatomical mask for normalization
    mask = torch.zeros_like(volume)
    mask[20:40, 82:103, 43:82] = 1
    
    # Normalize based on anatomical region mean
    mask_mean = volume[mask.bool()].mean()
    if mask_mean > 0:
        volume = volume / mask_mean
    
    return volume

def generate_brain_mask(volume):
    """
    Generate brain mask using Otsu thresholding and morphological operations.
    """
    # Convert to numpy for scikit-image operations
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    # Apply Otsu thresholding
    thresh = threshold_otsu(volume)
    mask = volume > thresh
    
    # Apply morphological closing to fill holes
    mask = binary_closing(mask, footprint=ball(2))
    
    # Convert back to tensor if input was tensor
    if isinstance(volume, torch.Tensor):
        mask = torch.from_numpy(mask)
    
    return mask

def process_volume(volume, target_shape=(64, 128, 128)):
    """
    Complete preprocessing pipeline:
    1. Resize to target shape
    2. Normalize intensities
    3. Generate and apply brain mask
    4. Add channel dimension
    """
    # Ensure we're working with a tensor
    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume)
    
    # Get device
    device = volume.device if isinstance(volume, torch.Tensor) else None
    
    # Resize
    volume = resize_volume(volume, target_shape)
    
    # Normalize intensities
    volume = normalize_intensity(volume)
    
    # Generate and apply brain mask
    mask = generate_brain_mask(volume)
    masked_volume = volume * mask
    
    # Add channel dimension if not present
    if masked_volume.dim() == 3:
        masked_volume = masked_volume.unsqueeze(0)
    
    # Move to correct device
    if device is not None:
        masked_volume = masked_volume.to(device)
    
    return masked_volume

def validate_processing(volume):
    """
    Validate the processing of a volume:
    - Check shape
    - Check value range
    - Check for NaN/Inf values
    """
    validation = {
        'shape': volume.shape,
        'min_val': float(volume.min()),
        'max_val': float(volume.max()),
        'mean_val': float(volume.mean()),
        'has_nan': torch.isnan(volume).any().item(),
        'has_inf': torch.isinf(volume).any().item()
    }
    
    return validation