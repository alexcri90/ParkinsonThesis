#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base model class with common functionality for all models.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path


class BaseModel(nn.Module):
    """
    Base model class with common functionality for all models.
    
    Args:
        name: Model name for saving/loading
    """
    def __init__(self, name="base_model"):
        super(BaseModel, self).__init__()
        self.name = name
    
    def save(self, directory, filename=None):
        """
        Save model weights to file.
        
        Args:
            directory: Directory to save the model
            filename: Optional filename, defaults to model name
        
        Returns:
            Path to saved model
        """
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f"{self.name}.pt"
        
        path = os.path.join(directory, filename)
        torch.save(self.state_dict(), path)
        return path
    
    def load(self, directory, filename=None):
        """
        Load model weights from file.
        
        Args:
            directory: Directory containing the model file
            filename: Optional filename, defaults to model name
            
        Returns:
            self with loaded weights
        """
        if filename is None:
            filename = f"{self.name}.pt"
        
        path = os.path.join(directory, filename)
        
        # Load to CPU first to avoid potential CUDA memory issues
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
        
        return self
    
    def _initialize_weights(self):
        """
        Initialize model weights with He initialization.
        https://arxiv.org/abs/1502.01852
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """
        Get the device the model is on.
        
        Returns:
            Device (CPU or GPU)
        """
        return next(self.parameters()).device


def get_model_path(model_name, root_dir="trained_models"):
    """
    Get path to model weights file.
    
    Args:
        model_name: Name of the model
        root_dir: Root directory for trained models
        
    Returns:
        Path to model weights file or None if not found
    """
    model_dir = os.path.join(root_dir, model_name)
    model_file = os.path.join(model_dir, f"{model_name}.pt")
    
    if os.path.exists(model_file):
        return model_file
    
    # Try to find the best checkpoint or epoch checkpoint
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        best_checkpoint = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        if os.path.exists(best_checkpoint):
            return best_checkpoint
        
        # If best checkpoint not found, try to find the latest epoch checkpoint
        epoch_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith("_epoch.pt")]
        if epoch_checkpoints:
            # Sort by epoch number
            epoch_checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
            return os.path.join(checkpoint_dir, epoch_checkpoints[-1])
    
    return None