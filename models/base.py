# models/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
from pathlib import Path
import logging

class BaseModel(nn.Module, ABC):
    """
    Base model class for all models in the project.
    Provides common functionality for model management, saving, and loading.
    """
    
    def __init__(self, name="base_model"):
        super(BaseModel, self).__init__()
        self.name = name
        self.logger = logging.getLogger(f"model.{name}")
    
    @abstractmethod
    def forward(self, x):
        """Forward pass (to be implemented by subclasses)"""
        pass
    
    def save(self, save_dir, epoch, optimizer=None, scheduler=None, losses=None, best=False):
        """
        Save model checkpoint
        
        Args:
            save_dir: Directory to save the model
            epoch: Current epoch number
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            losses: Dictionary of losses to save
            best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare filename
        filename = f"{self.name}_{'best' if best else 'epoch'}.pt"
        save_path = save_dir / filename
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_name': self.name
        }
        
        # Add optional items
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if losses is not None:
            checkpoint['losses'] = losses
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")
        
        return save_path
    
    def load(self, checkpoint_path, optimizer=None, scheduler=None):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
        
        Returns:
            Dictionary with loaded checkpoint info
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
    
    def find_latest_checkpoint(self, checkpoint_dir):
        """
        Find the latest checkpoint for this model in the given directory
        
        Args:
            checkpoint_dir: Directory to search for checkpoints
        
        Returns:
            Path to latest checkpoint or None if not found
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return None
        
        # Look for best model first
        best_path = checkpoint_dir / f"{self.name}_best.pt"
        if best_path.exists():
            return best_path
        
        # Look for epoch checkpoint
        epoch_path = checkpoint_dir / f"{self.name}_epoch.pt"
        if epoch_path.exists():
            return epoch_path
        
        return None
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _initialize_weights(self):
        """Initialize model weights (to be used by subclasses)"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)