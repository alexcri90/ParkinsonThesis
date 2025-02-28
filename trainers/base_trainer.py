#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base trainer with common training logic for all models.
"""

import os
import gc
import json
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda.amp as amp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler with patience"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


class BaseTrainer:
    """
    Base trainer with common training logic for all models.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object
        criterion: Loss function (default: MSELoss)
        optimizer: Optimizer (default: Adam)
        device: Device to train on (default: cuda if available, else cpu)
    """
    def __init__(self, model, train_loader, val_loader, config, 
                 criterion=None, optimizer=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set criterion
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Set scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=self.config.training.scheduler_factor,
            patience=self.config.training.scheduler_patience, 
            verbose=True
        )
        
        # Set early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience
        )
        
        # Set mixed precision scaler
        self.scaler = amp.GradScaler(enabled=self.config.training.use_mixed_precision)
        
        # Set output directory
        self.output_dir = os.path.join(
            self.config.general.output_dir,
            self.model.name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set checkpoint directory
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set log directory
        self.log_dir = os.path.join(self.output_dir, self.config.general.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Load checkpoint if available
        self._load_checkpoint()
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model: {self.model.name}")
        logger.info(f"Trainable parameters: {self.model.count_parameters():,}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Create config file
        self._save_config()
        
        start_time = time.time()
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_loss = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                # Evaluate on validation set
                val_loss = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(is_best=True)
                
                # Regular checkpoint
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint()
                
                # Plot progress
                if (epoch + 1) % 5 == 0:
                    self._plot_progress()
                
                # Check early stopping
                if self.early_stopping(val_loss):
                    logger.info("Early stopping triggered!")
                    break
                
                # Clean up
                gc.collect()
                torch.cuda.empty_cache()
            
            # Final checkpoint and plot
            self._save_checkpoint()
            self._plot_progress()
            
            # Final model save
            self.model.save(self.output_dir)
            
            logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
            
            # Return training history
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'total_epochs': self.current_epoch + 1
            }
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user!")
            logger.info("Saving checkpoint...")
            self._save_checkpoint()
            self._plot_progress()
            
            # Return training history so far
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'total_epochs': self.current_epoch + 1,
                'interrupted': True
            }
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        train_loss = AverageMeter()
        
        # Progress bar
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs} [Train]")
        
        # Zero gradients at epoch start
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            inputs = batch['volume'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with amp.autocast(enabled=self.config.training.use_mixed_precision):
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, inputs)
                # Scale loss by accumulation steps
                loss = loss / self.config.training.accumulation_steps
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update running loss
            train_loss.update(loss.item() * self.config.training.accumulation_steps)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{train_loss.avg:.6f}"})
            pbar.update()
            
            # Clean up
            del inputs, outputs, loss
        
        pbar.close()
        
        # Log training loss
        logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - Train Loss: {train_loss.avg:.6f}")
        
        return train_loss.avg
    
    def validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        val_loss = AverageMeter()
        
        # Progress bar
        pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data
                inputs = batch['volume'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, inputs)
                
                # Update running loss
                val_loss.update(loss.item())
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{val_loss.avg:.6f}"})
                pbar.update()
                
                # Clean up
                del inputs, outputs, loss
        
        pbar.close()
        
        # Log validation loss
        logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs} - Val Loss: {val_loss.avg:.6f}")
        
        return val_loss.avg
    
    def _save_checkpoint(self, is_best=False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.as_dict() if hasattr(self.config, 'as_dict') else self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model.name}_{self.current_epoch}_epoch.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model.name}_latest.pt"
        )
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint (if applicable)
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, 
                f"{self.model.name}_best.pt"
            )
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    def _load_checkpoint(self):
        """
        Load latest checkpoint if available.
        """
        latest_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model.name}_latest.pt"
        )
        
        if os.path.exists(latest_path):
            logger.info(f"Loading checkpoint: {latest_path}")
            
            # Load to CPU first to avoid CUDA memory issues
            checkpoint = torch.load(latest_path, map_location='cpu')
            
            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state dict
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state dict if available
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            logger.info(f"Resuming from epoch {self.current_epoch}")
        else:
            logger.info("No checkpoint found, starting training from scratch")
    
    def _save_config(self):
        """
        Save configuration to JSON file.
        """
        config_path = os.path.join(self.output_dir, "config.json")
        
        with open(config_path, 'w') as f:
            if hasattr(self.config, 'as_dict'):
                json.dump(self.config.as_dict(), f, indent=4)
            else:
                json.dump(self.config, f, indent=4)
    
    def _plot_progress(self):
        """
        Plot training and validation loss curves.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot all epochs
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot recent epochs
        if len(self.train_losses) > 10:
            plt.subplot(1, 2, 2)
            recent = 20  # Show last 20 epochs
            start = max(0, len(self.train_losses) - recent)
            plt.plot(
                range(start + 1, len(self.train_losses) + 1), 
                self.train_losses[start:], 
                label='Train Loss'
            )
            plt.plot(
                range(start + 1, len(self.val_losses) + 1), 
                self.val_losses[start:], 
                label='Val Loss'
            )
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Last {recent} Epochs')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(plot_path)
        plt.close()