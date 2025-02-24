# trainers/base_trainer.py
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import gc

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

class BaseTrainer(ABC):
    """Base trainer class for all models"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, 
                 device, config):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device to use (cuda or cpu)
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Setup directories
        self.output_dir = Path(config.get('output_dir', 'trained_models'))
        self.model_dir = self.output_dir / model.name
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.log_dir = self.model_dir / 'logs'
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"trainer.{model.name}")
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.early_stop_counter = 0
        self.early_stop_patience = config.get('early_stop_patience', 7)
        
        # Setup scheduler if provided
        self.scheduler = None
        if 'scheduler' in config:
            scheduler_type = config['scheduler'].get('type', 'reduce_lr')
            if scheduler_type == 'reduce_lr':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=config['scheduler'].get('factor', 0.5), 
                    patience=config['scheduler'].get('patience', 5),
                    verbose=True
                )
        
        # Save config
        self.save_config()
        
        # Try to load latest checkpoint
        self.load_checkpoint()
        
        # Additional setup
        self.setup()
    
    def setup(self):
        """Additional setup (to be implemented by subclasses)"""
        pass
    
    def save_config(self):
        """Save configuration to file"""
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def load_checkpoint(self):
        """Load the latest checkpoint if available"""
        # Find latest checkpoint
        checkpoint_path = self.model.find_latest_checkpoint(self.checkpoint_dir)
        
        if checkpoint_path is None:
            self.logger.info("No checkpoint found, starting from scratch")
            return False
        
        # Load checkpoint
        checkpoint = self.model.load(checkpoint_path, self.optimizer, self.scheduler)
        
        # Restore training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        self.current_epoch = self.start_epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore losses if available
        if 'losses' in checkpoint:
            self.train_losses = checkpoint['losses'].get('train_losses', [])
            self.val_losses = checkpoint['losses'].get('val_losses', [])
        
        self.logger.info(f"Resumed from checkpoint at epoch {self.start_epoch}")
        return True
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        # Prepare losses for saving
        losses = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save model
        checkpoint_path = self.model.save(
            self.checkpoint_dir, 
            epoch, 
            self.optimizer, 
            self.scheduler, 
            losses,
            best=is_best
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    @abstractmethod
    def train_epoch(self, epoch):
        """Train for one epoch (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def validate(self):
        """Validate the model (to be implemented by subclasses)"""
        pass
    
    def train(self, max_epochs):
        """Train the model for a specified number of epochs"""
        self.logger.info(f"Starting training from epoch {self.start_epoch} to {max_epochs}")
        self.logger.info(f"Training on device: {self.device}")
        
        # Track start time
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, max_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                train_loss = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Print progress
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.8f}")
                
                # Check early stopping
                if self.early_stop_counter >= self.early_stop_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Training finished
            total_time = time.time() - start_time
            self.logger.info(f"Training finished after {time.time() - start_time:.2f} seconds")
            
            # Plot and save learning curves
            self.plot_learning_curves()
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'total_time': total_time
            }
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
            self.save_checkpoint(self.current_epoch)
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'interrupted': True
            }
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise
    
    def plot_learning_curves(self):
        """Plot and save learning curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{self.model.name} Learning Curves')
        plt.grid(True)
        
        # Save figure
        plt.savefig(self.model_dir / 'learning_curves.png')
        plt.close()
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()