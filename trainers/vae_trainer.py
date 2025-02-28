# trainers/vae_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from trainers.base_trainer import BaseTrainer
from trainers.losses import VAELoss

class VAETrainer(BaseTrainer):
    """Trainer for Variational Autoencoder models"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        """
        Initialize the VAE trainer
        
        Args:
            model: The VAE model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            device: Device to use for training
            config: Configuration dictionary
        """
        super(VAETrainer, self).__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            config=config
        )
        
        # Create loss function
        self.kl_weight = config['model'].get('kl_weight', 0.01)
        self.criterion = VAELoss(
            reconstruction_type=config.get('reconstruction_loss', 'combined'),
            kl_weight=self.kl_weight
        )
        
        # KL annealing parameters
        self.use_kl_annealing = config.get('use_kl_annealing', True)
        self.kl_anneal_cycles = config.get('kl_anneal_cycles', 1)
        self.kl_anneal_start = config.get('kl_anneal_start', 0.0)
        self.kl_anneal_ratio = config.get('kl_anneal_ratio', 0.5)  # Fraction of epochs for annealing
        
        # Initialize tracking variables
        self.recon_losses = []
        self.kl_losses = []
        self.current_kl_weight = self.kl_anneal_start if self.use_kl_annealing else self.kl_weight
        
        # Set up a scheduler
        self.scheduler = self._initialize_scheduler()
        
        # Set up logging
        self.logger = logging.getLogger(f"trainer.{model.name}")
    
    def _train_epoch(self, epoch):
        """Train the model for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        batch_count = 0
        
        # Update KL weight if using annealing
        if self.use_kl_annealing:
            self.current_kl_weight = self._get_kl_weight(epoch)
            self.criterion.kl_weight = self.current_kl_weight
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = self.criterion(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config['training'], 'grad_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'kl_w': self.current_kl_weight
            })
            
            # Logging interval
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f"Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, "
                    f"KL: {kl_loss.item():.6f}, KL weight: {self.current_kl_weight:.6f}"
                )
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
        
        # Calculate epoch average losses
        epoch_loss /= batch_count
        epoch_recon_loss /= batch_count
        epoch_kl_loss /= batch_count
        
        # Store for tracking
        self.train_losses.append(epoch_loss)
        self.recon_losses.append(epoch_recon_loss)
        self.kl_losses.append(epoch_kl_loss)
        
        return epoch_loss
    
    def _validate_epoch(self, epoch):
        """Validate the model after an epoch"""
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                
                # Forward pass
                recon_batch, mu, logvar = self.model(data)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.criterion(recon_batch, data, mu, logvar)
                
                # Track losses
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                batch_count += 1
        
        # Calculate epoch average losses
        val_loss /= batch_count
        val_recon_loss /= batch_count
        val_kl_loss /= batch_count
        
        # Store for tracking
        self.val_losses.append(val_loss)
        
        # Log validation results
        self.logger.info(
            f"Validation after Epoch {epoch}: "
            f"Loss: {val_loss:.6f}, Recon: {val_recon_loss:.6f}, "
            f"KL: {val_kl_loss:.6f}"
        )
        
        return val_loss
    
    def _get_kl_weight(self, epoch):
        """Calculate KL weight for the current epoch using cyclical annealing"""
        if not self.use_kl_annealing:
            return self.kl_weight
        
        # Calculate the cycle progress
        cycle_length = self.num_epochs / self.kl_anneal_cycles
        cycle_idx = int(epoch / cycle_length)
        cycle_progress = (epoch - cycle_idx * cycle_length) / cycle_length
        
        # Annealing within each cycle
        if cycle_progress < self.kl_anneal_ratio:
            # Linear annealing from start to kl_weight
            progress = cycle_progress / self.kl_anneal_ratio
            return self.kl_anneal_start + progress * (self.kl_weight - self.kl_anneal_start)
        else:
            # Constant kl_weight for the rest of the cycle
            return self.kl_weight
    
    def visualize_reconstructions(self, data_loader, num_samples=4):
        """
        Generate and save reconstruction visualizations from a data loader
        
        Args:
            data_loader: DataLoader to get samples from
            num_samples: Number of samples to visualize
        
        Returns:
            Path to saved figure
        """
        self.model.eval()
        
        # Get some examples from the data loader
        examples = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                if batch_idx == 0:
                    examples = data[:num_samples].to(self.device)
                    labels = label[:num_samples]
                    break
        
        # Get model reconstructions
        recon_batch, _, _ = self.model(examples)
        
        # Get some random samples from the latent space
        random_samples = self.model.sample(num_samples=num_samples, device=self.device)
        
        # Create a figure with rows: original, reconstruction, random samples
        fig = plt.figure(figsize=(12, 12))
        
        # Plot original images
        for i in range(num_samples):
            # Get the middle slice for each volume
            original_vol = examples[i, 0].cpu().numpy()
            recon_vol = recon_batch[i, 0].cpu().numpy()
            random_vol = random_samples[i, 0].cpu().numpy()
            
            # Find middle indices for meaningful slices
            idx_depth = original_vol.shape[0] // 2
            
            # Plot original
            ax = fig.add_subplot(3, num_samples, i+1)
            ax.imshow(original_vol[idx_depth], cmap='gray')
            ax.set_title(f"Original {i+1}\nLabel: {labels[i]}")
            ax.axis('off')
            
            # Plot reconstruction
            ax = fig.add_subplot(3, num_samples, i+1+num_samples)
            ax.imshow(recon_vol[idx_depth], cmap='gray')
            ax.set_title(f"Reconstruction {i+1}")
            ax.axis('off')
            
            # Plot random sample
            ax = fig.add_subplot(3, num_samples, i+1+2*num_samples)
            ax.imshow(random_vol[idx_depth], cmap='gray')
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
        
        # Add super title
        plt.suptitle(f"VAE Reconstructions and Samples\nLatent Dim: {self.model.latent_dim}", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = self.model_dir / "reconstructions.png"
        plt.savefig(save_path)
        plt.close(fig)
        
        # Create a figure with loss curves
        fig = plt.figure(figsize=(12, 8))
        
        # Plot training and validation losses
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self.train_losses, label='Train')
        ax.plot(self.val_losses, label='Validation')
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot reconstruction and KL losses
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(self.recon_losses, label='Reconstruction')
        ax.plot(self.kl_losses, label='KL Divergence')
        ax.set_title('Component Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.model_dir / "learning_curves.png"
        plt.savefig(save_path)
        plt.close(fig)
        
        return save_path
    
    def visualize_latent_space(self, data_loader, num_samples=200):
        """
        Visualize the latent space by projecting samples and coloring by label
        
        Args:
            data_loader: DataLoader to get samples from
            num_samples: Maximum number of samples to visualize
        
        Returns:
            Path to saved figure
        """
        from sklearn.decomposition import PCA
        
        self.model.eval()
        
        # Collect latent representations and labels
        latent_points = []
        labels = []
        
        with torch.no_grad():
            for data, label in data_loader:
                # Encode batch
                data = data.to(self.device)
                mu, _ = self.model.encode(data)
                
                # Collect points
                latent_points.append(mu.cpu().numpy())
                labels.extend(label.numpy())
                
                if len(labels) >= num_samples:
                    break
        
        # Concatenate batches
        latent_array = np.concatenate(latent_points, axis=0)[:num_samples]
        labels_array = np.array(labels[:num_samples])
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_array)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Define a colormap for the three classes (Control, PD, SWEDD)
        colors = ['blue', 'red', 'green']
        labels_map = {0: 'Control', 1: 'PD', 2: 'SWEDD'}
        
        # Plot each class
        for i, label in enumerate(np.unique(labels_array)):
            idx = np.where(labels_array == label)
            plt.scatter(
                latent_2d[idx, 0], 
                latent_2d[idx, 1],
                c=colors[i % len(colors)],
                label=labels_map.get(label, f"Class {label}"),
                alpha=0.7
            )
        
        plt.title(f"VAE Latent Space Visualization\nLatent Dim: {self.model.latent_dim}", fontsize=14)
        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2f})")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save figure
        save_path = self.model_dir / "latent_space.png"
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test dataset
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        mse_values = []
        batch_count = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                recon_batch, mu, logvar = self.model(data)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.criterion(recon_batch, data, mu, logvar)
                
                # MSE for individual samples
                mse = torch.mean((recon_batch - data)**2, dim=(1, 2, 3, 4))
                mse_values.extend(mse.cpu().numpy())
                
                # Track losses
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
                batch_count += 1
        
        # Calculate average losses
        test_loss /= batch_count
        test_recon_loss /= batch_count
        test_kl_loss /= batch_count
        
        # Calculate MSE statistics
        mse_mean = np.mean(mse_values)
        mse_std = np.std(mse_values)
        
        # Log results
        self.logger.info(
            f"Test results: "
            f"Loss={test_loss:.6f}, Recon={test_recon_loss:.6f}, "
            f"KL={test_kl_loss:.6f}, MSE={mse_mean:.6f} (±{mse_std:.6f})"
        )
        
        # Create visualizations for the test set
        self.visualize_reconstructions(test_loader)
        self.visualize_latent_space(test_loader)
        
        # Return metrics dictionary
        return {
            'loss': test_loss,
            'recon_loss': test_recon_loss,
            'kl_loss': test_kl_loss,
            'mse_mean': float(mse_mean),
            'mse_std': float(mse_std)
        }