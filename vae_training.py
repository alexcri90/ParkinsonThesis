import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import numpy as np
from tqdm import tqdm

def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Compute VAE loss with reconstruction and KL divergence terms.
    Returns total loss and a dictionary with individual loss components.
    """
    # Reconstruction loss (binary cross entropy as our images are normalized between 0 and 1)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + kl_loss
    
    return total_loss, {
        'total_loss': total_loss.item(),
        'reconstruction_loss': recon_loss.item(),
        'kl_loss': kl_loss.item()
    }

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        x_recon, mu, log_var = model(batch)
        
        # Compute loss
        loss, loss_dict = vae_loss(x_recon, batch, mu, log_var)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict['total_loss']
        total_recon_loss += loss_dict['reconstruction_loss']
        total_kl_loss += loss_dict['kl_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'total_loss': f"{loss_dict['total_loss']/batch.size(0):.4f}",
            'recon_loss': f"{loss_dict['reconstruction_loss']/batch.size(0):.4f}",
            'kl_loss': f"{loss_dict['kl_loss']/batch.size(0):.4f}"
        })
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss
    }

def train_vae(model: nn.Module,
              train_loader: DataLoader,
              num_epochs: int,
              learning_rate: float,
              device: torch.device,
              save_path: str = 'vae_model.pt') -> Dict:
    """Train the VAE model."""
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'loss': [],
        'reconstruction_loss': [],
        'kl_loss': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        epoch_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Update history
        for k, v in epoch_metrics.items():
            history[k].append(v)
        
        # Print epoch metrics
        print(f"Average loss: {epoch_metrics['loss']:.4f}")
        print(f"Average reconstruction loss: {epoch_metrics['reconstruction_loss']:.4f}")
        print(f"Average KL loss: {epoch_metrics['kl_loss']:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_metrics['loss'],
            }, save_path)
    
    return history