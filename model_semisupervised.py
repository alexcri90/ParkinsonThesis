# model_semisupervised.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler

# Keep the original building blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels//8, 1)
        self.key = nn.Conv3d(in_channels, in_channels//8, 1)
        self.value = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        q = self.query(x).view(batch_size, -1, D*H*W)
        k = self.key(x).view(batch_size, -1, D*H*W)
        v = self.value(x).view(batch_size, -1, D*H*W)
        
        attention = F.softmax(torch.bmm(q.permute(0,2,1), k), dim=2)
        out = torch.bmm(v, attention.permute(0,2,1))
        out = out.view(batch_size, C, D, H, W)
        return self.gamma * out + x

class SemiSupervisedEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.res1 = ResidualBlock(32)
        
        # Deeper layers
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.res2 = ResidualBlock(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.res3 = ResidualBlock(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.res4 = ResidualBlock(256)
        
        # Attention mechanism
        self.attention = SelfAttention3D(256)
        
        # Calculate flatten size
        self.flatten_size = 256 * 8 * 8 * 8
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.res4(x)
        
        x = self.attention(x)
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.flatten_size = 256 * 8 * 8 * 8
        
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(32)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8, 8)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
        
        return x

class MetadataClassifier(nn.Module):
    def __init__(self, latent_dim, metadata_dims):
        """
        metadata_dims: dict containing the number of classes for each metadata field
        e.g., {'PatientSex': 4, 'StudyDescription': 88, ...}
        """
        super().__init__()
        
        self.classifiers = nn.ModuleDict()
        for field, num_classes in metadata_dims.items():
            self.classifiers[field] = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, z):
        return {field: classifier(z) for field, classifier in self.classifiers.items()}

class SemiSupervisedVAE(nn.Module):
    def __init__(self, latent_dim=128, metadata_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.metadata_dims = metadata_dims or {}
        
        self.encoder = SemiSupervisedEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.metadata_classifier = MetadataClassifier(latent_dim, metadata_dims)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        metadata_preds = self.metadata_classifier(z)
        return recon, mu, logvar, metadata_preds

def train_semisupervised_vae(model, train_loader, num_epochs, device, save_path,
                             metadata_weights=None, start_epoch=0, history=None):
    """
    Train the semi-supervised VAE with support for pausing/resuming training.
    
    Args:
        model: The semi-supervised VAE model.
        train_loader: DataLoader for training data.
        num_epochs: Total number of epochs to train.
        device: Device (CPU or GPU) to train on.
        save_path: Path for saving checkpoints.
        metadata_weights: dict with weight factors for each metadata field's loss.
        start_epoch: Epoch to resume training from.
        history: Dictionary holding training history (for resuming training).
    """
    print("Initializing/Resuming training...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = GradScaler()
    
    # Set default metadata weights if none provided
    if metadata_weights is None:
        metadata_weights = {field: 1.0 for field in model.metadata_dims}
    
    # Using BCEWithLogitsLoss with sum reduction for reconstruction loss.
    # Alternatively, you could use reduction='mean' to average over voxels.
    recon_criterion = nn.BCEWithLogitsLoss(reduction='sum')
    metadata_criteria = {
        field: nn.CrossEntropyLoss(reduction='mean')
        for field in model.metadata_dims
    }
    
    # Initialize or resume training history
    if history is None:
        history = {
            'loss': [], 'recon_loss': [], 'kl_loss': [],
            **{f'{field}_loss': [] for field in model.metadata_dims}
        }
    
    # KL annealing parameters
    annealing_epochs = 10  # Increase KL weight linearly over the first 10 epochs

    print(f"\nStarting training from epoch {start_epoch + 1} to {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in history.keys()}
        
        # Update KL weight: linearly increase until it reaches 1.0
        kl_weight = min((epoch + 1) / annealing_epochs, 1.0)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, metadata) in enumerate(pbar):
            try:
                data = data.to(device)
                metadata = {k: v.to(device) for k, v in metadata.items()}
                optimizer.zero_grad()
                
                # Forward pass
                with autocast():
                    recon, mu, logvar, metadata_preds = model(data)
                
                    # Compute reconstruction loss (option: use 'mean' reduction if preferred)
                    recon_loss = recon_criterion(recon, data)
                    # If using mean reduction, you might not need to further average per sample.
                    # For sum reduction, we divide later by total number of samples.
                    
                    # KL divergence loss
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # Metadata classification losses
                    metadata_losses = {
                        field: metadata_criteria[field](metadata_preds[field], metadata[field])
                        for field in model.metadata_dims
                    }
                    
                    # Total loss: combine reconstruction, KL, and metadata losses
                    loss = recon_loss + kl_weight * kl_loss + \
                           sum(metadata_weights[field] * m_loss for field, m_loss in metadata_losses.items())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update running epoch losses
                epoch_losses['loss'] += loss.item()
                epoch_losses['recon_loss'] += recon_loss.item()
                epoch_losses['kl_loss'] += kl_loss.item()
                for field, m_loss in metadata_losses.items():
                    epoch_losses[f'{field}_loss'] += m_loss.item()
                
                # Update progress bar with current batch losses
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('\nGPU out of memory! Clearing cache and skipping batch...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Normalize losses by the number of samples (or batches, as appropriate)
        num_samples = len(train_loader.dataset)
        for k in epoch_losses:
            epoch_losses[k] /= num_samples
            history[k].append(epoch_losses[k])
        
        # Update learning rate scheduler based on total epoch loss
        scheduler.step(epoch_losses['loss'])
        
        # Log summary every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs} summary:')
            for k, v in epoch_losses.items():
                print(f'  {k}: {v:.4f}')
            print(f'  KL Weight: {kl_weight:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint every 5 epochs
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_losses['loss'],
                'history': history
            }
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved at epoch {epoch+1} to {save_path}')
    
    return history