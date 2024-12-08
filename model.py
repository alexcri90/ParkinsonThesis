import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Calculate the size of flattened features
        self.flatten_size = 256 * 8 * 8 * 8  # For 128x128x128 input
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # Apply convolutions with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.flatten_size = 256 * 8 * 8 * 8  # Same as encoder
        
        # Fully connected layer to reshape
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        # Transposed convolutions
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(32)

    def forward(self, x):
        # Fully connected and reshape
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8, 8)
        
        # Apply transposed convolutions with ReLU and batch norm
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)  # No activation - will use BCEWithLogitsLoss
        
        return x

class DATScanVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

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
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def generate(self, num_samples=1, device=torch.device('cuda')):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decoder(z)
        samples = torch.sigmoid(logits)
        return samples

def train_vae(model, train_loader, num_epochs, learning_rate, device, save_path):
    print("Initializing training...")
    # Use a smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    print("Created optimizer")
    scaler = GradScaler('cuda')
    print("Created scaler")
    
    # Move loss function to the same device as the model
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    print("Created criterion")
    
    # Weight for KL loss (start small and increase gradually)
    kl_weight = 0.0
    
    # Initialize dictionary to store training history
    history = {'loss': [], 'kl_loss': [], 'recon_loss': []}
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_recon_loss = 0
        
        # Gradually increase KL weight
        kl_weight = min((epoch + 1) / 20, 1.0)
        
        print(f"Processing {len(train_loader)} batches...")
        for batch_idx, data in enumerate(train_loader):
            if batch_idx == 0:
                print(f"Starting first batch. Data shape: {data.shape}")
            
            try:
                data = data.to(device, non_blocking=True)
                if batch_idx == 0:
                    print("Data moved to GPU")
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast('cuda', enabled=True):
                    # Forward pass
                    recon_logits, mu, logvar = model(data)
                    if batch_idx == 0:
                        print("Forward pass completed")
                    
                    # Clip values to prevent numerical instability
                    mu = torch.clamp(mu, -10, 10)
                    logvar = torch.clamp(logvar, -10, 10)
                    
                    # Reconstruction loss (using BCEWithLogitsLoss)
                    recon_loss = criterion(recon_logits, data)
                    
                    # KL divergence loss with gradient clipping
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = torch.clamp(kl_loss, 0, 1e6)
                    
                    # Total loss with weighted KL term
                    loss = recon_loss + kl_weight * kl_loss
                    
                    if batch_idx == 0:
                        print("Loss computation completed")
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                if batch_idx == 0:
                    print("Backward pass completed")
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                if batch_idx == 0:
                    print("Optimizer step completed")
                
                # Update epoch losses
                epoch_loss += loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
                
                # Print batch progress
                if batch_idx % 10 == 0:
                    print(f'\rBatch {batch_idx}/{len(train_loader)} | '
                          f'Loss: {loss.item():.4f} | '
                          f'KL: {kl_loss.item():.4f} | '
                          f'Recon: {recon_loss.item():.4f}')
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                if "out of memory" in str(e):
                    print('\nGPU out of memory! Clearing cache and skipping batch...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
                # Update epoch losses
                epoch_loss += loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
                
                # Print batch progress
                if batch_idx % 10 == 0:
                    print(f'\rBatch {batch_idx}/{len(train_loader)} | '
                          f'Loss: {loss.item():.4f} | '
                          f'KL: {kl_loss.item():.4f} | '
                          f'Recon: {recon_loss.item():.4f}', end='')
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('\nGPU out of memory! Clearing cache and skipping batch...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        
        # Store in history
        history['loss'].append(avg_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['recon_loss'].append(avg_recon_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Average Loss: {avg_loss:.4f}')
            print(f'KL Loss: {avg_kl_loss:.4f}')
            print(f'Reconstruction Loss: {avg_recon_loss:.4f}')
            print(f'KL Weight: {kl_weight:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'history': history
            }, save_path)
    
    return history