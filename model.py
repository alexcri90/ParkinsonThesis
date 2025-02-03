import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import os

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

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Initial convolution block
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)  # Reduced from 32 to 16
        self.bn1 = nn.BatchNorm3d(16)
        self.res1 = ResidualBlock(16)
        
        # Second block
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # Reduced from 64 to 32
        self.bn2 = nn.BatchNorm3d(32)
        self.res2 = ResidualBlock(32)
        
        # Third block
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # Reduced from 128 to 64
        self.bn3 = nn.BatchNorm3d(64)
        self.res3 = ResidualBlock(64)
        
        # Fourth block
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # Reduced from 256 to 128
        self.bn4 = nn.BatchNorm3d(128)
        self.res4 = ResidualBlock(128)
        
        # Attention mechanism
        self.attention = SelfAttention3D(128)  # Updated to match new channel count
        
        # For input volume 128x128x128, after 4 stride-2 convs:
        # 128 -> 64 -> 32 -> 16 -> 8
        self.flatten_size = 128 * 8 * 8 * 8  # Reduced from 256 to 128
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        self.dropout = nn.Dropout3d(0.1)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Add assertions for debugging
        assert torch.isfinite(x).all(), "Input contains nan or inf"
        
        x = F.relu(self.bn1(self.conv1(x)))
        assert torch.isfinite(x).all(), "After conv1"
        x = self.res1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        assert torch.isfinite(x).all(), "After conv2"
        x = self.res2(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        assert torch.isfinite(x).all(), "After conv3"
        x = self.res3(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        assert torch.isfinite(x).all(), "After conv4"
        x = self.res4(x)
        
        x = self.attention(x)
        assert torch.isfinite(x).all(), "After attention"
        
        x = x.view(x.size(0), -1)
        
        # Scale down before FC layers
        x = x * 0.1
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.flatten_size = 128 * 8 * 8 * 8  # Reduced to match encoder
        
        self.dropout = nn.Dropout(0.1)
        
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(16)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Scale input to avoid explosion
        x = x * 0.1
        
        x = self.dropout(self.fc(x))
        x = x.view(x.size(0), 128, 8, 8, 8)
        
        x = self.dropout(F.relu(self.bn1(self.deconv1(x))))
        x = self.dropout(F.relu(self.bn2(self.deconv2(x))))
        x = self.dropout(F.relu(self.bn3(self.deconv3(x))))
        x = torch.sigmoid(self.deconv4(x))  # Added sigmoid for [0,1] output
        
        return x

class DATScanVAE(nn.Module):
    def __init__(self, latent_dim=128):
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
        recon = self.decoder(z)
        return recon, mu, logvar
    
    @torch.no_grad()
    def generate(self, num_samples=1, device=torch.device('cuda')):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decoder(z)
        samples = torch.sigmoid(logits)
        return samples

def train_vae(model, train_loader, num_epochs, learning_rate, device, save_path, start_epoch=0, history=None):
    """
    Revised training function for the DATScanVAE model (suitable for medical applications on an nVidia 4070Ti):
      - Total epochs: 150
      - Learning rate: 5e-5
      - Warmup: first 10 epochs with beta=0; then linearly increase beta to max_beta over 50 epochs.
      - Reconstruction loss computed as per-voxel MSE (reduction='mean') scaled by recon_scale (100.0)
      - Uses torch.amp.autocast with device_type='cuda'
      - Applies gradient clipping and detailed logging.
    """
    print("Initializing/Resuming training with medical-grade settings...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Use mean reduction for reconstruction loss (per voxel)
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    history = history or {'loss': [], 'kl_loss': [], 'recon_loss': []}
    
    # Extended KL warmup parameters:
    warmup_epochs = 10         # First 10 epochs: zero KL contribution.
    annealing_epochs = 50      # Then ramp up beta over the next 50 epochs.
    max_beta = 0.1             # Maximum beta value.
    
    recon_scale = 100.0        # Scaling factor for the reconstruction loss.
    max_grad_norm = 1.0        # Gradient clipping threshold.
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nStarting training from epoch {start_epoch+1} to {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        
        # Compute beta using the modified warmup schedule:
        if epoch < warmup_epochs:
            beta = 0.0
        else:
            beta = min(max_beta, ((epoch - warmup_epochs + 1) / annealing_epochs) * max_beta)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, data in enumerate(pbar):
            try:
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Use autocast with device_type='cuda'
                with torch.amp.autocast(device_type='cuda'):
                    recon, mu, logvar = model(data)
                    recon_loss = criterion(recon, data) * recon_scale
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss
                
                if torch.isnan(loss):
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'beta': f'{beta:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('\nGPU out of memory! Clearing cache and skipping batch...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        num_batches = max(1, len(train_loader))
        avg_loss = epoch_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        
        scheduler.step(avg_loss)
        
        history['loss'].append(avg_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['recon_loss'].append(avg_recon_loss)
        
        # Detailed logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Average Loss: {avg_loss:.4f}')
            print(f'  KL Loss: {avg_kl_loss:.4f}')
            print(f'  Reconstruction Loss: {avg_recon_loss:.4f}')
            print(f'  Beta: {beta:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint every 5 epochs and at the end.
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'history': history,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'beta': beta
            }
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
    
    return history