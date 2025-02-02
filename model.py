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
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.res1 = ResidualBlock(32)
        
        # Second block
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.res2 = ResidualBlock(64)
        
        # Third block
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.res3 = ResidualBlock(128)
        
        # Fourth block
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.res4 = ResidualBlock(256)
        
        # Attention mechanism remains the same
        self.attention = SelfAttention3D(256)
        
        # For input volume 128x128x128, after 4 stride-2 convs:
        # Depth: 128 -> 64 -> 32 -> 16 -> 8; Height/Width: 128 -> 64 -> 32 -> 16 -> 8
        self.flatten_size = 256 * 8 * 8 * 8  # = 256 * 512 = 131072
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
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
        # Use the new flatten size
        self.flatten_size = 256 * 8 * 8 * 8  # 131072
        
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        # Transposed convolutions to reconstruct to 128x128x128
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        
        self.deconv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        # Reshape to (batch, 256, 8, 8, 8)
        x = x.view(x.size(0), 256, 8, 8, 8)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
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
    print("Initializing/Resuming training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    history = history or {'loss': [], 'kl_loss': [], 'recon_loss': []}
    
    # Extend annealing to 50 epochs instead of 10
    annealing_epochs = 50
    
    max_grad_norm = 0.5
    
    print(f"\nStarting training from epoch {start_epoch+1} to {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        
        # Extended annealing: gradually ramp KL weight over 50 epochs
        kl_weight = min((epoch + 1) / annealing_epochs, 1.0)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, data in enumerate(pbar):
            try:
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                recon, mu, logvar = model(data)
                
                # Clamp for numerical stability
                mu = torch.clamp(mu, -10, 10)
                logvar = torch.clamp(logvar, -10, 10)
                
                # Reconstruction loss
                recon_loss = criterion(recon, data)
                
                # KL divergence computed by summing over latent dimensions
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                
                # Optionally, you can disable consistency loss if it is interfering:
                # random_noise = torch.randn_like(data) * 0.1
                # noisy_data = data + random_noise
                # noisy_recon, _, _ = model(noisy_data)
                # consistency_loss = torch.nn.functional.mse_loss(recon, noisy_recon)
                # loss = recon_loss + kl_weight * kl_loss + 0.01 * consistency_loss
                
                # For now, we omit consistency loss:
                loss = recon_loss + kl_weight * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_recon_loss += recon_loss.item()
                
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
        
        avg_loss = epoch_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        
        scheduler.step(avg_loss)
        
        history['loss'].append(avg_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['recon_loss'].append(avg_recon_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Average Loss: {avg_loss:.4f}')
            print(f'KL Loss: {avg_kl_loss:.4f}')
            print(f'Reconstruction Loss: {avg_recon_loss:.4f}')
            print(f'KL Weight: {kl_weight:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'history': history,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'kl_weight': kl_weight
            }
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
    
    return history

