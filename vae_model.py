import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with batch normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class Conv3DTransposeBlock(nn.Module):
    """Basic 3D transpose convolution block with batch normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, output_padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    """Encoder network for 3D DATSCAN images."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Starting from 128x128x128x1
        self.conv1 = Conv3DBlock(1, 16, stride=2)    # -> 64x64x64x16
        self.conv2 = Conv3DBlock(16, 32, stride=2)   # -> 32x32x32x32
        self.conv3 = Conv3DBlock(32, 64, stride=2)   # -> 16x16x16x64
        self.conv4 = Conv3DBlock(64, 128, stride=2)  # -> 8x8x8x128
        
        # Compute the flattened size
        self.flatten_size = 8 * 8 * 8 * 128
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Get mean and log variance
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var

class Decoder(nn.Module):
    """Decoder network for 3D DATSCAN images."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.flatten_size = 8 * 8 * 8 * 128
        
        # Fully connected layer
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        # Transpose convolutions
        self.conv1 = Conv3DTransposeBlock(128, 64)  # -> 16x16x16x64
        self.conv2 = Conv3DTransposeBlock(64, 32)   # -> 32x32x32x32
        self.conv3 = Conv3DTransposeBlock(32, 16)   # -> 64x64x64x16
        self.conv4 = Conv3DTransposeBlock(16, 1)    # -> 128x128x128x1
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Fully connected layer
        x = self.fc(z)
        
        # Reshape
        x = x.view(-1, 128, 8, 8, 8)
        
        # Transpose convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Final activation to ensure output is between 0 and 1
        return torch.sigmoid(x)

class DATScanVAE(nn.Module):
    """Complete VAE model for 3D DATSCAN images."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new DATSCAN images by sampling from the latent space."""
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            # Decode the samples
            samples = self.decoder(z)
        return samples