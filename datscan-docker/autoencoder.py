# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    """Spatial attention module that learns to focus on diagnostically relevant regions."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Compute attention maps
        attn = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)  # Scale to [0, 1]
        
        # Store attention map for visualization
        self.attention_map = attn.detach()
        
        # Apply attention
        return x * attn

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and leaky ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    """
    3D Encoder network with attention mechanism.
    No skip connections to maintain pure latent representation.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Initial feature extraction
        self.init_conv = ConvBlock(1, 32)
        
        # Downsampling path with progressive channel increase
        self.down1 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64)
        )
        
        # Attention after first downsampling
        self.attention1 = SpatialAttention(64)
        
        self.down2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128)
        )
        
        self.down3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256)
        )
        
        self.down4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512)
        )
        
        # Project to latent space
        self.flatten_size = 512 * 4 * 8 * 8  # For input of (64, 128, 128)
        self.fc = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        
        # Apply attention
        x = self.attention1(x)
        
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        # Flatten and project to latent space
        flat = x.view(x.size(0), -1)
        z = self.fc(flat)
        
        return z
    
    def get_attention_map(self):
        """Return the attention map for visualization."""
        if hasattr(self.attention1, 'attention_map'):
            return self.attention1.attention_map
        return None

class Decoder(nn.Module):
    """3D Decoder network with no skip connections."""
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.flatten_size = 512 * 4 * 8 * 8  # For output of (64, 128, 128)
        self.fc = nn.Linear(latent_dim, self.flatten_size)
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(512, 256),
            ConvBlock(256, 256)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(256, 128),
            ConvBlock(128, 128)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(128, 64),
            ConvBlock(64, 64)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock(64, 32),
            ConvBlock(32, 32)
        )
        
        # Final convolution with PReLU instead of sigmoid
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.final_activation = nn.PReLU(num_parameters=1)
        
    def forward(self, z):
        # Reshape from latent space
        x = self.fc(z)
        x = x.view(-1, 512, 4, 8, 8)
        
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        # Final convolution with PReLU activation
        x = self.final_conv(x)
        x = self.final_activation(x)
        
        return x

class RegionWeightedMSELoss(nn.Module):
    """MSE loss that assigns higher importance to striatal regions."""
    def __init__(self, striatal_weight=5.0, shape=(64, 128, 128), device='cuda'):
        super().__init__()
        self.striatal_weight = striatal_weight
        weight_map = self._create_weight_map(shape, striatal_weight)
        # Explicitly move to the specified device
        self.register_buffer('weight_map', weight_map.to(device))
        
    def _create_weight_map(self, shape, striatal_weight):
        weight_map = torch.ones((1, 1, *shape), dtype=torch.float32)
        # Striatal region coordinates
        weight_map[:, :, 20:40, 82:103, 43:82] = striatal_weight
        return weight_map
    
    def forward(self, pred, target):
        # Compute MSE loss
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Ensure weight_map is on the same device as mse
        weight_map = self.weight_map.to(mse.device)
        
        # Apply region weighting
        weighted_mse = mse * weight_map
        
        return weighted_mse.mean()

class StriatalMSELoss(nn.Module):
    """MSE loss specifically for the striatal region."""
    def __init__(self, shape=(64, 128, 128), device='cuda'):
        super().__init__()
        mask = self._create_striatal_mask(shape)
        self.register_buffer('striatal_mask', mask.to(device))
        
    def _create_striatal_mask(self, shape):
        mask = torch.zeros((1, 1, *shape), dtype=torch.float32)
        # Striatal region coordinates
        mask[:, :, 20:40, 82:103, 43:82] = 1.0
        return mask
    
    def forward(self, pred, target):
        # Compute MSE loss
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Ensure mask is on the same device as mse
        mask = self.striatal_mask.to(mse.device)
        
        # Mask to striatal region
        masked_mse = mse * mask
        
        # Normalize by number of pixels in the mask
        return masked_mse.sum() / mask.sum()

class Autoencoder(nn.Module):
    """3D Autoencoder with attention mechanism for DATSCAN images."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def get_attention_map(self):
        """Get attention map from encoder"""
        return self.encoder.get_attention_map()