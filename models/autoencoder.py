#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D Autoencoder model for DATSCAN images.
Implements encoder and decoder for 3D medical volumes.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from models.base import BaseModel


class ConvBlock(nn.Module):
    """
    Memory-efficient convolutional block with batch normalization and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        padding: Padding for convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)),
            ('bn', nn.BatchNorm3d(out_channels)),
            ('relu', nn.ReLU(inplace=True))  # inplace ReLU for memory efficiency
        ]))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    3D Encoder network optimized for 128³ input volumes.
    
    Args:
        latent_dim: Dimension of the latent space
        in_channels: Number of input channels (default: 1 for grayscale)
    """
    def __init__(self, latent_dim=128, in_channels=1):
        super().__init__()

        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, 32)  # 128 -> 128

        # Downsampling path with progressive channel increase
        self.down1 = nn.Sequential(
            ConvBlock(32, 64, stride=2),    # 128 -> 64
            ConvBlock(64, 64)
        )

        self.down2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),    # 64 -> 32
            ConvBlock(128, 128)
        )

        self.down3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),   # 32 -> 16
            ConvBlock(256, 256)
        )

        self.down4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),  # 16 -> 8
            ConvBlock(512, 512)
        )

        # Project to latent space
        self.flatten_size = 512 * 8 * 8 * 4
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        x = self.init_conv(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Flatten and project to latent space
        flat = torch.flatten(d4, start_dim=1)
        z = self.fc(flat)

        return z


class Decoder(nn.Module):
    """
    3D Decoder network optimized for 128³ output volumes.
    
    Args:
        latent_dim: Dimension of the latent space
        out_channels: Number of output channels (default: 1 for grayscale)
    """
    def __init__(self, latent_dim=128, out_channels=1):
        super().__init__()

        self.flatten_size = 512 * 8 * 8 * 4
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

        # Final convolution
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, z):
        # Reshape from latent space
        x = self.fc(z)
        x = x.view(-1, 512, 4, 8, 8)

        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final convolution
        x = self.final_conv(x)

        return x


class Autoencoder(BaseModel):
    """
    Memory-optimized 3D Autoencoder for 128³ medical volumes.
    
    Args:
        latent_dim: Dimension of the latent space
        in_channels: Number of input channels (default: 1 for grayscale)
        name: Model name for saving/loading
    """
    def __init__(self, latent_dim=128, in_channels=1, name="autoencoder"):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space (for generation)"""
        return self.decoder(z)
    
    def get_latent_dim(self):
        """Get the dimension of the latent space"""
        return self.latent_dim


# For testing purposes
if __name__ == "__main__":
    # Create model and test with dummy data
    model = Autoencoder(latent_dim=128)
    print(f"Model parameter count: {model.count_parameters():,}")
    
    # Create dummy input
    dummy_input = torch.randn(2, 1, 64, 128, 128)
    
    # Test forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape must match input shape"
    
    # Test encoder
    latent = model.encode(dummy_input)
    print(f"Latent shape: {latent.shape}")
    assert latent.shape == (2, 128), "Latent shape is incorrect"
    
    # Test decoder
    dummy_latent = torch.randn(2, 128)
    reconstruction = model.decode(dummy_latent)
    print(f"Reconstruction shape: {reconstruction.shape}")
    assert reconstruction.shape == dummy_input.shape, "Reconstruction shape must match input shape"
    
    print("Model test successful!")