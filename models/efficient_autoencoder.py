# models/efficient_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EfficientEncoder(nn.Module):
    """Efficient 3D Convolutional Encoder with reduced parameter count"""
    
    def __init__(self, latent_dim):
        super(EfficientEncoder, self).__init__()
        
        # Initial convolution block
        self.init_conv = ConvBlock(1, 16, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)
        )  # 32x64x64
        
        self.down2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )  # 16x32x32
        
        self.down3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )  # 8x16x16
        
        self.down4 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )  # 4x8x8
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool3d(1)  # Outputs 256x1x1x1
        
        # Bottleneck fully connected layers (much smaller now)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        # Global average pooling (replaces flattening)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 256]
        
        # Bottleneck FC layers
        x = self.fc(x)
        return x

class EfficientDecoder(nn.Module):
    """Efficient 3D Convolutional Decoder with proper dimensions"""
    
    def __init__(self, latent_dim):
        super(EfficientDecoder, self).__init__()
        
        # Bottleneck fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # Reshape to starting volume
        self.initial_shape = (256, 1, 1, 1)
        
        # Upsampling blocks with correct output dimensions
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=4, padding=0),  # 1x1x1 → 4x4x4
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )  # 4x4x4
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4x4 → 8x8x8
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )  # 8x8x8
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8x8 → 16x16x16
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1)
        )  # 16x16x16
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # 16x16x16 → 32x32x32
            ConvBlock(16, 16, kernel_size=3, stride=1, padding=1)
        )  # 32x32x32
        
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),  # 32x32x32 → 64x64x64
            ConvBlock(8, 8, kernel_size=3, stride=1, padding=1)
        )  # 64x64x64
        
        # Add final upsampling to get original dimensions
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(8, 8, kernel_size=(2,4,4), stride=(1,2,2), padding=(0,1,1)),  # 64x64x64 → 64x128x128
            ConvBlock(8, 8, kernel_size=3, stride=1, padding=1) 
        )  # 64x128x128
        
        # Final output convolution
        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Bottleneck FC layers
        x = self.fc(x)
        
        # Reshape to initial volume shape
        x = x.view(-1, *self.initial_shape)
        
        # Upsampling path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.final_up(x)
        
        # Final convolution
        x = torch.sigmoid(self.final_conv(x))
        return x

class EfficientAutoencoder(BaseModel):
    """Efficient 3D Convolutional Autoencoder with much lower parameter count"""
    
    def __init__(self, latent_dim=128, name="efficient_autoencoder"):
        super(EfficientAutoencoder, self).__init__(name=name)
        self.encoder = EfficientEncoder(latent_dim)
        self.decoder = EfficientDecoder(latent_dim)
        self.latent_dim = latent_dim
        self._initialize_weights()
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def get_model_info(self):
        """Return model information"""
        return {
            "name": self.name,
            "latent_dim": self.latent_dim,
            "parameters": self.count_parameters()
        }