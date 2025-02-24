# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class ResidualBlock3D(nn.Module):
    """3D Residual block with two convolutions and optional bottleneck."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """3D Convolutional Encoder with residual connections."""
    
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.res1 = ResidualBlock3D(32, 64, stride=2)    # 32x64x64
        self.res2 = ResidualBlock3D(64, 128, stride=2)   # 16x32x32
        self.res3 = ResidualBlock3D(128, 256, stride=2)  # 8x16x16
        self.res4 = ResidualBlock3D(256, 512, stride=2)  # 4x8x8
        self.flatten_size = 512 * 4 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    """3D Convolutional Decoder with residual connections."""
    
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * 4 * 8 * 8),
            nn.ReLU()
        )
        self.initial_shape = (512, 4, 8, 8)
        self.res1 = ResidualBlock3D(512, 256)
        self.up1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.res2 = ResidualBlock3D(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.res3 = ResidualBlock3D(128, 64)
        self.up3 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.res4 = ResidualBlock3D(64, 32)
        self.up4 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.final = nn.Conv3d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.initial_shape)
        x = self.up1(self.res1(x))
        x = self.up2(self.res2(x))
        x = self.up3(self.res3(x))
        x = self.up4(self.res4(x))
        x = torch.sigmoid(self.final(x))
        return x

class Autoencoder(BaseModel):
    """Complete 3D Convolutional Autoencoder."""
    
    def __init__(self, latent_dim=128, name="autoencoder"):
        super(Autoencoder, self).__init__(name=name)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
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