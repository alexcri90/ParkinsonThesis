# models/pure_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock3D(nn.Module):
    """Enhanced 3D Residual block with SE attention"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # SE block for attention
        self.use_se = use_se
        if use_se:
            self.se = SEBlock3D(out_channels)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class PureEncoder(nn.Module):
    """Pure 3D Convolutional Encoder with attention but no skip outputs"""
    
    def __init__(self, latent_dim):
        super(PureEncoder, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.res1 = ResidualBlock3D(32, 64, stride=2)    # 32x64x64
        self.res2 = ResidualBlock3D(64, 128, stride=2)   # 16x32x32
        self.res3 = ResidualBlock3D(128, 256, stride=2)  # 8x16x16
        self.res4 = ResidualBlock3D(256, 512, stride=2)  # 4x8x8
        
        # Focal attention for striatum region
        self.attention = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.flatten_size = 512 * 4 * 8 * 8
        
        # Deeper bottleneck layers for better latent representation
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, latent_dim)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Apply attention focusing on important regions
        att = self.attention(x)
        x = x * att
        
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x

class PureDecoder(nn.Module):
    """Pure 3D Convolutional Decoder with NO skip connections"""
    
    def __init__(self, latent_dim):
        super(PureDecoder, self).__init__()
        
        # Deeper bottleneck expansion for better latent representation
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512 * 4 * 8 * 8),
            nn.ReLU()
        )
        self.initial_shape = (512, 4, 8, 8)
        
        # Upsampling blocks with NO skip-connections
        self.res1 = ResidualBlock3D(512, 512)
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        
        self.res2 = ResidualBlock3D(256, 256)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        
        self.res3 = ResidualBlock3D(128, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        
        self.res4 = ResidualBlock3D(64, 64)
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        
        # Enhanced final layers with residual connections
        self.final_res = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Add a 1x1 conv to enhance feature selection
            nn.Conv3d(32, 16, kernel_size=1),
            nn.ReLU()
        )
        # Special attention to striatal regions
        self.striatal_attention = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.final = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.initial_shape)
        
        # First upsampling block
        x = self.res1(x)
        x = self.up1(x)
        
        # Second upsampling block
        x = self.res2(x)
        x = self.up2(x)
        
        # Third upsampling block
        x = self.res3(x)
        x = self.up3(x)
        
        # Fourth upsampling block
        x = self.res4(x)
        x = self.up4(x)
        
        # Final processing with attention to important regions
        x = self.final_res(x)
        
        # Apply striatal-specific attention
        att = self.striatal_attention(x)
        x = x * att + x  # Residual connection to preserve information
        
        x = torch.sigmoid(self.final(x))
        return x

class PureAutoencoder(BaseModel):
    """Pure 3D Convolutional Autoencoder with strict bottleneck and no skip connections."""
    
    def __init__(self, latent_dim=128, name="pure_autoencoder"):
        super(PureAutoencoder, self).__init__(name=name)
        self.encoder = PureEncoder(latent_dim)
        self.decoder = PureDecoder(latent_dim)
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