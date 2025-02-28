# models/pure_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel
from models.pure_autoencoder import PureEncoder, PureDecoder, ResidualBlock3D

class VAEEncoder(nn.Module):
    """Pure VAE encoder with residual blocks that outputs mean and log variance"""
    
    def __init__(self, latent_dim):
        super(VAEEncoder, self).__init__()
        
        # Base encoder features
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
        
        # Separate paths for mu and logvar to avoid information sharing
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc_mu = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Encode features
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Apply attention focusing on important regions
        att = self.attention(x)
        x = x * att
        
        x = x.view(-1, self.flatten_size)
        
        # Split into mu and logvar
        x = self.fc_shared(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class PureVAE(BaseModel):
    """Pure Variational Autoencoder with strict bottleneck and no skip connections"""
    
    def __init__(self, latent_dim=128, kl_weight=0.01, name="pure_vae"):
        super(PureVAE, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Create encoder and decoder
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = PureDecoder(latent_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # During evaluation, use the mean directly
            z = mu
        return z
    
    def forward(self, x):
        # Encode input
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def sample(self, num_samples=1, device=None):
        """Sample from the latent space"""
        if device is None:
            device = next(self.parameters()).device
            
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode samples
        samples = self.decode(z)
        return samples
    
    def generate_variations(self, x, num_variations=5, std_scale=0.5):
        """Generate variations of an input by sampling around its latent representation"""
        mu, logvar = self.encoder(x)
        variations = []
        
        # Generate samples around the input's latent mean
        for _ in range(num_variations):
            std = torch.exp(0.5 * logvar) * std_scale
            eps = torch.randn_like(std)
            z = mu + eps * std
            variation = self.decode(z)
            variations.append(variation)
            
        return variations
    
    def interpolate(self, x1, x2, steps=10):
        """Interpolate between two inputs in latent space"""
        mu1, _ = self.encode(x1.unsqueeze(0))
        mu2, _ = self.encode(x2.unsqueeze(0))
        
        # Create interpolation steps
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            z = mu1 * (1 - alpha) + mu2 * alpha
            interp = self.decode(z)
            interpolations.append(interp.squeeze(0))
            
        return torch.stack(interpolations)
    
    def get_model_info(self):
        """Return model information"""
        return {
            "name": self.name,
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
            "parameters": self.count_parameters()
        }