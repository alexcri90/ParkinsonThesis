# trainers/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    """Simple VAE loss combining MSE and KL divergence"""
    def __init__(self, kl_weight=0.01):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, mu, logvar):
        # Simple MSE reconstruction loss
        recon_loss = self.mse_loss(pred, target)
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with weighting
        total_loss = recon_loss + self.kl_weight * kl_div
        
        return total_loss, recon_loss, kl_div