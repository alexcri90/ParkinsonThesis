import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import pydicom
import numpy as np

class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dims, embedding_dim=32):
        super().__init__()
        self.metadata_dims = metadata_dims
        
        # Network for metadata
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dims, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.metadata_net(x)

class SupervisedVAE(nn.Module):
    def __init__(self, latent_dim=128, metadata_dims=10, num_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.metadata_dims = metadata_dims
        self.num_classes = num_classes
        
        # Original VAE components
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(metadata_dims)
        
        # Classifier head
        classifier_input_dim = latent_dim + 32  # latent_dim + metadata_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, metadata):
        # Encode image
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # Encode metadata
        metadata_embedding = self.metadata_encoder(metadata)
        
        # Combine latent representations for classification
        combined_features = torch.cat([z, metadata_embedding], dim=1)
        
        # Generate classification and reconstruction
        classification = self.classifier(combined_features)
        reconstruction = self.decoder(z)
        
        return reconstruction, classification, mu, logvar

def train_supervised_vae(model, train_loader, num_epochs, device, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Loss functions
    recon_criterion = nn.BCEWithLogitsLoss(reduction='sum')
    classification_criterion = nn.CrossEntropyLoss()
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': [], 'class_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        
        for batch_idx, (data, metadata, labels) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, classification, mu, logvar = model(data, metadata)
            
            # Calculate losses
            recon_loss = recon_criterion(recon_batch, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = classification_criterion(classification, labels)
            
            # Total loss (weighted sum)
            total_loss = recon_loss + kl_loss + 10.0 * class_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['kl_loss'] += kl_loss.item()
            epoch_losses['class_loss'] += class_loss.item()
        
        # Update scheduler
        scheduler.step(epoch_losses['total_loss'])
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, save_path)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            for loss_name, loss_value in epoch_losses.items():
                avg_loss = loss_value / len(train_loader.dataset)
                print(f"{loss_name}: {avg_loss:.4f}")
    
    return history