import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

class SemiSupervisedVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder components
        self.encoder = nn.ModuleDict({
            'conv1': nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            'bn1': nn.BatchNorm3d(16),
            'conv2': nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            'bn2': nn.BatchNorm3d(32),
            'conv3': nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            'bn3': nn.BatchNorm3d(64),
            'conv4': nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            'bn4': nn.BatchNorm3d(128)
        })
        
        # Latent space projections
        self.flatten_size = 128 * 8 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Decoder components
        self.decoder_fc = nn.Linear(latent_dim + num_classes, self.flatten_size)
        self.decoder = nn.ModuleDict({
            'deconv1': nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            'debn1': nn.BatchNorm3d(64),
            'deconv2': nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            'debn2': nn.BatchNorm3d(32),
            'deconv3': nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            'debn3': nn.BatchNorm3d(16),
            'deconv4': nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        })
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        # Encoder forward pass
        x = F.relu(self.encoder['bn1'](self.encoder['conv1'](x)))
        x = F.relu(self.encoder['bn2'](self.encoder['conv2'](x)))
        x = F.relu(self.encoder['bn3'](self.encoder['conv3'](x)))
        x = F.relu(self.encoder['bn4'](self.encoder['conv4'](x)))
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z, c):
        # Concatenate latent vector with class embedding
        z_c = torch.cat([z, F.one_hot(c, self.num_classes).float()], dim=1)
        
        # Decoder forward pass
        x = self.decoder_fc(z_c)
        x = x.view(x.size(0), 128, 8, 8, 8)
        
        x = F.relu(self.decoder['debn1'](self.decoder['deconv1'](x)))
        x = F.relu(self.decoder['debn2'](self.decoder['deconv2'](x)))
        x = F.relu(self.decoder['debn3'](self.decoder['deconv3'](x)))
        x = torch.sigmoid(self.decoder['deconv4'](x))
        
        return x
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        class_pred = self.classifier(z)
        
        return recon, class_pred, mu, logvar

def compute_loss(model, x, c, recon_scale=1.0, kl_scale=0.1, class_scale=1.0):
    """
    Compute the total loss for semi-supervised VAE with proper scaling
    """
    recon, class_pred, mu, logvar = model(x, c)
    
    # Reconstruction loss using BCE (better for image data)
    recon_loss = F.binary_cross_entropy(recon, x, reduction='mean') * recon_scale
    
    # KL divergence with numerical stability
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp().clamp(min=1e-8)) * kl_scale
    
    # Classification loss with label smoothing
    class_loss = F.cross_entropy(class_pred, c, label_smoothing=0.1) * class_scale
    
    # Early return if any component is NaN
    if torch.isnan(recon_loss) or torch.isnan(kl_loss) or torch.isnan(class_loss):
        raise ValueError("NaN detected in loss computation")
    
    total_loss = recon_loss + kl_loss + class_loss
    
    return total_loss, recon_loss, kl_loss, class_loss

def train_semisupervised_vae(model, train_loader, num_epochs, learning_rate, device,
                            save_path, start_epoch=0, history=None):
    """
    Training function for semi-supervised VAE (optimized for NVIDIA 4070Ti)
    """
    print("Initializing/Resuming training with medical-grade settings...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    history = history or {
        'loss': [], 'recon_loss': [], 'kl_loss': [], 
        'class_loss': [], 'class_acc': []
    }
    
    # Training parameters with more conservative scaling
    warmup_epochs = 5
    annealing_epochs = 20
    max_recon_scale = 1.0
    max_kl_scale = 0.1  # Start with smaller KL weight
    max_class_scale = 0.5
    max_grad_norm = 0.5  # More aggressive gradient clipping
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nStarting training from epoch {start_epoch+1} to {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_class_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Compute scaling factors
        if epoch < warmup_epochs:
            kl_scale = 0.0
            class_scale = 0.0
        else:
            kl_scale = min(max_kl_scale, 
                          ((epoch - warmup_epochs + 1) / annealing_epochs) * max_kl_scale)
            class_scale = min(max_class_scale, 
                            ((epoch - warmup_epochs + 1) / annealing_epochs) * max_class_scale)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, labels) in enumerate(pbar):
            try:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type='cuda'):
                    recon, class_pred, mu, logvar = model(data, labels)
                    recon_loss = F.mse_loss(recon, data, reduction='mean') * max_recon_scale
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * kl_scale
                    class_loss = F.cross_entropy(class_pred, labels) * class_scale
                    
                    loss = recon_loss + kl_loss + class_loss
                
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # Compute accuracy
                pred = class_pred.argmax(dim=1)
                correct_preds += (pred == labels).sum().item()
                total_samples += labels.size(0)
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_class_loss += class_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'class': f'{class_loss.item():.4f}',
                    'acc': f'{100 * correct_preds / total_samples:.2f}%'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('\nGPU out of memory! Clearing cache and skipping batch...')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Compute epoch metrics
        num_batches = max(1, len(train_loader))
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches
        epoch_acc = 100 * correct_preds / total_samples
        
        scheduler.step(avg_loss)
        
        # Update history
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['class_loss'].append(avg_class_loss)
        history['class_acc'].append(epoch_acc)
        
        # Detailed logging every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Average Loss: {avg_loss:.4f}')
            print(f'  Reconstruction Loss: {avg_recon_loss:.4f}')
            print(f'  KL Loss: {avg_kl_loss:.4f}')
            print(f'  Classification Loss: {avg_class_loss:.4f}')
            print(f'  Classification Accuracy: {epoch_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save gradient scaler state
                'loss': avg_loss,
                'history': history,
                'random_state': {
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
                    'numpy': np.random.get_state()
                }
            }
            
            # Save to temporary file first, then move to final location
            temp_path = save_path + '.tmp'
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, save_path)
            
            # Also save a numbered backup
            backup_path = f"{save_path}.e{epoch+1}"
            torch.save(checkpoint, backup_path)
            print(f"Checkpoint saved at epoch {epoch + 1} -> {save_path}")
            print(f"Backup saved -> {backup_path}")
    
    return history