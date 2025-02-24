# utils.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def process_volume(volume, target_shape=(64, 128, 128)):
    """Process a 3D volume with normalization and resizing."""
    # Convert to tensor if necessary
    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume)
    
    # Ensure float32
    volume = volume.float()
    
    # Normalize
    volume = volume - volume.min()
    if volume.max() > 0:
        volume = volume / volume.max()
    
    # Create anatomical mask for normalization
    mask = torch.zeros_like(volume)
    mask[20:40, 82:103, 43:82] = 1
    
    # Normalize based on anatomical region
    mask_mean = volume[mask.bool()].mean()
    if mask_mean > 0:
        volume = volume / mask_mean
    
    # Add channel dimension
    volume = volume.unsqueeze(0)
    
    return volume

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.size(0))
            
            # Update progress bar
            pbar.set_postfix({'train_loss': f'{losses.avg:.4f}'})
            
            # Clear GPU memory periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    return losses.avg

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            reconstruction, _ = model(data)
            loss = criterion(reconstruction, data)
            losses.update(loss.item(), data.size(0))
    
    return losses.avg

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs, checkpoint_dir, log_dir):
    """Complete training loop with checkpointing and logging"""
    
    # Create directories if they don't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        filename=Path(log_dir) / 'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=7,
        path=Path(checkpoint_dir) / 'best_model.pt'
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log progress
        message = f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}'
        print(message)
        logging.info(message)
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            logging.info("Early stopping triggered")
            break
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, Path(checkpoint_dir) / 'best_model.pt')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_reconstruction(model, test_loader, device):
    """Evaluate model reconstruction quality using MSE and SSIM"""
    model.eval()
    mse_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, _ = model(data)
            
            # Calculate MSE
            mse = nn.MSELoss()(reconstruction, data).item()
            mse_scores.append(mse)
            
            # Calculate SSIM for middle slices
            for i in range(data.size(0)):
                orig_mid = data[i, 0, data.size(2)//2].cpu().numpy()
                recon_mid = reconstruction[i, 0, reconstruction.size(2)//2].cpu().numpy()
                ssim_score = ssim(orig_mid, recon_mid, data_range=1.0)
                ssim_scores.append(ssim_score)
    
    return {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores)
    }

def visualize_reconstructions(model, test_loader, device, num_samples=5):
    """Visualize original vs reconstructed samples"""
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # Get samples
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        reconstructions, _ = model(data)
        
        # Plot comparisons
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
        
        for i in range(num_samples):
            # Plot original
            slice_idx = data.size(2)//2
            axes[i, 0].imshow(data[i, 0, slice_idx].cpu(), cmap='gray')
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')
            
            # Plot reconstruction
            axes[i, 1].imshow(reconstructions[i, 0, slice_idx].cpu(), cmap='gray')
            axes[i, 1].set_title(f'Reconstruction {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        return fig