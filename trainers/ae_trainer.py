# trainers/ae_trainer.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from trainers.base_trainer import BaseTrainer, AverageMeter

class AutoencoderTrainer(BaseTrainer):
    """Trainer class for autoencoder models"""
    
    def setup(self):
        """Additional setup specific to autoencoder training"""
        self.grad_clip = self.config.get('grad_clip', None)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()
        
        # Create progress bar
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch} [Train]")
        
        # Train loop
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(data)
            loss = self.criterion(reconstruction, data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{losses.avg:.6f}',
                'lr': f'{current_lr:.8f}'
            })
            pbar.update()
            
            # Clear GPU memory periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pbar.close()
        return losses.avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        losses = AverageMeter()
        
        # Create progress bar
        pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                reconstruction, _ = self.model(data)
                loss = self.criterion(reconstruction, data)
                
                # Update metrics
                losses.update(loss.item(), data.size(0))
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{losses.avg:.6f}'})
                pbar.update()
        
        pbar.close()
        return losses.avg
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        mse_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc="Evaluating"):
                data = data.to(self.device)
                reconstruction, _ = self.model(data)
                
                # Calculate MSE
                mse = torch.nn.functional.mse_loss(reconstruction, data).item()
                mse_scores.append(mse)
                
                # Calculate SSIM for middle slices
                for i in range(data.size(0)):
                    orig_mid = data[i, 0, data.size(2)//2].cpu().numpy()
                    recon_mid = reconstruction[i, 0, reconstruction.size(2)//2].cpu().numpy()
                    ssim_score = ssim(orig_mid, recon_mid, data_range=1.0)
                    ssim_scores.append(ssim_score)
        
        results = {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores)
        }
        
        return results
    
    def visualize_reconstructions(self, test_loader, num_samples=5):
        """Visualize original vs reconstructed samples"""
        self.model.eval()
        
        with torch.no_grad():
            # Get samples
            data, _ = next(iter(test_loader))
            data = data[:num_samples].to(self.device)
            reconstructions, _ = self.model(data)
            
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
            
            # Save figure
            plt.savefig(self.model_dir / 'reconstructions.png')
            
            return fig