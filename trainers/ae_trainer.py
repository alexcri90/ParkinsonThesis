#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Autoencoder-specific trainer with reconstruction visualization.
"""

import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoencoderTrainer(BaseTrainer):
    """
    Trainer for Autoencoder model with reconstruction visualization.
    
    Args:
        model: Autoencoder model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object
        criterion: Loss function (default: MSELoss)
        optimizer: Optimizer (default: Adam)
        device: Device to train on (default: cuda if available, else cpu)
    """
    def __init__(self, model, train_loader, val_loader, config, 
                 criterion=None, optimizer=None, device=None):
        super().__init__(model, train_loader, val_loader, config, 
                         criterion, optimizer, device)
        
        # Verify that model is an autoencoder
        assert hasattr(model, 'encode') and hasattr(model, 'decode'), \
            "Model must be an autoencoder with encode() and decode() methods"
    
    def train(self):
        """
        Train the autoencoder and also generate reconstructions.
        
        Returns:
            Training history dictionary
        """
        # Call base train method
        history = super().train()
        
        # Generate and visualize reconstructions
        self._visualize_reconstructions()
        
        # Analyze latent space
        self._analyze_latent_space()
        
        return history
    
    def _visualize_reconstructions(self, num_samples=5):
        """
        Generate and visualize reconstructions from the validation set.
        
        Args:
            num_samples: Number of samples to visualize
        """
        self.model.eval()
        
        # Find samples with different labels if possible
        unique_labels = {}
        visualized_samples = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                volumes = batch['volume'].to(self.device)
                labels = batch['label']
                
                # Generate reconstructions
                reconstructions = self.model(volumes)
                
                # Store samples with different labels
                for i in range(len(volumes)):
                    label = labels[i]
                    if label not in unique_labels and len(unique_labels) < num_samples:
                        unique_labels[label] = {
                            'original': volumes[i].cpu().numpy(),
                            'reconstruction': reconstructions[i].cpu().numpy(),
                            'label': label
                        }
                
                # If we have enough unique labels, break
                if len(unique_labels) >= num_samples:
                    break
                
                # If we can't find enough unique labels, add more samples
                if len(volumes) < num_samples and len(visualized_samples) < num_samples:
                    for i in range(min(num_samples - len(visualized_samples), len(volumes))):
                        visualized_samples.append({
                            'original': volumes[i].cpu().numpy(),
                            'reconstruction': reconstructions[i].cpu().numpy(),
                            'label': labels[i]
                        })
                
                break
        
        # Combine results, prioritizing unique labels
        final_samples = list(unique_labels.values())
        
        # Add more samples if needed
        if len(final_samples) < num_samples:
            needed = num_samples - len(final_samples)
            final_samples.extend(visualized_samples[:needed])
        
        # Create figure
        fig = plt.figure(figsize=(15, 4 * len(final_samples)))
        
        for i, sample in enumerate(final_samples):
            original = sample['original'][0]  # Remove channel dimension
            reconstruction = sample['reconstruction'][0]  # Remove channel dimension
            label = sample['label']
            
            # Get middle slices
            axial_idx = original.shape[0] // 2
            coronal_idx = original.shape[1] // 2
            sagittal_idx = original.shape[2] // 2
            
            # Plot original axial slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 1)
            ax.imshow(original[axial_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Original Axial')
            ax.set_axis_off()
            
            # Plot original coronal slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 2)
            ax.imshow(original[:, coronal_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Original Coronal')
            ax.set_axis_off()
            
            # Plot original sagittal slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 3)
            ax.imshow(original[:, :, sagittal_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Original Sagittal')
            ax.set_axis_off()
            
            # Plot reconstructed axial slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 4)
            ax.imshow(reconstruction[axial_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Reconstructed Axial')
            ax.set_axis_off()
            
            # Plot reconstructed coronal slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 5)
            ax.imshow(reconstruction[:, coronal_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Reconstructed Coronal')
            ax.set_axis_off()
            
            # Plot reconstructed sagittal slice
            ax = plt.subplot(len(final_samples), 6, i*6 + 6)
            ax.imshow(reconstruction[:, :, sagittal_idx], cmap='gray', vmin=0, vmax=4)
            if i == 0:
                ax.set_title('Reconstructed Sagittal')
            ax.set_axis_off()
            
            # Add label information
            plt.figtext(0.01, 0.93 - (i/len(final_samples) * 0.9), f"Label: {label}", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "reconstructions.png"), dpi=200)
        plt.close()
        
        logger.info(f"Reconstruction visualization saved to {os.path.join(self.output_dir, 'reconstructions.png')}")
    
    def _analyze_latent_space(self, max_samples=200):
        """
        Analyze the latent space of the autoencoder.
        
        Args:
            max_samples: Maximum number of samples to use for analysis
        """
        self.model.eval()
        
        # Collect latent vectors and labels
        latent_vectors = []
        labels = []
        paths = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Analyzing latent space"):
                volumes = batch['volume'].to(self.device)
                batch_labels = batch['label']
                batch_paths = batch['path']
                
                # Encode volumes
                latent = self.model.encode(volumes)
                
                # Add to lists
                latent_vectors.append(latent.cpu().numpy())
                labels.extend(batch_labels)
                paths.extend(batch_paths)
                
                # Check if we have enough samples
                if len(labels) >= max_samples:
                    break
        
        # Concatenate latent vectors
        latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
        labels = labels[:max_samples]
        paths = paths[:max_samples]
        
        # Calculate reconstruction errors for each sample
        recon_errors = []
        
        with torch.no_grad():
            for i in range(0, len(latent_vectors), self.val_loader.batch_size):
                batch_latent = torch.tensor(
                    latent_vectors[i:i+self.val_loader.batch_size],
                    device=self.device
                )
                batch_volumes = torch.stack([
                    torch.tensor(self.val_loader.dataset[idx]['volume'])
                    for idx in range(i, min(i+self.val_loader.batch_size, len(latent_vectors)))
                ]).to(self.device)
                
                reconstructions = self.model.decode(batch_latent)
                
                # Calculate MSE
                batch_errors = torch.mean(
                    (reconstructions - batch_volumes) ** 2,
                    dim=(1, 2, 3, 4)
                ).cpu().numpy()
                
                recon_errors.extend(batch_errors)
        
        # Save results
        results = {
            'latent_dim': self.model.get_latent_dim(),
            'recon_errors': recon_errors,
            'labels': labels,
            'paths': paths
        }
        
        with open(os.path.join(self.output_dir, "latent_analysis.json"), 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = {
                'latent_dim': results['latent_dim'],
                'recon_errors': [float(e) for e in results['recon_errors']],
                'labels': results['labels'],
                'paths': results['paths']
            }
            json.dump(serializable_results, f, indent=4)
        
        # Plot reconstruction errors by label
        plt.figure(figsize=(10, 6))
        
        # Group by label
        unique_labels = sorted(set(labels))
        label_errors = {label: [] for label in unique_labels}
        
        for label, error in zip(labels, recon_errors):
            label_errors[label].append(error)
        
        # Create boxplot
        plt.boxplot(
            [label_errors[label] for label in unique_labels],
            labels=unique_labels
        )
        plt.title("Reconstruction Errors by Label")
        plt.xlabel("Label")
        plt.ylabel("Reconstruction Error (MSE)")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(self.output_dir, "recon_errors_by_label.png"))
        plt.close()
        
        # Find outliers
        threshold = np.percentile(recon_errors, 95)  # Top 5% are outliers
        outliers = [(path, label, error) for path, label, error in zip(paths, labels, recon_errors) if error > threshold]
        
        # Sort outliers by error (highest first)
        outliers.sort(key=lambda x: x[2], reverse=True)
        
        # Save outliers to file
        with open(os.path.join(self.output_dir, "outliers.txt"), 'w') as f:
            f.write("Potential outliers (reconstruction error > 95th percentile):\n\n")
            for path, label, error in outliers:
                f.write(f"Path: {path}\nLabel: {label}\nError: {error:.6f}\n\n")
        
        logger.info(f"Latent space analysis completed. Results saved to {self.output_dir}")