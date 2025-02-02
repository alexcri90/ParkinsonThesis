#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import your VAE model and dataloader creation functions.
from model import DATScanVAE  # Ensure this file contains one clean definition of DATScanVAE.
from preprocessing import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train DATScan VAE')
    parser.add_argument('--csv', type=str, default='dicom_file_paths.csv',
                        help='Path to CSV file with image file paths and labels')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128],
                        help='Target shape for preprocessing (Z Y X)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Total number of epochs')
    parser.add_argument('--checkpoint', type=str, default='datscan_vae_checkpoint.pt',
                        help='Path to save checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    return parser.parse_args()

def load_data(csv_path, target_shape, batch_size, num_workers=2):
    # Load CSV (assumes same structure as your Notebook DataFrame)
    df = pd.read_csv(csv_path)
    dataloaders = create_dataloaders(
        df=df,
        batch_size=batch_size,
        target_shape=tuple(target_shape),
        normalize_method='minmax',
        apply_brain_mask=True,
        augment=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_workers=num_workers
    )
    # Combine all groups into one dataset
    combined_dataset = torch.utils.data.ConcatDataset([d.dataset for d in dataloaders.values()])
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return train_loader, df

def train_vae(model, train_loader, optimizer, scheduler, device, start_epoch, num_epochs, checkpoint_path):
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    history = {'loss': [], 'kl_loss': [], 'recon_loss': []}
    annealing_epochs = 10  # KL annealing duration
    max_grad_norm = 0.5

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        kl_weight = min((epoch + 1) / annealing_epochs, 1.0)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for data in pbar:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_logits, mu, logvar = model(data)
            mu = torch.clamp(mu, -10, 10)
            logvar = torch.clamp(logvar, -10, 10)
            recon_loss = criterion(recon_logits, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_kl_loss = epoch_kl_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)
        history['loss'].append(avg_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['recon_loss'].append(avg_recon_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f} | KL: {avg_kl_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL Weight: {kl_weight:.4f}")

        # Save checkpoint every 10 epochs and at the last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'kl_weight': kl_weight
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    return history

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    train_loader, df = load_data(args.csv, args.target_shape, args.batch_size)
    
    # Initialize model; use latent_dim from your design (e.g., 128)
    model = DATScanVAE(latent_dim=128).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    start_epoch = 0
    history = None
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', None)
        print(f"Resuming at epoch {start_epoch}")
    else:
        if args.resume:
            print("Checkpoint not found. Starting from scratch.")

    history = train_vae(model, train_loader, optimizer, scheduler, device, start_epoch, args.epochs, args.checkpoint)
    print("Training complete.")

if __name__ == '__main__':
    main()
