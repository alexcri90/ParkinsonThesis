# train.py
import os
import time
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Import custom modules
from dataset import create_efficient_dataloaders
from autoencoder import Autoencoder, RegionWeightedMSELoss, StriatalMSELoss

def clear_memory():
    """Clear memory and cache to free up resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (allocated), "
              f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB (reserved)")

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('training')

def train(args):
    """Train an autoencoder model."""
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting training with arguments: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    
    # Load data
    if args.data_csv and os.path.exists(args.data_csv):
        df = pd.read_csv(args.data_csv)
        logger.info(f"Loaded data CSV with {len(df)} entries")
    else:
        raise FileNotFoundError(f"Data CSV file not found: {args.data_csv}")
    
    # Create dataloaders
    train_loader, val_loader = create_efficient_dataloaders(
        df=df,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=not args.cpu
    )
    logger.info(f"Created dataloaders with batch size {args.batch_size}")
    
    # Create model
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    logger.info(f"Created autoencoder with latent dimension {args.latent_dim}")
    
    # Create loss functions
    criterion = RegionWeightedMSELoss(
        striatal_weight=args.striatal_weight,
        shape=(64, 128, 128),
        device=device
    )
    striatal_criterion = StriatalMSELoss(
        shape=(64, 128, 128),
        device=device
    )
    logger.info(f"Created loss functions with striatal weight {args.striatal_weight}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Created optimizer with learning rate {args.learning_rate}")
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    logger.info("Created learning rate scheduler")
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    best_striatal_loss = float('inf')
    train_losses = []
    val_losses = []
    striatal_losses = []
    patience_counter = 0
    
    # Load checkpoint if resuming training
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            striatal_losses = checkpoint['striatal_losses']
            best_val_loss = checkpoint['best_val_loss']
            best_striatal_loss = checkpoint['best_striatal_loss']
            patience_counter = checkpoint['patience_counter']
            logger.info(f"Resumed training from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            train_striatal_loss = 0
            
            # Initialize progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
            
            # Accumulation counter
            accum_iter = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_pbar):
                # Move batch to device
                volumes = batch['volume'].to(device)
                
                # Forward pass
                reconstructions, _ = model(volumes)
                
                # Calculate losses
                loss = criterion(reconstructions, volumes)
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                accum_iter += 1
                if accum_iter == args.gradient_accumulation or batch_idx == len(train_loader) - 1:
                    # Gradient clipping
                    if args.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_iter = 0
                
                # Calculate striatal loss for monitoring (detached from computation graph)
                with torch.no_grad():
                    striatal_loss = striatal_criterion(reconstructions, volumes)
                
                # Update metrics
                train_loss += loss.item() * args.gradient_accumulation
                train_striatal_loss += striatal_loss.item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation:.6f}',
                    'striatal': f'{striatal_loss.item():.6f}'
                })
                
                # Clear memory
                del volumes, reconstructions, loss, striatal_loss
                if batch_idx % 5 == 0:  # Clear cache periodically
                    torch.cuda.empty_cache()
            
            # Calculate average training losses
            avg_train_loss = train_loss / len(train_loader)
            avg_train_striatal_loss = train_striatal_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_striatal_loss = 0
            
            # Initialize progress bar
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_pbar:
                    # Move batch to device
                    volumes = batch['volume'].to(device)
                    
                    # Forward pass
                    reconstructions, _ = model(volumes)
                    
                    # Calculate losses
                    loss = criterion(reconstructions, volumes)
                    striatal_loss = striatal_criterion(reconstructions, volumes)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_striatal_loss += striatal_loss.item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'striatal': f'{striatal_loss.item():.6f}'
                    })
                    
                    # Clear memory
                    del volumes, reconstructions, loss, striatal_loss
            
            # Calculate average validation losses
            avg_val_loss = val_loss / len(val_loader)
            avg_val_striatal_loss = val_striatal_loss / len(val_loader)
            
            # Update learning rate scheduler
            if args.scheduler_metric == 'striatal':
                scheduler.step(avg_val_striatal_loss)
            else:
                scheduler.step(avg_val_loss)
            
            # Store losses
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            striatal_losses.append(avg_val_striatal_loss)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Log results
            logger.info(f"Epoch {epoch+1}/{args.epochs} completed in {elapsed_time:.2f}s")
            logger.info(f"Train Loss: {avg_train_loss:.6f}, Striatal: {avg_train_striatal_loss:.6f}")
            logger.info(f"Val Loss: {avg_val_loss:.6f}, Striatal: {avg_val_striatal_loss:.6f}")
            
            # Save latest checkpoint
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'striatal_losses': striatal_losses,
                'best_val_loss': best_val_loss,
                'best_striatal_loss': best_striatal_loss,
                'patience_counter': patience_counter
            }, checkpoint_path)
            
            # Check if this is the best model by validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_val.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'striatal_losses': striatal_losses,
                    'best_val_loss': best_val_loss,
                    'best_striatal_loss': best_striatal_loss
                }, best_checkpoint_path)
                logger.info(f"Saved best model (val loss) at epoch {epoch+1}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check if this is the best model by striatal loss
            if avg_val_striatal_loss < best_striatal_loss:
                best_striatal_loss = avg_val_striatal_loss
                best_striatal_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_striatal.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'striatal_losses': striatal_losses,
                    'best_val_loss': best_val_loss,
                    'best_striatal_loss': best_striatal_loss
                }, best_striatal_checkpoint_path)
                logger.info(f"Saved best model (striatal loss) at epoch {epoch+1}")
            
            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Clear memory at the end of each epoch
            clear_memory()
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'checkpoints', 'final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': args.latent_dim,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'striatal_losses': striatal_losses,
        'best_val_loss': best_val_loss,
        'best_striatal_loss': best_striatal_loss
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(striatal_losses, label='Striatal Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Striatal Region Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'learning_curves.png'))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'striatal_losses': striatal_losses,
        'best_val_loss': best_val_loss,
        'best_striatal_loss': best_striatal_loss
    }

def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train autoencoder for DATSCAN images")
    
    # Data parameters
    parser.add_argument('--data_csv', type=str, required=True, 
                        help='Path to CSV file with file paths and labels')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache processed volumes (optional)')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of latent space (default: 128)')
    parser.add_argument('--striatal_weight', type=float, default=5.0,
                        help='Weight factor for striatal regions in loss (default: 5.0)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer (default: 1e-5)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0, 0 to disable)')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of batches to accumulate gradients (default: 1)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping (default: 15, 0 to disable)')
    parser.add_argument('--scheduler_metric', type=str, default='striatal',
                        choices=['val', 'striatal'],
                        help='Metric to monitor for scheduler (default: striatal)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models/autoencoder',
                        help='Output directory for trained models (default: ./trained_models/autoencoder)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint if available')
    
    # Other parameters
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()