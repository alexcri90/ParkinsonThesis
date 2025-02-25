# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import logging
from pathlib import Path
import json
import pandas as pd
import gc

# Import modules
from dataset import DATSCANDataset
from models.autoencoder import Autoencoder
from trainers.ae_trainer import AutoencoderTrainer
from config import AUTOENCODER_CONFIG, VAE_CONFIG, Config

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def create_dataloaders(config):
    """Create data loaders for training, validation, and testing"""
    # Load file paths
    df = pd.read_csv("validated_file_paths.csv")
    
    # Create dataset
    dataset = DATSCANDataset(
        df['file_path'].tolist(), 
        df['label'].tolist(),
        preload=config['data'].get('preload', False),
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader

def train_autoencoder(config):
    """Train an autoencoder model"""
    logger = logging.getLogger("train_autoencoder")
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = Autoencoder(
        latent_dim=config['model']['latent_dim'],
        name=config['model']['name']
    ).to(device)
    
    logger.info(f"Created {model.name} model with {model.count_parameters():,} parameters")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(f"Data loaded. Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config
    )
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train(config['training']['num_epochs'])
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate(test_loader)
    logger.info(f"Test results: MSE={eval_results['mse_mean']:.6f} (±{eval_results['mse_std']:.6f}), "
                f"SSIM={eval_results['ssim_mean']:.6f} (±{eval_results['ssim_std']:.6f})")
    
    # Save evaluation results
    with open(trainer.model_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    trainer.visualize_reconstructions(test_loader)
    
    # Clean up
    del model, optimizer, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results, eval_results

def main():
    """Main function to parse arguments and run training"""
    parser = argparse.ArgumentParser(description='Train models for DATSCAN imaging')
    parser.add_argument('--model', type=str, default='autoencoder',
                        choices=['autoencoder', 'vae'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension size')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for trained models')
    parser.add_argument('--name', type=str, default=None,
                    help='Custom name for the model')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger("main")
    
    # Get configuration based on model type
    if args.model == 'autoencoder':
        config = AUTOENCODER_CONFIG.as_dict()
    elif args.model == 'vae':
        config = VAE_CONFIG.as_dict()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Load from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            for key, value in file_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    
    # Override with command-line arguments
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.latent_dim:
        config['model']['latent_dim'] = args.latent_dim
    if args.output_dir:
        config['directories']['output_dir'] = args.output_dir
    if args.name:
        config['model']['name'] = args.name
    
    # Print configuration
    logger.info(f"Training {args.model} with configuration:")
    logger.info(json.dumps(config, indent=2))
    
    # Train model
    try:
        if args.model == 'autoencoder':
            results, eval_results = train_autoencoder(config)
            
            # Print results
            logger.info(f"Training completed! Best validation loss: {results['best_val_loss']:.6f}")
            logger.info(f"Test results: MSE={eval_results['mse_mean']:.6f}, SSIM={eval_results['ssim_mean']:.6f}")
            
        elif args.model == 'vae':
            # TODO: Implement VAE training
            logger.info("VAE training not yet implemented")
        
        else:
            logger.error(f"Unknown model type: {args.model}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()