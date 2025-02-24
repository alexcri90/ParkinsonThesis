# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

from model import Autoencoder
from dataset import DATSCANDataset
from utils import train_model, evaluate_reconstruction, visualize_reconstructions

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 100
    LATENT_DIM = 256
    NUM_WORKERS = 4  # Adjust based on your CPU
    
    # Create directories for outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'training_output_{timestamp}')
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameters
    hyperparameters = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'latent_dim': LATENT_DIM,
        'num_workers': NUM_WORKERS
    }
    with open(output_dir / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    try:
        # Load data
        print("Loading data...")
        df = pd.read_csv("validated_file_paths.csv")
        
        # Create dataset
        dataset = DATSCANDataset(df['file_path'].tolist(), df['label'].tolist())
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        print(f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Initialize model
        print("Initializing model...")
        model = Autoencoder(latent_dim=LATENT_DIM).to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        print("Starting training...")
        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(training_results['train_losses'], label='Train Loss')
        plt.plot(training_results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(output_dir / 'loss_curves.png')
        plt.close()
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pt')['model_state_dict'])
        
        # Evaluate model
        print("Evaluating model...")
        evaluation_results = evaluate_reconstruction(model, test_loader, device)
        
        # Save evaluation results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        # Visualize reconstructions
        print("Generating visualization...")
        fig = visualize_reconstructions(model, test_loader, device)
        fig.savefig(output_dir / 'reconstructions.png')
        plt.close()
        
        print("Training completed!")
        print(f"Results saved in: {output_dir}")
        print("\nEvaluation Results:")
        print(f"MSE: {evaluation_results['mse_mean']:.4f} (±{evaluation_results['mse_std']:.4f})")
        print(f"SSIM: {evaluation_results['ssim_mean']:.4f} (±{evaluation_results['ssim_std']:.4f})")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()