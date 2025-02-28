# main.py
import os
import argparse
import torch
import gc
import json
import pandas as pd
from config import Config, AUTOENCODER_CONFIG, LOW_MEMORY_CONFIG, RTX_4070TI_CONFIG

def clear_memory():
    """Clear memory and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (allocated), "
              f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB (reserved)")

def train_model(config):
    """Train a model with the given configuration."""
    from train import train
    
    # Convert config to argparse namespace
    class Args:
        pass
    
    args = Args()
    
    # Add data parameters
    args.data_csv = config.data['data_csv']
    args.cache_dir = config.data['cache_dir']
    args.batch_size = config.data['batch_size']
    args.train_ratio = config.data['train_ratio']
    args.num_workers = config.data['num_workers']
    
    # Add model parameters
    args.latent_dim = config.model['latent_dim']
    args.striatal_weight = config.model['striatal_weight']
    
    # Add training parameters
    args.learning_rate = config.training['learning_rate']
    args.weight_decay = config.training['weight_decay']
    args.epochs = config.training['epochs']
    args.gradient_clip = config.training['gradient_clip']
    args.gradient_accumulation = config.training['gradient_accumulation']
    args.patience = config.training['patience']
    args.scheduler_metric = config.training['scheduler_metric']
    args.output_dir = config.training['output_dir']
    args.resume = config.training['resume']
    args.cpu = config.training.get('cpu', False)
    
    print("Starting training with the following configuration:")
    print(f"Data CSV: {args.data_csv}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    
    # Train the model
    results = train(args)
    return results

def evaluate_model(config):
    """Evaluate a model with the given configuration."""
    from evaluate import main as evaluate_main
    
    # Convert config to argparse namespace
    class Args:
        pass
    
    args = Args()
    
    # Add evaluation parameters
    args.model_path = config.evaluation['model_path']
    args.data_csv = config.data['data_csv']
    args.output_dir = config.evaluation['output_dir']
    args.batch_size = config.evaluation['batch_size']
    args.num_workers = config.evaluation['num_workers']
    args.cache_dir = config.data['cache_dir']
    args.cpu = config.training.get('cpu', False)
    
    print("Starting evaluation with the following configuration:")
    print(f"Model path: {args.model_path}")
    print(f"Data CSV: {args.data_csv}")
    print(f"Output directory: {args.output_dir}")
    
    # Evaluate the model
    evaluate_main()
    return True

def ensure_dataset(csv_path):
    """Ensure that the dataset CSV exists, if not create it from scan directory."""
    if os.path.exists(csv_path):
        print(f"Dataset CSV found at {csv_path}")
        return True
    
    print(f"Dataset CSV not found at {csv_path}")
    print("Attempting to create it...")
    
    # Ask for base directory
    base_dir = input("Enter the base directory containing PPMI Images folder: ")
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return False
    
    # Import necessary functions
    from dataset import create_efficient_dataloaders
    import pydicom
    import pandas as pd
    
    def collect_files(base_dir):
        """Collect DICOM files from the expected directories."""
        included_files = []
        excluded_files = []
        
        # Define the expected folders and corresponding labels
        expected_folders = {
            "PPMI_Images_PD": "PD",
            "PPMI_Images_SWEDD": "SWEDD",
            "PPMI_Images_Cont": "Control"
        }
        
        # Iterate over immediate subdirectories in base_dir
        for folder_name in expected_folders.keys():
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder_path}")
                
                # Recursively traverse the expected folder
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(".dcm"):
                            full_path = os.path.join(root, file)
                            
                            # Exclude any file with "br_raw" in its full path
                            if "br_raw" in full_path:
                                excluded_files.append(full_path)
                                print(f"Excluding raw file: {full_path}")
                            else:
                                included_files.append((full_path, expected_folders[folder_name]))
            else:
                print(f"Folder not found: {folder_path}")
        
        return included_files, excluded_files
    
    # Collect files
    included_files, excluded_files = collect_files(base_dir)
    
    if not included_files:
        print("No DICOM files found in the expected directories")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(included_files, columns=["file_path", "label"])
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Created dataset CSV at {csv_path} with {len(df)} files")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DATSCAN Autoencoder Project")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    train_parser.add_argument('--preset', type=str, choices=['default', 'rtx4070ti', 'lowmem'],
                             default='rtx4070ti', help='Configuration preset to use if no config file')
    train_parser.add_argument('--data_csv', type=str, help='Path to CSV file with file paths and labels')
    train_parser.add_argument('--output_dir', type=str, help='Output directory for trained models')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    eval_parser.add_argument('--data_csv', type=str, help='Path to CSV file with file paths and labels')
    eval_parser.add_argument('--output_dir', type=str, help='Output directory for evaluation results')
    
    # Docker setup command
    docker_parser = subparsers.add_parser('docker', help='Set up Docker environment')
    docker_parser.add_argument('--output_dir', type=str, default='./datscan-docker',
                              help='Output directory for Docker setup')
    
    # Memory cleanup command
    subparsers.add_parser('cleanup', help='Clear memory caches')
    
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'train':
        # Load configuration
        if args.config and os.path.exists(args.config):
            config = Config(config_file=args.config)
        else:
            # Use preset
            if args.preset == 'rtx4070ti':
                config = RTX_4070TI_CONFIG
            elif args.preset == 'lowmem':
                config = LOW_MEMORY_CONFIG
            else:
                config = AUTOENCODER_CONFIG
        
        # Override with command line arguments
        if args.data_csv:
            config.data['data_csv'] = args.data_csv
        if args.output_dir:
            config.training['output_dir'] = args.output_dir
        if args.resume:
            config.training['resume'] = True
        
        # Ensure dataset exists
        if not ensure_dataset(config.data['data_csv']):
            print("Dataset CSV not found and could not be created. Exiting...")
            return
        
        # Show configuration
        print("Using configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        
        # Train model
        train_model(config)
    
    elif args.command == 'evaluate':
        # Create configuration for evaluation
        config = Config()
        
        # Set evaluation parameters
        config.evaluation['model_path'] = args.model_path
        
        if args.data_csv:
            config.data['data_csv'] = args.data_csv
        
        if args.output_dir:
            config.evaluation['output_dir'] = args.output_dir
        
        # Ensure dataset exists
        if not ensure_dataset(config.data['data_csv']):
            print("Dataset CSV not found and could not be created. Exiting...")
            return
        
        # Evaluate model
        evaluate_model(config)
    
    elif args.command == 'docker':
        # Set up Docker environment
        from docker_setup import setup_docker_environment
        
        class DockerArgs:
            pass
        
        docker_args = DockerArgs()
        docker_args.output_dir = args.output_dir
        
        setup_docker_environment(docker_args)
    
    elif args.command == 'cleanup':
        # Clear memory
        clear_memory()
        print("Memory cleared")

if __name__ == "__main__":
    main()