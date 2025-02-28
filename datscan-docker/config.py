# config.py
import os
import json
import argparse

class Config:
    """Configuration class for DATSCAN project."""
    
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON configuration file
            **kwargs: Override configuration parameters
        """
        # Default configuration
        self.data = {
            'data_csv': 'validated_file_paths.csv',
            'cache_dir': 'processed_cache',
            'train_ratio': 0.8,
            'batch_size': 8,
            'num_workers': 4,
            'pin_memory': True
        }
        
        self.model = {
            'name': 'autoencoder',
            'latent_dim': 128,
            'striatal_weight': 5.0
        }
        
        self.training = {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'gradient_accumulation': 1,
            'patience': 15,
            'scheduler_metric': 'striatal',
            'output_dir': './trained_models/autoencoder',
            'resume': False,
            'cpu': False
        }
        
        self.evaluation = {
            'model_path': None,
            'output_dir': './evaluation_results',
            'batch_size': 8,
            'num_workers': 4
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with kwargs
        self.update(**kwargs)
    
    def load_from_file(self, config_file):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update configuration
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(getattr(self, section), dict):
                getattr(self, section).update(values)
            else:
                setattr(self, section, values)
    
    def save_to_file(self, output_file):
        """Save configuration to JSON file."""
        config_data = {
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'evaluation': self.evaluation
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def update(self, **kwargs):
        """Update configuration with provided values."""
        for key, value in kwargs.items():
            parts = key.split('.')
            if len(parts) == 2:
                section, param = parts
                if hasattr(self, section) and isinstance(getattr(self, section), dict):
                    getattr(self, section)[param] = value
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'evaluation': self.evaluation
        }
    
    def get_parser(self):
        """Create argument parser with configuration options."""
        parser = argparse.ArgumentParser(description="DATSCAN autoencoder training and evaluation")
        
        # Configuration file
        parser.add_argument('--config', type=str, help='Path to configuration file')
        
        # Data parameters
        parser.add_argument('--data.data_csv', type=str, help='Path to CSV file with file paths and labels')
        parser.add_argument('--data.cache_dir', type=str, help='Directory to cache processed volumes')
        parser.add_argument('--data.train_ratio', type=float, help='Ratio of training data')
        parser.add_argument('--data.batch_size', type=int, help='Batch size for training')
        parser.add_argument('--data.num_workers', type=int, help='Number of workers for data loading')
        
        # Model parameters
        parser.add_argument('--model.name', type=str, help='Model name (autoencoder)')
        parser.add_argument('--model.latent_dim', type=int, help='Dimension of latent space')
        parser.add_argument('--model.striatal_weight', type=float, help='Weight factor for striatal regions in loss')
        
        # Training parameters
        parser.add_argument('--training.epochs', type=int, help='Number of epochs to train')
        parser.add_argument('--training.learning_rate', type=float, help='Learning rate')
        parser.add_argument('--training.weight_decay', type=float, help='Weight decay for optimizer')
        parser.add_argument('--training.gradient_clip', type=float, help='Gradient clipping value')
        parser.add_argument('--training.gradient_accumulation', type=int, help='Number of batches to accumulate gradients')
        parser.add_argument('--training.patience', type=int, help='Patience for early stopping')
        parser.add_argument('--training.scheduler_metric', type=str, help='Metric to monitor for scheduler')
        parser.add_argument('--training.output_dir', type=str, help='Output directory for trained models')
        parser.add_argument('--training.resume', action='store_true', help='Resume training from latest checkpoint if available')
        parser.add_argument('--training.cpu', action='store_true', help='Force using CPU even if GPU is available')
        
        # Evaluation parameters
        parser.add_argument('--evaluation.model_path', type=str, help='Path to the trained model checkpoint')
        parser.add_argument('--evaluation.output_dir', type=str, help='Output directory for evaluation results')
        parser.add_argument('--evaluation.batch_size', type=int, help='Batch size for evaluation')
        parser.add_argument('--evaluation.num_workers', type=int, help='Number of workers for data loading during evaluation')
        
        return parser
    
    def from_args(self, args):
        """Update configuration from parsed arguments."""
        # Load from config file if provided
        if args.config:
            self.load_from_file(args.config)
        
        # Update from arguments
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None and key != 'config':
                self.update(**{key: value})
        
        return self

# Default configurations for different use cases
DEFAULT_CONFIG = Config()

AUTOENCODER_CONFIG = Config(
    model={'name': 'autoencoder', 'latent_dim': 128, 'striatal_weight': 5.0},
    training={'learning_rate': 3e-5, 'epochs': 200, 'batch_size': 8, 'gradient_accumulation': 4}
)

# Configuration optimized for limited RAM
LOW_MEMORY_CONFIG = Config(
    data={'batch_size': 4, 'num_workers': 2},
    model={'latent_dim': 64, 'striatal_weight': 5.0},
    training={'gradient_accumulation': 8, 'learning_rate': 1e-4}
)

# Configuration optimized for RTX 4070 Ti
RTX_4070TI_CONFIG = Config(
    data={'batch_size': 16, 'num_workers': 6},
    model={'latent_dim': 128, 'striatal_weight': 5.0},
    training={'gradient_accumulation': 2, 'learning_rate': 3e-5}
)