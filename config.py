# config.py
import os
import json
from pathlib import Path

class Config:
    """Configuration class for training and model parameters"""
    
    # Default configurations
    DEFAULTS = {
        # Data settings
        'data': {
            'batch_size': 4,
            'num_workers': 6,  # Optimized for 8-core CPU
            'pin_memory': True,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'preload': False,
            'cache_dir': 'data_cache'
        },
        
        # Model settings
        'model': {
            'name': 'autoencoder',
            'latent_dim': 128
        },
        
        # Training settings
        'training': {
            'num_epochs': 100,
            'learning_rate': 1e-5,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'early_stop_patience': 10
        },
        
        # Scheduler settings
        'scheduler': {
            'type': 'reduce_lr',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-7
        },
        
        # Directories
        'directories': {
            'output_dir': 'trained_models',
            'data_dir': 'data'
        },
        
        # Device settings
        'device': 'cuda',
        
        # Logging and visualization
        'logging': {
            'log_interval': 10,
            'save_interval': 1,
            'tensorboard': True
        }
    }
    
    def __init__(self, config_path=None, **kwargs):
        """
        Initialize configuration
        
        Args:
            config_path: Path to JSON configuration file
            **kwargs: Override default settings
        """
        # Start with default configuration
        self.config = self._nested_dict_copy(self.DEFAULTS)
        
        # Load from file if provided
        if config_path is not None:
            self._load_from_file(config_path)
        
        # Override with kwargs
        self._update_nested_dict(self.config, kwargs)
    
    def _nested_dict_copy(self, d):
        """Create a deep copy of a nested dictionary"""
        if not isinstance(d, dict):
            return d
        return {k: self._nested_dict_copy(v) for k, v in d.items()}
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary with another dictionary"""
        if not isinstance(u, dict):
            return u
        
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _load_from_file(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            self._update_nested_dict(self.config, file_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
    
    def save(self, save_path):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def __getitem__(self, key):
        """Get configuration value by key"""
        if key in self.config:
            return self.config[key]
        
        # Try to handle nested keys (e.g., 'model.latent_dim')
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    raise KeyError(f"Key '{key}' not found in configuration")
            return current
        
        raise KeyError(f"Key '{key}' not found in configuration")
    
    def get(self, key, default=None):
        """Get configuration value with default fallback"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def __setitem__(self, key, value):
        """Set configuration value by key"""
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        self._update_nested_dict(self.config, kwargs)
    
    def as_dict(self):
        """Return configuration as dictionary"""
        return self._nested_dict_copy(self.config)
    
    def __str__(self):
        return json.dumps(self.config, indent=2)
    
    def __repr__(self):
        return self.__str__()


# Create configurations for different models
AUTOENCODER_CONFIG = Config(
    model={'name': 'autoencoder', 'latent_dim': 128},
    training={'learning_rate': 1e-5, 'num_epochs': 100}
)

VAE_CONFIG = Config(
    model={'name': 'vae', 'latent_dim': 128, 'kl_weight': 0.01},
    training={'learning_rate': 5e-5, 'num_epochs': 150}
)