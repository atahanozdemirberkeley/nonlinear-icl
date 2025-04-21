# src/config.py
import os
import yaml
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
import copy

@dataclass
class ModelConfig:
    n_dims: int = 10                   # Number of input dimensions
    n_positions: int = 100             # Number of positions (context length)
    n_embd: int = 128                  # Size of embeddings
    n_layer: int = 4                   # Number of transformer layers
    n_head: int = 4                    # Number of attention heads
    dropout: float = 0.0               # Dropout rate
    d_model: int = 128                # External name in yaml
    n_layers: int = 4                 # External name in yaml
    n_heads: int = 4                  # External name in yaml
    d_token: int = None               # Add support for d_token parameter used in ICLTransformer
    init_scale: float = 0.02          # Scale for weight initialization
    kernel_type: str = "softmax"       # Type of kernel to use (relu, gelu, softmax)


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3       # Changed to match lr in yaml
    n_batches: int = 1000             # Added to match yaml
    n_epochs: int = 20                # Added to match yaml
    train_steps: int = 10000          # Total number of training steps
    steps_per_epoch: int = 500        # Number of training steps per epoch
    eval_every: int = 100             # How often to evaluate on validation set
    save_every: int = 500             # How often to save checkpoints
    keep_every_steps: int = 2000      # How often to save model snapshots
    patience: int = 5                 # Early stopping patience
    seed: int = 42                    # Random seed
    warmup_steps: int = 1000          # Learning rate warmup steps
    weight_decay: float = 0.01        # Weight decay for optimizer
    grad_clip: float = 1.0            # Gradient clipping value
    n_val_tasks: int = 5              # Number of validation tasks
    n_test_tasks: int = 0             # Number of test tasks
    save_dir: str = None              # Directory to save model checkpoints
    
    # Curriculum learning parameters
    min_points: int = 10              # Minimum number of points to start with
    min_dims: int = 2                 # Minimum number of dimensions to start with
    point_schedule: float = 0.5       # Fraction of training to reach max points
    dim_schedule: float = 0.7         # Fraction of training to reach max dimensions
    num_training_examples: int = None # Limit number of training examples (for seed management)
    num_unique_distributions: int = None # Limit to this many unique distributions (None = unlimited)
    
    
@dataclass
class TaskConfig:
    name: str = "gaussian"            # Changed from task_name to name to match gaussian.yaml
    task_scale: float = 1.0
    n_dims: int = 2                   # Added to match gaussian.yaml
    n_samples_train: int = 10         # Added to match gaussian.yaml
    n_samples_test: int = 10          # Added to match gaussian.yaml
    
    
@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project_name: str = "kernel-icl"
    log_every: int = 100
    

@dataclass
class Config:
    """
    Configuration class that loads from YAML and provides attribute-style access.
    """
    def __init__(self):
        # Model params
        self.model = SimpleNamespace(
            n_dims=10,           # Input dimensionality
            n_positions=10,      # Number of sequence positions (k+1)
            d_token=None,        # Token dimension (if None, use n_dims + 1)
            d_model=128,         # Model hidden dimension
            n_heads=4,           # Number of attention heads
            kernel_type='relu'   # Kernel type: 'relu', 'gelu', or 'softmax'
        )
        
        # Training params
        self.training = SimpleNamespace(
            batch_size=64,               # Batch size for training
            learning_rate=5e-4,          # Learning rate
            train_steps=10000,           # Number of training steps
            min_dims=5,                  # Minimum dimensions for curriculum
            dim_schedule=0.5,            # % of training to reach max dimensions (0 to disable)
            num_unique_distributions=None, # Limit to this many unique distributions (None = unlimited)
            grad_clip=1.0,               # Gradient clipping value (0 to disable)
            n_val_tasks=10,              # Number of validation tasks
            eval_every=500,              # Evaluate every this many steps
            save_every=1000,             # Save checkpoint every this many steps
        )
        
        # Task params
        self.task = SimpleNamespace(
            name='gaussian',      # Task name: 'gaussian', 'linear', 'quadratic', etc.
            task_scale=0.25       # Scale for task weights
        )
        
        # Logging params
        self.logging = SimpleNamespace(
            use_wandb=False,        # Whether to use wandb
            project_name='prob-icl', # Project name for wandb
            log_every=100           # Log to wandb every this many steps
        )
        
        # Output params
        self.output_dir = 'runs'    # Output directory for logs and checkpoints
    
    @classmethod
    def load(cls, path):
        """Load configuration from YAML file."""
        config = cls()
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Using defaults.")
            return config
        
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update config with values from YAML
        for section, params in yaml_config.items():
            if not hasattr(config, section):
                setattr(config, section, SimpleNamespace())
            
            section_config = getattr(config, section)
            
            # Check if params is a dictionary before trying to iterate over it
            if params is not None and isinstance(params, dict):
                for key, value in params.items():
                    # Special case for learning_rate - always convert to float
                    if key == 'learning_rate':
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass
                    # Try to convert string numeric values to proper numeric types
                    elif isinstance(value, str):
                        try:
                            # Try to convert to float (handles both regular decimals and scientific notation)
                            value = float(value)
                            # Convert to int if it's a whole number
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            # If conversion fails, keep as string
                            pass
                    
                    setattr(section_config, key, value)
            elif params is not None:
                # Handle case where params is a string or other non-dict value
                setattr(config, section, params)
        
        return config
    
    def save(self, path):
        """Save configuration to YAML file."""
        # Convert to dictionary
        config_dict = {}
        for section_name in dir(self):
            if section_name.startswith('_'):
                continue
            
            section = getattr(self, section_name)
            if not isinstance(section, SimpleNamespace):
                continue
            
            config_dict[section_name] = {}
            for key in dir(section):
                if key.startswith('_'):
                    continue
                
                value = getattr(section, key)
                if callable(value):
                    continue
                
                config_dict[section_name][key] = value
        
        # Save to file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def copy(self):
        """Create a deep copy of the config."""
        return copy.deepcopy(self)