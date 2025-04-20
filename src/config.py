# src/config.py
import os
import yaml
from dataclasses import dataclass, asdict, field

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


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3       # Changed to match lr in yaml
    n_batches: int = 1000             # Added to match yaml
    n_epochs: int = 20                # Added to match yaml
    train_steps: int = 10000          # Kept for backward compatibility
    eval_every: int = 100             # Changed to match yaml
    save_every: int = 500             # Changed to match yaml 
    patience: int = 5
    seed: int = 42
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    
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
    # Use field with default_factory to avoid mutable default issue
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "outputs"
    
    def save(self, path):
        """Save config to a YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path):
        """Load config from a YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        config = cls()
        
        # Load model config directly - no need for field name mapping anymore
        # We now exactly match the parameter names from Greg Yang's implementation
        config.model = ModelConfig(**config_dict.get('model', {}))
        
        # Map train to training for backwards compatibility
        train_dict = config_dict.get('train', config_dict.get('training', {}))
        if 'lr' in train_dict:
            train_dict['learning_rate'] = train_dict['lr']
            train_dict.pop('lr')  # Remove the 'lr' key to avoid unexpected keyword argument error
            
        config.training = TrainingConfig(**train_dict)
        
        # Load task config
        config.task = TaskConfig(**config_dict.get('task', {}))
        
        # Load logging config
        config.logging = LoggingConfig(**config_dict.get('logging', {}))
        
        config.output_dir = config_dict.get('output_dir', "outputs")
        
        return config