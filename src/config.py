# src/config.py
import os
import yaml
from dataclasses import dataclass, asdict, field

@dataclass
class ModelConfig:
    n_dims: int = 10
    n_positions: int = 100
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    train_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000
    
    
@dataclass
class TaskConfig:
    task_name: str = "rff"
    input_dim: int = 10
    rff_dim: int = 100
    sigma: float = 0.5
    context_length: int = 10
    
    
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
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.task = TaskConfig(**config_dict.get('task', {}))
        config.logging = LoggingConfig(**config_dict.get('logging', {}))
        config.output_dir = config_dict.get('output_dir', "outputs")
        
        return config