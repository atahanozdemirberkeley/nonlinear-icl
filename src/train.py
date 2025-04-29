# src/train.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import torch.nn.functional as F
import random

from model import GPT2ICLModel
from custom_model import CustomICLModel
from tasks import TaskSampler
from config import Config
from utils import set_random_seed


def create_prompt_sequence(xs, ys, config):
    """Create prompt sequence for the model."""
    # Get the expected token dimensionality from config
    d_token = config.model.d_token if config.model.d_token is not None else config.model.n_dims + 1
    
    # Ensure ys has the right shape (batch_size, seq_length, 1)
    if ys.dim() == 2:
        ys = ys.unsqueeze(-1)
    
    # Calculate total padding needed to reach d_token
    padding_size = d_token - (xs.shape[2] + 1)
    
    if padding_size > 0:
        # Add padding to match the expected d_token size
        padding = torch.zeros(xs.shape[0], xs.shape[1], padding_size, device=xs.device)
        
        # Concatenate xs, ys, and padding along the feature dimension
        prompt_seq = torch.cat([xs, ys, padding], dim=2)
    else:
        # Simple concatenation if no additional padding needed
        prompt_seq = torch.cat([xs, ys], dim=2)

    return prompt_seq

def compute_loss_all_prefixes(model, xs, ys, config):
    """
    Compute average MSE across all prefix lengths without leaking the
    current target label into the prompt.
    """
    # Ensure ys has shape (B, L, 1)
    if ys.dim() == 2:
        ys = ys.unsqueeze(-1)

    losses = []
    model_type = getattr(config.model, "model_type", "gpt2")
    seq_len = xs.shape[1]

    for i in range(1, seq_len):
        # ----- build prefix inputs -----
        prefix_xs = xs[:, : i + 1]                  # (B, i+1, D_x)

        # context labels y₀ … y_{i‑1}, plus dummy at position i
        dummy     = torch.zeros_like(ys[:, :1])     # (B, 1, 1)
        prefix_ys = torch.cat([ys[:, :i], dummy], dim=1)  # (B, i+1, 1)

        prompt = create_prompt_sequence(prefix_xs, prefix_ys, config)

        # ----- model forward -----
        if model_type == "mlp":
            prediction = model(prompt)[:, i]        # predict y_i
        else:                                       # transformer/gpt2
            prediction = model(prompt)[:, -1]       # last position

        # ----- loss for this prefix -----
        target = ys[:, i].squeeze(-1)               # (B,)
        loss   = F.mse_loss(prediction.squeeze(-1), target)
        losses.append(loss)

    return torch.stack(losses).mean()

class Curriculum:
    """Curriculum learning for gradually increasing task difficulty (dimensions only)."""
    def __init__(self, config):
        self._min_dims = config.training.min_dims
        self._max_dims = config.model.n_dims
        self.n_dims = self._min_dims
        self._total_steps = config.training.train_steps
        # get the maximum at total_steps * dim_schedule
        self._dim_schedule = config.training.dim_schedule
    
    def update(self, step):
        """Update curriculum based on current step (only for dimensions)."""
        # Update number of dimensions
        if self._dim_schedule > 0:
            dim_progress = min(1.0, step / (self._total_steps * self._dim_schedule))
            self.n_dims = min(
                self._max_dims, 
                int(self._min_dims + (self._max_dims - self._min_dims) * dim_progress)
            )

def sample_seeds(total_seeds, count):
    """Sample unique random seeds from 0 to total_seeds-1."""
    seeds = set()
    while len(seeds) < count:
        seeds.add(random.randint(0, total_seeds - 1))
    return list(seeds)


def train_model(config):
    """
    Train an in-context learning model with curriculum learning.
    """

    set_random_seed(config.training.seed)
    print("--> train_model(): Random seed to ", config.training.seed)

    output_dir = os.path.join(
        config.output_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)
    print("--> train_model(): Output directory: ", output_dir)
    config.save(os.path.join(output_dir, "config.yaml"))
    print("--> train_model(): Config saved to ", os.path.join(output_dir, "config.yaml"))
    
    if config.logging.use_wandb:
        assert os.environ.get("WANDB_API_KEY", None) is not None, (
            "Please set the WANDB_API_KEY environment variable to use wandb logging. "
            "You can get your API key from https://wandb.ai/"
        )
        wandb.init(
            project=config.logging.project_name,
            config=vars(config),
            name= (
                f"task={config.task.name}_scale={config.task.task_scale}_"
                f"xrange={config.task.x_range}_freqmin={config.task.freq_min}_"
                f"freqmax={config.task.freq_max}_nlayer={config.model.n_layer}_"
                f"npos={config.model.n_positions}_ndims={config.model.n_dims}_"
                f"dmodel={config.model.d_model}_nheads={config.model.n_heads}_"
                f"lr={config.training.learning_rate}_batch={config.training.batch_size}_"
                f"trainsteps={config.training.train_steps}"
            ),
        )
    else:
        print("--> train_model(): Wandb logging is disabled. Set --logging.use_wandb True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> train_model(): using device {device}")
    
    # d_token is the input token dimension, and this is different from the d_model.
    # we will have a learnable lienar projection from d_token to d_model (> d_token)
    # d_token is n_dims + 1, becuase it is concat. of input (n_dims) and output (1)
    if config.model.d_token is None:
        config.model.d_token = config.model.n_dims + 1
    assert config.model.d_token == config.model.n_dims + 1, "d_token should be n_dims + 1"
    
    # Load the model
    model_type = getattr(config.model, 'model_type', None)
    assert model_type is not None, "Model type must be specified in the config."
    if model_type == 'gpt2':        
        model = GPT2ICLModel(
            d_token=config.model.d_token,
            n_positions=config.model.n_positions,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layer=config.model.n_layer,
        ).to(device)
        print(f"--> train_model(): Using GPT2-style model with {config.model.n_layer} layers")

    elif model_type == "custom_transformer":
        model = CustomICLModel(
            d_token=config.model.d_token,
            d_model=config.model.d_model,
            d_ff=4*config.model.d_model,
            n_layers=config.model.n_layer,
            n_head=config.model.n_heads,
            d_qkv=config.model.d_model // config.model.n_heads,
            dropout_attn=config.model.dropout_attn,
            dropout_ffn=config.model.dropout_ffn,
            nonlin="SiLU",
        ).to(device)
        print(f"--> train_model(): Using custom transformer model.")

    elif model_type == 'mlp':
        from model import MLPICLModel
        hidden_dim = getattr(config.model, 'hidden_dim', 256)
        n_layers = getattr(config.model, 'n_layers', 4)        
        model = MLPICLModel(
            d_token=config.model.d_token,
            n_positions=config.model.n_positions,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        ).to(device)
        print(f"--> train_model(): Using MLP model with {n_layers} layers and {hidden_dim} hidden dimensions")

    else:
        raise ValueError(f"--> train_model(): Unknown model type {model_type}")
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Initialize optimizer
    learning_rate = float(config.training.learning_rate)
    weight_decay = float(config.training.weight_decay) if hasattr(config.training, 'weight_decay') else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create validation tasks using task_sampler
    print(f"Creating {config.training.n_val_tasks} validation tasks with {config.model.n_dims} dimensions")
    task_sampler = TaskSampler(config)
    val_tasks = [task_sampler() for _ in range(config.training.n_val_tasks)]

    # Training state
    curriculum = Curriculum(config)
    starting_step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    steps = []
     
    # Training loop
    pbar = tqdm(range(starting_step, config.training.train_steps))
    
    for step in pbar:
        model.train()
        
        # # Get task sampler for current curriculum dimensions
        # task_kwargs = {
        #     "scale": getattr(config.task, 'task_scale', 0.25),
        #     "num_tasks": getattr(config.training, 'pool_size', None)  # Support task pools
        # }
        
        # # Add task-specific parameters
        # if config.task.name == 'kernel_rff':
        #     task_kwargs["lengthscale"] = getattr(config.task, 'lengthscale', 1.0)
        #     task_kwargs["rff_dim"] = getattr(config.task, 'rff_dim', 128)
        
        import pdb; pdb.set_trace()
        # get attributes from config, except for those starting with "_"

        current_config = config.update_from_args(curriculum.
        import pdb; pdb.set_trace()        
        
        # Handle seeds like in the original repo
        task_args = {}
        if num_unique_distributions is not None:
            # Sample seeds for this batch
            seeds = sample_seeds(num_unique_distributions, config.training.batch_size)
            # Use seeds directly
            task_args["seeds"] = seeds
        
        # Create new task for this iteration
        task = task_sampler(**task_args)
        
        # Sample data with full number of points
        xs, ys = task.sample(config.training.batch_size, curriculum.n_points)
        xs, ys = xs.to(device), ys.to(device)
        
        # Create prompt sequence for in-context learning
        prompt_seq = create_prompt_sequence(xs, ys, config)

        # Forward pass and compute loss
        optimizer.zero_grad()
        loss = compute_loss_all_prefixes(model, xs, ys, config)
        
        # Save training loss for plotting
        train_losses.append(loss.item())
        steps.append(step)
        
        # Backward pass and optimization
        loss.backward()
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        
        # Update curriculum
        curriculum.update(step)
        
        # Log to wandb
        if config.logging.use_wandb and step % config.logging.log_every == 0:
            wandb.log({
                "train/loss": loss.item(),
                "n_dims": curriculum.n_dims_truncated,
                "step": step,
            })
        
        # Update progress bar
        pbar.set_description(f"loss {loss.item():.6f}, dims {curriculum.n_dims_truncated}")
        
        # Validation and checkpoint saving
        if step % config.training.eval_every == 0 or step == config.training.train_steps - 1:
            model.eval()
            val_losses_for_step = []
            
            # Generate fresh validation tasks each time
            temp_val_tasks = []
            for i in range(config.training.n_val_tasks):
                try:
                    temp_val_task = val_task_sampler()
                    temp_val_tasks.append(temp_val_task)
                except Exception as e:
                    print(f"Error creating validation task {i}: {e}")
            
            with torch.no_grad():
                # Use the fresh validation tasks
                for val_task in temp_val_tasks:
                    # Sample data from validation task
                    val_xs, val_ys = val_task.sample(config.training.batch_size, config.model.n_positions)
                    val_xs, val_ys = val_xs.to(device), val_ys.to(device)
                    
                    # Compute validation loss
                    val_loss = compute_loss_all_prefixes(model, val_xs, val_ys, config)
                    val_losses_for_step.append(val_loss.item())
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses_for_step)
            val_losses.append(avg_val_loss)
            
            # Log validation results
            print(f"Step {step}, Val Loss: {avg_val_loss:.6f}")
            
            if config.logging.use_wandb:
                wandb.log({
                    "val/loss": avg_val_loss,
                    "step": step,
                })
            
            # Save model if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                print(f"New best model saved (val_loss: {avg_val_loss:.6f})")
            
        # Save checkpoint periodically
        if step % config.training.save_every == 0 and step > 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }, os.path.join(output_dir, "checkpoint.pt"))
    
    # Plot and save training curves
    plt.figure(figsize=(10, 6))
    
    # Plot train loss with actual step numbers
    plt.plot(steps, train_losses, label="Train Loss")
    
    # Fix for validation steps calculation
    # If we have 0 in the validation steps, we need to handle it specially
    val_step_indices = [i for i in range(len(steps)) if i % config.training.eval_every == 0 or i == len(steps) - 1]
    val_steps = [steps[i] for i in val_step_indices if i < len(steps)][:len(val_losses)]
    
    # Make sure val_steps and val_losses have the same length
    if len(val_steps) > len(val_losses):
        val_steps = val_steps[:len(val_losses)]
    elif len(val_steps) < len(val_losses):
        val_losses = val_losses[:len(val_steps)]
        
    # Plot val loss with actual step numbers
    plt.plot(val_steps, val_losses, label="Val Loss", marker="o")
    
    # Set x-axis to show a reasonable number of step values
    if steps:
        # Only show around 10 tick marks regardless of how many steps there are
        max_step = max(steps)
        step_size = max(1, max_step // 10)
        tick_positions = list(range(0, max_step + 1, step_size))
        plt.xticks(tick_positions)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{config.task.name} (d={config.model.n_dims})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    # Save loss data for reproducibility
    loss_data = {
        'steps': steps,
        'train_losses': train_losses,
        'val_steps': val_steps,
        'val_losses': val_losses
    }
    torch.save(loss_data, os.path.join(output_dir, "loss_data.pt"))
    
    # Close wandb
    if config.logging.use_wandb:
        wandb.finish()
    
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description="Train ICL model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')    
    args, additional = parser.parse_known_args()
    
    # Load configuration from file and override with CLI arguments
    config = Config.load(args.config)
    print(f"Loaded configuration from {args.config}")
    config.update_from_args(additional)
    
    # Always use full position length (k+1) to match the mathematical objective
    config.training.min_points = config.model.n_positions
    config.training.point_schedule = 0
    print(f"Using full sequence length of {config.model.n_positions} positions for all training")
    
    # generate a model-specific and task-specific output directory
    output_dir = f"{config.output_dir}_d{config.model.n_dims}"
    
    # Add task-specific parameters
    if config.task.name == "sinusoidal":
        # For sinusoidal tasks, include amplitude scale, x_range, and frequency range
        task_scale = getattr(config.task, 'task_scale', 0.25)
        x_range = getattr(config.task, 'x_range', 5.0)
        freq_min = getattr(config.task, 'freq_min', 0.5)
        freq_max = getattr(config.task, 'freq_max', 2.0)
        output_dir = f"{output_dir}_s{task_scale}_x{x_range}_f{freq_min}-{freq_max}"
    elif config.task.name == "kernel_rff":
        # For kernel_rff tasks, include scale, lengthscale, and rff_dim
        task_scale = getattr(config.task, 'task_scale', 0.25)
        lengthscale = getattr(config.task, 'lengthscale', 1.0)
        rff_dim = getattr(config.task, 'rff_dim', 128)
        output_dir = f"{output_dir}_s{task_scale}_l{lengthscale}_rff{rff_dim}"
    else:
        # For other tasks, just include scale
        task_scale = getattr(config.task, 'task_scale', 0.25)
        output_dir = f"{output_dir}_s{task_scale}"
    
    config.output_dir = output_dir
    
    # Train model
    model, output_dir = train_model(config)
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()