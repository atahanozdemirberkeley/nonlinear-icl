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
from tasks import get_task, get_task_sampler
from config import Config

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
    """Compute loss for all prefixes of the sequence."""
    losses = []
    
    # For each prefix length i from 0 to seq_length-2
    for i in range(1, xs.shape[1]):
        # Take first i+1 examples
        prefix_xs = xs[:, :i+1]
        prefix_ys = ys[:, :i+1]
        
        # Create prompt sequence
        prompt = create_prompt_sequence(prefix_xs, prefix_ys, config)
        
        # Get prediction for the last position
        prediction = model(prompt)[:, -1]  # Predict at the last position
        
        # Compute loss against the true label
        target = ys[:, i]
        if target.dim() == 2:
            target = target.squeeze(-1)  # Remove feature dimension if present
        
        loss = F.mse_loss(prediction, target)
        losses.append(loss)
    
    # Average loss across all prefix lengths
    return torch.stack(losses).mean()

class Curriculum:
    """Curriculum learning for gradually increasing task difficulty (dimensions only)."""
    def __init__(self, config):
        self.min_dims = config.training.min_dims
        self.max_dims = config.model.n_dims
        
        # Always use full number of positions (no curriculum on points)
        self.n_points = config.model.n_positions
        
        # Current curriculum state for dimensions
        self.n_dims_truncated = self.min_dims
        
        # Curriculum schedule
        self.total_steps = config.training.train_steps
        self.dim_schedule = config.training.dim_schedule  # % of training to reach max dims
    
    def update(self, step):
        """Update curriculum based on current step (only for dimensions)."""
        # Update number of dimensions
        if self.dim_schedule > 0:
            dim_progress = min(1.0, step / (self.total_steps * self.dim_schedule))
            self.n_dims_truncated = min(self.max_dims, 
                                    int(self.min_dims + (self.max_dims - self.min_dims) * dim_progress))

def sample_seeds(total_seeds, count):
    """Sample unique random seeds from 0 to total_seeds-1."""
    seeds = set()
    while len(seeds) < count:
        seeds.add(random.randint(0, total_seeds - 1))
    return list(seeds)

def train_model(config):
    """Train an in-context learning model with curriculum learning."""
    # No global seed setting - each run will use different random distributions
    print("Random seeds enabled - each run will use different random distributions")
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config with all parameters
    config.save(os.path.join(output_dir, "config.yaml"))
    
    # Initialize wandb if enabled
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.project_name,
            config=vars(config),
            name=f"{config.task.name}_d{config.model.n_dims}_n{config.model.n_positions}"
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate d_token if not specified
    if config.model.d_token is None:
        config.model.d_token = config.model.n_dims + 1
    
    # Create model - use GPT2 model like in the original repo
    # Get number of layers from config or use default of 12
    n_layer = getattr(config.model, 'n_layer', 12)
    
    model = GPT2ICLModel(
        d_token=config.model.d_token,
        n_positions=config.model.n_positions,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layer=n_layer
    ).to(device)
    print(f"Using GPT2-style model with {n_layer} layers")
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    learning_rate = float(config.training.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create validation tasks using task_sampler
    print(f"Creating {config.training.n_val_tasks} validation tasks with {config.model.n_dims} dimensions")
    val_tasks = []
    val_task_sampler = get_task_sampler(
        config.task.name, 
        config.model.n_dims, 
        config.training.batch_size,
        scale=getattr(config.task, 'task_scale', 0.25)
    )
    
    for i in range(config.training.n_val_tasks):
        try:
            # Each validation task gets a completely random distribution
            val_task = val_task_sampler()
            val_tasks.append(val_task)
        except Exception as e:
            print(f"Error creating validation task {i}: {e}")
    
    # Create curriculum
    curriculum = Curriculum(config)
    print(f"Using {curriculum.n_points} points for all training, curriculum on dimensions: {curriculum.min_dims} → {curriculum.max_dims}")
    
    # Training state
    starting_step = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    steps = []
    
    # Number of unique distributions to use during training (if limited)
    num_unique_distributions = getattr(config.training, 'num_unique_distributions', None)
    # For backward compatibility
    if num_unique_distributions is None:
        num_unique_distributions = getattr(config.training, 'num_training_examples', None)
        if num_unique_distributions is not None:
            print(f"Warning: 'num_training_examples' is deprecated, use 'num_unique_distributions' instead")
    
    # Training loop
    pbar = tqdm(range(starting_step, config.training.train_steps))
    
    for step in pbar:
        model.train()
        
        # Get task sampler for current curriculum dimensions
        task_sampler = get_task_sampler(
            config.task.name, 
            curriculum.n_dims_truncated,
            config.training.batch_size,
            scale=getattr(config.task, 'task_scale', 0.25),
            num_tasks=getattr(config.training, 'pool_size', None)  # Support task pools
        )
        
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
    
    # Set x-axis to show actual step numbers
    if steps:
        plt.xticks(steps)  # Force x-axis to show all steps
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{config.task.name} (d={config.model.n_dims})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    # Close wandb
    if config.logging.use_wandb:
        wandb.finish()
    
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description="Train ICL model for probability distributions")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--task_name', type=str, help='Override task name from config')
    parser.add_argument('--n_dims', type=int, help='Override input dimensions')
    parser.add_argument('--task_scale', type=float, help='Override task scale parameter')
    parser.add_argument('--steps', type=int, help='Override number of training steps')
    parser.add_argument('--min_dims', type=int, help='Override minimum dimensions for curriculum')
    parser.add_argument('--dim_schedule', type=float, help='Override dimension schedule (0 to disable)')
    parser.add_argument('--x_range', type=float, help='Override x range for sinusoidal tasks')
    parser.add_argument('--freq_min', type=float, help='Override minimum frequency for sinusoidal tasks')
    parser.add_argument('--freq_max', type=float, help='Override maximum frequency for sinusoidal tasks')
    
    args = parser.parse_args()
    
    # Load configuration from file
    config = Config.load(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override parameters if provided
    if args.task_name:
        config.task.name = args.task_name
    
    if args.n_dims:
        config.model.n_dims = args.n_dims
        # Update d_token to match dimensions
        config.model.d_token = config.model.n_dims + 1
    
    if args.task_scale:
        config.task.task_scale = args.task_scale
    
    if args.steps:
        config.training.train_steps = args.steps
    
    if args.min_dims:
        config.training.min_dims = args.min_dims
    
    if args.dim_schedule is not None:
        config.training.dim_schedule = args.dim_schedule
    
    # Process sinusoidal task specific parameters
    if args.x_range:
        config.task.x_range = args.x_range
    
    if args.freq_min:
        config.task.freq_min = args.freq_min
    
    if args.freq_max:
        config.task.freq_max = args.freq_max
    
    # Always use full position length (k+1) to match the mathematical objective
    config.training.min_points = config.model.n_positions
    config.training.point_schedule = 0
    print(f"Using full sequence length of {config.model.n_positions} positions for all training")
    
    # Update output directory to include task-specific parameters
    base_output_dir = config.output_dir
    
    # Add dimensions to output directory name
    output_dir = f"{base_output_dir}_d{config.model.n_dims}"
    
    # Add task-specific parameters
    if config.task.name == "sinusoidal":
        # For sinusoidal tasks, include amplitude scale, x_range, and frequency range
        task_scale = getattr(config.task, 'task_scale', 0.25)
        x_range = getattr(config.task, 'x_range', 5.0)
        freq_min = getattr(config.task, 'freq_min', 0.5)
        freq_max = getattr(config.task, 'freq_max', 2.0)
        output_dir = f"{output_dir}_s{task_scale}_x{x_range}_f{freq_min}-{freq_max}"
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