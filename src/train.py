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

from model import ICLModel
from tasks import get_task
from data_generator import generate_samples_for_task
from config import Config

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_prompt_sequence(xs, ys, config):
    """Create prompt sequence for the model.
    
    Args:
        xs: Input vectors of shape (batch_size, n_positions, n_dims)
        ys: Output values of shape (batch_size, n_positions, 1)
        config: Config object
        
    Returns:
        Prompt sequence tensor of shape (batch_size, n_positions, d_token)
    """
    # Get the expected token dimensionality from config
    d_token = config.model.d_token if config.model.d_token is not None else config.model.n_dims + 1
    
    # Get actual input dimensions
    actual_dims = xs.shape[2]
    
    # Ensure ys has the right shape (batch_size, n_positions, 1)
    if ys.dim() == 2:
        ys_prep = ys.unsqueeze(-1)  # Add feature dimension
    else:
        ys_prep = ys  # Already in the expected format
    
    # Create the prompt sequence by concatenating xs and ys
    batch_size, n_positions = xs.shape[0], xs.shape[1]
    
    # Calculate total padding needed to reach d_token
    padding_size = d_token - (actual_dims + 1)
    
    if padding_size > 0:
        # Add padding to match the expected d_token size
        padding = torch.zeros(batch_size, n_positions, padding_size, device=xs.device)
        
        # Concatenate xs, ys, and padding along the feature dimension
        prompt_seq = torch.cat([xs, ys_prep, padding], dim=2)
    else:
        # Simple concatenation if no additional padding needed
        prompt_seq = torch.cat([xs, ys_prep], dim=2)
    
    # Debug info for the first batch in the first epoch
    if not hasattr(create_prompt_sequence, "debug_printed"):
        print(f"DEBUG - xs shape: {xs.shape}")
        print(f"DEBUG - ys shape: {ys.shape}")
        print(f"DEBUG - ys_prep shape: {ys_prep.shape}")
        print(f"DEBUG - config n_dims: {config.model.n_dims}")
        print(f"DEBUG - actual input dims: {actual_dims}")
        print(f"DEBUG - d_token from config: {d_token}")
        print(f"DEBUG - padding_size: {padding_size}")
        print(f"DEBUG - prompt_seq shape: {prompt_seq.shape}")
        create_prompt_sequence.debug_printed = True
        
    return prompt_seq

def compute_loss_all_positions(model, prompt_seq, ys, start_pos=1):
    """
    Compute loss for all positions starting from start_pos.
    
    Args:
        model: ICL model
        prompt_seq: Prompt sequence tensor [batch_size, n_points, d_token]
        ys: Target tensor [batch_size, n_points]
        start_pos: Starting position for loss computation (usually 1)
        
    Returns:
        Average loss over all positions
    """
    # Forward pass through the model to get predictions for all positions
    predictions = model(prompt_seq)
    
    # Compute loss for all positions starting from start_pos
    # We don't compute loss for position 0 since there's no prediction task there
    losses = []
    for pos in range(start_pos, prompt_seq.shape[1]):
        # Get predictions and targets for current position
        pred = predictions[:, pos]
        target = ys[:, pos]
        
        # Compute MSE loss for this position
        pos_loss = F.mse_loss(pred, target)
        losses.append(pos_loss)
    
    # Average loss across all positions
    avg_loss = torch.stack(losses).mean()
    return avg_loss, torch.stack(losses)

class Curriculum:
    """
    Curriculum learning for gradually increasing task difficulty.
    """
    def __init__(self, config):
        self.config = config
        self.min_points = config.training.min_points
        self.max_points = config.model.n_positions
        self.min_dims = config.training.min_dims
        self.max_dims = config.model.n_dims
        
        # Current curriculum state
        self.n_points = self.min_points
        self.n_dims_truncated = self.min_dims
        
        # Curriculum schedule
        self.total_steps = config.training.train_steps
        self.point_schedule = config.training.point_schedule  # % of training to reach max points
        self.dim_schedule = config.training.dim_schedule  # % of training to reach max dims
    
    def update(self, step):
        """Update curriculum based on current step."""
        # Update number of points
        if self.point_schedule > 0:
            point_progress = min(1.0, step / (self.total_steps * self.point_schedule))
            self.n_points = min(self.max_points, 
                               int(self.min_points + (self.max_points - self.min_points) * point_progress))
        
        # Update number of dimensions
        if self.dim_schedule > 0:
            dim_progress = min(1.0, step / (self.total_steps * self.dim_schedule))
            self.n_dims_truncated = min(self.max_dims, 
                                       int(self.min_dims + (self.max_dims - self.min_dims) * dim_progress))

def sample_seeds(total_seeds, count):
    """Sample unique random seeds."""
    seeds = set()
    while len(seeds) < count:
        seeds.add(random.randint(0, total_seeds - 1))
    return list(seeds)

def train_model(config):
    """
    Train an in-context learning model with curriculum learning.
    
    Args:
        config: Configuration object
        
    Returns:
        Trained model and output directory
    """
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Add n_dims and task_scale to the config that is saved (for reference)
    # This ensures these values are recorded with the model even though
    # they're not in the original config file
    config_copy_for_saving = config.copy()
    if not hasattr(config_copy_for_saving.model, 'n_dims'):
        config_copy_for_saving.model.n_dims = config.model.n_dims
    if not hasattr(config_copy_for_saving.task, 'task_scale'):
        config_copy_for_saving.task.task_scale = getattr(config.task, 'task_scale', 0.25)
    
    # Save config with all parameters
    config_copy_for_saving.save(os.path.join(output_dir, "config.yaml"))
    print(f"Saved complete config with n_dims={config.model.n_dims}, task_scale={getattr(config.task, 'task_scale', 0.25)}")
    
    # Always disable dimension curriculum and set min_dims to n_dims
    config.training.min_dims = config.model.n_dims
    config.training.dim_schedule = 0
    print(f"Dimension curriculum disabled. Using fixed {config.training.min_dims} dimensions for all training.")
    
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
        print(f"Setting d_token to {config.model.d_token} (n_dims + 1)")
    
    # Create model
    model = ICLModel(
        d_token=config.model.d_token,
        n_positions=config.model.n_positions,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        kernel_type=config.model.kernel_type
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    # Create validation data
    print(f"Creating {config.training.n_val_tasks} validation tasks with {config.model.n_dims} dimensions")
    val_tasks = []
    
    # Create validation tasks with different random seeds
    for i in range(config.training.n_val_tasks):
        try:
            # Each validation task gets a different random seed
            val_seed = None  # Don't specify seed to get random distributions
            
            val_task = get_task(config.task.name, config.model.n_dims, 
                               batch_size=config.training.batch_size, 
                               scale=getattr(config.task, 'task_scale', 0.25),
                               seed=val_seed)
            val_tasks.append(val_task)
        except Exception as e:
            print(f"Error creating validation task {i}: {e}")
    
    if not val_tasks:
        print("WARNING: No validation tasks were created. Training will proceed without validation.")
    
    # Create curriculum
    curriculum = Curriculum(config)
    
    # Print initial curriculum state
    print(f"Initial curriculum: {curriculum.n_dims_truncated} dimensions, {curriculum.n_points} points")
    print(f"Curriculum will increase to: {curriculum.max_dims} dimensions, {curriculum.max_points} points")
    
    # Ensure that min_dims is not higher than max_dims
    if curriculum.min_dims > curriculum.max_dims:
        print(f"Warning: min_dims ({curriculum.min_dims}) is higher than max_dims ({curriculum.max_dims})")
        print(f"Setting min_dims to max_dims ({curriculum.max_dims})")
        curriculum.min_dims = curriculum.max_dims
        curriculum.n_dims_truncated = curriculum.max_dims
    
    # Load checkpoint if exists
    starting_step = 0
    state_path = os.path.join(output_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        # Update curriculum to match saved state
        for i in range(starting_step + 1):
            curriculum.update(i)
    
    # Keep track of best validation loss for saving best model
    best_val_loss = float('inf')
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    steps = []
    
    # Add task_scale to config.task if not present
    if not hasattr(config.task, 'task_scale'):
        config.task.task_scale = 0.25
        print(f"No task_scale specified, using default: {config.task.task_scale}")
    
    # Training loop
    pbar = tqdm(range(starting_step, config.training.train_steps))
    
    for step in pbar:
        model.train()
        
        # Sample seeds for this batch if limited training examples
        if config.training.num_training_examples is not None:
            task_seeds = sample_seeds(config.training.num_training_examples, config.training.batch_size)
            xs_seeds = [s for s in task_seeds]
            ys_seeds = [s + 1 for s in task_seeds]  # Offset by 1 to ensure different sequences
        else:
            task_seeds = None
            xs_seeds = None
            ys_seeds = None
        
        # Sample a new task instance with curriculum-controlled dimensions
        # Print debug info on first step
        if step == starting_step:
            print(f"Creating task with {curriculum.n_dims_truncated} dimensions (max: {curriculum.max_dims})")
            print(f"Task scale: {config.task.task_scale}")
        
        try:
            # Generate a unique random seed for this batch
            batch_seed = None  # Don't specify seed to get random sampling
            
            # Get task with the current curriculum dimensions and random seed
            task = get_task(config.task.name, curriculum.n_dims_truncated, 
                            batch_size=config.training.batch_size, 
                            scale=config.task.task_scale,
                            seed=batch_seed)  # Use random seed each time
            
            # Sample data with curriculum-controlled number of points
            xs, ys = task.sample(config.training.batch_size, curriculum.n_points, 
                               xs_seeds=xs_seeds, ys_seeds=ys_seeds)
            
            # Debug on first step
            if step == starting_step:
                print(f"Sampled data: xs.shape={xs.shape}, ys.shape={ys.shape}")
                
            # Ensure dimensions match what we expect
            if xs.shape[2] != curriculum.n_dims_truncated:
                print(f"WARNING: Dimension mismatch! Expected {curriculum.n_dims_truncated}, got {xs.shape[2]}")
                # Force the task to use the correct dimensions
                print(f"Recreating task with explicit dimensions {curriculum.n_dims_truncated}")
                task = get_task(config.task.name, int(curriculum.n_dims_truncated), 
                                batch_size=config.training.batch_size, 
                                scale=config.task.task_scale,
                                seed=config.training.seed)
                xs, ys = task.sample(config.training.batch_size, curriculum.n_points, 
                                   xs_seeds=xs_seeds, ys_seeds=ys_seeds)
                print(f"After fix: xs.shape={xs.shape}, ys.shape={ys.shape}")
        except Exception as e:
            print(f"Error creating task: {e}")
            print(f"Attempting to create task with max dimensions {curriculum.max_dims}")
            # Fall back to maximum dimensions
            task = get_task(config.task.name, curriculum.max_dims, 
                            batch_size=config.training.batch_size, 
                            scale=config.task.task_scale,
                            seed=config.training.seed)
            xs, ys = task.sample(config.training.batch_size, curriculum.n_points)
        
        xs, ys = xs.to(device), ys.to(device)
        
        # Create prompt sequence

        batch, n_pos = ys.size()
# create zeros for the "no label" at position 0
        zero = torch.zeros(batch, 1, device=ys.device)
        # drop the last true y so we can shift in a zero at front
        ys_shifted = torch.cat([zero, ys[:, :-1]], dim=1)

        # prompt_seq = create_prompt_sequence(xs, ys, config)
        prompt_seq = create_prompt_sequence(xs, ys_shifted, config)

        # Forward pass and compute loss
        optimizer.zero_grad()
        loss, point_wise_losses = compute_loss_all_positions(model, prompt_seq, ys)
        
        # Save training loss for plotting
        train_losses.append(loss.item())
        steps.append(step)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Update curriculum
        curriculum.update(step)
        
        # Calculate baseline loss (naive solution - linear in dimensions)
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        
        # Log to wandb
        if config.logging.use_wandb and step % config.logging.log_every == 0:
            point_wise_tags = list(range(curriculum.n_points))
            wandb.log({
                "train/step_loss": loss.item(),
                "excess_loss": loss.item() / max(baseline_loss, 1e-8),
                "pointwise/loss": dict(
                    zip(point_wise_tags, point_wise_losses.cpu().numpy())
                ),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
                "step": step,
            })
        
        # Update progress bar
        pbar.set_description(f"loss {loss.item():.6f}, points {curriculum.n_points}, dims {curriculum.n_dims_truncated}")
        
        # Validation and checkpoint saving
        if step % config.training.eval_every == 0 or step == config.training.train_steps - 1:
            model.eval()
            val_losses_for_step = []
            
            with torch.no_grad():
                for i, val_task in enumerate(val_tasks):
                    try:
                        # Sample data from validation task
                        val_xs, val_ys = val_task.sample(config.training.batch_size, config.model.n_positions)
                        val_xs, val_ys = val_xs.to(device), val_ys.to(device)
                        
                        # Debug info on first validation
                        if step == starting_step and i == 0:
                            print(f"Validation data: val_xs.shape={val_xs.shape}, val_ys.shape={val_ys.shape}")
                        
                        # Ensure validation data has correct dimensions
                        if val_xs.shape[2] != config.model.n_dims:
                            print(f"WARNING: Validation dimension mismatch! Expected {config.model.n_dims}, got {val_xs.shape[2]}")
                            # Skip this validation task
                            continue
                        
                        # Create prompt sequence for validation
                        val_prompt_seq = create_prompt_sequence(val_xs, val_ys, config)
                        
                        # Compute validation loss
                        val_loss, _ = compute_loss_all_positions(model, val_prompt_seq, val_ys)
                        val_losses_for_step.append(val_loss.item())
                    except Exception as e:
                        print(f"Error during validation for task {i}: {e}")
            
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
                
                model_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, model_path)
                print(f"New best model saved to {model_path}")
            
        # Save checkpoint periodically
        if step % config.training.save_every == 0 and step > 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }
            torch.save(training_state, state_path)
            print(f"Checkpoint saved at step {step}")
            
        # Save model snapshots periodically
        if config.training.keep_every_steps > 0 and step % config.training.keep_every_steps == 0 and step > 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_{step}.pt"))
            print(f"Model snapshot saved at step {step}")
    
    # Final evaluation on test set
    if config.training.n_test_tasks > 0:
        model.eval()
        test_task_seeds = list(range(
            config.training.seed + 1 + config.training.n_val_tasks,
            config.training.seed + 1 + config.training.n_val_tasks + config.training.n_test_tasks
        ))
        test_tasks = [get_task(config.task.name, config.model.n_dims, batch_size=config.training.batch_size, seed=s) for s in test_task_seeds]
        test_losses = []
        
        with torch.no_grad():
            for test_task in test_tasks:
                test_xs, test_ys = test_task.sample(config.training.batch_size, config.model.n_positions)
                test_xs, test_ys = test_xs.to(device), test_ys.to(device)
                
                # Create prompt sequence for testing
                test_prompt_seq = create_prompt_sequence(test_xs, test_ys, config)
                
                # Compute test loss
                test_loss, _ = compute_loss_all_positions(model, test_prompt_seq, test_ys)
                test_losses.append(test_loss.item())
        
        # Calculate average test loss
        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.6f}")
        
        # Log to wandb
        if config.logging.use_wandb:
            wandb.log({"test/loss": avg_test_loss})
    
    # Plot and save training curves
    plot_training_curves(steps, train_losses, val_losses, output_dir, config)
    
    # Save loss data for later comparison
    save_training_data(steps, train_losses, val_losses, output_dir, config)
    
    # Close wandb run
    if config.logging.use_wandb:
        wandb.finish()
    
    return model, output_dir

def plot_training_curves(steps, train_losses, val_losses, output_dir, config):
    """
    Plot training and validation loss curves
    
    Args:
        steps: List of training steps
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save plots
        config: Config object for metadata
    """
    plt.figure(figsize=(12, 6))
    
    # Filter out validation steps to match validation data points
    val_steps = [steps[i] for i in range(0, len(steps), config.training.eval_every)]
    if len(val_steps) > len(val_losses):
        val_steps = val_steps[:len(val_losses)]
    elif len(val_steps) < len(val_losses):
        val_losses = val_losses[:len(val_steps)]
    
    plt.plot(steps, train_losses, label="Training Loss", color="blue", alpha=0.7)
    plt.plot(val_steps, val_losses, label="Validation Loss", color="red", marker="o", linestyle="-", markersize=3)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{config.task.name.capitalize()} Distribution (d={config.model.n_dims}, scale={getattr(config.task, 'task_scale', 0.25)})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    # Add curriculum information
    plt.figtext(0.5, 0.01, 
                f"Curriculum: {config.training.min_points}→{config.model.n_positions} points, " 
                f"{config.training.min_dims}→{config.model.n_dims} dims",
                ha="center", fontsize=10)
    
    # Save plot
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    
    print(f"Training curve saved to {loss_plot_path}")

def save_training_data(steps, train_losses, val_losses, output_dir, config):
    """
    Save training data for later comparison
    
    Args:
        steps: List of training steps
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save data
        config: Config object for metadata
    """
    training_data = {
        "task": config.task.name,
        "dims": config.model.n_dims,
        "task_scale": getattr(config.task, "task_scale", 0.25),
        "steps": steps,
        "train_losses": train_losses,
        "val_steps": [steps[i] for i in range(0, len(steps), config.training.eval_every)][:len(val_losses)],
        "val_losses": val_losses,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Create experiments directory if it doesn't exist
    experiments_dir = os.path.join("experiments", config.task.name)
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Save data as numpy arrays for easy loading
    data_path = os.path.join(experiments_dir, 
                            f"{config.task.name}_d{config.model.n_dims}_s{getattr(config.task, 'task_scale', 0.25)}.npz")
    np.savez(data_path, **training_data)
    
    print(f"Training data saved to {data_path}")

def main():
    parser = argparse.ArgumentParser(description="Train ICL model for probability distributions")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--task_name', type=str, help='Override task name from config')
    parser.add_argument('--n_dims', type=int, help='Override input dimensions')
    parser.add_argument('--task_scale', type=float, help='Override task scale parameter')
    parser.add_argument('--steps', type=int, help='Override number of training steps')
    
    # Only keep the point-related curriculum parameters
    parser.add_argument('--min_points', type=int, help='Override minimum points for curriculum')
    parser.add_argument('--point_schedule', type=float, help='Override point schedule (0 to disable)')
    
    args = parser.parse_args()
    
    # Load configuration from file
    config = Config.load(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override parameters if provided
    if args.task_name:
        config.task.name = args.task_name
        print(f"Overriding task name to: {args.task_name}")
    
    # Track if we need to update d_token
    dims_changed = False
    
    if args.n_dims:
        config.model.n_dims = args.n_dims
        dims_changed = True
        print(f"Overriding input dimensions to: {args.n_dims}")
    
    if args.task_scale:
        config.task.task_scale = args.task_scale
        print(f"Overriding task scale to: {args.task_scale}")
    
    if args.steps:
        config.training.train_steps = args.steps
        print(f"Overriding training steps to: {args.steps}")
    
    # Only override point-related curriculum parameters
    if args.min_points:
        config.training.min_points = args.min_points
        print(f"Overriding minimum curriculum points to: {args.min_points}")
    
    if args.point_schedule is not None:
        config.training.point_schedule = args.point_schedule
        print(f"Overriding point schedule to: {args.point_schedule}")
        
        # Provide helpful message if disabling curriculum
        if args.point_schedule == 0:
            print(f"Point curriculum disabled. Using fixed {config.training.min_points} points.")
    
    # Always recalculate d_token if dimensions changed or if it wasn't specified
    if dims_changed or not hasattr(config.model, 'd_token') or config.model.d_token is None:
        config.model.d_token = config.model.n_dims + 1
        print(f"Setting d_token to {config.model.d_token} (n_dims + 1)")
    else:
        # Ensure d_token is at least n_dims + 1
        min_d_token = config.model.n_dims + 1
        if config.model.d_token < min_d_token:
            print(f"Warning: d_token ({config.model.d_token}) is less than n_dims + 1 ({min_d_token})")
            print(f"Setting d_token to {min_d_token}")
            config.model.d_token = min_d_token
    
    # Update output directory to include dimensions and scale
    base_output_dir = config.output_dir
    config.output_dir = f"{base_output_dir}_d{config.model.n_dims}_s{getattr(config.task, 'task_scale', 0.25)}"
    
    # Train model
    model, output_dir = train_model(config)
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()