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

    # Ensure ys has the right shape (batch_size, n_positions, 1)
    if ys.dim() == 2:
        ys_prep = ys.unsqueeze(-1)  # Add feature dimension
    else:
        ys_prep = ys  # Already in the expected format
    
    # Create the prompt sequence by concatenating xs and ys
    batch_size, n_positions = xs.shape[0], xs.shape[1]
    
    if d_token > config.model.n_dims + 1:
        # Add padding to match the expected d_token size
        padding_size = d_token - (config.model.n_dims + 1)
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
        print(f"DEBUG - d_token from config: {d_token}")
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
    
    # Save config
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
    val_task_seeds = list(range(config.training.seed + 1, config.training.seed + 1 + config.training.n_val_tasks))
    val_tasks = [get_task(config.task.name, config.model.n_dims, batch_size=config.training.batch_size, seed=s) for s in val_task_seeds]
    
    # Create curriculum
    curriculum = Curriculum(config)
    
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
        task = get_task(config.task.name, curriculum.n_dims_truncated, 
                        batch_size=config.training.batch_size, seed=config.training.seed)
        
        # Sample data with curriculum-controlled number of points
        xs, ys = task.sample(config.training.batch_size, curriculum.n_points, 
                            xs_seeds=xs_seeds, ys_seeds=ys_seeds)
        xs, ys = xs.to(device), ys.to(device)
        
        # Create prompt sequence
        prompt_seq = create_prompt_sequence(xs, ys, config)
        
        # Forward pass and compute loss
        optimizer.zero_grad()
        loss, point_wise_losses = compute_loss_all_positions(model, prompt_seq, ys)
        
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
            val_losses = []
            
            with torch.no_grad():
                for val_task in val_tasks:
                    val_xs, val_ys = val_task.sample(config.training.batch_size, config.model.n_positions)
                    val_xs, val_ys = val_xs.to(device), val_ys.to(device)
                    
                    # Create prompt sequence for validation
                    val_prompt_seq = create_prompt_sequence(val_xs, val_ys, config)
                    
                    # Compute validation loss
                    val_loss, _ = compute_loss_all_positions(model, val_prompt_seq, val_ys)
                    val_losses.append(val_loss.item())
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses)
            
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
    
    # Close wandb run
    if config.logging.use_wandb:
        wandb.finish()
    
    return model, output_dir

def main():
    parser = argparse.ArgumentParser(description="Train ICL model for probability distributions")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--task_name', type=str, help='Override task name from config')
    args = parser.parse_args()
    
    # Load configuration from file
    config = Config.load(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override task name if provided
    if args.task_name:
        config.task.name = args.task_name
        print(f"Overriding task name to: {args.task_name}")
    
    # Train model
    model, output_dir = train_model(config)
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()