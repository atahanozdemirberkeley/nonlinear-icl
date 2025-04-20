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

from model import GregTransformer
from tasks import get_task
from data_generator import generate_samples_for_task
from config import Config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(config):
    """
    Train an in-context learning model for probability distributions.
    
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
    
    # Create model
    model = GregTransformer(
        n_dims=config.model.n_dims,
        n_positions=config.model.n_positions,
        n_embd=config.model.n_embd,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create scheduler with warmup
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < config.training.warmup_steps:
            return float(step) / float(max(1, config.training.warmup_steps))
        else:
            # Cosine decay from 1.0 to 0.1
            progress = float(step - config.training.warmup_steps) / float(
                max(1, config.training.train_steps - config.training.warmup_steps)
            )
            return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create sample task for validation
    val_task = get_task(
        config.task.name,
        config.model.n_dims,
        config.training.batch_size,
        scale=config.task.task_scale
    )
    
    # Generate validation data
    val_xs = generate_samples_for_task(
        val_task, 
        config.model.n_positions, 
        config.model.n_dims, 
        config.training.batch_size, 
        device
    )
    val_ys = val_task.evaluate(val_xs)
    
    # Training loop
    losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Configure curriculum learning
    # Start with a small number of context points and gradually increase
    min_ctx_points = 5  # Start with just a few examples
    max_ctx_points = config.model.n_positions // 2  # Maximum context points
    increase_interval = config.training.train_steps // 10  # Increase points every 10% of training
    current_ctx_points = min_ctx_points
    
    # Set up progress bar
    pbar = tqdm(range(config.training.train_steps))
    
    for step in pbar:
        # Update curriculum - gradually increase context points
        if step > 0 and step % increase_interval == 0 and current_ctx_points < max_ctx_points:
            current_ctx_points = min(current_ctx_points + 5, max_ctx_points)
            print(f"Curriculum updated: now using {current_ctx_points} context points")
        
        # Create new task instance for each batch
        task = get_task(
            config.task.name,
            config.model.n_dims,
            config.training.batch_size,
            scale=config.task.task_scale
        )
        
        # Generate samples
        xs = generate_samples_for_task(
            task, 
            current_ctx_points, 
            config.model.n_dims, 
            config.training.batch_size, 
            device
        )
        
        # Compute ground truth
        ys = task.evaluate(xs)
        
        # Set model to training mode
        model.train()
        
        # Use the last ys value as target
        target = ys[:, -1]
        
        # Forward pass using exact API from in-context-learning
        pred = model(xs, ys)[:, -1]  # Get prediction for the last position
        
        # Use mean squared error loss
        loss = nn.MSELoss()(pred, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.weight_decay)
        
        optimizer.step()
        scheduler.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Update progress bar
        pbar.set_description(f"Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Log to wandb
        if config.logging.use_wandb and step % config.logging.log_every == 0:
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "step": step
            })
        
        # Evaluate on validation set every eval_every steps
        if step % config.training.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Create validation data
                if step == 0:
                    print(f"Validation xs shape: {val_xs.shape}")
                    print(f"Validation ys shape: {val_ys.shape}")
                
                # Get validation prediction and loss
                val_target = val_ys[:, -1]
                val_pred = model(val_xs, val_ys)[:, -1]
                val_loss = nn.MSELoss()(val_pred, val_target)
                
                # Log validation loss
                if config.logging.use_wandb:
                    wandb.log({"val_loss": val_loss.item(), "step": step})
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                        'val_loss': val_loss.item(),
                        'config': config
                    }, os.path.join(output_dir, f"best_model.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= config.training.patience:
                        print(f"Early stopping at step {step}")
                        break
        
        # Save checkpoint every save_every steps
        if config.training.save_every > 0 and step % config.training.save_every == 0 and step > 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'config': config
            }, checkpoint_path)
    
    # Save final model
    torch.save({
        'step': config.training.train_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss.item(),
        'config': config
    }, os.path.join(output_dir, "final_model.pt"))
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {config.task.name}')
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
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