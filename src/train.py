# src/train.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR

from model import ICLTransformer, SimpleICLTransformer
from kernels import generate_rff_function, KernelRidgeRegression
from data import create_dataloader
from eval import evaluate_model, compare_with_krr
from config import Config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def kernel_mse_loss(pred, target, scale=1.0):
    """
    MSE loss with kernel re-weighting - helps with difficult examples
    """
    diff = pred - target
    # Apply RBF kernel-like re-weighting
    weight = torch.exp(-(diff**2) / (2 * scale**2))
    
    # Compute weighted MSE but scale it to be closer to regular MSE for better interpretability
    raw_loss = (diff**2) * weight
    scaling_factor = torch.mean(weight).detach()  # Detach to avoid affecting gradients
    return torch.mean(raw_loss) / (scaling_factor + 1e-8)  # Add epsilon to avoid division by zero

def train_model(config):
    """
    Train an in-context learning model with RFF functions.
    
    Args:
        config: Training configuration
        
    Returns:
        Trained model
    """
    # Set seed for reproducibility
    set_seed(42)
    
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
            name=f"rff_d{config.task.input_dim}_k{config.task.context_length}"
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate a function to learn using RFF - for visualization, create a fixed 2D function
    f, feature_map = generate_rff_function(
        config.task.input_dim, 
        config.task.rff_dim, 
        config.task.sigma
    )
    
    # Create model
    d_token = config.task.input_dim + 1  # input + output
    model = ICLTransformer(
        d_token,
        d_model=config.model.n_embd,
        n_layers=config.model.n_layer,
        n_heads=config.model.n_head,
        dropout=0.1  # Add dropout for regularization
    ).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Create validation dataset upfront (fixed function)
    # Use explicit dtype=torch.float32 to avoid dtype mismatches
    val_inputs = torch.rand(100, config.task.context_length, config.task.input_dim, dtype=torch.float32) * 2 - 1
    
    # Convert to numpy, apply function, and convert back with explicit dtype
    val_outputs_np = np.array([f(x.cpu().numpy()) for x in val_inputs])
    val_outputs = torch.tensor(val_outputs_np, dtype=torch.float32)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use OneCycleLR scheduler with longer warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=config.training.train_steps,
        pct_start=0.2,  # 20% warmup - increased from 10%
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Create dataloader - create new samples for each batch
    dataloader = create_dataloader(
        f,
        config.task.input_dim,
        config.task.context_length,
        n_samples=config.training.train_steps * config.training.batch_size,
        batch_size=config.training.batch_size
    )
    
    # Training loop
    losses = []
    raw_mses = []  # Track raw MSE values for comparison
    eval_mses = []
    best_eval_mse = float('inf')
    patience_counter = 0
    patience = 15  # Increased early stopping patience
    no_improvement_steps = 0  # Count steps without improvement
    
    # Create loss function with adaptive scaling
    loss_scale = 1.0
    
    # Set up progress bar
    pbar = tqdm(enumerate(dataloader), total=config.training.train_steps)
    
    for step, (prompt_seq, target) in pbar:
        if step >= config.training.train_steps:
            break
            
        # Move data to device
        prompt_seq = prompt_seq.to(device)
        target = target.to(device)
        
        # Set model to training mode
        model.train()
        
        # Forward pass
        pred = model(prompt_seq)
        
        # Track raw MSE for monitoring real performance
        raw_mse = torch.mean((pred - target)**2).item()
        raw_mses.append(raw_mse)
        
        # Dynamically adjust loss_scale based on prediction range
        if step % 100 == 0 and step > 0:
            with torch.no_grad():
                # Use recent prediction variance to set scale
                loss_scale = max(0.1, min(5.0, torch.var(pred).sqrt().item() * 2))
        
        # Use kernel MSE loss for better handling of difficult examples
        loss = kernel_mse_loss(pred, target, scale=loss_scale)
        
        # Add smoothness regularization for better kernel behavior
        # Fix: Move the smoothness regularization to a separate step to avoid gradient issues
        if step > 1000:  # Apply after initial training
            # Create a perturbed version of the input with small noise
            with torch.no_grad():  # Don't track gradients for creating perturbed inputs
                # Fix shape mismatch - create noise with proper dimensions
                noise = torch.randn_like(prompt_seq[:, -1:, :config.task.input_dim]) * 0.01
                perturbed_prompt = prompt_seq.clone().detach()  # Detach to create a fresh tensor
                
                # Correctly apply noise to last position only
                # The shape should be [batch_size, 1, input_dim]
                perturbed_prompt = perturbed_prompt.clone()  # Create a copy to modify
                last_inputs = perturbed_prompt[:, -1:, :config.task.input_dim].clone()
                perturbed_last_inputs = last_inputs + noise
                
                # Reconstruct the perturbed prompt with the modified last position
                perturbed_prompt = torch.cat([
                    perturbed_prompt[:, :-1, :],  # All but last position
                    torch.cat([  # Modified last position
                        perturbed_last_inputs,
                        perturbed_prompt[:, -1:, config.task.input_dim:]
                    ], dim=2)
                ], dim=1)
            
            # Get predictions on original and perturbed inputs
            perturbed_pred = model(perturbed_prompt)
            
            # Calculate smoothness loss
            smoothness_loss = torch.mean((perturbed_pred - pred)**2) * 0.1
            loss = loss + smoothness_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track progress
        losses.append(loss.item())
        
        # Update progress bar to show both weighted loss and actual MSE
        pbar.set_description(f"Loss: {loss.item():.6f} | MSE: {raw_mse:.6f} (scale={loss_scale:.2f})")
        
        # Log metrics
        if config.logging.use_wandb and step % config.logging.log_every == 0:
            wandb.log({
                "loss": loss.item(),
                "raw_mse": raw_mse,
                "step": step,
                "learning_rate": scheduler.get_last_lr()[0],
                "loss_scale": loss_scale
            })
        
        # Evaluate model
        if step % config.training.eval_every == 0 or step == config.training.train_steps - 1:
            # Set model to evaluation mode
            model.eval()
            
            # Evaluate on the fixed validation set
            with torch.no_grad():
                val_mses = []
                for i in range(len(val_inputs)):
                    # Prepare input sequence
                    val_seq = []
                    for j in range(config.task.context_length - 1):
                        # Explicitly use float32 for all tensor operations
                        val_seq.append(torch.cat([
                            val_inputs[i, j], 
                            torch.tensor([val_outputs[i, j]], dtype=torch.float32)
                        ]))
                    # Add query point
                    val_seq.append(torch.cat([
                        val_inputs[i, -1], 
                        torch.tensor([0.0], dtype=torch.float32)
                    ]))
                    # Stack and ensure float32 dtype
                    val_prompt = torch.stack(val_seq).to(dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Get prediction
                    val_pred = model(val_prompt)
                    
                    # Calculate MSE
                    val_mse = ((val_pred - val_outputs[i, -1].to(device)) ** 2).item()
                    val_mses.append(val_mse)
                
                avg_val_mse = np.mean(val_mses)
                eval_mses.append(avg_val_mse)
            
            # Also evaluate on random samples
            mse = evaluate_model(
                model, f, config.task.input_dim, config.task.context_length, 
                n_samples=20, device=device
            )
            
            print(f"Step {step} | Val MSE: {avg_val_mse:.6f} | Test MSE: {mse:.6f} | Train MSE: {np.mean(raw_mses[-100:]):.6f}")
            
            if config.logging.use_wandb:
                wandb.log({
                    "val_mse": avg_val_mse, 
                    "test_mse": mse,
                    "train_mse": np.mean(raw_mses[-100:]),
                    "step": step
                })
            
            # Save best model
            if avg_val_mse < best_eval_mse:
                best_eval_mse = avg_val_mse
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                patience_counter = 0
                no_improvement_steps = 0
                
                # When we find a significantly better model, also try a higher learning rate
                if step > 1000 and no_improvement_steps > 3:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 1.5
                    print(f"Increasing learning rate to {optimizer.param_groups[0]['lr']:.8f}")
            else:
                patience_counter += 1
                no_improvement_steps += 1
                
                # If we haven't improved for a while, reduce learning rate
                if no_improvement_steps % 3 == 0 and step > 1000:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                    print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.8f}")
                    no_improvement_steps = 0  # Reset counter
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {step} steps")
                break
        
        # Save model checkpoint
        if step % config.training.save_every == 0 and step > 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'best_val_mse': best_eval_mse
            }, os.path.join(output_dir, f"checkpoint_{step}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pt"))
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
    
    # Plot training curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Weighted Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    plt.subplot(1, 3, 2)
    plt.plot(raw_mses)
    plt.title("Training MSE (Unweighted)")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    
    plt.subplot(1, 3, 3)
    eval_steps = list(range(0, len(eval_mses) * config.training.eval_every, config.training.eval_every))
    plt.plot(eval_steps, eval_mses)
    plt.title("Validation MSE")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    
    # If 2D input, visualize the learned function
    if config.task.input_dim == 2:
        visualize_2d_function(model, f, device, output_dir)
    
    # Compare with KRR baseline using the best model
    print("Comparing with KRR baseline...")
    results = compare_with_krr(
        model, feature_map, config.task.input_dim, config.task.context_length,
        n_samples=20  # More samples for better comparison
    )
    
    # Calculate average MSEs
    avg_transformer_mse = np.mean(results['transformer'])
    avg_krr_mses = {alpha: np.mean(mses) for alpha, mses in results['krr'].items()}
    
    print(f"Average Transformer MSE: {avg_transformer_mse:.6f}")
    for alpha, mse in avg_krr_mses.items():
        print(f"Average KRR MSE (alpha={alpha}): {mse:.6f}")
    
    # Log comparison results
    if config.logging.use_wandb:
        wandb.log({
            "avg_transformer_mse": avg_transformer_mse,
            **{f"avg_krr_mse_alpha_{alpha}": mse for alpha, mse in avg_krr_mses.items()}
        })
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Plot transformer results
    plt.scatter(range(len(results['transformer'])), results['transformer'], 
               label='Transformer', marker='o', s=100, color='blue')
    
    # Plot KRR results for each alpha
    for alpha, mse_list in results['krr'].items():
        plt.scatter(range(len(mse_list)), mse_list, 
                   label=f'KRR (α={alpha})', marker='x', s=100)
    
    plt.yscale('log')
    plt.xlabel('Function Index')
    plt.ylabel('MSE (log scale)')
    plt.title('In-Context Learning vs. Kernel Ridge Regression')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparison_results.png"), dpi=300)
    
    return model


def visualize_2d_function(model, true_function, device, output_dir):
    """Visualize the learned function in 2D"""
    # Create a 2D grid of points
    n = 50
    x1 = np.linspace(-1, 1, n)
    x2 = np.linspace(-1, 1, n)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.stack([X1.flatten(), X2.flatten()], axis=1)
    
    # Generate true function values
    true_values = true_function(points).reshape(n, n)
    
    # Generate context points
    context_size = 15
    np.random.seed(42)
    context_x = np.random.uniform(-1, 1, (context_size, 2))
    context_y = true_function(context_x)
    
    # Create a model prediction function
    def predict(x):
        model.eval()
        with torch.no_grad():
            # Create sequence with context points
            tokens = []
            for i in range(context_size):
                token = np.concatenate((context_x[i], [context_y[i]]), axis=0)
                tokens.append(token)
            
            # Add the query point
            all_preds = []
            
            # Make predictions for each point in batches
            batch_size = 100
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_preds = []
                
                for j in range(len(batch_x)):
                    # Clone context tokens
                    seq = tokens.copy()
                    # Add query point
                    query = np.concatenate((batch_x[j], [0.0]), axis=0)
                    seq.append(query)
                    
                    # Convert to tensor
                    prompt_seq = torch.tensor(np.stack(seq, axis=0), 
                                            dtype=torch.float32, 
                                            device=device).unsqueeze(0)
                    
                    # Get prediction
                    pred = model(prompt_seq).item()
                    batch_preds.append(pred)
                
                all_preds.extend(batch_preds)
            
            return np.array(all_preds)
    
    # Get model predictions
    pred_values = predict(points).reshape(n, n)
    
    # Create plots
    plt.figure(figsize=(18, 6))
    
    # True function
    plt.subplot(1, 3, 1)
    plt.pcolormesh(X1, X2, true_values, shading='auto', cmap='viridis')
    plt.colorbar(label='f(x)')
    plt.scatter(context_x[:, 0], context_x[:, 1], c='red', s=50, edgecolor='white', label='Context points')
    plt.title('True Function')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    
    # Predicted function
    plt.subplot(1, 3, 2)
    plt.pcolormesh(X1, X2, pred_values, shading='auto', cmap='viridis')
    plt.colorbar(label='f(x)')
    plt.scatter(context_x[:, 0], context_x[:, 1], c='red', s=50, edgecolor='white', label='Context points')
    plt.title('ICL Transformer Prediction')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    
    # Difference
    plt.subplot(1, 3, 3)
    plt.pcolormesh(X1, X2, np.abs(true_values - pred_values), shading='auto', cmap='magma')
    plt.colorbar(label='|Difference|')
    plt.scatter(context_x[:, 0], context_x[:, 1], c='black', s=50, edgecolor='white', label='Context points')
    plt.title('Absolute Difference')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2d_function_viz.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Create default config
    config = Config()
    
    # Train model
    model = train_model(config)