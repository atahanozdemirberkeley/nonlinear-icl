# src/eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import torch.nn as nn

from model import GPT2ICLModel  # Use the same model as training
from tasks import get_task
from config import Config
from train import create_prompt_sequence  # Import create_prompt_sequence function

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NNBaseline:
    """k-Nearest Neighbors baseline model"""
    def __init__(self, n_neighbors, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN-{n_neighbors}"

    def predict(self, xs, ys):
        """
        Make predictions using k-NN
        
        Args:
            xs: Input tensor [batch_size, n_points, n_dims]
            ys: Target tensor [batch_size, n_points]
            
        Returns:
            Predictions tensor [batch_size, n_points]
        """
        batch_size, n_points, _ = xs.shape
        predictions = torch.zeros_like(ys)
        
        for i in range(n_points):
            if i == 0:
                # For first point, predict zero
                predictions[:, 0] = 0.0
                continue
                
            # For each point, use previous points as training data
            for b in range(batch_size):
                train_xs = xs[b, :i]
                train_ys = ys[b, :i]
                test_x = xs[b, i:i+1]
                
                # Compute distances
                dists = torch.sqrt(((train_xs - test_x)**2).sum(dim=1))
                
                if self.weights == "uniform":
                    weights = torch.ones_like(dists)
                else:  # "distance"
                    weights = 1.0 / (dists + 1e-10)
                    # Handle exact matches (infinite weights)
                    inf_mask = torch.isinf(weights)
                    if inf_mask.any():
                        weights[inf_mask] = 0.0
                        weights[inf_mask] = 1.0
                
                # Get indices of k nearest neighbors
                k = min(self.n_neighbors, i)
                _, indices = torch.topk(dists, k, largest=False)
                
                # Compute weighted average
                weights_sum = weights[indices].sum()
                if weights_sum > 0:
                    predictions[b, i] = (weights[indices] * train_ys[indices]).sum() / weights_sum
                else:
                    predictions[b, i] = train_ys.mean()  # Fallback to mean
        
        return predictions

class AveragingBaseline:
    """Simple averaging baseline model"""
    def __init__(self):
        self.name = "Averaging"

    def predict(self, xs, ys):
        """
        Make predictions by averaging previous outputs
        
        Args:
            xs: Input tensor [batch_size, n_points, n_dims]
            ys: Target tensor [batch_size, n_points]
            
        Returns:
            Predictions tensor [batch_size, n_points]
        """
        batch_size, n_points, _ = xs.shape
        predictions = torch.zeros_like(ys)
        
        for i in range(n_points):
            if i == 0:
                # For first point, predict zero
                predictions[:, 0] = 0.0
            else:
                # For other points, predict mean of previous points
                predictions[:, i] = torch.mean(ys[:, :i], dim=1)
        
        return predictions

def evaluate_model(model, task_name, n_dims, batch_size, n_positions, task_scale=0.25, n_runs=10, device="cuda", baselines=None, config=None):
    """
    Evaluate the model on a specific task with varying numbers of context examples.
    
    Args:
        model: Trained model
        task_name: Name of the task
        n_dims: Input dimension
        batch_size: Batch size for evaluation
        n_positions: Maximum number of data points per sequence
        task_scale: Scale for task sampling (should match training)
        n_runs: Number of evaluation runs
        device: Device to use
        baselines: List of baseline models to compare against
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create baseline models if not provided
    if baselines is None:
        baselines = [
            NNBaseline(n_neighbors=1),
            NNBaseline(n_neighbors=3),
            AveragingBaseline()
        ]
    
    # Initialize metrics tensor to gather results for all points and all runs
    # Shape: [n_runs * batch_size, n_positions]
    all_transformer_metrics = torch.zeros(n_runs * batch_size, n_positions)
    baseline_metrics = {}
    for baseline in baselines:
        baseline_metrics[baseline.name] = torch.zeros(n_runs * batch_size, n_positions)
    
    # Get x_range and frequency range from config if available
    x_range = getattr(config.task, 'x_range', 5.0) if config else 5.0
    freq_min = getattr(config.task, 'freq_min', 0.5) if config else 0.5
    freq_max = getattr(config.task, 'freq_max', 2.0) if config else 2.0
    
    # Create frequency and x_range info string for printing
    freq_info = f"frequency range: {freq_min}-{freq_max}" if task_name == "sinusoidal" else ""
    x_info = f"x_range: {x_range}" if task_name == "sinusoidal" else ""
    
    print(f"Evaluating {task_name} task with {n_runs} runs, {batch_size} examples per run")
    print(f"Task parameters: scale={task_scale}, {freq_info}, {x_info}")
    print(f"Evaluating in-context performance from 0 to {n_positions-1} context examples")
    
    batch_offset = 0
    for run in tqdm(range(n_runs), desc="Evaluation runs"):
        # Create task with the same scale as training but with a random seed
        # This ensures we evaluate on unseen distributions
        seed = None  # Set to None to get random sampling
        
        # Add all relevant parameters for sinusoidal task
        task_kwargs = {"scale": task_scale, "seed": seed}
        if task_name == "sinusoidal" and config:
            task_kwargs["x_range"] = x_range
            task_kwargs["freq_min"] = freq_min
            task_kwargs["freq_max"] = freq_max
            
        task = get_task(task_name, n_dims, batch_size=batch_size, **task_kwargs)
        
        # Generate samples
        xs, ys = task.sample(batch_size, n_positions)
        xs, ys = xs.to(device), ys.to(device)
        
        # Evaluate transformer for all points
        with torch.no_grad():
            # Get predictions for all positions
            all_transformer_preds = []
            
            # For each context length (0 to n_positions-1), get prediction for position i
            for i in range(n_positions):
                if i == 0:
                    # No context, predict 0
                    pred_i = torch.zeros(batch_size, device=device)
                else:
                    # Use context up to position i
                    xs_i = xs[:, :i+1]
                    ys_i = ys[:, :i+1]
                    
                    # Create prompt sequence
                    prompt_seq = create_prompt_sequence(xs_i, ys_i, config)
                    
                    # Get prediction for position i
                    pred_i = model(prompt_seq)[:, i]
                
                # Store prediction
                all_transformer_preds.append(pred_i)
            
            # Stack predictions [batch_size, n_positions]
            transformer_preds = torch.stack(all_transformer_preds, dim=1)
            
            # Calculate MSE for each point
            point_wise_metrics = ((transformer_preds - ys) ** 2).cpu()
            
            # Store results
            all_transformer_metrics[batch_offset:batch_offset+batch_size, :] = point_wise_metrics
            
        # Evaluate baselines
        xs_cpu = xs.cpu()
        ys_cpu = ys.cpu()
        
        for baseline in baselines:
            baseline_preds = baseline.predict(xs_cpu, ys_cpu)
            point_wise_errors = ((baseline_preds - ys_cpu) ** 2)
            baseline_metrics[baseline.name][batch_offset:batch_offset+batch_size, :] = point_wise_errors
        
        batch_offset += batch_size
    
    # Calculate metrics
    results = {}
    
    # Transformer metrics
    mean_errors = all_transformer_metrics.mean(dim=0)
    std_errors = all_transformer_metrics.std(dim=0) / torch.sqrt(torch.tensor(all_transformer_metrics.size(0)))
    
    results["transformer"] = {
        "average_mse": mean_errors.mean().item(),
        "mean_error_by_context": mean_errors.tolist(),
        "std_error_by_context": std_errors.tolist(),
        "few_shot_mse": mean_errors[1:6].mean().item() if len(mean_errors) > 5 else float('nan'),
        "many_shot_mse": mean_errors[6:].mean().item() if len(mean_errors) > 6 else float('nan'),
    }
    
    # Baseline metrics
    for baseline in baselines:
        mean_errors = baseline_metrics[baseline.name].mean(dim=0)
        std_errors = baseline_metrics[baseline.name].std(dim=0) / torch.sqrt(torch.tensor(baseline_metrics[baseline.name].size(0)))
        
        results[baseline.name] = {
            "average_mse": mean_errors.mean().item(),
            "mean_error_by_context": mean_errors.tolist(),
            "std_error_by_context": std_errors.tolist(),
            "few_shot_mse": mean_errors[1:6].mean().item() if len(mean_errors) > 5 else float('nan'),
            "many_shot_mse": mean_errors[6:].mean().item() if len(mean_errors) > 6 else float('nan'),
        }
    
    return results

def plot_learning_curves(results, output_dir, task_name="sinusoidal", n_dims=1):
    """
    Plot learning curves for all models showing how error decreases with more context.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
        task_name: Name of the task
        n_dims: Number of input dimensions
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, data in results.items():
        mean_errors = data["mean_error_by_context"]
        std_errors = data["std_error_by_context"]
        
        # Calculate confidence intervals (mean ± std error)
        low_errors = [m - s for m, s in zip(mean_errors, std_errors)]
        high_errors = [m + s for m, s in zip(mean_errors, std_errors)]
        
        x = np.arange(len(mean_errors))
        plt.plot(x, mean_errors, label=model_name, linewidth=2)
        plt.fill_between(
            x,
            low_errors,
            high_errors,
            alpha=0.2
        )
    
    plt.xlabel("Number of In-Context Examples", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.title(f"In-Context Learning Performance on {task_name} (d={n_dims})", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300)
    plt.close()
    
    # Plot normalized version to show relative improvement
    plt.figure(figsize=(12, 8))
    
    for model_name, data in results.items():
        mean_errors = data["mean_error_by_context"]
        std_errors = data["std_error_by_context"]
        
        # Calculate confidence intervals (mean ± std error)
        low_errors = [m - s for m, s in zip(mean_errors, std_errors)]
        high_errors = [m + s for m, s in zip(mean_errors, std_errors)]
        
        # Normalize by first error
        if mean_errors[0] > 0:
            normalized_errors = np.array(mean_errors) / mean_errors[0]
            normalized_low = np.array(low_errors) / mean_errors[0]
            normalized_high = np.array(high_errors) / mean_errors[0]
            
            x = np.arange(len(normalized_errors))
            plt.plot(x, normalized_errors, label=model_name, linewidth=2)
            plt.fill_between(
                x,
                normalized_low,
                normalized_high,
                alpha=0.2
            )
    
    plt.xlabel("Number of In-Context Examples", fontsize=14)
    plt.ylabel("Normalized Error (relative to zero-shot)", fontsize=14)
    plt.title("Normalized In-Context Learning Performance", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "normalized_learning_curves.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate In-Context Learning Model for Probability Distributions")
    parser.add_argument('--config', type=str, default='configs/gaussian.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of evaluation runs')
    parser.add_argument('--task_name', type=str, help='Override task name from config')
    parser.add_argument('--n_dims', type=int, help='Override input dimensions')
    parser.add_argument('--n_positions', type=int, default=41, help='Number of positions for evaluation (default: 60)')
    parser.add_argument('--task_scale', type=float, help='Override task scale for evaluation')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration from file
    config = Config.load(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override task name if provided
    if args.task_name:
        config.task.name = args.task_name
        print(f"Overriding task name to: {args.task_name}")
    
    # Override dimensions if provided
    dims_changed = False
    if args.n_dims:
        config.model.n_dims = args.n_dims
        dims_changed = True
        print(f"Overriding input dimensions to: {args.n_dims}")
    
    # Override task scale if provided
    if args.task_scale:
        config.task.task_scale = args.task_scale
        print(f"Overriding task scale to: {args.task_scale}")
    
    # Override n_positions for evaluation
    original_n_positions = None
    if args.n_positions:
        # Store original value for creating the model
        original_n_positions = config.model.n_positions
        # Set new value for evaluation
        config.model.n_positions = args.n_positions
        print(f"Setting evaluation positions to: {args.n_positions}")
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Calculate d_token if not specified or dimensions changed
    if dims_changed or config.model.d_token is None:
        config.model.d_token = config.model.n_dims + 1
        print(f"Setting d_token to {config.model.d_token} (n_dims + 1)")
    else:
        # Ensure d_token is at least n_dims + 1
        min_d_token = config.model.n_dims + 1
        if config.model.d_token < min_d_token:
            print(f"Warning: d_token ({config.model.d_token}) is less than n_dims + 1 ({min_d_token})")
            print(f"Setting d_token to {min_d_token}")
            config.model.d_token = min_d_token
    
    # For loading the model, use the ORIGINAL n_positions to match the pretrained model
    model_n_positions = original_n_positions or config.model.n_positions
    
    # Use GPT2ICLModel to match the model used in training
    # Get number of layers from config (default to 12 if not specified)
    n_layer = getattr(config.model, 'n_layer', 12)
    
    model = GPT2ICLModel(
        d_token=config.model.d_token,
        n_positions=model_n_positions,  # Use original position count
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layer=n_layer
    ).to(device)
    
    # Load model weights
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights directly")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Falling back to using the same number of positions as the checkpoint")
        
        # Try to detect the model's actual n_positions from the error message
        import re
        error_str = str(e)
        match = re.search(r'torch\.Size\(\[(\d+), \d+\]\)', error_str)
        if match:
            detected_positions = int(match.group(1))
            print(f"Detected {detected_positions} positions in checkpoint")
            
            # Recreate model with detected positions
            model = GPT2ICLModel(
                d_token=config.model.d_token,
                n_positions=detected_positions,
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                n_layer=n_layer
            ).to(device)
            
            # Try loading again
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Update config to use detected positions
            config.model.n_positions = detected_positions
            print(f"Successfully loaded model with {detected_positions} positions")
        else:
            raise
    
    print(f"Loaded model from {args.model_path}")
    print(f"Model configuration:")
    print(f"  Task: {config.task.name}")
    print(f"  Input dimensions: {config.model.n_dims}")
    print(f"  Model dimensions: {config.model.d_model}")
    print(f"  Layers: {config.model.n_layer}")
    print(f"  Heads: {config.model.n_heads}")
    
    # Evaluate model
    task_scale = getattr(config.task, 'task_scale', 0.25)  # Get task_scale or default to 0.25
    
    # Print evaluation parameters
    print(f"\nEvaluation parameters:")
    print(f"  Task: {config.task.name}")
    print(f"  Dimensions: {config.model.n_dims}")
    print(f"  Positions: {config.model.n_positions}")
    print(f"  Task scale: {task_scale}")
    print(f"  Number of runs: {args.n_runs}")
    
    results = evaluate_model(
        model=model,
        task_name=config.task.name,
        n_dims=config.model.n_dims,
        batch_size=32,  # Fixed batch size for evaluation
        n_positions=config.model.n_positions,
        task_scale=task_scale,
        n_runs=args.n_runs,
        device=device,
        config=config  # Pass config to the evaluation function
    )
    
    # Save results
    with open(os.path.join(args.output_dir, f"{config.task.name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning curves
    plot_learning_curves(results, args.output_dir, config.task.name, config.model.n_dims)
    
    # Print detailed summary to terminal
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {config.task.name} (d={config.model.n_dims})")
    print("="*50)
    
    # Print result summary for each model
    for model_name, data in results.items():
        print(f"\n{model_name} Performance:")
        mean_errors = data["mean_error_by_context"]
        
        # Print key context points (start, middle, end)
        print(f"  Zero-shot (0 examples): {mean_errors[0]:.6f}")
        
        # Print few-shot (points 5, 10, 20)
        if len(mean_errors) > 5:
            print(f"  Few-shot (5 examples): {mean_errors[5]:.6f}")
        if len(mean_errors) > 10:
            print(f"  Few-shot (10 examples): {mean_errors[10]:.6f}")
        if len(mean_errors) > 20:
            print(f"  Medium-shot (20 examples): {mean_errors[20]:.6f}")
        
        # Print final context
        print(f"  Full-context ({len(mean_errors)-1} examples): {mean_errors[-1]:.6f}")
        
        # Calculate improvement
        if mean_errors[0] > 0:
            rel_improvement = (mean_errors[0] - mean_errors[-1]) / mean_errors[0] * 100
            print(f"  Relative improvement: {rel_improvement:.2f}%")
    
    # If transformer is in results, compare to baselines
    if "transformer" in results:
        print("\nComparison to Baselines:")
        transformer_final = results["transformer"]["mean_error_by_context"][-1]
        for model_name, data in results.items():
            if model_name != "transformer":
                baseline_final = data["mean_error_by_context"][-1]
                ratio = transformer_final / baseline_final if baseline_final > 0 else float('inf')
                if ratio < 1:
                    print(f"  Transformer outperforms {model_name} by {(1/ratio - 1)*100:.2f}%")
                else:
                    print(f"  {model_name} outperforms Transformer by {(ratio - 1)*100:.2f}%")
    
    print("\nResults saved to:", os.path.join(args.output_dir, f"{config.task.name}_results.json"))
    print("Learning curves saved to:", os.path.join(args.output_dir, "learning_curves.png"))
    print("="*50)

if __name__ == "__main__":
    main()