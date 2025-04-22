# src/eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime

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

def evaluate_model(model, task_name, n_dims, batch_size, n_positions, task_scale=1.0, n_runs=10, device="cuda", baselines=None, config=None):
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
    
    # Get model type from config
    model_type = getattr(config.model, 'model_type', 'gpt2') if config else 'gpt2'
    
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
                    
                    # FIX: Create masked ys to prevent label leakage
                    # Use context up to position i-1, add a dummy for position i
                    dummy = torch.zeros_like(ys[:, :1])
                    ys_i_masked = torch.cat([ys[:, :i], dummy], dim=1)
                    
                    # Create prompt sequence with masked target
                    prompt_seq = create_prompt_sequence(xs_i, ys_i_masked, config)
                    
                    # Get prediction for position i (the last position with the dummy token)
                    if model_type == 'mlp':
                        # For MLP model - it returns predictions for all positions
                        pred_i = model(prompt_seq)[:, i]
                    else:
                        # For transformer models - they return prediction at the last position
                        pred_i = model(prompt_seq)[:, -1]
                
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
    
    # Save with consistent naming pattern
    plt.savefig(os.path.join(output_dir, f"{task_name}_d{n_dims}_learning_curve.png"), dpi=300)
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
    
    # Save with consistent naming pattern
    plt.savefig(os.path.join(output_dir, f"{task_name}_d{n_dims}_normalized_learning_curve.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL model for probability distributions")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model or checkpoint')
    parser.add_argument('--task_name', type=str, default='sinusoidal', help='Task name for evaluation')
    parser.add_argument('--n_dims', type=int, default=1, help='Input dimensions for evaluation')
    parser.add_argument('--task_scale', type=float, default=1, help='Scale for task sampling')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of evaluation runs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Base directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_type', type=str, default='gpt2', help='Model type (gpt2 or mlp)')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get task-specific output dir and timestamp from model path
    model_dir = os.path.dirname(args.model_path)  # e.g., .../sinusoidal_d1_s1.0_x5.0_f0.2-5.0/20250422_013544
    timestamp = os.path.basename(model_dir)  # e.g., 20250422_013544
    task_dir_name = os.path.basename(os.path.dirname(model_dir))  # e.g., sinusoidal_d1_s1.0_x5.0_f0.2-5.0
    
    # Mirror the exact directory structure but in eval_results
    # eval_results/sinusoidal_d1_s1.0_x5.0_f0.2-5.0/20250422_013544
    eval_dir = os.path.join(args.output_dir, task_dir_name, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Determine config path from model directory
    config_path = os.path.join(model_dir, "config.yaml")
    
    # Try to load the configuration from the model directory
    config = None
    model_params = {}
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = Config.load(config_path)
        
        # Extract model parameters from config
        model_params = {
            "d_model": config.model.d_model,
            "n_heads": config.model.n_heads,
            "n_layer": getattr(config.model, 'n_layer', 12),
            "n_positions": config.model.n_positions,
            "n_dims": config.model.n_dims,
            "d_token": getattr(config.model, 'd_token', None),
            "model_type": getattr(config.model, 'model_type', 'gpt2')
        }
        
        print(f"Found model configuration: d_model={model_params['d_model']}, "
              f"n_dims={model_params['n_dims']}, model_type={model_params['model_type']}")
    else:
        # If no config.yaml, try to parse from directory/file name
        print("No config.yaml found. Attempting to parse parameters from model path.")
        
        # Extract dimensions from directory name (e.g., sinusoidal_d1, d2, etc.
        dir_name = os.path.basename(os.path.dirname(args.model_path))
        
        # Look for patterns like d1, d2, etc.
        import re
        d_match = re.search(r'd(\d+)', dir_name)
        if d_match:
            n_dims = int(d_match.group(1))
            print(f"Extracted n_dims={n_dims} from directory name")
            args.n_dims = n_dims  # Update n_dims
        
        # Default parameters if we couldn't extract from path
        model_params = {
            "d_model": 256,  # Default to 256 as in most configs
            "n_heads": 8,
            "n_layer": 12,
            "n_positions": 41,
            "n_dims": args.n_dims,  # Use provided or extracted n_dims
            "model_type": args.model_type
        }
        
        print(f"Using default model configuration: d_model={model_params['d_model']}, "
              f"n_dims={model_params['n_dims']}, model_type={model_params['model_type']}")
    
    # Calculate d_token if not specified
    if 'd_token' not in model_params or model_params['d_token'] is None:
        model_params['d_token'] = model_params['n_dims'] + 1
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with the correct architecture parameters
    if model_params['model_type'] == 'gpt2':
        model = GPT2ICLModel(
            d_token=model_params['d_token'],
            n_positions=model_params['n_positions'],
            d_model=model_params['d_model'],
            n_heads=model_params['n_heads'],
            n_layer=model_params['n_layer']
        ).to(device)
    elif model_params['model_type'] == 'mlp':
        from model import MLPICLModel
        hidden_dim = getattr(config.model, 'hidden_dim', 256) if config else 256
        n_layers = getattr(config.model, 'n_layers', 4) if config else 4
        
        model = MLPICLModel(
            d_token=model_params['d_token'],
            n_positions=model_params['n_positions'],
            hidden_dim=hidden_dim,
            n_layers=n_layers
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_params['model_type']}")
    
    # Load model state
    print(f"Loading model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
        return
    
    # Task parameters for evaluation
    task_kwargs = {}
    if config and config.task.name == "sinusoidal":
        # Add sinusoidal-specific parameters if available
        task_kwargs["x_range"] = getattr(config.task, 'x_range', 5.0)
        task_kwargs["freq_min"] = getattr(config.task, 'freq_min', 0.5)
        task_kwargs["freq_max"] = getattr(config.task, 'freq_max', 2.0)
    
    # Evaluate model
    print(f"Evaluating model on {args.task_name} task with {args.n_runs} runs")
    results = evaluate_model(
        model=model,
        task_name=args.task_name,
        n_dims=model_params['n_dims'],  # Use the model's n_dims
        batch_size=args.batch_size,
        n_positions=model_params['n_positions'],
        task_scale=args.task_scale,
        n_runs=args.n_runs,
        device=device,
        config=config
    )
    
    # Plot learning curves
    plot_learning_curves(results, eval_dir, task_name=args.task_name, n_dims=model_params['n_dims'])
    
    # Save results to the timestamped directory
    results_file = os.path.join(eval_dir, f"{args.task_name}_d{model_params['n_dims']}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Display results summary
    print("\n======= EVALUATION RESULTS =======")
    print(f"Task: {args.task_name}, Dimensions: {model_params['n_dims']}, Scale: {args.task_scale}")
    print(f"Number of runs: {args.n_runs}, Points per sequence: {model_params['n_positions']}")
    print("\nMean Squared Error (MSE) by context length:")
    
    transformer_errors = results["transformer"]["mean_error_by_context"]
    for i, err in enumerate(transformer_errors):
        print(f"  Context {i}: {err:.6f}")
    
    print("\nAverage MSE by model:")
    for model_name, model_results in results.items():
        print(f"  {model_name}: {model_results['average_mse']:.6f}")
    
    print(f"\nFew-shot MSE (contexts 1-5): {results['transformer']['few_shot_mse']:.6f}")
    print(f"Many-shot MSE (contexts 6+): {results['transformer']['many_shot_mse']:.6f}")
    print(f"\nFull details saved to {results_file}")
    
    # Update file paths for learning curve plots
    learning_curve_path = os.path.join(eval_dir, f"{args.task_name}_d{model_params['n_dims']}_learning_curve.png")
    normalized_curve_path = os.path.join(eval_dir, f"{args.task_name}_d{model_params['n_dims']}_normalized_learning_curve.png")
    
    print(f"Learning curve plots saved to:")
    print(f"  {learning_curve_path}")
    print(f"  {normalized_curve_path}")
    print("===================================")

if __name__ == "__main__":
    main()