# src/eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import torch.nn as nn

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

def evaluate_model(model, task_name, n_dims, batch_size, n_positions, task_scale=0.25, n_runs=10, device="cuda", baselines=None):
    """
    Evaluate the model on a specific task with varying numbers of context examples.
    This follows Greg Yang's evaluation approach as closely as possible.
    
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
    
    batch_offset = 0
    for run in tqdm(range(n_runs), desc="Evaluation runs"):
        # Create task with the same scale as training
        task = get_task(task_name, n_dims, batch_size, scale=task_scale)
        
        # Generate samples
        xs = generate_samples_for_task(task, n_positions, n_dims, batch_size, device)
        ys = task.evaluate(xs)
        
        # Evaluate transformer for all points
        with torch.no_grad():
            # Get predictions for all positions
            all_preds = model(xs, ys)
            
            # Calculate MSE for each point
            point_wise_metrics = ((all_preds - ys.unsqueeze(-1)) ** 2).squeeze(-1).cpu()
            
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
    
    # Calculate metrics using Greg Yang's aggregation function
    results = {}
    results["transformer"] = aggregate_metrics(all_transformer_metrics)
    
    for baseline in baselines:
        results[baseline.name] = aggregate_metrics(baseline_metrics[baseline.name])
    
    return results

def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Aggregates metrics following Greg Yang's approach.
    Takes a tensor of shape [n_samples, n_points] and returns statistics.
    """
    results = {}
    results["mean_error_by_context"] = metrics.mean(dim=0).tolist()
    results["std_error_by_context"] = metrics.std(dim=0, unbiased=True).tolist()
    
    # Calculate bootstrap confidence intervals
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :].tolist()
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :].tolist()
    
    return results

def plot_learning_curves(results, output_dir):
    """
    Plot learning curves for all models showing how error decreases with more context.
    This matches Greg Yang's visualization approach.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, data in results.items():
        mean_errors = data["mean_error_by_context"]
        
        # Get confidence intervals from bootstrap or standard error
        if "bootstrap_low" in data and "bootstrap_high" in data:
            low_errors = data["bootstrap_low"]
            high_errors = data["bootstrap_high"]
        else:
            std_errors = data["std_error_by_context"]
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
    plt.title("In-Context Learning Performance on Gaussian Distribution", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300)
    plt.close()
    
    # Plot normalized version to show relative improvement
    plt.figure(figsize=(12, 8))
    
    for model_name, data in results.items():
        mean_errors = data["mean_error_by_context"]
        
        # Get confidence intervals
        if "bootstrap_low" in data and "bootstrap_high" in data:
            low_errors = data["bootstrap_low"]
            high_errors = data["bootstrap_high"]
        else:
            std_errors = data["std_error_by_context"]
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
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint if available
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        # Use model parameters from checkpoint config but task from current config
        config.model = checkpoint_config.model
        print("Using model configuration from checkpoint")
    
    print(f"Model configuration:")
    print(f"  Task: {config.task.name}")
    print(f"  Input dimensions: {config.model.n_dims}")
    print(f"  Embedding dimension: {config.model.n_embd}")
    print(f"  Layers: {config.model.n_layer}")
    print(f"  Heads: {config.model.n_head}")
    
    # Initialize model with the correct parameters
    model = GregTransformer(
        n_dims=config.model.n_dims,
        n_positions=config.model.n_positions, 
        n_embd=config.model.n_embd,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head
    ).to(device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    task_scale = getattr(config.task, 'task_scale', 0.25)  # Get task_scale or default to 0.25
    results = evaluate_model(
        model=model,
        task_name=config.task.name,
        n_dims=config.model.n_dims,
        batch_size=32,  # Fixed batch size for evaluation
        n_positions=config.model.n_positions,
        task_scale=task_scale,
        n_runs=args.n_runs,
        device=device
    )
    
    # Save results
    with open(os.path.join(args.output_dir, f"{config.task.name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning curves
    plot_learning_curves(results, args.output_dir)
    
    # Print summary
    print("\nEvaluation Results:")
    for model_name, data in results.items():
        mean_final = data["mean_error_by_context"][-1]
        print(f"{model_name}: Final Error = {mean_final:.6f}")
    
    # Print relative improvement
    transformer_errors = results["transformer"]["mean_error_by_context"]
    if transformer_errors[0] > 0:
        relative_improvement = (transformer_errors[0] - transformer_errors[-1]) / transformer_errors[0] * 100
        print(f"\nTransformer relative improvement: {relative_improvement:.2f}%")

if __name__ == "__main__":
    main()