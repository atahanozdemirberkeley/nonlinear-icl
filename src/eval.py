# src/eval.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from kernels import KernelRidgeRegression

def evaluate_model(model, f, input_dim, context_length, n_samples=100, device="cuda"):
    """
    Evaluate the model on new data.
    
    Args:
        model: Trained model
        f: Target function
        input_dim: Input dimension
        context_length: Number of context points
        n_samples: Number of evaluation samples
        device: Torch device
        
    Returns:
        float: Mean squared error
    """
    model.eval()
    
    mse_losses = []
    with torch.no_grad():
        for _ in range(n_samples):
            # Generate a new sequence
            xs = np.random.uniform(-1, 1, (context_length + 1, input_dim))
            ys = f(xs)
            
            # Prepare the input sequence
            tokens = []
            for i in range(context_length):
                token = np.concatenate((xs[i], [ys[i]]), axis=0)
                tokens.append(token)
            
            # Add the query point (with placeholder y=0)
            last = np.concatenate((xs[-1], [0.0]), axis=0)
            tokens.append(last)
            
            # Convert to tensor and get prediction
            prompt_seq = torch.tensor(np.stack(tokens, axis=0), 
                                     dtype=torch.float32, 
                                     device=device).unsqueeze(0)
            pred = model(prompt_seq).item()
            
            # True target
            target = ys[-1]
            
            # Calculate error
            mse = (pred - target) ** 2
            mse_losses.append(mse)
    
    avg_mse = np.mean(mse_losses)
    return avg_mse


def compare_with_krr(model, feature_map, input_dim, context_length, n_samples=10, alphas=[0.1, 1.0, 10.0]):
    """
    Compare model performance with Kernel Ridge Regression.
    
    Args:
        model: Trained model
        feature_map: RFF feature map
        input_dim: Input dimension
        context_length: Number of context points
        n_samples: Number of comparison samples
        alphas: List of regularization parameters to try
        
    Returns:
        dict: Comparison results
    """
    device = next(model.parameters()).device
    model.eval()
    
    results = {
        'transformer': [],
        'krr': {alpha: [] for alpha in alphas},
        'functions': []
    }
    
    for i in range(n_samples):
        # Generate a random function
        weights = np.random.randn(feature_map.rff_dim)
        
        def f(x):
            return feature_map.transform(x) @ weights
        
        results['functions'].append(f)
        
        # Evaluate transformer on this function
        transformer_mse = evaluate_model(model, f, input_dim, context_length, 
                                        n_samples=20, device=device)
        results['transformer'].append(transformer_mse)
        
        # Compare with KRR for different alphas
        for alpha in alphas:
            # Generate context points
            xs = np.random.uniform(-1, 1, (context_length, input_dim))
            ys = f(xs)
            
            # Fit KRR
            krr = KernelRidgeRegression(feature_map, alpha)
            krr.fit(xs, ys)
            
            # Evaluate on new points
            test_xs = np.random.uniform(-1, 1, (20, input_dim))
            test_ys = f(test_xs)
            preds = krr.predict(test_xs)
            
            # Compute MSE
            krr_mse = np.mean((preds - test_ys) ** 2)
            results['krr'][alpha].append(krr_mse)
    
    return results


def plot_comparison(results):
    """
    Plot comparison results.
    
    Args:
        results: Results from compare_with_krr
    """
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
    
    plt.savefig('comparison_results.png', dpi=300)
    plt.close()