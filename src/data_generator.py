import torch
import numpy as np
from scipy import stats

def generate_gaussian_samples(batch_size, n_points, n_dims, means, std=1.0):
    """
    Generate samples from Gaussian distributions with specified means
    
    Args:
        batch_size: Number of batches
        n_points: Number of samples per batch
        n_dims: Dimensionality of each sample
        means: Tensor of means [batch_size, n_dims]
        std: Standard deviation
        
    Returns:
        Tensor of samples [batch_size, n_points, n_dims]
    """
    # Expand means to match output shape
    expanded_means = means.unsqueeze(1).expand(batch_size, n_points, n_dims)
    
    # Generate samples
    samples = expanded_means + torch.randn(batch_size, n_points, n_dims) * std
    
    return samples

def generate_poisson_samples(batch_size, n_points, n_dims, rates):
    """
    Generate samples from Poisson distributions with specified rates
    
    Args:
        batch_size: Number of batches
        n_points: Number of samples per batch
        n_dims: Dimensionality of each sample
        rates: Tensor of rate parameters [batch_size, 1]
        
    Returns:
        Tensor of samples [batch_size, n_points, n_dims]
    """
    # Expand rates to match output shape
    expanded_rates = rates.unsqueeze(1).expand(batch_size, n_points, n_dims)
    
    # Generate Poisson samples
    samples = torch.poisson(expanded_rates)
    
    return samples

def generate_bernoulli_samples(batch_size, n_points, n_dims, probs):
    """
    Generate samples from Bernoulli distributions with specified probabilities
    
    Args:
        batch_size: Number of batches
        n_points: Number of samples per batch
        n_dims: Dimensionality of each sample
        probs: Tensor of probability parameters [batch_size, 1]
        
    Returns:
        Tensor of samples [batch_size, n_points, n_dims]
    """
    # Expand probs to match output shape
    expanded_probs = probs.unsqueeze(1).expand(batch_size, n_points, n_dims)
    
    # Generate Bernoulli samples
    samples = torch.bernoulli(expanded_probs)
    
    return samples

def generate_exponential_samples(batch_size, n_points, n_dims, rates):
    """
    Generate samples from Exponential distributions with specified rates
    
    Args:
        batch_size: Number of batches
        n_points: Number of samples per batch
        n_dims: Dimensionality of each sample
        rates: Tensor of rate parameters [batch_size, 1]
        
    Returns:
        Tensor of samples [batch_size, n_points, n_dims]
    """
    # Expand rates to match output shape
    expanded_rates = rates.unsqueeze(1).expand(batch_size, n_points, n_dims)
    
    # Generate using exponential formula: -log(U)/rate where U ~ Uniform(0,1)
    u = torch.rand(batch_size, n_points, n_dims)
    samples = -torch.log(u) / expanded_rates
    
    return samples

def generate_gamma_samples(batch_size, n_points, n_dims, shape, scale):
    """
    Generate samples from Gamma distributions with specified shape and scale
    
    Args:
        batch_size: Number of batches
        n_points: Number of samples per batch
        n_dims: Dimensionality of each sample
        shape: Tensor of shape parameters [batch_size, 1]
        scale: Tensor of scale parameters [batch_size, 1]
        
    Returns:
        Tensor of samples [batch_size, n_points, n_dims]
    """
    # Expand parameters to match output shape
    expanded_shape = shape.unsqueeze(1).expand(batch_size, n_points, n_dims)
    expanded_scale = scale.unsqueeze(1).expand(batch_size, n_points, n_dims)
    
    # Generate Gamma samples using PyTorch's distribution
    gamma_dist = torch.distributions.Gamma(expanded_shape, 1.0/expanded_scale)
    samples = gamma_dist.sample()
    
    return samples

def generate_samples_for_task(task, n_samples, n_dims, batch_size, device):
    """
    Generate input samples for a probability distribution task.
    
    Args:
        task: Task instance 
        n_samples: Number of samples per batch
        n_dims: Input dimension
        batch_size: Number of batches
        device: Torch device
        
    Returns:
        Tensor of shape [batch_size, n_samples, n_dims]
    """
    # Generate random input samples from a standard normal distribution
    samples = torch.randn(batch_size, n_samples, n_dims)
    
    # Normalize the samples to have unit norm
    samples = samples / (torch.norm(samples, dim=2, keepdim=True) + 1e-8)
    
    # Scale samples to be in a reasonable range
    samples = samples * 0.5
    
    return samples.to(device) 