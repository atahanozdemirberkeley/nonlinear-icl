import torch
import math
import numpy as np

class Task:
    def __init__(self, n_dims, batch_size, seed=None):
        self.n_dims = n_dims
        self.batch_size = batch_size
        
        # Set seed if provided
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def evaluate(self, xs):
        raise NotImplementedError

    def sample(self, batch_size, n_points, xs_seeds=None, ys_seeds=None):
        """
        Sample inputs and corresponding outputs.
        
        Args:
            batch_size: Number of batches
            n_points: Number of points per batch
            xs_seeds: Optional list of seeds for input sampling
            ys_seeds: Optional list of seeds for output sampling
            
        Returns:
            xs: Input tensor [batch_size, n_points, n_dims]
            ys: Output tensor [batch_size, n_points]
        """
        # Set temporary random states if seeds are provided
        prev_state = None
        if xs_seeds is not None:
            assert len(xs_seeds) == batch_size, "Number of seeds must match batch size"
            prev_state = torch.get_rng_state()
            
        # Sample input points
        xs = torch.randn(batch_size, n_points, self.n_dims)
        
        # Reset random state if we changed it
        if prev_state is not None:
            torch.set_rng_state(prev_state)
        
        # Set seed for outputs if provided
        if ys_seeds is not None:
            assert len(ys_seeds) == batch_size, "Number of seeds must match batch size"
            prev_state = torch.get_rng_state()
            for i, seed in enumerate(ys_seeds):
                torch.manual_seed(seed)
                # This is required for some generators that need randomness
                
        # Get outputs from task
        ys = self.evaluate(xs)
        
        # Reset random state if we changed it
        if prev_state is not None:
            torch.set_rng_state(prev_state)
            
        return xs, ys

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, scale=1, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        self.w = torch.randn(batch_size, n_dims, 1) * scale

    def evaluate(self, xs):
        w = self.w.to(xs.device)
        return (xs @ w)[:, :, 0]

class QuadraticRegression(Task):
    def __init__(self, n_dims, batch_size, scale=1, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        self.w1 = torch.randn(batch_size, n_dims, 1) * scale
        self.w2 = torch.randn(batch_size, n_dims, 1) * scale

    def evaluate(self, xs):
        w1 = self.w1.to(xs.device)
        w2 = self.w2.to(xs.device)
        return (xs @ w1)[:, :, 0] + (xs ** 2 @ w2)[:, :, 0]

class ReLUNetwork(Task):
    def __init__(self, n_dims, batch_size, hidden_size=100, scale=1, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        self.hidden_size = hidden_size
        self.w1 = torch.randn(batch_size, n_dims, hidden_size) * scale
        self.w2 = torch.randn(batch_size, hidden_size, 1) * scale

    def evaluate(self, xs):
        w1 = self.w1.to(xs.device)
        w2 = self.w2.to(xs.device)
        hidden = torch.relu(xs @ w1)
        return (hidden @ w2)[:, :, 0]

class GaussianDistribution(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        # Create random weights for linear function that determines mean
        self.w = torch.randn(batch_size, n_dims) * scale
        self.b = torch.randn(batch_size) * scale
        # Set a fixed reasonable variance
        self.sigma = torch.ones(batch_size) * 0.1
        
    def evaluate(self, xs):
        """
        Sample from a Gaussian distribution with mean determined by w*x + b
        """
        w = self.w.to(xs.device)
        b = self.b.to(xs.device)
        sigma = self.sigma.to(xs.device)
        
        # Calculate mean (mu) for each example as w*x + b
        mu = torch.bmm(xs, w.unsqueeze(2)).squeeze(2) + b.unsqueeze(1)
        
        # Sample from the Gaussian distribution
        eps = torch.randn_like(mu)
        samples = mu + eps * sigma.unsqueeze(1)
        
        return samples

class PoissonDistribution(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        # Create random weights for linear function that determines lambda
        self.w = torch.randn(batch_size, n_dims) * scale
        self.b = torch.randn(batch_size) * scale
        
    def evaluate(self, xs):
        """
        Sample from a Poisson distribution with lambda determined by exp(w*x + b)
        """
        w = self.w.to(xs.device)
        b = self.b.to(xs.device)
        
        # Calculate lambda parameter (must be positive)
        lam = torch.exp(torch.bmm(xs, w.unsqueeze(2)).squeeze(2) + b.unsqueeze(1))
        lam = torch.clamp(lam, min=0.1, max=20.0)  # Clamp to reasonable values
        
        # Sample from Poisson distribution
        # Using Gaussian approximation for simplicity (valid for large lambda)
        samples = torch.sqrt(lam) * torch.randn_like(lam) + lam
        samples = torch.round(torch.clamp(samples, min=0))
        
        return samples

class BernoulliDistribution(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        # Create random weights for linear function that determines probability
        self.w = torch.randn(batch_size, n_dims) * scale
        self.b = torch.randn(batch_size) * scale
        
    def evaluate(self, xs):
        """
        Sample from a Bernoulli distribution with probability determined by sigmoid(w*x + b)
        """
        w = self.w.to(xs.device)
        b = self.b.to(xs.device)
        
        # Calculate probability using sigmoid
        logits = torch.bmm(xs, w.unsqueeze(2)).squeeze(2) + b.unsqueeze(1)
        p = torch.sigmoid(logits)
        
        # Sample from Bernoulli distribution
        samples = torch.bernoulli(p)
        
        return samples

class ExponentialDistribution(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        # Create random weights for linear function that determines rate
        self.w = torch.randn(batch_size, n_dims) * scale
        self.b = torch.randn(batch_size) * scale
        
    def evaluate(self, xs):
        """
        Sample from an Exponential distribution with rate parameter determined by exp(w*x + b)
        """
        w = self.w.to(xs.device)
        b = self.b.to(xs.device)
        
        # Calculate rate parameter (must be positive)
        rate = torch.exp(torch.bmm(xs, w.unsqueeze(2)).squeeze(2) + b.unsqueeze(1))
        rate = torch.clamp(rate, min=0.1, max=10.0)  # Clamp to reasonable values
        
        # Sample from Exponential distribution using inverse transform sampling
        u = torch.rand_like(rate)
        samples = -torch.log(u) / rate
        
        return samples

class GammaDistribution(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None):
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        # Create random weights for linear functions that determine shape and rate
        self.w_shape = torch.randn(batch_size, n_dims) * scale
        self.b_shape = torch.randn(batch_size) * scale
        self.w_rate = torch.randn(batch_size, n_dims) * scale
        self.b_rate = torch.randn(batch_size) * scale
        
    def evaluate(self, xs):
        """
        Sample from a Gamma distribution with shape and rate parameters
        """
        w_shape = self.w_shape.to(xs.device)
        b_shape = self.b_shape.to(xs.device)
        w_rate = self.w_rate.to(xs.device)
        b_rate = self.b_rate.to(xs.device)
        
        # Calculate shape parameter (must be positive)
        shape = torch.exp(torch.bmm(xs, w_shape.unsqueeze(2)).squeeze(2) + b_shape.unsqueeze(1))
        shape = torch.clamp(shape, min=0.5, max=5.0)  # Clamp to reasonable values
        
        # Calculate rate parameter (must be positive)
        rate = torch.exp(torch.bmm(xs, w_rate.unsqueeze(2)).squeeze(2) + b_rate.unsqueeze(1))
        rate = torch.clamp(rate, min=0.5, max=5.0)  # Clamp to reasonable values
        
        # Sample from Gamma distribution
        # Using approximation based on sum of exponentials for integer shape values
        # For non-integer shape values, this is a reasonable approximation
        samples = torch.zeros_like(shape)
        for i in range(int(shape.max().item()) + 1):
            # Add exponential samples based on shape
            mask = (shape >= i).float()
            u = torch.rand_like(rate)
            samples += mask * (-torch.log(u) / rate)
        
        return samples

def get_task(task_name, n_dims, batch_size, **kwargs):
    task_map = {
        'linear': LinearRegression,
        'quadratic': QuadraticRegression,
        'relu': ReLUNetwork,
        'gaussian': GaussianDistribution,
        'poisson': PoissonDistribution,
        'bernoulli': BernoulliDistribution,
        'exponential': ExponentialDistribution,
        'gamma': GammaDistribution
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}")
    return task_map[task_name](n_dims, batch_size, **kwargs) 