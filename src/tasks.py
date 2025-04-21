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
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super().__init__(n_dims, batch_size, seed=None)  # Ignore the seed parameter in parent class
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(batch_size, n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(batch_size, n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b
        
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

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

class SinusoidalRegression(Task):
    def __init__(self, n_dims, batch_size, scale=1.0, seed=None, seeds=None, x_range=5.0, 
                 freq_min=0.5, freq_max=2.0, **kwargs):
        """
        Initialize the SinusoidalRegression task.
        
        Args:
            n_dims: Input dimensionality (only the first dimension is used)
            batch_size: Number of samples in a batch
            scale: Scale factor for amplitudes
            seed: Random seed for reproducibility (global)
            seeds: Optional list of seeds (one per batch element)
            x_range: Range for input sampling (-x_range, x_range)
            freq_min: Minimum frequency for the sine wave
            freq_max: Maximum frequency for the sine wave
            **kwargs: Additional arguments (ignored, for compatibility with other tasks)
        """
        super().__init__(n_dims, batch_size, seed=seed)
        self.scale = scale
        self.x_range = x_range
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        # Each batch gets different sinusoid parameters
        # If seeds are provided, use them for reproducibility
        if seeds is not None:
            assert len(seeds) == batch_size, "Number of seeds must match batch size"
            
            # Create sinusoid parameters deterministically for each batch element
            self.amplitude = torch.zeros(batch_size)
            self.frequency = torch.zeros(batch_size)
            self.phase = torch.zeros(batch_size)
            
            for i, seed in enumerate(seeds):
                generator = torch.Generator()
                generator.manual_seed(seed)
                self.amplitude[i] = torch.empty(1, generator=generator).uniform_(0.5, 1.5) * scale
                self.frequency[i] = torch.empty(1, generator=generator).uniform_(freq_min, freq_max)
                self.phase[i] = torch.empty(1, generator=generator).uniform_(0, 2 * math.pi)
        else:
            # Random parameters (based on global seed if provided)
            self.amplitude = torch.empty(batch_size).uniform_(0.5, 1.5) * scale
            self.frequency = torch.empty(batch_size).uniform_(freq_min, freq_max)
            self.phase = torch.empty(batch_size).uniform_(0, 2 * math.pi)
    
    def sample(self, batch_size, n_points, xs_seeds=None, ys_seeds=None):
        """
        Override sample method to use uniform distribution for x values in range (-x_range, x_range).
        This ensures even coverage across multiple periods of the sine wave.
        
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
            
        # Sample input points uniformly based on x_range parameter
        xs = torch.empty(batch_size, n_points, self.n_dims).uniform_(-self.x_range, self.x_range)
        
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
        
    def evaluate(self, xs):
        """
        Apply sinusoidal function y = A sin(ωx + φ)
        Using just the first dimension of x if n_dims > 1
        """
        # Use just the first dimension if input is multi-dimensional
        x = xs[:, :, 0]  # Shape: [batch_size, n_points]
        
        # Move parameters to device
        amplitude = self.amplitude.to(xs.device).unsqueeze(1)  # Shape: [batch_size, 1]
        frequency = self.frequency.to(xs.device).unsqueeze(1)  # Shape: [batch_size, 1]
        phase = self.phase.to(xs.device).unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Calculate sin(ωx + φ)
        y = amplitude * torch.sin(frequency * x + phase)
        
        return y

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



def get_task_sampler(task_name, n_dims, batch_size, scale=1.0, num_tasks=None):
    """
    Creates a task sampler function that returns new task instances with optional seeds.
    
    Args:
        task_name: Name of the task
        n_dims: Number of dimensions for input
        batch_size: Batch size
        scale: Scale parameter for task
        num_tasks: If provided, creates a pool of tasks to sample from
        
    Returns:
        A function that creates task instances
    """
    # Create pool dict if num_tasks is provided
    pool_dict = None
    if num_tasks is not None:
        if task_name == "linear":
            pool_dict = LinearRegression.generate_pool_dict(n_dims, num_tasks)
    
    def task_sampler(seed=None, seeds=None):
        task_args = {
            "n_dims": n_dims,
            "batch_size": batch_size,
            "scale": scale,
        }
        
        # Only add pool_dict for tasks that support it (currently only linear)
        if pool_dict is not None and task_name == "linear":
            task_args["pool_dict"] = pool_dict
        
        # Handle multiple seeds (for training with limited distribution pool)
        if seeds is not None:
            task_args["seeds"] = seeds
        
        # Handle single seed - convert to list of seeds
        if seed is not None:
            if task_name == "linear":  # Only for tasks that support per-batch seeds
                # Create a list of seeds derived from the base seed
                generator = torch.Generator().manual_seed(seed)
                derived_seeds = [torch.randint(0, 1000000, (1,), generator=generator).item() 
                                for _ in range(batch_size)]
                task_args["seeds"] = derived_seeds
            else:
                # For tasks that only support a global seed
                task_args["seed"] = seed
            
        task = get_task(task_name, **task_args)
        return task
        
    return task_sampler

def get_task(task_name, n_dims, batch_size, **kwargs):
    task_map = {
        'linear': LinearRegression,
        'quadratic': QuadraticRegression,
        'sinusoidal': SinusoidalRegression,
        'relu': ReLUNetwork
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}")
    return task_map[task_name](n_dims, batch_size, **kwargs) 