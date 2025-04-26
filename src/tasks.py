import math
from typing import TYPE_CHECKING, Callable, Tuple, Optional, Dict, Any, final

import torch
import numpy as np

from utils import set_random_seed
from config import Config

class Task:
    def __init__(self, n_dims: int, seed: Optional[int] = None, **kwargs):
        self.n_dims = n_dims
        if seed is not None:
            set_random_seed(seed)

    def evaluate(self, xs: torch.Tensor, add_noise: bool, noise_scale: float = 0.0) -> torch.Tensor:
        """
        Semantics:
            Evaluate the task on input xs.
            Optionally add noise of mean 0 and standard deviation noise_scale.
        """
        raise NotImplementedError

    def sample(self, batch_size: int, n_points: int,
               add_noise: bool, noise_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Semantics:
            Sample inputs and corresponding outputs.
        
        Args:
            batch_size: Number of batches
            n_points: Number of points (~ sequence length) per batch
            
        Returns:
            xs: Input tensor [batch_size, n_points, n_dims]
            ys: Output tensor [batch_size, n_points]
        """            
        xs = torch.randn(batch_size, n_points, self.n_dims)
        return xs, self.evaluate(xs, add_noise=add_noise, noise_scale=noise_scale)


@final
class SinusoidalRegression(Task):
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        task_scale: float, 
        x_range: float, 
        freq_min: float,
        freq_max: float,
        seed: Optional[int] = None,
        **kwargs):
        """
        Initialize the SinusoidalRegression task.
        
        Args:
            n_dims: Input dimensionality (only the first dimension is used)
            scale: Scale factor for amplitudes
            x_range: Range for input sampling (-x_range, x_range)
            freq_min: Minimum frequency for the sine wave
            freq_max: Maximum frequency for the sine wave
            batch_size: Samples in a single batch
        """
        super().__init__(n_dims=n_dims, seed=seed)
        self.scale = task_scale
        self.x_range = x_range
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.amplitude = torch.empty(batch_size).uniform_(0.5, 1.5) * self.scale
        self.frequency = torch.empty(batch_size).uniform_(freq_min, freq_max)
        self.phase = torch.empty(batch_size).uniform_(0, 2 * math.pi)

    def evaluate(self, xs: torch.Tensor, add_noise: bool, noise_scale: float = 0.0) -> torch.Tensor:
        """
        Apply sinusoidal function y = A sin(ωx + φ)
        """
        assert xs.ndim == 2 or (xs.ndim == 3 and xs.shape[2] == 1), (
            "--> SinusoidalRegression::evaluate(): Input must be 2D or 3D with last dimension of size 1."
        )
        assert xs.shape[0] == self.amplitude.shape[0], (
            "--> SinusoidalRegression::evaluate(): Batch size of input must match predefined batch size."
        )
        if xs.ndim == 3:
            xs = xs.squeeze(-1)
        
        amplitude = self.amplitude.to(xs.device).unsqueeze(1)
        frequency = self.frequency.to(xs.device).unsqueeze(1)
        phase = self.phase.to(xs.device).unsqueeze(1)
        y = amplitude * torch.sin(frequency * xs + phase)
        if add_noise:
            y += torch.randn_like(y) * noise_scale
        
        return y
    
    def sample(self, batch_size: int, n_points: int,
               add_noise: bool, noise_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override sample method to use uniform distribution for x values in range (-x_range, x_range).
        This ensures even coverage across multiple periods of the sine wave.
        a
        Args:
            batch_size: Number of batches
            n_points: Number of points per batch
            
        Returns:
            xs: Input tensor [batch_size, n_points, n_dims]
            ys: Output tensor [batch_size, n_points]
        """
        xs = torch.empty(batch_size, n_points, self.n_dims).uniform_(-self.x_range, self.x_range)        
        return xs, self.evaluate(xs, add_noise=add_noise, noise_scale=noise_scale)


@final
class TaskSampler:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def get_task(task_name: str, n_dims: int, batch_size: int, **kwargs) -> Task:

        if task_name == 'sinusoidal':
            task_scale = kwargs.pop('task_scale', None)
            x_range = kwargs.pop('x_range', None)
            freq_min = kwargs.pop('freq_min', None)
            freq_max = kwargs.pop('freq_max', None)
            assert (task_scale is not None and x_range is not None and
                    freq_min is not None and freq_max is not None), (
                "--> TaskSampler::get_task(): Missing required parameters for SinusoidalRegression."
            )
            return SinusoidalRegression(
                n_dims=n_dims,
                batch_size=batch_size,
                task_scale=task_scale,
                x_range=x_range,
                freq_min=freq_min,
                freq_max=freq_max,
                **kwargs
            )
        else:
            raise NotImplementedError()

    def __call__(self) -> Task:
        return TaskSampler.get_task(
            task_name=self.config.task.name,
            n_dims=self.config.model.n_dims,
            batch_size=self.config.training.batch_size,
            **{k: v for k, v in vars(self.config.task).items() if k not in ['name']}
        )




# class LinearRegression(Task):
#     def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
#         """scale: a constant by which to scale the randomly sampled weights."""
#         super().__init__(n_dims, batch_size, seed=None)  # Ignore the seed parameter in parent class
#         self.scale = scale

#         if pool_dict is None and seeds is None:
#             self.w_b = torch.randn(batch_size, n_dims, 1)
#         elif seeds is not None:
#             self.w_b = torch.zeros(batch_size, n_dims, 1)
#             generator = torch.Generator()
#             assert len(seeds) == batch_size
#             for i, seed in enumerate(seeds):
#                 generator.manual_seed(seed)
#                 self.w_b[i] = torch.randn(n_dims, 1, generator=generator)
#         else:
#             assert "w" in pool_dict
#             indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
#             self.w_b = pool_dict["w"][indices]

#     def evaluate(self, xs_b):
#         w_b = self.w_b.to(xs_b.device)
#         ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
#         return ys_b
        
#     @staticmethod
#     def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
#         return {"w": torch.randn(num_tasks, n_dims, 1)}

# class QuadraticRegression(Task):
#     def __init__(self, n_dims, batch_size, scale=1, seed=None):
#         super().__init__(n_dims, batch_size, seed=seed)
#         self.scale = scale
#         self.w1 = torch.randn(batch_size, n_dims, 1) * scale
#         self.w2 = torch.randn(batch_size, n_dims, 1) * scale

#     def evaluate(self, xs):
#         w1 = self.w1.to(xs.device)
#         w2 = self.w2.to(xs.device)
#         return (xs @ w1)[:, :, 0] + (xs ** 2 @ w2)[:, :, 0]


# class ReLUNetwork(Task):
#     def __init__(self, n_dims, batch_size, hidden_size=100, scale=1, seed=None):
#         super().__init__(n_dims, batch_size, seed=seed)
#         self.scale = scale
#         self.hidden_size = hidden_size
#         self.w1 = torch.randn(batch_size, n_dims, hidden_size) * scale
#         self.w2 = torch.randn(batch_size, hidden_size, 1) * scale

#     def evaluate(self, xs):
#         w1 = self.w1.to(xs.device)
#         w2 = self.w2.to(xs.device)
#         hidden = torch.relu(xs @ w1)
#         return (hidden @ w2)[:, :, 0]

# class KernelRFFRegression(Task):
#     def __init__(self, n_dims, batch_size, pool_dict, scale=1.0, seed=None, seeds=None, **kwargs):
#         """
#         In-context linear regression in a fixed RFF feature space.

#         Args:
#             pool_dict: {'phi_weight': Tensor[D,n_dims], 'phi_bias': Tensor[D]}
#             scale: scale for target w sampling
#             seeds: optional list of seeds for w sampling
#         """
#         super().__init__(n_dims, batch_size, seed)
#         self.scale = scale
#         self.phi_weight = pool_dict['phi_weight']  # [D, n_dims]
#         self.phi_bias = pool_dict['phi_bias']      # [D]
#         D = self.phi_weight.shape[0]
#         # Sample linear weights in feature space
#         if seeds is None:
#             self.w_b = torch.randn(batch_size, D, 1) * scale
#         else:
#             assert len(seeds) == batch_size
#             self.w_b = torch.zeros(batch_size, D, 1)
#             for i, sd in enumerate(seeds):
#                 g = torch.Generator().manual_seed(sd)
#                 self.w_b[i] = torch.randn(D, 1, generator=g) * scale
    
#     def sample(self, batch_size, n_points, xs_seeds=None, ys_seeds=None):
#         """
#         Override sample method to use standard Gaussian for input distribution.
#         This creates cleaner RBF kernel mappings than uniform distributions.
        
#         Args:
#             batch_size: Number of batches
#             n_points: Number of points per batch
#             xs_seeds: Optional list of seeds for input sampling
#             ys_seeds: Optional list of seeds for output sampling
            
#         Returns:
#             xs: Input tensor [batch_size, n_points, n_dims]
#             ys: Output tensor [batch_size, n_points]
#         """
#         # Set temporary random states if seeds are provided
#         prev_state = None
#         if xs_seeds is not None:
#             assert len(xs_seeds) == batch_size, "Number of seeds must match batch size"
#             prev_state = torch.get_rng_state()
            
#         # Sample input points from standard Gaussian
#         xs = torch.randn(batch_size, n_points, self.n_dims)
        
#         # Reset random state if we changed it
#         if prev_state is not None:
#             torch.set_rng_state(prev_state)
        
#         # Set seed for outputs if provided
#         if ys_seeds is not None:
#             assert len(ys_seeds) == batch_size, "Number of seeds must match batch size"
#             prev_state = torch.get_rng_state()
#             for i, seed in enumerate(ys_seeds):
#                 torch.manual_seed(seed)
#                 # This is required for some generators that need randomness
                
#         # Get outputs from task
#         ys = self.evaluate(xs)
        
#         # Reset random state if we changed it
#         if prev_state is not None:
#             torch.set_rng_state(prev_state)
            
#         return xs, ys

#     def evaluate(self, xs):
#         # xs: [batch_size, n_points, n_dims]
#         # Compute RFF features: phi(x) = sqrt(2/D) * cos(x W^T + b)
#         batch, N, _ = xs.shape
#         W = self.phi_weight.to(xs.device)       # [D,n_dims]
#         b = self.phi_bias.to(xs.device)         # [D]
#         x_flat = xs.reshape(batch * N, -1)      # [batch*N, n_dims]
#         proj = x_flat @ W.t() + b               # [batch*N, D]
#         phi = torch.cos(proj) * math.sqrt(2.0 / W.shape[0])
#         phi = phi.reshape(batch, N, -1)         # [batch, N, D]
#         # Compute y = phi @ w
#         w = self.w_b.to(xs.device)              # [batch, D,1]
#         y = (phi @ w)[:, :, 0]                  # [batch, N]
#         return y