import torch
import math

class Task:
    def __init__(self, n_dims, batch_size):
        self.n_dims = n_dims
        self.batch_size = batch_size

    def evaluate(self, xs):
        raise NotImplementedError

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, scale=1):
        super().__init__(n_dims, batch_size)
        self.scale = scale
        self.w = torch.randn(batch_size, n_dims, 1) * scale

    def evaluate(self, xs):
        w = self.w.to(xs.device)
        return (xs @ w)[:, :, 0]

class QuadraticRegression(Task):
    def __init__(self, n_dims, batch_size, scale=1):
        super().__init__(n_dims, batch_size)
        self.scale = scale
        self.w1 = torch.randn(batch_size, n_dims, 1) * scale
        self.w2 = torch.randn(batch_size, n_dims, 1) * scale

    def evaluate(self, xs):
        w1 = self.w1.to(xs.device)
        w2 = self.w2.to(xs.device)
        return (xs @ w1)[:, :, 0] + (xs ** 2 @ w2)[:, :, 0]

class ReLUNetwork(Task):
    def __init__(self, n_dims, batch_size, hidden_size=100, scale=1):
        super().__init__(n_dims, batch_size)
        self.scale = scale
        self.hidden_size = hidden_size
        self.w1 = torch.randn(batch_size, n_dims, hidden_size) * scale
        self.w2 = torch.randn(batch_size, hidden_size, 1) * scale

    def evaluate(self, xs):
        w1 = self.w1.to(xs.device)
        w2 = self.w2.to(xs.device)
        hidden = torch.relu(xs @ w1)
        return (hidden @ w2)[:, :, 0]

def get_task(task_name, n_dims, batch_size, **kwargs):
    task_map = {
        'linear': LinearRegression,
        'quadratic': QuadraticRegression,
        'relu': ReLUNetwork
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}")
    return task_map[task_name](n_dims, batch_size, **kwargs) 