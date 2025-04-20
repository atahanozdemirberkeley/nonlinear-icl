# src/kernels.py
import numpy as np
import torch
import torch.nn as nn

class RFFFeatureMap:
    """
    Random Fourier Features (RFF) for approximating RBF kernels.
    
    This class implements the Random Fourier Features technique to approximate
    an RBF kernel, allowing efficient computation of kernel operations.
    
    Args:
        input_dim (int): Dimension of the input data
        rff_dim (int): Dimension of the RFF feature space
        sigma (float): Bandwidth parameter for the RBF kernel
    """
    def __init__(self, input_dim, rff_dim, sigma):
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.W = np.random.normal(scale=1.0/(sigma**2), size=(rff_dim, input_dim))
        self.b = np.random.uniform(0, 2 * np.pi, size=(rff_dim,))
        
    def transform(self, x):
        """
        Transform input data using RFF.
        
        Args:
            x: Input data of shape (..., input_dim)
            
        Returns:
            RFF transformed data of shape (..., rff_dim)
        """
        proj = x @ self.W.T + self.b
        return np.sqrt(2.0 / self.rff_dim) * np.cos(proj)

    def to_torch(self):
        """Convert the RFF feature map to a PyTorch module"""
        return TorchRFFFeatureMap(self.input_dim, self.rff_dim, self.sigma, self.W, self.b)


class TorchRFFFeatureMap(nn.Module):
    """PyTorch implementation of Random Fourier Features"""
    def __init__(self, input_dim, rff_dim, sigma, W=None, b=None):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim = rff_dim
        self.sigma = sigma
        
        if W is None:
            W = np.random.normal(scale=1.0/(sigma**2), size=(rff_dim, input_dim))
        if b is None:
            b = np.random.uniform(0, 2 * np.pi, size=(rff_dim,))
            
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False)
        
    def forward(self, x):
        """
        Transform input data using RFF.
        
        Args:
            x: Input data tensor of shape (..., input_dim)
            
        Returns:
            RFF transformed data of shape (..., rff_dim)
        """
        proj = torch.matmul(x, self.W.T) + self.b
        return torch.sqrt(torch.tensor(2.0 / self.rff_dim)) * torch.cos(proj)


class KernelRidgeRegression:
    """
    Kernel Ridge Regression implementation.
    
    Args:
        feature_map: RFF feature map for kernel approximation
        alpha (float): Regularization parameter
    """
    def __init__(self, feature_map, alpha):
        self.feature_map = feature_map
        self.alpha = alpha
        self.weights = None
        self.X_features = None
        
    def fit(self, X, y):
        """
        Fit the kernel ridge regression model.
        
        Args:
            X: Input data of shape (n_samples, input_dim)
            y: Target values of shape (n_samples,)
        """
        # Transform inputs with RFF
        X_features = self.feature_map.transform(X)
        
        # Solve the ridge regression
        n_samples = X.shape[0]
        A = X_features.T @ X_features + self.alpha * np.eye(self.feature_map.rff_dim)
        b = X_features.T @ y
        self.weights = np.linalg.solve(A, b)
        self.X_features = X_features
        
    def predict(self, X_new):
        """
        Make predictions on new data.
        
        Args:
            X_new: New data of shape (n_samples, input_dim)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        X_new_features = self.feature_map.transform(X_new)
        return X_new_features @ self.weights


def generate_rff_function(input_dim, rff_dim, sigma):
    """
    Generate a random function using RFF features.
    
    Args:
        input_dim (int): Dimension of the input data
        rff_dim (int): Dimension of the RFF feature space
        sigma (float): Bandwidth parameter for the RBF kernel
        
    Returns:
        tuple: (function, feature_map)
    """
    feature_map = RFFFeatureMap(input_dim, rff_dim, sigma)
    weights = np.random.randn(rff_dim)
    
    def f(x):
        """Evaluate the random function at input points x"""
        return feature_map.transform(x) @ weights
    
    return f, feature_map