# src/data.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ICLDataset(Dataset):
    """
    In-Context Learning Dataset.
    
    This dataset generates sequences of (x, f(x)) pairs for in-context learning,
    where f is a function generated using Random Fourier Features.
    
    Args:
        f: Function to learn
        input_dim (int): Dimension of the input data
        context_length (int): Number of context points
        n_samples (int): Number of samples in the dataset
    """
    def __init__(self, f, input_dim, context_length, n_samples):
        self.f = f
        self.input_dim = input_dim
        self.context_length = context_length
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate k+1 random points
        xs = np.random.uniform(-1, 1, (self.context_length + 1, self.input_dim))
        ys = self.f(xs)
        
        # Format as tokens
        tokens = []
        
        # Create context points (x, y) pairs
        for i in range(self.context_length):
            token = np.concatenate((xs[i], [ys[i]]), axis=0)
            tokens.append(token)
        
        # Add the query point x with placeholder y=0
        query = np.concatenate((xs[-1], [0.0]), axis=0)
        tokens.append(query)
        
        prompt_seq = np.stack(tokens, axis=0)
        target = ys[-1]
        
        return torch.tensor(prompt_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def create_dataloader(f, input_dim, context_length, n_samples, batch_size, shuffle=True):
    """
    Create a dataloader for in-context learning.
    
    Args:
        f: Function to learn
        input_dim (int): Dimension of the input data
        context_length (int): Number of context points
        n_samples (int): Number of samples in the dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = ICLDataset(f, input_dim, context_length, n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)