# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from copy import deepcopy

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer models.
    Based on the Transformer paper's sinusoidal encoding.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class FourierFeatureProjection(nn.Module):
    """
    Random Fourier Features projection layer for better handling of RBF-like functions.
    """
    def __init__(self, input_dim, output_dim, scale=10.0):
        super().__init__()
        # Initialize random projection matrix
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim//2) * scale,
            requires_grad=True
        )
        
    def forward(self, x):
        # Project input to higher dimension
        projection = x @ self.weight
        
        # Apply sin and cos
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class GregTransformer(nn.Module):
    """
    Exact implementation of the TransformerModel from Greg Yang's in-context-learning repository.
    """
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(GregTransformer, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


class SimpleICLTransformer(nn.Module):
    """
    Simple Transformer for In-Context Learning.
    
    A lightweight transformer using standard PyTorch modules.
    
    Args:
        d_token (int): Dimension of the token (input_dim + 1)
        d_model (int): Model dimension
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
    """
    def __init__(self, d_token, d_model=256, n_layers=1, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(d_token, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, prompt_seq):
        """
        Forward pass through the model.
        
        Args:
            prompt_seq (torch.Tensor): Input sequence of shape [batch_size, seq_len, d_token]
            
        Returns:
            torch.Tensor: Predictions for the last token
        """
        h = self.input_proj(prompt_seq)
        for layer in self.layers:
            h = layer(h, h)  # causal self-attention
        return self.output_proj(h[:, -1, :]).squeeze(-1)