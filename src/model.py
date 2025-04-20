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
    def __init__(self, d_model, max_len=100):
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


class ICLTransformer(nn.Module):
    """
    In-Context Learning Transformer model with Fourier feature enhancement.
    
    Args:
        d_token (int): Dimension of the token (input_dim + 1)
        d_model (int): Model dimension
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
    """
    def __init__(self, d_token, d_model=256, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_token = d_token
        self.d_model = d_model
        
        # Optional Fourier feature projection for input - helps with RBF approximation
        self.fourier_proj = FourierFeatureProjection(d_token, d_model*2)
        
        # Input projection (after Fourier features)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # GPT-2 configuration
        config = GPT2Config(
            vocab_size=1,  # Not used for continuous inputs
            n_positions=128,  # Maximum context length (increased)
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            activation_function="gelu_new",  # Better activation
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=True,
            scale_attn_weights=True
        )
        
        # GPT-2 model backbone
        self.transformer = GPT2Model(config)
        
        # Output network with layer normalization and multiple layers - designed for kernel approximation
        self.output_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize all weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with appropriate scales"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, prompt_seq):
        """
        Forward pass through the model.
        
        Args:
            prompt_seq (torch.Tensor): Input sequence of shape [batch_size, seq_len, d_token]
            
        Returns:
            torch.Tensor: Predictions of shape [batch_size]
        """
        batch_size, seq_len, _ = prompt_seq.shape
        
        # Apply Fourier feature projection first
        h = self.fourier_proj(prompt_seq)
        
        # Project to model dimension
        h = self.input_proj(h)
        
        # Apply positional encoding
        h = self.pos_encoder(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        # Create attention mask (allows each position to attend to all prior positions)
        attention_mask = torch.ones(batch_size, seq_len).to(prompt_seq.device)
        
        # Forward pass through transformer
        outputs = self.transformer(
            inputs_embeds=h,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Project to output using more sophisticated output network
        # Get the representation at the last position (query position)
        query_hidden = hidden_states[:, -1]
        
        # Project to scalar output
        prediction = self.output_net(query_hidden).squeeze(-1)
        
        return prediction


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