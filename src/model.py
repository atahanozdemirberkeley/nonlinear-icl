# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from copy import deepcopy
import math

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
    Transformer model based on Greg Yang's architecture.
    """
    def __init__(self, d_model, n_positions, n_heads=4, kernel_type='relu'):
        super().__init__()
        self.d_model = d_model
        self.n_positions = n_positions
        self.n_heads = n_heads
        self.kernel_type = kernel_type
        
        # Define head dimension
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Create attention mask (causal/triangular)
        mask = torch.tril(torch.ones(n_positions, n_positions))
        self.register_buffer("mask", mask.view(1, 1, n_positions, n_positions))
    
    def _compute_kernel_function(self, scores):
        """Apply the specified kernel function to attention scores."""
        if self.kernel_type == 'relu':
            return torch.relu(scores)
        elif self.kernel_type == 'gelu':
            return nn.functional.gelu(scores)
        elif self.kernel_type == 'softmax':
            return torch.nn.functional.softmax(scores, dim=-1)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def forward(self, x, inds=None):
        """
        Args:
            x: Input tensor of shape [batch_size, n_positions, d_model]
            inds: Optional indices to compute self-attention for specific positions
                  If None, compute self-attention for all positions
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute specific positions or all
        if inds is not None:
            # Only compute self-attention for specific positions
            positions_to_compute = inds
        else:
            # Compute for all positions
            positions_to_compute = list(range(seq_len))
        
        # First residual block: self-attention
        residual = x
        x = self.norm1(x)
        
        # Multi-head self-attention
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply kernel function
        attn_weights = self._compute_kernel_function(scores)
        
        # Compute weighted sum
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Project back to d_model dimension
        context = self.output_proj(context)
        
        # Add residual connection
        x = residual + context
        
        # Second residual block: feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        # If indices were provided, only return those positions
        if inds is not None:
            return x[:, inds, :]
        
        return x


class ICLModel(nn.Module):
    """
    In-context learning model based on Greg Yang's approach.
    """
    def __init__(self, d_token, n_positions, d_model=128, n_heads=4, kernel_type='relu'):
        super().__init__()
        self.d_token = d_token
        self.n_positions = n_positions
        self.d_model = d_model
        self.n_heads = n_heads
        self.kernel_type = kernel_type
        
        # Input projection
        self.input_proj = nn.Linear(d_token, d_model)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_positions, d_model))
        
        # Transformer layers
        self.transformer = GregTransformer(
            d_model=d_model,
            n_positions=n_positions,
            n_heads=n_heads,
            kernel_type=kernel_type
        )
        
        # Output projection (to scalar prediction)
        self.output_proj = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize linear layers
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, inds=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, n_positions, d_token]
            inds: Optional indices to compute predictions for specific positions
                  If None, compute predictions for all positions
                  
        Returns:
            Predictions tensor of shape [batch_size, n_positions] if inds is None
            or [batch_size, len(inds)] if inds is provided
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimension
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer layers
        x = self.transformer(x, inds=inds)
        
        # Project to scalar predictions
        predictions = self.output_proj(x).squeeze(-1)
        
        return predictions


class GPT2ICLModel(nn.Module):
    """
    In-context learning model based on GPT2, matching the original in-context-learning repo.
    
    This uses the HuggingFace Transformers GPT2Model as the backbone.
    """
    def __init__(self, d_token, n_positions, d_model=256, n_heads=8, n_layer=12):
        super().__init__()
        self.d_token = d_token
        self.n_positions = n_positions
        self.d_model = d_model
        
        # Configure the GPT2 model
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_heads,
            resid_pdrop=0.0,  # No dropout
            embd_pdrop=0.0,   # No dropout
            attn_pdrop=0.0,   # No dropout
            use_cache=False
        )
        
        # Input projection
        self.input_proj = nn.Linear(d_token, d_model)
        
        # GPT2 backbone
        self._backbone = GPT2Model(configuration)
        
        # Output projection (to scalar prediction)
        self.output_proj = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layers
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, n_positions, d_token]
            
        Returns:
            Predictions tensor of shape [batch_size, n_positions]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimension
        x = self.input_proj(x)
        # x = x + self._backbone.wpe.weight[:seq_len, :]
        
        # Apply GPT2 model
        output = self._backbone(inputs_embeds=x).last_hidden_state
        
        # Project to scalar predictions
        predictions = self.output_proj(output).squeeze(-1)
        import pdb; pdb.set_trace()
        return predictions


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


class MLPICLModel(nn.Module):
    """
    Simple MLP model for in-context learning without attention.
    Uses sequential feed-forward layers to process the input sequence.
    """
    def __init__(self, d_token, n_positions, hidden_dim=256, n_layers=4):
        super().__init__()
        self.d_token = d_token
        self.n_positions = n_positions
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create separate MLPs for each sequence length
        self.mlps = nn.ModuleList()
        
        # Create MLPs for all possible prefix lengths
        for i in range(1, n_positions + 1):
            # Input size for this prefix length
            input_size = i * d_token
            
            # Create an MLP for this prefix length
            layers = []
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_dim, 1))
            
            self.mlps.append(nn.Sequential(*layers))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_token]
        Returns:
            Predictions for all positions
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.n_positions, device=x.device)
        
        # Process for the current sequence length
        if seq_len <= self.n_positions:
            # Flatten the sequence
            x_flat = x.reshape(batch_size, -1)
            
            # Use the appropriate MLP for this sequence length
            mlp_idx = seq_len - 1  # 0-indexed
            output = self.mlps[mlp_idx](x_flat)
            
            # The model predicts the next value after the sequence
            if seq_len < self.n_positions:
                outputs[:, seq_len] = output.squeeze(-1)
            
        return outputs