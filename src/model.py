import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class InContextModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer, n_head):
        super().__init__()
        self.n_dims = n_dims
        
        # Create GPT-2 configuration
        config = GPT2Config(
            vocab_size=1,  # We'll use continuous values
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * n_embd,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=0,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )
        
        # Initialize GPT-2 model
        self.model = GPT2LMHeadModel(config)
        
        # Input embedding layer
        self.input_embedding = nn.Linear(n_dims, n_embd)
        
        # Output projection layer
        self.output_projection = nn.Linear(n_embd, n_dims)

    def forward(self, xs, ys=None):
        # xs: [batch_size, n_points, n_dims]
        # ys: [batch_size, n_points] or None
        
        batch_size, n_points, n_dims = xs.shape
        
        # Create input embeddings
        x_emb = self.input_embedding(xs)  # [batch_size, n_points, n_embd]
        
        if ys is not None:
            # During training, we have both input and output
            y_emb = self.input_embedding(ys.unsqueeze(-1))  # [batch_size, n_points, n_embd]
            
            # Interleave x and y embeddings
            combined = torch.stack([x_emb, y_emb], dim=2)  # [batch_size, n_points, 2, n_embd]
            combined = combined.view(batch_size, 2 * n_points, -1)  # [batch_size, 2*n_points, n_embd]
            
            # Get model outputs
            outputs = self.model(inputs_embeds=combined)
            hidden_states = outputs.last_hidden_state
            
            # Project back to n_dims
            predictions = self.output_projection(hidden_states[:, 1::2])  # Take every other position
            return predictions
        else:
            # During inference, we only have input
            outputs = self.model(inputs_embeds=x_emb)
            hidden_states = outputs.last_hidden_state
            
            # Project back to n_dims
            predictions = self.output_projection(hidden_states)
            return predictions 