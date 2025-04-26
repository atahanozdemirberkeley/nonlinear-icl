import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_qkv, dropout_attn):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv

        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head * d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        self.layernorm = nn.LayerNorm(
            normalized_shape=d_model,
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
        )

        self.p_dropout_attn = dropout_attn

    def forward(self, x, mask):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch size, length, d_model]
          mask: controlling the attention pattern, shape [batch size, length, length]
        Returns:
          A single tensor containing the output from this layer
        """

        assert x.dim() == 3, "MHA forward(): x must be a 3D tensor"
        assert mask.dim() == 3, "MHA forward(): mask must be a 3D tensor"
        assert (
            x.shape[0] == mask.shape[0] and x.shape[1] == mask.shape[1]
        ), "MHA forward(): x and mask must have the same batch size and length"
        sequence_len = x.shape[1]

        # shape of x = [batch size, length, d_model]
        # shape of self.w_{q,k,v} = [n_head, d_model, d_qkv]
        # shape of {q,k,v} vector = [batch_size, n_head, length, d_qkv]
        q = torch.einsum("bld,hde->bhle", x, self.w_q)
        k = torch.einsum("bld,hde->bhle", x, self.w_k)
        v = torch.einsum("bld,hde->bhle", x, self.w_v)

        # shape of attention score = [batch_size, n_head, sequence_len, sequence_len]
        # dim=2 is query index, dim=3 is key index
        attention_logit = torch.einsum("bhle,bhme->bhlm", q, k) / math.sqrt(self.d_qkv)

        # shape of mask = [batch size, sequence_len, sequence_len]
        # mask[i,j] == True when if we want masking the i-th query to the j-th key
        mask = mask.unsqueeze(1)  # unsqueeze at head dim
        attention_logit = attention_logit.masked_fill(mask, -1e9)

        # shape of attention_prob = [batch_size, n_head, sequence_len, sequence_len]
        attention_prob = F.dropout(
            F.softmax(attention_logit, dim=-1),
            p=self.p_dropout_attn,
            training=self.training,
        )
        # shape of v = [batch_size, n_head, sequence_len, d_qkv]
        # shape of attention_output = [batch_size, n_head, sequence_len, d_qkv]
        attention_output = torch.einsum("bhlm,bhme->bhle", attention_prob, v)

        # shape of self.w_o = [n_head, d_qkv, d_model]
        # shape of concat_output = [batch size, length, d_model]
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(
            -1, sequence_len, self.n_head * self.d_qkv
        )
        concat_output = torch.einsum("ble,ed->bld", attention_output, self.w_o)
        return self.layernorm(
            x + F.dropout(concat_output, p=self.p_dropout_attn, training=self.training)
        )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ffn, nonlin):
        super().__init__()

        assert nonlin in ["ReLU", "SiLU", "GELU"], f"Unsupported nonlin_fn: {nonlin}"

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            getattr(nn, nonlin)(),
            nn.Dropout(p=dropout_ffn),
            nn.Linear(d_ff, d_model),
        )
        self.layernorm = nn.LayerNorm(
            normalized_shape=d_model,
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
        )
        self.d_model = d_model
        self.d_ff = d_ff
        self.nonlin = nonlin
        self.p_dropout_ffn = dropout_ffn

    def forward(self, x):
        return self.layernorm(
            x + F.dropout(self.ffn(x), p=self.p_dropout_ffn, training=self.training)
        )


class CustomTransformerEncoder(nn.Module):
    def __init__(
        self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
        dropout_attn=0.1, dropout_ffn=0.1, nonlin="ReLU",
    ):
        super().__init__()

        self.attention_layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttention(
                    d_model=d_model, n_head=n_head, d_qkv=d_qkv, dropout_attn=dropout_attn
                ),
                PositionwiseFeedForward(
                    d_model=d_model, d_ff=d_ff, dropout_ffn=dropout_ffn, nonlin=nonlin
                )
            ]) for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        """Runs the Transformer encoder.
        Args:
            x: the input to the Transformer, a tensor of shape [batch size, length, d_model]
            mask: controlling the attention pattern, shape [batch size, length, length]
        Returns:
            output of shape [batch size, length, d_model]
        """
        for attention_layer, ffn_layer in self.attention_layers:
            x = ffn_layer(attention_layer(x, mask))
        return x

    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")


class CustomICLModel(nn.Module):
    def __init__(
        self, d_token, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
        dropout_attn=0.1, dropout_ffn=0.1, nonlin="ReLU"
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_token, d_model)
        self.encoder = CustomTransformerEncoder(
            d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_head=n_head, d_qkv=d_qkv,
            dropout_attn=dropout_attn, dropout_ffn=dropout_ffn, nonlin=nonlin
        )
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        """Runs the ICL model.
        Args:
            x: the input to the model, a tensor of shape [batch size, length, d_token]
            mask: controlling the attention pattern, shape [batch size, length, length]
        Returns:
            output of shape [batch size, length, 1]
        """
        return self.output_proj(self.encoder(self.input_proj(x), mask))