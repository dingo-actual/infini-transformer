from typing import Optional

import torch
from torch import nn

from compressive_memory import CompressiveMemory


class InfiniTransformer(nn.Module):
    """Transformer layer with compressive memory."""
    def __init__(self, dim_input: int, dim_hidden: int, dim_key: int, dim_value: int, num_heads: int, segment_len: int, update="linear", dropout: float = 0.0):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            segment_len (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(InfiniTransformer, self).__init__()
        
        # Multi-head attention
        self.attn = CompressiveMemory(dim_input, dim_key, dim_value, num_heads, segment_len, update)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_input)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            mask (torch.Tensor, optional): Attention mask of shape (seq_len, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        
        # Apply multi-head attention, followed by MLP and layer normalization with residual connection.
        x_ = self.attn(x, mask)
        x_ = self.mlp(x_)
        
        return self.layer_norm(x_ + x)
