import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Implements rotary positional embeddings (RoPE) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    (https://arxiv.org/abs/2104.09864)"""
    def __init__(self, dim: int, seq_len: int, dim_embedding_pct: float = 0.5, base: int = 10000):
        """Instantiate the module.

        Args:
            dim (int): Key/Value dimension of the attention layer.
            seq_len (int): Maximum sequence length.
            dim_embedding_pct (float): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
            base (int, optional): Base used for calculating thetas. Defaults to 10000.
        """
        super(RotaryPositionalEmbeddings, self).__init__()
        
        # Record input parameters
        self.dim = dim
        self.effective_dim = int(dim * dim_embedding_pct)
        self.seq_len = seq_len
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        
        # Initialize cos and sin component matrices
        self._calculate_cos_sin_components()
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg
        
    def _calculate_cos_sin_components(self, offset: int = 0) -> None:
        """Calculate the cosine and sine component matrices for the rotary positional embeddings.
        Uses multidimensional extension of theta as defined in Sec 3.2.2 as well as equation (34)
        from the RoFormer paper

        Args:
            offset (int, optional): Position multiple offset. Defaults to 0.
        """
        # Calculate matrix of angles: thetas[i,j] = base^(-2 * ceil(i/2)) * (j + offset)
        # Shape: (effective_dim, seq_len)
        thetas = torch.repeat_interleave(
            (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1))).unsqueeze(-1).repeat((1, self.seq_len)), 
            repeats=2, 
            dim=0
        )
        thetas *= torch.arange(1 + offset, self.seq_len + 1 + offset).unsqueeze(0)
        
        # Calculate cosine and sine of thetas and reshape for downstream use
        self.cos_component = thetas.cos().unsqueeze(0).unsqueeze(0)
        self.sin_component = thetas.sin().unsqueeze(0).unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if self.dim_embedding_pct < 1.0:
            x_pos = x[..., :self.effective_dim]
            x_pass = x[..., self.effective_dim:]
        else:
            x_pos = x
        # If the sequence length is less than the maximum sequence length, perform calculations
        # with truncated cos_component and sin_component, along the sequence axis
        if x.size(2) < self.seq_len:
            x_cos = self.cos_component[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.sin_component[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        # Otherwise, perform calculations with the full cos_component and sin_component
        else:
            x_cos = self.cos_component.repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.sin_component.repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out