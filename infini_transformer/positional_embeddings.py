from typing import Optional

import torch
from torch import nn



class RoPEEmbeddings(nn.Module):
    """Implements rotary positional embeddings (RoPE) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    (https://arxiv.org/abs/2104.09864).
    
    Modifications have been made to make it compatible with both Infini-Attention
    and Mixture-of-Experts."""
    def __init__(self, dim: int, seq_len: int, dim_embedding_pct: float = 0.5, base: int = 10000):
        """Instantiate the module.

        Args:
            dim (int): Key/Value dimension of the attention layer.
            seq_len (int): Maximum sequence length.
            dim_embedding_pct (float): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
            base (int, optional): Base used for calculating thetas. Defaults to 10000.
        """
        super(RoPEEmbeddings, self).__init__()
        
        # Record input parameters
        self.dim = dim
        self.effective_dim = int(dim * dim_embedding_pct)
        self.seq_len = seq_len
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        self.last_offset = 0
        
        # Initialize cos and sin component matrices
        self._calculate_cos_sin_components()
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg
        
    def _calculate_cos_sin_components(self, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> None:
        """Calculate the cosine and sine component matrices for the rotary positional embeddings.
        Uses multidimensional extension of theta as defined in Sec 3.2.2 as well as equation (34)
        from the RoFormer paper

        Args:
            offset (int, optional): Position offset for Infini-Former compatibility. Defaults to 0.
            select_mask (Optional[torch.Tensor], optional): Mask to select a subset of the positional embeddings for Mixture-of-Depths compatibility. Defaults to None.
        """
        if select_mask is None:
            # Calculate matrix of angles: thetas[i,j] = base^(-2 * ceil(i/2)) * (j + offset)
            thetas = torch.repeat_interleave(
                (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1))).unsqueeze(-1).repeat((1, self.seq_len)), 
                repeats=2, 
                dim=0
            )
            # Multiply by index positions, then transpose to get correct shape
            thetas *= torch.arange(1 + offset, self.seq_len + 1 + offset).unsqueeze(0)
            thetas = thetas.transpose(0, 1)
            
            # Calculate cosine and sine of thetas and reshape for downstream use
            # Shape: (1, 1, seq_len, effective_dim)
            self.cos_component = thetas.cos().unsqueeze(0).unsqueeze(0)
            self.sin_component = thetas.sin().unsqueeze(0).unsqueeze(0)
        else:
            # (n_obs, select_seq_len)
            select_ixs = 1 + offset + torch.argwhere(select_mask)[:, 1].view((select_mask.size(0), -1))
            # (n_obs, select_seq_len, effective_dim)
            select_ixs = select_ixs.unsqueeze(-1).repeat((1, 1, self.effective_dim))
            # (effective_dim, select_seq_len)
            thetas = torch.repeat_interleave(
                (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1))).unsqueeze(-1).repeat((1, select_ixs.size(1))), 
                repeats=2, 
                dim=0
            )
            # (n_obs, select_seq_len, effective_dim)
            thetas = thetas.transpose(0, 1).unsqueeze(0).repeat((select_mask.size(0), 1, 1))
            thetas *= select_ixs
            
            # Calculate cosine and sine of thetas and reshape for downstream use
            self.cos_component = thetas.cos().unsqueeze(1)
            self.sin_component = thetas.sin().unsqueeze(1)
        
    def forward(self, x: torch.Tensor, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
            offset (int, optional): Position offset for Infini-Former compatibility. Defaults to 0.
            select_mask (Optional[torch.Tensor], optional): Mask to select a subset of the positional embeddings for Mixture-of-Depths compatibility. Defaults to None.

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if offset != self.last_offset:
            self._calculate_cos_sin_components(offset=offset, select_mask=select_mask)
            self.last_offset = offset
            cos_sin_recalculated = True
        else:
            cos_sin_recalculated = False
        
        if self.dim_embedding_pct < 1.0:
            x_pos = x[..., :self.effective_dim]
            x_pass = x[..., self.effective_dim:]
        else:
            x_pos = x
        
        # If no selection mask is specified, add embeddings as usual
        if select_mask is None:
            # If the sequence length is less than the maximum sequence length, perform calculations
            # with truncated cos_component and sin_component, along the sequence axis
            if x.size(2) < self.seq_len:
                x_cos = self.cos_component[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.sin_component[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1)
            # Otherwise, perform calculations with the full cos_component and sin_component
            else:
                x_cos = self.cos_component.repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.sin_component.repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        # If a selection mask is specified, incorporate it into the positional embeddings
        else:
            if not cos_sin_recalculated:
                self._calculate_cos_sin_components(offset=offset, select_mask=select_mask)
                self.last_offset = offset
            x_cos = self.cos_component.repeat(1, x_pos.size(1), 1, 1) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.sin_component.repeat(1, x_pos.size(1), 1, 1)
            
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out