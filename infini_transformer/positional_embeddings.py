import math
from typing import Optional

import torch
from torch import nn



class PositionEmbeddings(nn.Module):
    """Base class for different types of positional embeddings."""
    def __init__(self):
        super(PositionEmbeddings, self).__init__()
        
    def forward(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class RoPEEmbeddings(PositionEmbeddings):
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
        
        # Initialize theta matrix
        self._calculate_thetas()
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg
        
    def _calculate_thetas(self, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> None:
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
            self.thetas = thetas.transpose(0, 1).unsqueeze(0).unsqueeze(0)
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
            self.thetas = thetas.unsqueeze(1)
        
    def forward(self, x: torch.Tensor, total_seq_len: int = 0, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
            total_seq_len (int, optional): Unused input for YaRN compatibility. Defaults to 0.
            offset (int, optional): Position offset for Infini-Former compatibility. Defaults to 0.
            select_mask (Optional[torch.Tensor], optional): Mask to select a subset of the positional embeddings for Mixture-of-Depths compatibility. Defaults to None.

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if offset != self.last_offset:
            self._calculate_thetas(offset=offset, select_mask=select_mask)
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
                x_cos = self.thetas.cos()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.thetas.sin()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1)
            # Otherwise, perform calculations with the full cos_component and sin_component
            else:
                x_cos = self.thetas.cos().repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.thetas.sin().repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        # If a selection mask is specified, incorporate it into the positional embeddings
        else:
            if not cos_sin_recalculated:
                self._calculate_thetas(offset=offset, select_mask=select_mask)
                self.last_offset = offset
            x_cos = self.thetas.cos().repeat(1, x_pos.size(1), 1, 1) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.thetas.sin().repeat(1, x_pos.size(1), 1, 1)
            
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out
    
class YaRNEmbeddings(PositionEmbeddings):
    """Implements Yet Another RoPE ExtensioN (YaRN) as described in the paper:
    "YaRN: Efficient Context Window Extension of Large Language Models" by Peng et al.
    (https://arxiv.org/abs/2309.00071).
    
    Modifications have been made to make it compatible with both Infini-Attention
    and Mixture-of-Experts."""
    def __init__(
        self, 
        dim: int, 
        seq_len: int, 
        context_len: int,
        context_len_ext: int,
        dim_embedding_pct: float = 0.5, 
        base: int = 10000,
        alpha: int = 1,
        beta: int = 32,
        length_scale: Optional[float] = None
    ):
        """Instantiate the module.

        Args:
            dim (int): Key/Value dimension of the attention layer.
            seq_len (int): Maximum sequence length.
            context_len (int): Length of the context window.
            context_len_ext (int): Extended length of the context window.
            dim_embedding_pct (float): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
            base (int, optional): Base used for calculating thetas. Defaults to 10000.
            alpha (int, optional): Interpolation minimum for dynamic scaling. Defaults to 1.
            beta (int, optional): Interpolation maximum for dynamic scaling. Defaults to 32.
            len_scale (Optional[float], optional): Length scale for attention calculation. Defaults to None.
        """
        super(YaRNEmbeddings, self).__init__()
        
        # Record input parameters
        self.dim = dim
        self.effective_dim = int(dim * dim_embedding_pct)
        self.seq_len = seq_len
        self.context_len = context_len
        self.context_len_ext = context_len_ext
        self.dim_embedding_pct = dim_embedding_pct
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.length_scale = length_scale
        
        self.last_offset = -1
        
        # Initialize sin component indices for input tensor
        # Indices for rearranging the input follow the pattern [1, 0, 3, 2, 5, 4, ...]
        # Indices that need to be negated in calculating the positional embeddings are [0, 2, 4, ...]
        self.ixs_sin = torch.empty(self.effective_dim, dtype=torch.long)
        self.ixs_sin_neg = 2 * torch.arange(self.effective_dim // 2)
        self.ixs_sin[self.ixs_sin_neg] = self.ixs_sin_neg + 1
        self.ixs_sin[self.ixs_sin_neg + 1] = self.ixs_sin_neg
        
    def _scale_factor(self, seq_len: int) -> float:
        """Calculate the scale factor for the given sequence length from section 3.3 in the paper.

        Args:
            seq_len (int): The sequence length to calculate the scale factor for.

        Returns:
            float: The scale factor for the given sequence length.
        """
        return max(1., seq_len / self.context_len)
    
    def _base_ext(self, seq_len: int) -> float:
        """Calculate the extended base from equation (16) in the paper.

        Args:
            seq_len (int): The sequence length to calculate the extended base for.

        Returns:
            float: The extended base for the given sequence length.
        """
        return self.base * (self._scale_factor(seq_len) ** (self.dim / (self.dim - 2)))
        
    def _wavelength_d(self, d: torch.Tensor) -> torch.Tensor:
        """Calculate the wavelength for the given dimension index tensor from equation (13) in the paper.

        Args:
            d (torch.Tensor): Tensor of dimension indices to calculate the wavelengths for.

        Returns:
            torch.Tensor: The wavelengths of the given dimension index tensor.
        """
        return 2. * math.pi * self.base ** (2 * d / self.dim)
    
    def _wavelength_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Calculate the wavelength of the given theta tensor from equation (13) in the paper.

        Args:
            theta (torch.Tensor): The theta tensor to calculate the wavelengths for.

        Returns:
            torch.Tensor: The wavelengths of the given theta tensor.
        """
        return 2. * math.pi / theta
    
    def _wavelength_context_ratio(self, wavelength: torch.Tensor) -> torch.Tensor:
        """Calculate the wavelength ratio for the context extension from equation (17) in the paper.

        Args:
            wavelength (torch.Tensor): The wavelengths to calculate the ratio for.

        Returns:
            torch.Tensor: The wavelength ratio for the context extension.
        """
        return self.context_len / wavelength
    
    def _ramp(self, ratio: torch.Tensor) -> torch.Tensor:
        """Calculate the ramp function from equation (18) in the paper.

        Args:
            ratio (torch.Tensor): The wavelength ratio to calculate the ramp function for.

        Returns:
            torch.Tensor: The ramp function values for the given wavelength ratio.
        """
        out = torch.zeros_like(ratio, device=ratio.device, dtype=torch.float)
        interp_mask = torch.logical_and(ratio >= self.alpha, ratio <= self.beta)
        one_mask = ratio > self.beta
        out[interp_mask] = (ratio[interp_mask] - self.alpha) / (self.beta - self.alpha)
        out[one_mask] = 1.
        return out
        
    def _calculate_thetas(self, total_seq_len: int, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> None:
        """Calculate the cosine and sine component matrices for the rotary positional embeddings.
        Uses multidimensional extension of theta as defined in Sec 3.2.2 as well as equation (34)
        from the RoFormer paper

        Args:
            total_seq_len (int): The total sequence length to calculate the thetas for.
            offset (int, optional): Position offset for Infini-Former compatibility. Defaults to 0.
            select_mask (Optional[torch.Tensor], optional): Mask to select a subset of the positional embeddings for Mixture-of-Depths compatibility. Defaults to None.
        """
        if select_mask is None:
            # Calculate matrix of angles
            # Shape: (effective_dim, seq_len)
            thetas = torch.repeat_interleave(
                (self.base ** (-2. * torch.arange(1, self.effective_dim//2 + 1) / self.dim)).unsqueeze(-1).repeat((1, self.seq_len)), 
                repeats=2, 
                dim=0
            )
            ramp = self._ramp(self._wavelength_context_ratio(self._wavelength_theta(thetas)))
            scale = self._scale_factor(total_seq_len)
            length_scale = 0.1 * math.log(scale) + 1. if self.length_scale is None else self.length_scale
            thetas = ((1. - ramp) * thetas / scale + ramp * thetas) * length_scale
            # Multiply by index positions, then transpose to get correct shape
            thetas *= torch.arange(1 + offset, self.seq_len + 1 + offset).unsqueeze(0)
            self.thetas = thetas.transpose(0, 1).unsqueeze(0).unsqueeze(0)
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
            ramp = self._ramp(self._wavelength_context_ratio(self._wavelength_theta(thetas)))
            scale = self._scale_factor(total_seq_len)
            length_scale = 0.1 * math.log(scale) + 1. if self.length_scale is None else self.length_scale
            thetas = ((1. - ramp) * thetas / scale + ramp * thetas) * length_scale
            # (n_obs, select_seq_len, effective_dim)
            thetas = thetas.transpose(0, 1).unsqueeze(0).repeat((select_mask.size(0), 1, 1))
            thetas *= select_ixs
            self.thetas = thetas.unsqueeze(1)
        
    def forward(self, x: torch.Tensor, total_seq_len: int, offset: int = 0, select_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor. Uses a multidimensional
        extension of equation (34) of the RoFormer paper.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
            total_seq_len (int): The total sequence length of the input.
            offset (int, optional): Position offset for Infini-Former compatibility. Defaults to 0.
            select_mask (Optional[torch.Tensor], optional): Mask to select a subset of the positional embeddings for Mixture-of-Depths compatibility. Defaults to None.

        Returns:
            torch.Tensor: Transformed input tensor with rotary positional embeddings applied.
        """
        if offset != self.last_offset:
            self._calculate_thetas(total_seq_len=total_seq_len, offset=offset, select_mask=select_mask)
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
                x_cos = self.thetas.cos()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.thetas.sin()[:, :, :x_pos.size(2), :].repeat(x_pos.size(0), x_pos.size(1), 1, 1)
            # Otherwise, perform calculations with the full cos_component and sin_component
            else:
                x_cos = self.thetas.cos().repeat(x_pos.size(0), x_pos.size(1), 1, 1) * x_pos
                x_sin = x_pos[..., self.ixs_sin]
                x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
                x_sin *= self.thetas.sin().repeat(x_pos.size(0), x_pos.size(1), 1, 1)
        # If a selection mask is specified, incorporate it into the positional embeddings
        else:
            if not cos_sin_recalculated:
                self._calculate_thetas(total_seq_len=total_seq_len, offset=offset, select_mask=select_mask)
                self.last_offset = offset
            x_cos = self.thetas.cos().repeat(1, x_pos.size(1), 1, 1) * x_pos
            x_sin = x_pos[..., self.ixs_sin]
            x_sin[..., self.ixs_sin_neg] = -x_sin[...,self.ixs_sin_neg]
            x_sin *= self.thetas.sin().repeat(1, x_pos.size(1), 1, 1)
            
        # If the sequence length is less than the maximum sequence length, concatenate positionally embedded
        # entries with original entries, otherwise return the positionally embedded entries
        if self.dim_embedding_pct < 1.0:
            out = torch.cat([x_cos + x_sin, x_pass], dim=-1)
        else:
            out = x_cos + x_sin
        
        return out