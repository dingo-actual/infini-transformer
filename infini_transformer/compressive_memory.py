from typing import Optional

import torch
from torch import nn


class CompressiveMemory(nn.Module):
    """Implements the Compressive Transformer memory module."""

    def __init__(self, dim_input: int, dim_key: int, dim_value: int, num_heads: int, segment_len: int, update: str = "linear"):
        """Initialize module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_heads (int): Number of attention heads.
            segment_len (int): Segment length (must be a factor of the input sequence length).
            update (str, optional): Type of memory update rule to use ("linear" or "delta"). Defaults to "linear".
        """
        super(CompressiveMemory, self).__init__()

        # Record input parameters
        self.num_heads = num_heads
        self.segment_len = segment_len

        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.update = update

        # Projections for stacked SDP attention
        self.proj_k = nn.Linear(dim_input, num_heads * dim_key, bias=False)
        self.proj_v = nn.Linear(dim_input, num_heads * dim_value, bias=False)
        self.proj_q = nn.Linear(dim_input, num_heads * dim_key, bias=False)

        # Initialize betas for weighted average of dot-product and memory-based attention
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_value))

        # Projection for output
        self.proj_out = nn.Linear(num_heads * dim_value, dim_input, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies Scaled Dot-Product Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            mask (torch.Tensor, optional): Attention mask of shape (seq_len, seq_len). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape

        num_segments, rem = divmod(seq_len, self.segment_len)

        if rem != 0:
            raise ValueError(
                "Sequence length must be divisible by segment length.")

        out = []

        # Initialize mem and normalization
        # !!! Initialization was never specified in the paper, so this is an educated guess
        mem = torch.zeros(1, self.num_heads, self.dim_key, self.dim_value)
        z = torch.zeros(batch_size, self.num_heads, self.dim_value, 1)
        
        # Project the input tensor to get the key, value, and query tensors
        k_full = self.proj_k(x).unsqueeze(1).view(
            (batch_size, self.num_heads, self.segment_len * num_segments, self.dim_key))
        v_full = self.proj_v(x).unsqueeze(1).view(
            (batch_size, self.num_heads, self.segment_len * num_segments, self.dim_value))
        q_full = self.proj_q(x).unsqueeze(1).view(
            (batch_size, self.num_heads, self.segment_len * num_segments, self.dim_key))
        
        # If attention mask is given, resize it to match the segment tensors
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.num_heads, 1, 1))

        for ix in range(num_segments):
            ix_lo = ix * self.segment_len
            ix_hi = ix_lo + self.segment_len

            # Extract segment from key, value and query tensors
            k = k_full[:, :, ix_lo:ix_hi, :]
            v = v_full[:, :, ix_lo:ix_hi, :]
            q = q_full[:, :, ix_lo:ix_hi, :]
            
            # Pre-calculate sigma(q) for updating memory and calculating attention
            # shape: (batch_size, num_heads, segment_len, dim_key)
            sigma_q = (nn.functional.elu(q) + 1.0)

            # Apply normalization term update
            z = z + (nn.functional.elu(k) + 1.0).sum(dim=-2, keepdim=True)

            # Apply SDP attention
            scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_key))
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores.masked_fill_(torch.logical_not(mask), float('-inf'))
                else:
                    scores += mask.view((batch_size, self.num_heads, seq_len, seq_len))
            att_dot = nn.functional.softmax(scores, dim=-1) @ v

            # Calculate normalized linear attention
            # shape: (batch_size, num_heads, segment_len, dim_value)
            att_mem = (sigma_q @ mem) / (sigma_q @ z)

            # Apply mem update
            sigma_k = nn.functional.elu(k) + 1.0
            if self.update == "linear":
                mem = mem + sigma_k.transpose(-2, -1) @ v
            elif self.update == "delta":
                mem = mem + \
                    sigma_k.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))

            # Calculate weighted average of dot-product and memory-based attention
            att = nn.functional.sigmoid(
                self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot
            att = att.view((batch_size, self.segment_len,
                           self.num_heads * self.dim_value))

            # Append output to buffer
            out.append(self.proj_out(att))

        # Return concatenated full sequence from buffer
        return torch.concat(out, dim=1)


def demo_compressive_memory():
    # Example usage
    dim_input = 512
    dim_key = 64
    dim_value = 64
    num_heads = 8
    segment_len = 32
    update = "linear"

    model = CompressiveMemory(
        dim_input, dim_key, dim_value, num_heads, segment_len, update)

    # Generate some random input
    batch = torch.randn(4, 128, dim_input)
    mask = torch.tril(torch.ones(segment_len, segment_len), diagonal=0).bool()
    model(batch, mask)


if __name__ == "__main__":
    demo_compressive_memory()
