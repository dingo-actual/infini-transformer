from typing import Optional, Tuple

import torch
from torch import nn

from compressive_memory import CompressiveMemory


class InfiniTransformer(nn.Module):
    """Transformer layer with compressive memory."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_key: int,
        dim_value: int,
        num_heads: int,
        segment_len: int,
        update="linear",
        dropout: float = 0.0
    ):
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
        self.attn = CompressiveMemory(
            dim_input, dim_key, dim_value, num_heads, segment_len, update)
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


class MoDInfiniTransformer(InfiniTransformer):
    """Mixture-of-Depths Infini-Transformer Layer."""

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_key: int,
        dim_value: int,
        num_heads: int,
        segment_len: int,
        sampling_factor: int,
        sampling_hidden_dim: int,
        update="linear",
        dropout: float = 0.0
    ):
        """Instantiate module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            segment_len (int): Segment length for the CompressiveMemory.
            sampling_factor (int): Reciprocal of the sampling rate for the Mixture-of-Depths mechanism.
            sampling_hidden_dim (int): Hidden dimension for the sampling MLP.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.

        Raises:
            ValueError: Segment length not divisible by sampling factor.
        """
        # Check that segment length is divisible by sampling factor; if not, raise an error
        if segment_len % sampling_factor != 0:
            raise ValueError(
                "Segment length must be divisible by sampling factor.")

        # Initialize ordinary InfiniTransformer, but with segment length reduced by sampling_factor
        super(MoDInfiniTransformer, self).__init__(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_key=dim_key,
            dim_value=dim_value,
            num_heads=num_heads,
            segment_len=segment_len // sampling_factor,
            update=update,
            dropout=dropout
        )

        # Record additional init arguments for forward pass
        self.segment_len = segment_len // sampling_factor
        self.full_segment_len = segment_len
        self.sampling_factor = sampling_factor
        self.dim_input = dim_input

        # Projection for tensor of logits when sampling
        self.proj_sampling = nn.Linear(dim_input, 1)

        # MLP for predicting selection of tokens
        self.mlp_sampling = nn.Sequential(
            nn.Linear(dim_input, sampling_hidden_dim),
            nn.ReLU(),
            nn.Linear(sampling_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
            torch.Tensor: Token selection mask of shape (batch_size * seq_len, 1).
            torch.Tensor: Predicted token selection scores of shape (batch_size * seq_len, 1) or None.
        """
        # Initialize sample_scores_pred to None
        sample_scores_pred = None

        # Calculate number of total segments, samples
        batch_size, seq_len, _ = x.shape
        num_segments = seq_len // self.full_segment_len
        samples_per_segment = self.full_segment_len // self.sampling_factor

        # Initialize list of token sample masks
        sample_masks = []

        # If training, use linear embedding for sample scores
        if self.training:
            sample_scores = self.proj_sampling(x).squeeze(-1)
            sample_scores_pred = self.mlp_sampling(x).squeeze(-1)
        # Otherwise, use MLP for sample scores
        else:
            sample_scores = self.mlp_sampling(x).squeeze(-1)

        # For each segment, sample the tokens with the highest scores
        for seg_num in range(num_segments):
            # Compute segment indices
            ix_lo = seg_num * self.full_segment_len
            ix_hi = ix_lo + self.full_segment_len

            # Argsort by sample scores to get indices of tokens to keep
            sort_ixs = torch.argsort(
                sample_scores[:, ix_lo:ix_hi], dim=1, descending=True)

            # Convert token indices to a binary mask
            sample_mask_seg = torch.zeros_like(
                sample_scores[:, ix_lo:ix_hi], device=x.device)
            sample_mask_seg.scatter_(
                dim=1, index=sort_ixs[:, :samples_per_segment], value=1.0)

            sample_masks.append(sample_mask_seg)

        # Combine segment masks into a single mask
        sample_mask = torch.cat(sample_masks, dim=1).bool()

        # Apply multi-head attention to sample, followed by MLP
        sample_shape = (batch_size, self.segment_len * num_segments, self.dim_input)
        x_ = x[sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))].view(sample_shape)
        x_ = self.attn(x_, mask)
        x_ = self.mlp(x_)

        # Add result of attended tokens to the result (equivalent to making the result for non-attended tokens zero)
        x[sample_mask.unsqueeze(-1).repeat((1, 1, self.dim_input))] += x_.view(-1)

        # If sample_scores and sample_mask are not None, flatten them
        if sample_scores_pred is not None:
            sample_scores_pred = sample_scores_pred.view((-1, 1))
        if sample_mask is not None:
            sample_mask = sample_mask.view((-1, 1)).float()

        return self.layer_norm(x), sample_mask, sample_scores_pred


def demo_mod_infini_transformer():
    """
    Demonstrates the usage of the MoDInfiniTransformer class.
    """
    # Define the model parameters
    dim_input = 512
    dim_hidden = 2048
    dim_key = 64
    dim_value = 64
    num_heads = 8
    segment_len = 2048
    sampling_factor = 8
    sampling_hidden_dim = 1024
    update = "linear"
    dropout = 0.1

    # Define batch dimensions
    seq_len = 4096
    batch_size = 2

    # Create causal attention mask
    attn_mask = torch.tril(
        torch.ones(
            (
                segment_len // sampling_factor,
                segment_len // sampling_factor
            )
        ),
        diagonal=0
    ).bool()

    # Create the MoDInfiniTransformer layer
    layer = MoDInfiniTransformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_key=dim_key,
        dim_value=dim_value,
        num_heads=num_heads,
        segment_len=segment_len,
        sampling_factor=sampling_factor,
        sampling_hidden_dim=sampling_hidden_dim,
        update=update,
        dropout=dropout
    )

    # Generate dummy batch
    x = torch.randn(batch_size, seq_len, dim_input)

    # Test outputs for the case where the net is training
    layer.train()
    x_att, sample_mask, sample_scores_pred = layer(x, mask=attn_mask)

    # Test output for the case where the net is not training
    layer.eval()
    x_att, sample_mask, sample_scores_pred = layer(x, mask=attn_mask)


if __name__ == "__main__":
    demo_mod_infini_transformer()
