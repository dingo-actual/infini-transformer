import torch
from torch import nn

from compressive_memory import CompressiveMemory


class Decoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_key, dim_value, num_heads, segment_len, update="linear", dropout: float = 0.0):
        super(Decoder, self).__init__()
        
        self.attn = CompressiveMemory(dim_input, dim_key, dim_value, num_heads, segment_len, update)
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim_input)
        
    def forward(self, x):
        x_ = self.attn(x)
        x_ = self.mlp(x_)
        
        return self.layer_norm(x_ + x)
