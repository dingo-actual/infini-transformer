# tests/test_transformer.py

import torch
from infini_transformer.transformer import InfiniTransformer, MoDInfiniTransformer

def test_infini_transformer():
    dim_input = 512
    dim_hidden = 2048
    dim_key = 64
    dim_value = 64
    num_heads = 8
    activation = "ffngeglu"
    segment_len = 2048
    update = "linear"
    dropout = 0.1

    layer = InfiniTransformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_key=dim_key,
        dim_value=dim_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        update=update,
        dropout=dropout
    )

    batch_size = 2
    seq_len = 4096
    x = torch.randn(batch_size, seq_len, dim_input)

    layer.eval()  # Set the layer to evaluation mode
    x_att = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)

def test_mod_infini_transformer():
    dim_input = 768
    dim_hidden = 3072
    dim_key = 96
    dim_value = 96
    num_heads = 12
    activation = "gelu"
    segment_len = 1024
    sampling_factor = 8
    update = "delta"
    dropout = 0.2

    layer = MoDInfiniTransformer(
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        dim_key=dim_key,
        dim_value=dim_value,
        num_heads=num_heads,
        activation=activation,
        segment_len=segment_len,
        sampling_factor=sampling_factor,
        update=update,
        dropout=dropout
    )

    batch_size = 4
    seq_len = 2048
    x = torch.randn(batch_size, seq_len, dim_input)

    layer.train()  # Set the layer to training mode
    x_att, sample_mask, sample_scores_pred = layer(x)

    assert x_att.shape == (batch_size, seq_len, dim_input)
    assert sample_mask.shape == (batch_size * seq_len, 1)
    assert sample_scores_pred.shape == (batch_size * seq_len, 1)