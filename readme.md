# Infini-Transformer

- [Infini-Transformer](#infini-transformer)
  - [Overview](#overview)
  - [Features](#features)
  - [Directory structure](#directory-structure)
  - [Getting Started](#getting-started)
  - [Usage](#usage)
    - [`CompressiveMemory`](#compressivememory)
    - [`InfiniTransformer`](#infinitransformer)
    - [`MoDInfiniTransformer`](#modinfinitransformer)
    - [`RoPEEmbeddings`](#ropeembeddings)
    - [`YaRNEmbeddings`](#yarnembeddings)
    - [Example Usage](#example-usage)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Overview

Infini-Transformer ([https://arxiv.org/abs/2404.07143](https://arxiv.org/abs/2404.07143)) is a powerful and versatile transformer model designed for a wide range of natural language processing tasks. It leverages state-of-the-art techniques and architectures to achieve exceptional performance and scalability to infinite context lengths.

## Features

- Scalable architecture for handling long sequences
- Large-scale pre-training on diverse datasets
- Support for multiple downstream tasks, including text classification, question answering, and language generation
- Efficient fine-tuning for task-specific adaptation
- Includes a Mixture-of-Depths ([https://arxiv.org/abs/2404.02258](https://arxiv.org/abs/2404.02258)) transformer layer that incorporates Infini-Attention
- Implementation of RoPE ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)) that conforms to Infini-Attention's and Mixture-of-Depth's memory-efficient designs (PoSE to be added soon)
- - Implementation of YaRN ([https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)) that conforms to Infini-Attention's and Mixture-of-Depth's memory-efficient designs (PoSE to be added soon)

## Directory structure

```default
infini-transformer/
│
├── infini_transformer/
│   ├── __init__.py
│   ├── transformer.py
│   ├── compressive_memory.py
│   ├── positional_embedder.py
│   └── activations.py
│
├── examples/
│   ├── __init__.py
│   └── modinfiniformer.py
│
├── tests/
│   ├── __init__.py
│   └── test_transformer.py
│
├── LICENSE
├── README.md
├── requirements.txt
├── MANIFEST.in
└── pyproject.toml
```

## Getting Started

To get started with Infini-Transformer, you can clone the repository and install it from source:

```bash
git clone https://github.com/dingo-actual/infini-transformer.git
cd infini-transformer
pip install -e .
```

## Usage

### `CompressiveMemory`

The `CompressiveMemory` module is a key component of the Infini-Transformer architecture. It is designed to handle long sequences efficiently by compressing and storing the input tokens in a memory matrix and normalization vector. This allows the model to maintain a large context window while keeping the memory usage bounded.

It performs a variant of multi-head self-attention with a recurrent update step by dividing the input tensor along its sequence dimension (which is assumed to be dimension 1). It begins by performing learned linear projections of the input into key, query and value tensors, from which it extracts segments for each recurrent step.

At each recurrent step, it calculates a learned linear combination of linear attention (which uses the memory and normalization matrices) and SDP attention. It then updates the memory matrix and normalization vector using the current step's key and value matrices, along with the current memory matrix and normalization vector. Before output, the combined attention tensor is stacked along all heads, then projected back to the input dimension.

The outputs from each recurrent step are concatenated along the sequence dimension (dimension 1) to produce the final output tensor.

The update for the memory matrix has two variants: linear and delta.

The linear update rule is:
$$M_t = M_{t-1} + \bigl(\textrm{ELU}(K_{t-1}\bigr) + 1)^TV_{t-1}$$

The delta update rule is:
$$M_t = M_{t-1} + \bigl(\textrm{ELU}(K_{t-1}) + 1\bigr)^T \biggl( V_{t-1} - \frac{(\textrm{ELU}(K_{t-1}) + 1)M_{t-1}}{(\textrm{ELU}(K_{t-1}) + 1)z_{t-1}}\biggr)$$

Where $M_i$ is the memory matrix and $z_i$ is the normalization vector at step $i$. The $K$ and $V$ matrices are subscripted to indicate the recurrent steps they correspond to.

Computations are stacked along the embedding dimension whenever possible to make use of multi-head attention in an efficient manner.

The `CompressiveMemory` module takes the following arguments:

- `dim_input`: The input dimension of the tensors.
- `dim_key`: The dimension of the key tensor and query tensors.
- `dim_value`: The dimension of the value tensor.
- `num_heads`: The number of attention heads.
- `segment_len`: The length of each segment in the recurrent attention computation.
- `sampling_factor`: The sampling factor used if using Mixture-of-Depths (use None if not using Mixture-of-Depths). (Default is None.)
- `update`: The type of update to use for the memory matrix. Can be "linear" or "delta". (Default is "linear".)
- `causal`: Whether to use causal attention in SDP calculations (where each position can only attend to previous positions). (Default is False.)
- `positional_embedder`: An optional `PositionEmbeddings` object: `RoPEEmbeddings` or `YaRNEmbeddings` (Default is None.)
- `init_state_learnable`: Whether the initial memory state and normalization vector are learnable parameters. (Default is False.)

Example usage of the `CompressiveMemory` module is as follows:

```python
import torch

from infini_transformer.compressive_memory import CompressiveMemory


cm = CompressiveMemory(
    dim_input=768,
    dim_key=64,
    dim_value=64,
    num_heads=8,
    segment_len=2048,
    sampling_factor=None,
    update="linear",
    causal=True,
    positional_embedder="rope",
    init_state_learnable=False
)

batch = torch.randn(
    2, # batch size
    65536, # sequence length
    768 # input dimension
)

output = cm(batch)
```

During training, no special handling of the output is required.

### `InfiniTransformer`

The `InfiniTransformer` class implements a variation on the original transformer the utilizes `CompressiveMemory` in place of standard self-attention. This allows the model to efficiently handle long sequences by compressing and storing the input tokens in a memory matrix and normalization vector. It makes use of the `CompressiveMemory` module to perform a variant of multi-head self-attention with a recurrent update step.

The primary difference between `InfiniTransformer` and an ordinary transformer is the replacement of `CompressiveMemory` for the standard multi-head self-attention mechanism.

The `InfiniTransformer` module takes the following arguments:

- `dim_input`: The input dimension of the tensors.
- `dim_hidden`: The hidden dimension of the MLP applied after multi-head self-attention.
- `dim_key`: The dimension of the key tensor and query tensors.
- `dim_value`: The dimension of the value tensor.
- `num_heads`: The number of attention heads.
- `activation`: The nonlinear activation function to apply in the MLP. The following activations are supported:
  
  - `"relu"`: ReLU activation
  - `"abs"`: Absolute value activation
  - `"gelu"`: Gaussian Error Linear Unit (GELU) activation
  - `"swish"`: Swish activation
  - `"swiglu"`: SwiGLU activation
  - `"geglu"`: Gated Gaussian Error Linear Unit (GeGELU) activation
  - `"ffnglu"`: Feed-Forward Network with Gated Linear Unit (FFNGLU) activation
  - `"ffngeglu"`: Feed-Forward Network with Gated Gaussian Error Linear Unit (FFNGeGLU) activation
  - `"ffnswiglu"`: Feed-Forward Network with Swish Gated Linear Unit (FFNSwiGLU) activation

- `segment_len`: The length of each segment in the recurrent attention computation.
- `update`: The type of update to use for the memory matrix. Can be "linear" or "delta". (Default is "linear".)
- `causal`: Whether to use causal attention in SDP calculations (where each position can only attend to previous positions). (Default is False.)
- `positional_embedder`: An optional `PositionEmbeddings` object: `RoPEEmbeddings` or `YaRNEmbeddings` (Default is None.)
- `init_state_learnable`: Whether the initial memory state and normalization vector are learnable parameters. (Default is False.)
- `dropout`: The dropout rate to apply in the MLP. (Default is 0.0.)

Example usage of the `InfiniTransformer` module is as follows:

```python
import torch

from infini_transformer import InfiniTransformer


tfm = InfiniTransformer(
    dim_input=768,
    dim_hidden=2048,
    dim_key=64,
    dim_value=64,
    num_heads=8,
    activation="ffngeglu",
    segment_len=2048,
    update="delta",
    causal=True,
    positional_embedder=None,
    init_state_learnable=False,
    dropout=0.1
)

batch = torch.randn(
    2, # batch size
    65536, # sequence length
    768 # input dimension
)

output = tfm(batch)
```

During training, no special handling of the output is required.

### `MoDInfiniTransformer`

The `MoDInfiniTransformer` module extends the `InfiniTransformer` module to incorporate Mixture-of-Depths (Raposo, et. al; [https://arxiv.org/abs/2404.02258](https://arxiv.org/abs/2404.02258)). A `MoDInfiniTransformer` block takes a learned linear projection of its input to a single dimension, and uses the tokens with the top-k highest values for the operations performed by `InfiniTransformer`, adding all remaining tokens to the residual connection. This allows the model to focus its capacity on the most important parts of the input sequence, reducing overall computation and memory requirements even further than `InfiniTransformer` alone.

The top-k selection would ordinarily cause segments within the recurrent loop to have different lengths. We avoid this by dividing the selection evenly amongst all segments.

Due to the non-causal nature of the top-k selection, at inference time the scores produced during projection to 1 dimension are taken to be logits for independent binary classifiers. As such, we train the model with an additional term added to the loss for each `ModInfiniFormer` layer, which is the binary cross-entropy loss between the logits and the top-k tokens selected during training.

As such, the output from `ModInfiniTransformer` is a tuple consisting of three tensors:

- The usual output tensor which matches the dimensions of the input tensor
- A tensor of shape `(batch_size * sequence_length, 1)`, which represents a binary mask of top-k tokens selected during training. This will be the target for our additional binary cross-entropy loss.
- A tensor of shape `(batch_size * sequence_length, 1)` of logits corresponding to the binary mask above. This represents the scores used to select the top-k tokens and is considered the prediction for the additional binary cross-entropy loss.

At inference time, the second and third elements of the tuple can be safely ignored, as all token selection logic is handled within the `MoDInfiniTransformer` module itself.

> **IMPORTANT NOTE**: The binary-classifier-based token selection mechanism for inference has no guarantee of selecting the same number of tokens for each element in a batch. If left unchecked, this would result in a ragged array, which is currently unsupported by PyTorch. The current solution in place is to force the batch size to 1 and concatenate forward passes over single observations. We are aware this is sub-optimal and hope to address it in the near future.

The `MoDInfiniTransformer` module takes the following arguments:

- `dim_input`: The input dimension of the tensors.
- `dim_hidden`: The hidden dimension of the MLP applied after multi-head self-attention.
- `dim_key`: The dimension of the key tensor and query tensors.
- `dim_value`: The dimension of the value tensor.
- `num_heads`: The number of attention heads.
- `activation`: The nonlinear activation function to apply in the MLP. The following activations are supported:
  
  - `"relu"`: ReLU activation
  - `"abs"`: Absolute value activation
  - `"gelu"`: Gaussian Error Linear Unit (GELU) activation
  - `"swish"`: Swish activation
  - `"swiglu"`: SwiGLU activation
  - `"geglu"`: Gated Gaussian Error Linear Unit (GeGELU) activation
  - `"ffnglu"`: Feed-Forward Network with Gated Linear Unit (FFNGLU) activation
  - `"ffngeglu"`: Feed-Forward Network with Gated Gaussian Error Linear Unit (FFNGeGLU) activation
  - `"ffnswiglu"`: Feed-Forward Network with Swish Gated Linear Unit (FFNSwiGLU) activation

- `segment_len`: The length of each segment in the recurrent attention computation.
- `sampling_factor`: A numeric value in the interval (1, `segment_len`) that determines the number of tokens to select from each segment during the top-k selection. A larger value of `sampling_factor` results in fewer tokens being selected.
- `update`: The type of update to use for the memory matrix. Can be "linear" or "delta". (Default is "linear".)
- `causal`: Whether to use causal attention in SDP calculations (where each position can only attend to previous positions). (Default is False.)
- `positional_embedder`: An optional `PositionEmbeddings` object: `RoPEEmbeddings` or `YaRNEmbeddings` (Default is None.)
- `init_state_learnable`: Whether the initial memory state and normalization vector are learnable parameters. (Default is False.)
- `dropout`: The dropout rate to apply in the MLP. (Default is 0.0.)

Example usage of the `InfiniTransformer` module is as follows:

```python
import torch

from infini_transformer import MoDInfiniTransformer


tfm = MoDInfiniTransformer(
    dim_input=768,
    dim_hidden=2048,
    dim_key=64,
    dim_value=64,
    num_heads=8,
    activation="ffngeglu",
    segment_len=2048,
    sampling_factor=8,
    update="delta",
    causal=True,
    init_state_learnable=False,
    positional_embedder=None,
    dropout=0.1
)

batch = torch.randn(
    2, # batch size
    65536, # sequence length
    768 # input dimension
)

output, select_target, select_pred = tfm(batch)
```

During training, we must account for the additional outputs from `MoDInfiniFormer` so we can use them for the binary cross-entropy loss. See [infini_transformer/example/modinfiniformer.py](infini_transformer/example/modinfiniformer.py) for an example of how to incorporate the additional outputs into both the overall model output and the training loop.

### `RoPEEmbeddings`

The `RoPEEmbeddings` module applies RoPE from the paper, "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su, et. al. ([https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)). Once instantiated, it can be passed to the `InfiniTransformer` or `MoDInfiniTransformer` modules as the `positional_embedder` parameter, which passes it through to `CompressiveMemory`, where the position-aware embeddings are applied to the key and query tensors.

The `RoPEEmbeddings` module takes the following arguments:

- `dim`: The dimension of the key/value tensors.
- `seq_len`: The maximum length of the input sequence to `CompressiveMemory` (this must match `CompressiveMemory`'s `segment_len` parameter).
- `dim_embeddings_pct`: The proportion of the key/value tensor dimension to use for the position-aware embeddings. For example, if `dim` is 64 and `dim_embeddings_pct` is 0.5, then 32 dimensions will be used for the position-aware embeddings. (Default is 0.5.)
- `base`: The base value to use for the position embedding angles. (Default is 10000.)

Example usage of the `RoPEEmbeddings` module is as follows:

```python
import torch

from infini_transformer import InfiniTransformer
from infini_transformer import RoPEEmbeddings

embedder = RoPEEmbeddings(
  dim=64, # must match dim_key parameter in InfiniTransformer
  seq_len=2048, # must match segment_len parameter in InfiniTransformer
  dim_embeddings_pct=0.5,
  base=10000
)

tfm = InfiniTransformer(
    dim_input=768,
    dim_hidden=2048,
    dim_key=64, # must match dim parameter in RoPEEmbeddings
    dim_value=64,
    num_heads=8,
    activation="ffngeglu",
    segment_len=2048, # must match seq_len parameter in RoPEEmbeddings
    update="delta",
    causal=True,
    positional_embedder=embedder,
    init_state_learnable=False,
    dropout=0.1
)

batch = torch.randn(
    2, # batch size
    65536, # sequence length
    768 # input dimension
)

output = tfm(batch)
```

### `YaRNEmbeddings`

The `YaRNEmbeddings` module applies YaRN from the paper, "YaRN: Efficient Context Window Extension of Large Language Models" by Peng, et. al. ([https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)). Once instantiated, it can be passed to the `InfiniTransformer` or `MoDInfiniTransformer` modules as the `positional_embedder` parameter, which passes it through to `CompressiveMemory`, where the position-aware embeddings are applied to the key and query tensors.

The `YaRNEmbeddings` module takes the following arguments:

- `dim`: The dimension of the key/value tensors.
- `seq_len`: The maximum length of the input sequence to `CompressiveMemory` (this must match `CompressiveMemory`'s `segment_len` parameter).
- `context_len`: Context length used during training.
- `context_len_ext`: Context length to extend to.
- `dim_embeddings_pct`: The proportion of the key/value tensor dimension to use for the position-aware embeddings. For example, if `dim` is 64 and `dim_embeddings_pct` is 0.5, then 32 dimensions will be used for the position-aware embeddings. (Default is 0.5.)
- `base`: The base value to use for the position embedding angles. (Default is 10000.)
- `alpha`: Interpolation minimum for dynamic scaling. (Default is 1.)
- `beta`: Interpolation minimum for dynamic scaling. (Default is 32.)
- `len_scale`: Length scale for attention calculation. Defaults to None (automatically calculated).

Example usage of the `YaRNEmbeddings` module is as follows:

```python
import torch

from infini_transformer import InfiniTransformer
from infini_transformer import YaRNEmbeddings

embedder = YaRNEmbeddings(
  dim=64, # must match dim_key in InfiniTransformer
  seq_len=2048, # must match segment_len parameter in InfiniTransformer
  context_len=32768,
  context_len_ext=65536,
  dim_embeddings_pct=0.5,
  base=10000,
  alpha=1,
  beta=32,
  len_scale=None
)

tfm = InfiniTransformer(
    dim_input=768,
    dim_hidden=2048,
    dim_key=64, # must match dim in YaRNEmbeddings
    dim_value=64,
    num_heads=8,
    activation="ffngeglu",
    segment_len=2048, # must match seq_len parameter in YaRNEmbeddings
    update="delta",
    causal=True,
    positional_embedder=embedder,
    init_state_learnable=False,
    dropout=0.1
)

batch = torch.randn(
    2, # batch size
    65536, # sequence length
    768 # input dimension
)

output = tfm(batch)
```

### Example Usage

Please see [infini_transformer/example/modinfiniformer.py](infini_transformer/example/modinfiniformer.py) for an example of a model and training routine using the `MoDInfiniTransformer` module.

More examples will be forthcoming.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the researchers and developers whose work has inspired and contributed to the development of Infini-Transformer and Mixture-of-Depths Transformer.

Also, we'd like to give special thanks to all the contributors, collaborators and people who have given feedback. Your efforts have made what was a rough outline of an implementation into something actually usable.

If you have any questions or need further assistance, please feel free to reach out to me at [ryan@beta-reduce.net](ryan@beta-reduce.net).
