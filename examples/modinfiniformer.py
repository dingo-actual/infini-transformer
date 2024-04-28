from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from infini_transformer import MoDInfiniTransformer, InfiniTransformer


class NextTokenModel(nn.Module):
    """
    An Infini-Transformer based model for next token prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        dim_hidden: int,
        dim_key: int,
        dim_value: int,
        num_heads: int,
        segment_len: int,
        sampling_factor: int,
        update="linear",
        causal: bool = False,
        init_state_learnable: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize the module.
        Parameters:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the embedding space.
            num_layers (int): Number of Infini-transformer layers.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            segment_len (int): Segment length for the CompressiveMemory.
            sampling_factor (int): Reciprocal of the sampling rate for the Mixture-of-Depths mechanism.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False
            init_state_learnable (bool, optional): Whether the initial state of the CompressiveMemory should be learnable. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super(NextTokenModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        transformers = []
        for layer_n in range(num_layers):
            if layer_n % 2 == 0:
                transformers.append(
                    MoDInfiniTransformer(
                        dim_input=embedding_dim,
                        dim_hidden=dim_hidden,
                        dim_key=dim_key,
                        dim_value=dim_value,
                        num_heads=num_heads,
                        segment_len=segment_len,
                        sampling_factor=sampling_factor,
                        update=update,
                        causal=causal,
                        init_state_learnable=init_state_learnable,
                        dropout=dropout
                    )
                )
            else:
                transformers.append(
                    InfiniTransformer(
                        dim_input=embedding_dim,
                        dim_hidden=dim_hidden,
                        dim_key=dim_key,
                        dim_value=dim_value,
                        num_heads=num_heads,
                        segment_len=segment_len,
                        update=update,
                        causal=causal,
                        init_state_learnable=init_state_learnable,
                        dropout=dropout
                    )
                )
            self.transformers = nn.ModuleList(transformers)

            self.proj_final = nn.Linear(embedding_dim, vocab_size)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Forward pass of the model for computing the next token probabilities and other outputs if applicable.

        Parameters:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            Tuple containing the probability distribution over the next tokens, actual modification tokens (if any), and predicted modification tokens (if any).
        """
        mod_token_actuals = []
        mod_token_preds = []

        x = self.embedding(x)
        for ix, transformer in enumerate(self.transformers):
            if ix % 2 == 0:
                x, mod_token_actual, mod_token_pred = transformer(x)
                mod_token_actuals.append(mod_token_actual)
                mod_token_preds.append(mod_token_pred)
            else:
                x = transformer(x)

        x = self.proj_final(x)

        next_token_probs = self.softmax(x)

        return next_token_probs, mod_token_actuals, mod_token_preds


def train_model(
    model: NextTokenModel,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    epochs: int,
    device: str
):
    # Switch model to training mode and move to the specified device
    model = model.train()
    model = model.to(device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Learning rate scheduler
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Main loss function
    loss_fn_main = nn.CrossEntropyLoss().to(device)
    # Auxiliary loss function
    loss_fn_sampling = nn.functional.binary_cross_entropy_with_logits

    for epoch in range(epochs):
        for ix, batch in enumerate(dataloader_train):
            # Move data batch to specified device
            batch = batch.to(device)
            # Target labels for training
            target = batch[:, 1:].clone()
            # Generate predictions and auxiliary outputs from model
            preds, mod_actuals, mod_preds = model(batch)

            # Calculate the main loss (Cross-Entropy)
            loss_main = loss_fn_main(input=preds[:, :-1, :], target=target)

            # Calculate auxiliary loss
            loss_aux = torch.tensor(0.0, device=device)
            for mod_actual, mod_pred in zip(mod_actuals, mod_preds):
                loss_aux += loss_fn_sampling(input=mod_pred, target=mod_actual)

            # Total loss is the sum of main and auxiliary losses
            loss = loss_main + loss_aux

            # Clear gradients, perform backpropagation, and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f'Epoch: {epoch + 1}/epochs ({ix + 1}/{len(dataloader_train)})  |  Training Loss: {loss_main.detach().cpu().item():.6f}\r',
                end=""
            )

        # Update the learning rate schedule after each epoch
        lr_schedule.step()

        # Validation phase
        with torch.no_grad():
            total_loss = 0.0
            num_obs = 0

            for batch in dataloader_val:
                batch = batch.to(device)
                target = batch[:, 1:].clone()

                preds, _, _ = model(batch)

                total_loss += loss_fn_main(input=preds[:, :-1, :], target=target).detach().cpu().item() * batch.size(0)
                num_obs += batch.size(0)

            # Calculate the average validation loss
            val_loss = total_loss / num_obs

            print(
                f'\nEpoch: {epoch + 1}/{epochs}  |  Validation Loss -- (CE): {val_loss:.6f}'
            )

    return model
