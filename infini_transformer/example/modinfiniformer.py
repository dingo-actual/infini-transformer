from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import MoDInfiniTransformer, InfiniTransformer


class NextTokenModel(nn.Module):
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
        mod_token_actuals = []
        mod_token_preds = []
        
        x = self.embedding(x)
        for ix, transformer in enumerate(self.transformers):
            if ix % 2 == 0:
                x, mod_token_actual, mod_token_pred = transformer(x, self.attention_mask_mod)
                mod_token_actuals.append(mod_token_actual)
                mod_token_preds.append(mod_token_pred)
            else:
                x = transformer(x, self.attention_mask)
                
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
    model = model.train()
    model = model.to(device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn_main = nn.CrossEntropyLoss().to(device)
    loss_fn_sampling = nn.functional.binary_cross_entropy_with_logits
            
    for epoch in range(epochs):
        for ix, batch in enumerate(dataloader_train):
            batch = batch.to(device)
            target = batch[:, 1:].clone()
            
            preds, mod_actuals, mod_preds = model(batch)
            
            loss_main = loss_fn_main(input=preds[:, :-1, :], target=target)
            loss_aux = torch.tensor(0.0, device=device)
            for mod_actual, mod_pred in zip(mod_actuals, mod_preds):
                loss_aux += loss_fn_sampling(input=mod_pred, target=mod_actual)
                
            loss = loss_main + loss_aux
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(
                f'Epoch: {epoch+1}/epochs ({ix+1}/{len(dataloader_train)})  |  Training Loss: {loss_main.detach().cpu().item():.6f}\r',
                end=""
            )
            
        lr_schedule.step()
        
        with torch.no_grad():
            total_loss = 0.0
            num_obs = 0
            
            for batch in dataloader_val:
                batch = batch.to(device)
                target = batch[:, 1:].clone()

                preds, _, _ = model(batch)

                total_loss += loss_fn_main(input=preds[:, :-1, :], target=target).detach().cpu().item() * batch.size(0)
                num_obs += batch.size(0)
                
            val_loss = total_loss / num_obs
            
            print(
                f'\nEpoch: {epoch+1}/{epochs}  |  Validation Loss -- (CE): {val_loss:.6f}'
            )

    return model
