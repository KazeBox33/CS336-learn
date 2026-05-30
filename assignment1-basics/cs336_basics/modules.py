from __future__ import annotations

import math

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
            self,
            in_features:int,
            out_features:int,
            device:torch.device|None =None,
            dtype: torch.dtype| None =None,
    ):
        super().__init__()

        self.in_features=in_features
        self.out_features=out_features

        self.weight=nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std) # 初始化Linear 层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    




class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings:int,
            embedding_dim:int,
            device:torch.device|None=None,
            dtype: torch.dtype|None=None,
    ):
        super().__init__()

        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim

        self.weight=nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor) ->torch.Tensor:
        return self.weight[token_ids]

        

class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model:int,
        eps:float=1e-5,
        device:torch.device |None = None,
        dtype:torch.dtype|None =None 
    ):
        super().__init__()


