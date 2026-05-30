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
        
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)

        rms=torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)
        result=x/rms*self.weight.to(torch.float32)

        return result.to(in_dtype)


def silu(x:torch.Tensor) ->torch.Tensor:
    return x*torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,
        device: torch.device| None=None,
        dtype:torch.dtype| None=None,
    ):
        super().__init__()

        self.d_model=d_model
        self.d_ff=d_ff

        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype )
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        return self.w2(silu(self.w1(x))*self.w3(x))
    

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor|None=None
)->torch.Tensor:
    d_k=Q.shape[-1]
    scores=Q@K.transpose(-2,-1)/math.sqrt(d_k)

    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))

    attention_weights=torch.softmax(scores,dim=-1)

    return attention_weights @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            device:torch.device|None=None,
            dtype: torch.dtype| None=None,
    ): 
        super().__init__()

        assert d_model%num_heads==0

        