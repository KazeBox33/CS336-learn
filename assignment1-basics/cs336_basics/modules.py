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
            max_seq_len:int|None =None,
            theta:float|None=None,
            device:torch.device|None=None,
            dtype: torch.dtype| None=None,
    ): 
        super().__init__()

        assert d_model%num_heads==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads

        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.o_proj=Linear(d_model,d_model,device=device,dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope=None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None) ->torch.Tensor:
        *batch_dims,sequence_length,_=x.shape
        
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.view(*batch_dims,sequence_length,self.num_heads,self.d_head)
        k=k.view(*batch_dims,sequence_length,self.num_heads,self.d_head)
        v=v.view(*batch_dims,sequence_length,self.num_heads,self.d_head)

        q=q.transpose(-3,-2)  #(batch,sequence, head, dim) -> (batch,head,sequence,dim)  保证了多头注意力相互隔离
        k=k.transpose(-3,-2)
        v=v.transpose(-3,-2)

        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(sequence_length,device=x.device)
            q=self.rope(q,token_positions)
            k=self.rope(k,token_positions)

        mask=torch.tril(
            torch.ones(sequence_length,sequence_length,device=x.device,dtype=torch.bool)
        )

        attn_output=scaled_dot_product_attention(q,k,v,mask)

        attn_output=attn_output.transpose(-3,-2)  #转换回来了
        attn_output=attn_output.contiguous().view(*batch_dims,sequence_length,self.d_model) 
        #contiguous() 是因为 transpose 后 tensor 的内存布局可能不是连续的，直接 view 可能报错。所以先让它变成连续内存。

        return self.o_proj(attn_output)



class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta:float,
            d_k:int, # d_k表示的是维度
            max_seq_len:int,
            device:torch.device|None =None,
    ):
        super().__init__()

        assert d_k %2 ==0 # Rope需要两两作为一组维度

        self.theta =theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len

        dim_indices=torch.arange(0,d_k,2,device=device).float() 
        inv_freq=1.0/(theta**(dim_indices/d_k))  # 算频率

        position=torch.arange(max_seq_len,device=device).float()
        angles=torch.outer(position,inv_freq) #外积

        self.register_buffer("cos",torch.cos(angles),persistent=False)
        self.register_buffer("sin",torch.sin(angles),persistent=False)

    def forward(self,x:torch.Tensor,token_positions: torch.Tensor) -> torch.Tensor:
        cos=self.cos[token_positions]
        sin=self.sin[token_positions]

        x_even=x[...,0::2]
        x_odd=x[...,1::2]

        rotated_even=x_even*cos-x_odd*sin
        rotated_odd=x_even*sin+x_odd*cos

        result=torch.empty_like(x)
        result[...,0::2]=rotated_even
        result[...,1::2]=rotated_odd
        return result
    
def softmax(x:torch.Tensor,dim:int) ->torch.Tensor:
    x_max=torch.max(x,dim=dim,keepdim=True).values
    x_shifted=x-x_max
    exp_x=torch.exp(x_shifted)
    return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)


# 下一部分是Transformer Block

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            d_ff:int,
            max_seq_len:int,
            theta:float,
            device:torch.device|None=None,
            dtype:torch.dtype|None=None,
    ):
        super().__init__()

        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)
        self.attn=MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        self.ln2=RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        x:torch.Tensor,
        token_positions:torch.Tensor|None=None,
    )-> torch.Tensor:
        x=x+self.attn(self.ln1(x),token_positions)
        x=x+self.ffn(self.ln2(x))

        return x
    
# Transformer Language Model

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            context_length:int,  # 表示模型一次性最多能看多少个token
            d_model:int,
            num_layers:int,
            num_heads:int,
            d_ff:int,
            rope_theta:float,
            device:torch.device|None=None,
            dtype:torch.dtype|None=None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings=Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.layers=nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
        self.lm_head=Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim < 1:
            raise ValueError("token_ids must have at least one dimension")

        sequence_length = token_ids.shape[-1]

        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence_length {sequence_length} exceeds context_length {self.context_length}"
            )

        token_positions=torch.arange(
                sequence_length,
                device=token_ids.device
            )
        
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x=layer(x,token_positions)

        x=self.ln_final(x)
        logits=self.lm_head(x)

        return logits


# 第四部分 反向传播梯度更新

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: #loss = -log(softmax(logits)[target])   可以化简成 loss = log(sum(exp(logits))) - logits[target]
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = torch.gather(
        logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return (log_normalizer - target_logits).mean()
