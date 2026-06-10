from __future__ import annotations

import torch

from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def sample_from_logits(
        logits:torch.Tensor,
        temperature:float=1.0,
        top_p:float|None=None, #表示可以不采用 top-p 采样
) -> int:
    if temperature==0:  #直接选最大概率token
        return int(torch.argmax(logits).item()) #直接取里面的最大值

    probs=torch.softmax(logits/temperature,dim=-1)   #如果temperature 小于1 ，分数差距会被放大， 模型更保守。 如果大于1，分数差距会被缩小，模型更加随机

    if top_p is not None:
        sorted_probs,sorted_indices=torch.sort(probs,descending=True)  # 返回 排序后的概率 和 排序前的位置， descending=True 从大到小排序
        cumulative_probs=torch.cumsum(sorted_probs,dim=-1) #相加 比如sorted_probs = [0.6, 0.3, 0.1] 累加后cumulative_probs = [0.6, 0.9, 1.0]
        
        keep=cumulative_probs-sorted_probs<=top_p # 它表示：保留那些“加入当前 token 之前，累计概率还没有超过 top_p”的 token。

        filtered_probs=torch.zeros_like(probs)
        filtered_probs[sorted_indices[keep]]=sorted_probs[keep] #把位置回归
        probs=filtered_probs/filtered_probs.sum() #归一化

    next_token=torch.multinomial(probs,num_samples=1)
    return int(next_token.item())
        
        
@torch.no_grad()
def generate(
    model:TransformerLM,
    tokenizer:Tokenizer,
    prompt:str,
    max_new_tokens:int,
    temperature:float=1.0,
    top_p:float|None=None,
    device:str|torch.device="cpu",
    end_token:str="<|endoftext|>",
)-> str:
    token_ids=tokenizer.encode(prompt)
    end_token_id=tokenizer.token_to_id[end_token.encode("utf-8")]

    was_training=model.training
    model.eval()

    for _ in range(max_new_tokens):
        input_ids=token_ids[-model.context_length:]
        x=torch.tensor(input_ids,dtype=torch.long,device=device).unsqueeze(0) #需要保持维度为(batch_size,sequence)

        logits=model(x)
        next_logits=logits[0,-1]  #只需要最后一个位置的
        
        next_token_id=sample_from_logits(next_logits,temperature,top_p) 
        token_ids.append(next_token_id)

        if next_token_id ==end_token_id:
            break

    if was_training:
        model.train()

    return tokenizer.decode(token_ids)
