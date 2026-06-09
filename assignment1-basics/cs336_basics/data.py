from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch 

def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    context_length:int,
    device:str,
)-> tuple[torch.Tensor,torch.Tensor]:
    max_start=len(dataset)-context_length
    starts=torch.randint(0,max_start,(batch_size,)) #从众多 start 起点中，选出batch_size序列个起点出来 ，从 0 到 max_start-1中取数

    x=torch.stack( #输入，  stack会把多个形状相同的tensor堆起来，增加一个新的维度
        [torch.from_numpy(dataset[start:start+context_length].astype(np.int64)) for start in starts]
    )  #最终得到 x.shape==(batch_size,context_length)

    y=torch.stack( #结果
        [torch.from_numpy(dataset[start+1:start+context_length+1].astype(np.int64)) for start in starts]
    )

    return x.to(device),y.to(device)
