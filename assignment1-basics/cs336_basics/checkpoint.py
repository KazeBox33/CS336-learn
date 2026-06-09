from __future__ import annotations

import os
from typing import BinaryIO,IO

import torch


def save_checkpoint(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:str|os.PathLike|BinaryIO|IO[bytes],  # 可能传入 字符串路径，Path对象，二进制文件对象，bytes IO 对象
)-> None:
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration
    }