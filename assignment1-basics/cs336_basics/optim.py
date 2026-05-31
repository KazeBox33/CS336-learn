from __future__ import annotations

import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr:float=1e-3,
            betas:tuple[float,float]=(0.9,0.95),
            eps:float=1e-8,
            weight_decay:float=0.01,
    ):
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params,defaults)
