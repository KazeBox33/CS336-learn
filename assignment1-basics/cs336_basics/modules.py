from __future__ import annotations

import math

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(
            self,
            in_feartures:int,
            out_features:int,
            device:torch.device|None =None,
            dtype: torch.dtype| None =None,
    ):
        super().__init__()

        self.in_features=self.in_features
        self.out_features=out_features

        self.weight=nn.Paramter(
            torch.empty(
                out_features,
                in_feartures,
                device=device,
                dtype=dtype
            )

        )