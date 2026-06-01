from __future__ import annotations

import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr:float=1e-3,  #学习率
            betas:tuple[float,float]=(0.9,0.95), #两个动量
            eps:float=1e-8,
            weight_decay:float=0.01, #权重衰减
    ):
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params,defaults)
    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:  # 在初始创建时会创建这个param_groups
            lr = group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad
                 
                state=self.state[p]   #optimizer基类维护的字典 用来给参数保存优化器自己的历史状态

                if len(state)==0: #表示第一次优化
                    state["step"]=0
                    state["exp_avg"]=torch.zeros_like(p)
                    state["exp_avg_sq"]=torch.zeros_like(p)

                state["step"]+=1

                exp_avg=state["exp_avg"]
                exp_avg_sq=state["exp_avg_sq"]

                