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

            for p in group["params"]: # 表明的是每一组参数 
                if p.grad is None:
                    continue
                grad=p.grad    #反向传播后更新梯度
                 
                state=self.state[p]   #optimizer基类维护的字典 用来给参数保存优化器自己的历史状态

                if len(state)==0: #表示第一次优化
                    state["step"]=0
                    state["exp_avg"]=torch.zeros_like(p)
                    state["exp_avg_sq"]=torch.zeros_like(p)

                state["step"]+=1

                exp_avg=state["exp_avg"]  #一阶动量， 用来维护历史平均方向
                exp_avg_sq=state["exp_avg_sq"] #二阶动量， 用来维护历史平均大小

                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1) #更新一阶动量
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2) #更新二阶动量
                
                bias_correction1=1-beta1**state["step"] # 用于偏差修正的
                bias_correction2=1-beta2**state["step"]

                corrected_exp_avg=exp_avg/bias_correction1
                corrected_exp_avg_sq=exp_avg_sq/bias_correction2
                
                update=corrected_exp_avg/(torch.sqrt(corrected_exp_avg_sq)+eps) #获取更新参数

                p.data.mul_(1-lr*weight_decay)
                p.data.add_(update,alpha=-lr)

        return loss