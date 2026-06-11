from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import numpy as np
import torch


from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.modules import TransformerLM,cross_entropy
from cs336_basics.optim import AdamW,get_lr_cosine_schedule,gradient_clipping
from cs336_basics.experiment import ExperimentLogger

@dataclass
class  TrainConfig:
    #数据相关
    train_data_path:str #训练集
    valid_data_path:str #测试集
    checkpoint_path:str
    #模型超参数
    vocab_size:int
    context_length:int
    d_model:int
    num_layers:int
    num_heads:int
    d_ff:int
    rope_theta:float
    #训练控制参数
    batch_size:int
    max_iters:int
    eval_interval:int  #每隔多少步评估一次
    eval_iters:int     #每次评估平均多少个batch
    save_interval:int  #每隔多少步保存checkpoint
    #优化器相关
    max_learning_rate:float
    min_learning_rate:float
    warmup_iters:int
    cosine_cycle_iters:int
    weight_decay:float
    betas:tuple[float,float]
    eps:float
    max_l2_norm:float

    device:str="cpu"
    log_path: str|None=None
    

def load_token_data(path:str) ->np.ndarray:
    return np.load(path,mmap_mode="r") #从磁盘加载token id 数据，并且用memory-mapped模式避免一次性读完整个大文件  ,不会一次性把整个文件全部读进RAM,而是在访问某一段时才读取那一段

def build_model(config: TrainConfig) ->TransformerLM:
    model=TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=torch.device(config.device)
    )
    return model


def build_optimizer(model:torch.nn.Module, config:TrainConfig)-> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )

@torch.no_grad() #这里不跟踪梯度
def estimate_loss(
    model:torch.nn.Module,
    dataset:np.ndarray,
    config:TrainConfig,
)-> float:
    if config.eval_iters<=0:
        raise ValueError("eval_iters must be positive")
    
    was_training=model.training
    model.eval()

    losses=[]
    for _ in range(config.eval_iters):
        x,y=get_batch(
            dataset,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits=model(x)
        loss=cross_entropy(logits,y)
        losses.append(loss.item()) # .item() 是将当前的 损失这个数加入到 losses这个数组当中

    if was_training:
        model.train()

    return sum(losses)/len(losses)


#加载数据，创建模型 创建 optimizer  
# 每一步训练：
#   get_batch
#   forward
#   loss
#   backward
#   gradient clipping
#   learning rate schedule
#   optimizer step
#   定期eval 评估 
#   定期 save checkpoint


def set_learning_rate(optimizer:torch.optim.Optimizer,lr :float) -> None:
    for group in optimizer.param_groups:
        group["lr"]=lr


def train(config:TrainConfig)-> None:
    train_data=load_token_data(config.train_data_path)
    valid_data=load_token_data(config.valid_data_path)

    model=build_model(config)
    optimizer=build_optimizer(model,config)

    logger=ExperimentLogger(config.log_path) if config.log_path is not None else None

    checkpoint_path=Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)

    model.train()

    for it in range(config.max_iters):
        lr=get_lr_cosine_schedule(
            it,
            config.max_learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.cosine_cycle_iters,
        )
        set_learning_rate(optimizer,lr)

        x,y=get_batch(
            train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device
        )

        optimizer.zero_grad()
        logits=model(x)
        loss=cross_entropy(logits,y)
        loss.backward()
        
        gradient_clipping(model.parameters(),config.max_l2_norm)
        optimizer.step()

        if it%config.eval_interval==0:
            train_loss=estimate_loss(model,train_data,config)
            valid_loss=estimate_loss(model,valid_data,config)
            print(
                f"iter {it}: "
                f"train loss {train_loss:.4f}, "
                f"valid loss {valid_loss:.4f}, "
                f"lr {lr:.6e}"
            )

            if logger is not None :
                logger.log(
                    {
                        "step": it,
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "lr": lr,
                    }
                )

        if it>0 and it%config.save_interval==0:
            save_checkpoint(model,optimizer,it,checkpoint_path)

    save_checkpoint(model,optimizer,config.max_iters,checkpoint_path)
