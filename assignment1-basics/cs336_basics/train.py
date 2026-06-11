from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    model_name:str="transformer_lm"
    seed:int=2025
    dry_run:bool=False
    

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


def count_parameters(model:torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_non_embedding_parameters(model:torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "token_embedding" not in name and "embedding" not in name
    )


def format_int(value:int) -> str:
    return f"{value:_}"


def format_training_config(
    config: TrainConfig,
    model: torch.nn.Module,
    train_tokens: int,
    valid_tokens: int,
) -> str:
    tokens_per_step=config.batch_size * config.context_length
    total_train_tokens=tokens_per_step * config.max_iters
    approx_epochs=total_train_tokens / train_tokens if train_tokens > 0 else 0.0
    total_params=count_parameters(model)
    non_embedding_params=count_non_embedding_parameters(model)

    return f"""@dataclass
class ModelConfig:
    model_name: str = "{config.model_name}"
    vocab_size: int = {format_int(config.vocab_size)}
    context_length: int = {format_int(config.context_length)}
    d_model: int = {format_int(config.d_model)}
    num_layers: int = {format_int(config.num_layers)}
    num_heads: int = {format_int(config.num_heads)}
    d_ff: int = {format_int(config.d_ff)}
    rope_theta: float = {config.rope_theta:g}
    total_parameters: int = {format_int(total_params)}
    non_embedding_parameters: int = {format_int(non_embedding_params)}


@dataclass
class TrainingConfig:
    batch_size: int = {format_int(config.batch_size)}
    max_iters: int = {format_int(config.max_iters)}
    tokens_per_step: int = {format_int(tokens_per_step)}
    total_tokens_processed: int = {format_int(total_train_tokens)}
    approx_epochs: float = {approx_epochs:.4f}
    train_tokens: int = {format_int(train_tokens)}
    valid_tokens: int = {format_int(valid_tokens)}
    train_data_path: str = "{config.train_data_path}"
    valid_data_path: str = "{config.valid_data_path}"

    # Optimizer related parameters
    betas: tuple = {config.betas}
    weight_decay: float = {config.weight_decay:g}
    max_lr: float = {config.max_learning_rate:g}
    min_lr: float = {config.min_learning_rate:g}
    warmup_iters: int = {format_int(config.warmup_iters)}
    cosine_cycle_iters: int = {format_int(config.cosine_cycle_iters)}
    eps: float = {config.eps:g}
    max_grad_norm: float = {config.max_l2_norm:g}

    # Logging & checkpointing
    eval_interval: int = {format_int(config.eval_interval)}
    eval_iters: int = {format_int(config.eval_iters)}
    save_interval: int = {format_int(config.save_interval)}
    checkpoint_path: str = "{config.checkpoint_path}"
    log_path: str = "{config.log_path}"

    # Others
    device: str = "{config.device}"
    seed: int = {config.seed}
    dry_run: bool = {config.dry_run}
"""


def write_training_config(
    config: TrainConfig,
    model: torch.nn.Module,
    train_tokens: int,
    valid_tokens: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    config_text=format_training_config(config,model,train_tokens,valid_tokens)
    print("\nConfiguration\n")
    print(config_text)
    (output_dir / "config.txt").write_text(config_text,encoding="utf-8")

    payload=asdict(config)
    payload["tokens_per_step"]=config.batch_size * config.context_length
    payload["total_tokens_processed"]=payload["tokens_per_step"] * config.max_iters
    payload["train_tokens"]=int(train_tokens)
    payload["valid_tokens"]=int(valid_tokens)
    payload["approx_epochs"]=payload["total_tokens_processed"] / train_tokens if train_tokens > 0 else 0.0
    payload["total_parameters"]=count_parameters(model)
    payload["non_embedding_parameters"]=count_non_embedding_parameters(model)
    (output_dir / "config.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

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
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available() and config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    train_data=load_token_data(config.train_data_path)
    valid_data=load_token_data(config.valid_data_path)

    model=build_model(config)
    optimizer=build_optimizer(model,config)

    logger=ExperimentLogger(config.log_path) if config.log_path is not None else None

    checkpoint_path=Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    write_training_config(
        config,
        model,
        train_tokens=len(train_data),
        valid_tokens=len(valid_data),
        output_dir=checkpoint_path.parent,
    )
    if config.dry_run:
        print("Dry run enabled: configuration was written, training will not start.")
        return

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
