import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice

import torch
from torch.nn import functional as F


def compute_sft_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean next-token loss over a packed batch whose labels are already shifted."""
    input_ids, labels = batch["input_ids"], batch["labels"]
    if input_ids.ndim != 2 or input_ids.shape != labels.shape or input_ids.numel() == 0:
        raise ValueError("input_ids and labels must share a non-empty (batch_size, sequence_length) shape.")

    device = next(model.parameters()).device
    # Do not pass labels to the model: causal-LM loss helpers may shift them again.
    logits = model(input_ids=input_ids.to(device)).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), # 前面的维度合并
        labels.to(device).reshape(-1),
    )


def sft_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[dict[str, torch.Tensor]],# 一组小批次
    max_grad_norm: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Accumulate packed-token means and update once, including partial final groups.

    The caller supplies up to its configured accumulation count of CPU batches.
    Every label is a valid packed token. Weight by token counts, not batch counts.
    """
    if not microbatches:
        raise ValueError("Provide at least one SFT microbatch.")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    token_counts = [batch["labels"].numel() for batch in microbatches]
    if any(count == 0 for count in token_counts):
        raise ValueError("SFT microbatches must contain at least one token.")
    total_tokens = sum(token_counts)
    total_loss = torch.zeros((), device=next(model.parameters()).device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for batch, count in zip(microbatches, token_counts, strict=True):
            loss = compute_sft_loss(model, batch)
            weighted_loss = loss * (count / total_tokens)
            weighted_loss.backward()
            total_loss += weighted_loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm if max_grad_norm is not None else math.inf,
            error_if_nonfinite=True,
        )
        optimizer.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    return total_loss, {
        "grad_norm": grad_norm.detach(),
        "num_tokens": total_tokens,
        "num_microbatches": len(microbatches),
    }


def create_sft_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, indexed by completed optimizer updates.

    The optimizer's initial learning rates are the peaks. With warmup, the first
    update uses zero LR; after total_updates updates the next LR is zero.
    Create once for the entire run, not once per epoch.
    """
    if not isinstance(total_updates, int) or total_updates <= 0:
        raise ValueError("total_updates must be a positive integer.")
    if not isinstance(warmup_updates, int) or not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be an integer in [0, total_updates).")

    def lr_factor(completed_updates: int) -> float:
        if completed_updates < warmup_updates:
            return completed_updates / warmup_updates
        progress = min((completed_updates - warmup_updates) / (total_updates - warmup_updates), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def run_sft_epoch( # 验证的，不反向也不更新参数
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict[str, torch.Tensor]],
    gradient_accumulation_steps: int, # 梯度累计的次数
    max_grad_norm: float | None = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Iterator[tuple[int, dict[str, float | int]]]:
    """Lazily train one epoch, yielding metrics after each optimizer update.

    Update numbers start at 1 within this epoch. Empty inputs yield no updates.
    Keep only one accumulation group of CPU batches, including a partial tail.
    """
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError("The scheduler must belong to the training optimizer.")
    batch_iterator = iter(batches)
    update = 0
    while microbatches := list(islice(batch_iterator, gradient_accumulation_steps)):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        loss, metadata = sft_train_step(model, optimizer, microbatches, max_grad_norm)
        if scheduler is not None:
            scheduler.step()
        update += 1
        yield update, {
            "loss": loss.item(),
            "grad_norm": metadata["grad_norm"].item(),
            "num_tokens": metadata["num_tokens"],
            "num_microbatches": metadata["num_microbatches"],
            "learning_rate": learning_rate,
        }


@torch.no_grad()
def evaluate_sft(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, float | int]:
    """Evaluate packed-token mean loss without changing parameters or gradients."""
    was_training = model.training
    total_loss = torch.zeros((), device=next(model.parameters()).device)
    total_tokens = 0
    num_batches = 0
    model.eval()
    try:
        for batch in batches:
            loss = compute_sft_loss(model, batch)
            num_tokens = batch["labels"].numel()
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            num_batches += 1
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("SFT validation requires at least one token.")
    return {
        "val_loss": (total_loss / total_tokens).item(),
        "num_tokens": total_tokens,
        "num_batches": num_batches,
    }
