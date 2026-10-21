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


def run_sft_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict[str, torch.Tensor]],
    gradient_accumulation_steps: int, # 梯度累计的次数
    max_grad_norm: float | None = 1.0,
) -> Iterator[tuple[int, dict[str, float | int]]]:
    """Lazily train one epoch, yielding metrics after each optimizer update.

    Update numbers start at 1 within this epoch. Empty inputs yield no updates.
    Keep only one accumulation group of CPU batches, including a partial tail.
    """
    if not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")
    batch_iterator = iter(batches)
    update = 0
    while microbatches := list(islice(batch_iterator, gradient_accumulation_steps)):
        loss, metadata = sft_train_step(model, optimizer, microbatches, max_grad_norm)
        update += 1
        yield update, {
            "loss": loss.item(),
            "grad_norm": metadata["grad_norm"].item(),
            "num_tokens": metadata["num_tokens"],
            "num_microbatches": metadata["num_microbatches"],
        }
