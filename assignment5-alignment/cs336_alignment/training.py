from collections.abc import Callable
import math
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)
from cs336_alignment.rewards import compute_group_normalized_rewards, compute_rollout_rewards
from cs336_alignment.tokenization import tokenize_prompt_and_output


def _prepare_rollout_batch(
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int, # 每道题的回答量
    advantage_eps: float = 1e-6,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, float]]:
    """Prepare CPU token batches and standard GRPO advantages before microbatching."""
    batch_size = len(rollout_responses)
    if batch_size == 0:
        raise ValueError("The rollout batch must contain at least one response.")
    if len(repeated_prompts) != batch_size or len(repeated_ground_truths) != batch_size:
        raise ValueError("Prompts, responses, and ground truths must have the same length.")
    if group_size <= 0 or batch_size % group_size != 0:
        raise ValueError("group_size must be positive and divide the rollout batch size.")

    raw_rewards, reward_metadata = compute_rollout_rewards( # 算奖励
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, advantage_metadata = compute_group_normalized_rewards(  # 算advantage
        raw_rewards, group_size, advantage_eps=advantage_eps
    )
    token_batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    reward_metadata.update(advantage_metadata)
    return token_batch, advantages, reward_metadata


def _accumulate_microbatch_gradients(
    model: torch.nn.Module,
    token_batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    gradient_accumulation_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Accumulate gradients and return detached logs; caller handles mode and updates."""
    batch_size = advantages.numel()
    if advantages.ndim != 1 or batch_size == 0:
        raise ValueError("advantages must be a non-empty one-dimensional tensor.")
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("gradient_accumulation_steps must be between 1 and batch_size.")
    shape = token_batch["input_ids"].shape
    if len(shape) != 2 or shape[0] != batch_size or any(
        token_batch[key].shape != shape for key in ("labels", "response_mask")
    ):
        raise ValueError("Token tensors must share shape (batch_size, sequence_length).")

    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    start = 0
    for microbatch_advantages in advantages.tensor_split(gradient_accumulation_steps): # 均衡切分
        end = start + microbatch_advantages.numel()
        microbatch = {key: tensor[start:end].to(device) for key, tensor in token_batch.items()} # 字典推导式
        scores = get_response_log_probs(
            model, microbatch["input_ids"], microbatch["labels"], return_token_entropy=True
        ) # 算 label的 log_prob
        # Entropy is only logged; do not retain its separate autograd branch.
        token_entropy = scores.pop("token_entropy").detach()
        token_losses, _ = compute_policy_gradient_loss(
            microbatch_advantages.to(device), scores["log_probs"]
        )
        loss = aggregate_loss_across_microbatch(token_losses, microbatch["response_mask"]) # 得出这个批次的loss
        weighted_loss = loss * ((end - start) / batch_size)
        weighted_loss.backward()
        total_loss += weighted_loss.detach()
        total_entropy += token_entropy.masked_fill(~microbatch["response_mask"], 0.0).sum()
        start = end
    response_tokens = token_batch["response_mask"].sum().clamp_min(1).to(device)
    return total_loss, {"mean_token_entropy": total_entropy / response_tokens}


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Perform one standard on-policy GRPO update and return detached logging values."""
    if (baseline, advantage_normalizer, importance_reweighting_method, loss_normalization) != (
        "mean", "std", "none", "sequence"
    ):
        raise NotImplementedError("This step supports only standard on-policy GRPO.")
    if not 1 <= gradient_accumulation_steps <= len(rollout_responses):
        raise ValueError("gradient_accumulation_steps must be between 1 and the rollout batch size.")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    if not math.isfinite(advantage_eps) or advantage_eps <= 0:
        raise ValueError("advantage_eps must be finite and positive.")

    token_batch, advantages, metadata = _prepare_rollout_batch(
        tokenizer, reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths,
        group_size, advantage_eps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, accumulation_metadata = _accumulate_microbatch_gradients(
        model, token_batch, advantages, gradient_accumulation_steps
    )
    # An infinite limit reports the norm without clipping when clipping is disabled.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else math.inf,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, {
        **metadata,
        **accumulation_metadata,
        "grad_norm": grad_norm.detach(),
    }
