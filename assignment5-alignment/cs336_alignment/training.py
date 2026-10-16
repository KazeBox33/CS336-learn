import math
from collections.abc import Callable
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)
from cs336_alignment.rewards import (
    compute_group_normalized_rewards,
    compute_rollout_rewards,
)
from cs336_alignment.tokenization import tokenize_prompt_and_output


def _prepare_rollout_batch(
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int, # 每道题的回答量
    advantage_eps: float = 1e-6,
    baseline: Literal["mean", "none"] = "mean",
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, float]]:
    """Prepare CPU tokens and full-group advantages before pruning or microbatching."""
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
        raw_rewards, group_size, baseline=baseline, advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    token_batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    reward_metadata.update(advantage_metadata)
    return token_batch, advantages, reward_metadata


def _accumulate_microbatch_gradients(
    model: torch.nn.Module,
    token_batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backpropagate nonzero advantages with the original batch's loss denominator.

    Entropy is measured only on retained responses. For an all-zero batch, dense
    language-model parameters receive explicit zero gradients so optimizer state
    and weight decay still advance as they do after a zero-loss backward pass.
    """
    batch_size = advantages.numel()
    if advantages.ndim != 1 or batch_size == 0:
        raise ValueError("advantages must be a non-empty one-dimensional tensor.")
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("gradient_accumulation_steps must be between 1 and batch_size.")
    if loss_normalization not in ("sequence", "constant"):
        raise NotImplementedError("Only sequence and constant loss normalization are implemented.")
    if loss_normalization == "constant" and (
        normalization_constant is None
        or not math.isfinite(normalization_constant)
        or normalization_constant <= 0
    ):
        raise ValueError("normalization_constant must be finite and positive for constant normalization.")
    shape = token_batch["input_ids"].shape
    if len(shape) != 2 or shape[0] != batch_size or any(
        token_batch[key].shape != shape for key in ("labels", "response_mask")
    ):
        raise ValueError("Token tensors must share shape (batch_size, sequence_length).")

    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    response_tokens = torch.zeros((), device=device)
    active_indices = advantages.nonzero(as_tuple=True)[0] # 找到 advantage不为0的下标
    if active_indices.numel() == 0:
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter) # 无梯度变化
    microbatch_sizes = [chunk.numel() for chunk in advantages.tensor_split(gradient_accumulation_steps)]
    start = 0
    for microbatch_size in microbatch_sizes:
        indices = active_indices[start:start + microbatch_size]
        if indices.numel() == 0:
            break
        microbatch_advantages = advantages[indices]
        microbatch = {key: tensor[indices].to(device) for key, tensor in token_batch.items()} # 字典推导式
        scores = get_response_log_probs(
            model, microbatch["input_ids"], microbatch["labels"], return_token_entropy=True
        ) # 算 label的 log_prob
        # Entropy is only logged; do not retain its separate autograd branch.
        token_entropy = scores.pop("token_entropy").detach()
        token_losses, _ = compute_policy_gradient_loss(
            microbatch_advantages.to(device), scores["log_probs"]
        )
        loss = aggregate_loss_across_microbatch( # 得出这个批次的loss
            token_losses, microbatch["response_mask"], loss_normalization, normalization_constant
        )
        # Sequence means retain the original batch size; constant sums already use Z.
        weighted_loss = (
            loss * (indices.numel() / batch_size) if loss_normalization == "sequence" else loss # advantage 为0的就不算了
        )
        weighted_loss.backward()
        total_loss += weighted_loss.detach()
        total_entropy += token_entropy.masked_fill(~microbatch["response_mask"], 0.0).sum()
        response_tokens += microbatch["response_mask"].sum()
        start += indices.numel()
    return total_loss, {"mean_token_entropy": total_entropy / response_tokens.clamp_min(1)}


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
    """Perform one on-policy update using the requested advantage and loss variants."""
    if importance_reweighting_method != "none":
        raise NotImplementedError("This step supports only on-policy training without reweighting.")
    if not 1 <= gradient_accumulation_steps <= len(rollout_responses):
        raise ValueError("gradient_accumulation_steps must be between 1 and the rollout batch size.")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    token_batch, advantages, metadata = _prepare_rollout_batch(
        tokenizer, reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths,
        group_size, advantage_eps, baseline, advantage_normalizer,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, accumulation_metadata = _accumulate_microbatch_gradients(
        model, token_batch, advantages, gradient_accumulation_steps,
        loss_normalization, normalization_constant,
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
