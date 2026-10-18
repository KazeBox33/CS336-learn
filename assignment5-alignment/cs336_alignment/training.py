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
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
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
    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise NotImplementedError("Unknown importance reweighting method.")
    if importance_reweighting_method == "gspo" and loss_normalization != "sequence":
        raise NotImplementedError("GSPO currently supports only sequence loss normalization.")
    if importance_reweighting_method != "none" and (
        old_log_probs is None or old_log_probs.shape != shape
    ):
        raise ValueError("old_log_probs must align with the full token batch.")
    if importance_reweighting_method in ("grpo", "gspo") and (
        cliprange is None or not math.isfinite(cliprange) or cliprange < 0
    ):
        raise ValueError("cliprange must be finite and non-negative for clipped reweighting.")

    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    response_tokens = torch.zeros((), device=device)
    clipped_units = torch.zeros((), device=device)
    clip_units = torch.zeros((), device=device)
    active_indices = advantages.nonzero(as_tuple=True)[0] # 找到 advantage不为0的下标
    if active_indices.numel() == 0:
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter) # 无梯度变化
    microbatch_sizes = [chunk.numel() for chunk in advantages.tensor_split(gradient_accumulation_steps)]
    start = 0
    for microbatch_size in microbatch_sizes:
        indices = active_indices[start:start + microbatch_size] # advantage 不为0的索引
        if indices.numel() == 0:
            break
        microbatch_advantages = advantages[indices]
        microbatch = {key: tensor[indices].to(device) for key, tensor in token_batch.items()} # 字典推导式
        scores = get_response_log_probs(
            model, microbatch["input_ids"], microbatch["labels"], return_token_entropy=True
        ) # 算 label的 log_prob
        # Entropy is only logged; do not retain its separate autograd branch.
        token_entropy = scores.pop("token_entropy").detach()
        microbatch_old_log_probs = None
        if importance_reweighting_method != "none":
            # Pruning must select the same response rows from both policies' scores.
            microbatch_old_log_probs = old_log_probs.detach()[indices.to(old_log_probs.device)].to(device) # 把行索引放到旧评分所在的设备
        token_losses, loss_metadata = compute_policy_gradient_loss(
            microbatch_advantages.to(device), scores["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=microbatch_old_log_probs,
            cliprange=cliprange,
            response_mask=microbatch["response_mask"], # 给gspo 准备的mask
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
        if "clip_unit_count" in loss_metadata:
            clipped_units += loss_metadata["clipped_unit_count"]
            clip_units += loss_metadata["clip_unit_count"]
        start += indices.numel()
    metadata = {"mean_token_entropy": total_entropy / response_tokens.clamp_min(1)}
    if importance_reweighting_method in ("grpo", "gspo"):
        metadata.update(
            clipped_unit_count=clipped_units, clip_unit_count=clip_units,
            clip_fraction=clipped_units / clip_units.clamp_min(1),
        )
    return total_loss, metadata


def _update_policy_from_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Update once using already-tokenized responses and already-computed advantages."""
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metadata = _accumulate_microbatch_gradients(
        model, token_batch, advantages, gradient_accumulation_steps,
        loss_normalization, normalization_constant,
        importance_reweighting_method, old_log_probs, cliprange,
    )
    # An infinite limit reports the norm without clipping when clipping is disabled.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else math.inf,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, {**metadata, "grad_norm": grad_norm.detach()}


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
    """Perform one update, optionally reweighting against caller-cached sampling scores."""
    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise NotImplementedError("Unknown importance reweighting method.")
    if not 1 <= gradient_accumulation_steps <= len(rollout_responses):
        raise ValueError("gradient_accumulation_steps must be between 1 and the rollout batch size.")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    token_batch, advantages, metadata = _prepare_rollout_batch(
        tokenizer, reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths,
        group_size, advantage_eps, baseline, advantage_normalizer,
    )
    loss, update_metadata = _update_policy_from_batch(
        model, optimizer, token_batch, advantages, gradient_accumulation_steps,
        max_grad_norm, importance_reweighting_method, old_log_probs, cliprange,
        loss_normalization, normalization_constant,
    )
    return loss, {**metadata, **update_metadata}


def train_rollout_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batch: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Consume a prepared CPU rollout batch once, updating independently per minibatch.

    Advantages must be computed over the original groups before splitting. Cached
    old scores must use the same full-batch token layout and remain fixed throughout.
    Loss and gradient norm are update means; entropy is retained-response-token weighted.
    """
    if advantages.ndim != 1 or advantages.numel() == 0:
        raise ValueError("advantages must be a non-empty one-dimensional tensor.")
    batch_size = advantages.numel()
    if train_batch_size <= 0 or batch_size % train_batch_size != 0:
        raise ValueError("train_batch_size must be positive and divide the rollout batch size.")
    if not 1 <= gradient_accumulation_steps <= train_batch_size:
        raise ValueError("gradient_accumulation_steps must be between 1 and train_batch_size.")
    shape = token_batch["input_ids"].shape
    if len(shape) != 2 or shape[0] != batch_size or any(
        token_batch[key].shape != shape for key in ("labels", "response_mask")
    ):
        raise ValueError("Token tensors must share shape (batch_size, sequence_length).")
    if importance_reweighting_method != "none" and (
        old_log_probs is None or old_log_probs.shape != shape
    ):
        raise ValueError("old_log_probs must align with the full rollout token batch.")

    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach() if old_log_probs is not None else None
    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_norm = torch.zeros_like(total_loss)
    max_norm = torch.zeros_like(total_loss)
    total_entropy = torch.zeros_like(total_loss)
    total_tokens = torch.zeros_like(total_loss)
    clipped_units = torch.zeros_like(total_loss)
    clip_units = torch.zeros_like(total_loss)
    num_updates = batch_size // train_batch_size
    for start in range(0, batch_size, train_batch_size):
        stop = start + train_batch_size
        minibatch = {key: tensor[start:stop] for key, tensor in token_batch.items()}
        minibatch_advantages = advantages[start:stop]
        minibatch_old = old_log_probs[start:stop] if old_log_probs is not None else None
        loss, metadata = _update_policy_from_batch(
            model, optimizer, minibatch, minibatch_advantages,
            gradient_accumulation_steps, max_grad_norm,
            importance_reweighting_method, minibatch_old, cliprange,
        )
        total_loss += loss / num_updates
        total_norm += metadata["grad_norm"] / num_updates
        max_norm = torch.maximum(max_norm, metadata["grad_norm"])
        active_tokens = minibatch["response_mask"][minibatch_advantages != 0].bool().sum().to(device) # 有效token数
        total_entropy += metadata["mean_token_entropy"] * active_tokens
        total_tokens += active_tokens
        if "clip_unit_count" in metadata:
            clipped_units += metadata["clipped_unit_count"]
            clip_units += metadata["clip_unit_count"]
    result = {
        "grad_norm": total_norm,
        "grad_norm_max": max_norm,
        "mean_token_entropy": total_entropy / total_tokens.clamp_min(1),
        "optimizer_steps": num_updates,
    }
    if importance_reweighting_method in ("grpo", "gspo"):
        result.update(
            clipped_unit_count=clipped_units, clip_unit_count=clip_units,
            clip_fraction=clipped_units / clip_units.clamp_min(1),
        )
    return total_loss, result
