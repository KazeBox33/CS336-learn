import math
from typing import Literal

import torch
from torch.nn import functional as F


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score next-token labels without detaching the policy's computation graph."""
    if input_ids.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same two-dimensional shape.")

    logits = model(input_ids=input_ids).logits
    # Normalize in FP32 even when the model produces low-precision logits.
    all_log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs = all_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = -(all_log_probs.exp() * all_log_probs).sum(dim=-1)
    return result


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # （batch_size,） 回答奖励
    policy_log_probs: torch.Tensor, # (batch_size, max_length)
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return token losses; response masking and aggregation happen separately."""
    if importance_reweighting_method not in ("none", "noclip", "grpo"):
        raise NotImplementedError("Only none, noclip, and grpo reweighting are implemented.")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")

    weights = raw_rewards_or_advantages
    if weights.ndim == 1:
        weights = weights.unsqueeze(-1)
    if weights.shape != (policy_log_probs.shape[0], 1):
        raise ValueError("Rewards or advantages must have shape (batch_size,) or (batch_size, 1).")

    # Treat rewards and the sampling policy as fixed, but retain current-policy gradients.
    weights = weights.detach()
    if importance_reweighting_method == "none":
        return -weights * policy_log_probs, {}

    if old_log_probs is None or old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs must have the same shape as policy_log_probs.")
    ratio = (policy_log_probs - old_log_probs.detach()).exp()
    objective = weights * ratio
    if importance_reweighting_method == "grpo":
        if cliprange is None or not math.isfinite(cliprange) or cliprange < 0:
            raise ValueError("cliprange must be finite and non-negative for grpo reweighting.")
        clipped_ratio = ratio.clamp(1 - cliprange, 1 + cliprange)
        objective = torch.minimum(objective, weights * clipped_ratio)
    return -objective, {}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, # (B,T) 上面的损失矩阵
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Reduce masked token losses using per-response means or a fixed total denominator."""
    if loss_normalization not in ("sequence", "constant"):
        raise NotImplementedError("Only sequence and constant loss normalization are implemented.")
    if per_token_policy_gradient_loss.ndim != 2 or mask.shape != per_token_policy_gradient_loss.shape:
        raise ValueError("Loss and mask must have the same (batch_size, sequence_length) shape.")
    if per_token_policy_gradient_loss.shape[0] == 0:
        raise ValueError("The microbatch must contain at least one response.")

    response_mask = mask.bool()
    masked_losses = per_token_policy_gradient_loss.masked_fill(~response_mask, 0.0)
    if loss_normalization == "constant": # 常数
        if normalization_constant is None or (
            not math.isfinite(normalization_constant) or normalization_constant <= 0
        ):
            raise ValueError("normalization_constant must be finite and positive for constant normalization.")
        return masked_losses.sum() / normalization_constant
    # Empty responses contribute zero rather than an undefined 0 / 0.
    response_lengths = response_mask.sum(dim=1).clamp_min(1) # 统计长度
    sequence_losses = masked_losses.sum(dim=1) / response_lengths # 逐token损失，先对每个回答取平均
    return sequence_losses.mean() # 再对回答取平均
