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


@torch.no_grad()
def get_old_log_probs( # 旧模型的评分
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    microbatch_size: int,
) -> torch.Tensor:
    """Cache CPU token scores before updating the policy that generated a rollout batch."""
    if input_ids.ndim != 2 or input_ids.shape != labels.shape or input_ids.shape[0] == 0:
        raise ValueError("input_ids and labels must share a non-empty (batch_size, sequence_length) shape.")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive.")

    device = next(model.parameters()).device
    was_training = model.training
    log_prob_chunks = []
    try:
        model.eval()
        for start in range(0, input_ids.shape[0], microbatch_size):
            scores = get_response_log_probs(
                model,
                input_ids[start:start + microbatch_size].to(device),
                labels[start:start + microbatch_size].to(device),
            )
            log_prob_chunks.append(scores["log_probs"].cpu())
    finally:
        model.train(was_training)
    return torch.cat(log_prob_chunks, dim=0) # 沿维度拼接


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # （batch_size,） 回答奖励
    policy_log_probs: torch.Tensor, # (batch_size, max_length)
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return token losses; response masking and aggregation happen separately."""
    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise NotImplementedError("Unknown importance reweighting method.")
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
    log_ratio = policy_log_probs - old_log_probs.detach()
    if importance_reweighting_method == "gspo":
        if response_mask is None or response_mask.shape != policy_log_probs.shape:
            raise ValueError("response_mask must match policy_log_probs for gspo reweighting.")
        response_tokens = response_mask.bool()
        # Average log-ratios before exponentiating; never multiply token ratios.
        if log_ratio.dtype in (torch.float16, torch.bfloat16):
            log_ratio = log_ratio.float() # 换成32位
        response_lengths = response_tokens.sum(dim=-1, keepdim=True).clamp_min(1) # 回复长度
        mean_log_ratio = log_ratio.masked_fill(~response_tokens, 0.0).sum(dim=-1, keepdim=True) / response_lengths
        ratio = mean_log_ratio.exp().expand_as(policy_log_probs)
    else:
        ratio = log_ratio.exp()
    objective = weights * ratio
    if importance_reweighting_method in ("grpo", "gspo"): # grpo 和 gspo 还要裁剪一下
        if cliprange is None or not math.isfinite(cliprange) or cliprange < 0:
            raise ValueError("cliprange must be finite and non-negative for clipped reweighting.")
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
