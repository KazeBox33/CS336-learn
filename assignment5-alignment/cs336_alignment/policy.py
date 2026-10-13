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
    """Return on-policy token losses; masking and aggregation happen separately."""
    if importance_reweighting_method != "none":
        raise NotImplementedError("This step supports only on-policy loss without reweighting.")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")

    weights = raw_rewards_or_advantages
    if weights.ndim == 1:
        weights = weights.unsqueeze(-1)
    if weights.shape != (policy_log_probs.shape[0], 1):
        raise ValueError("Rewards or advantages must have shape (batch_size,) or (batch_size, 1).")

    # Treat rewards as fixed estimator weights, but retain the policy gradient.
    per_token_loss = -weights.detach() * policy_log_probs
    return per_token_loss, {}
