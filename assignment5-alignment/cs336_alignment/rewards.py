from collections.abc import Callable
from typing import Literal

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str], # 回答
    repeated_ground_truths: list[str],# 答案
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score paired rollouts and summarize their total and format rewards."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Each rollout response must have exactly one ground truth.")
    if not rollout_responses:
        raise ValueError("The rollout batch must contain at least one response.")

    rewards = []
    format_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True):
        scores = reward_fn(response, ground_truth)
        rewards.append(scores["reward"])
        format_rewards.append(scores["format_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
    }
    return raw_rewards, metadata # 返回个一维奖励向量和metadata


def compute_group_normalized_rewards( # advantage_normalization
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute standard GRPO advantages within each prompt's response group."""
    if baseline != "mean" or advantage_normalizer != "std":
        raise NotImplementedError("This step supports only mean baseline and std normalization.")
    if raw_rewards.ndim != 1 or raw_rewards.numel() == 0:
        raise ValueError("raw_rewards must be a non-empty one-dimensional tensor.")
    if group_size <= 0 or raw_rewards.numel() % group_size != 0:
        raise ValueError("group_size must be positive and divide the number of rewards.")
    if advantage_eps <= 0:
        raise ValueError("advantage_eps must be positive.")

    grouped_rewards = raw_rewards.float().reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    # Sample std matches the assignment; singleton groups have no relative signal.
    group_stds = grouped_rewards.std(
        dim=1, keepdim=True, correction=1 if group_size > 1 else 0
    ) # 使用样本标准差
    advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    metadata = {
        "mean_reward": grouped_rewards.mean().item(),
        "mean_group_std": group_stds.mean().item(),
    }
    return advantages.reshape(-1), metadata
