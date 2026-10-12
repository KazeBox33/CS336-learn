from collections.abc import Callable

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
    return raw_rewards, metadata
