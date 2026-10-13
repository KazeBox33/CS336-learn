from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizerBase

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
