from collections.abc import Callable

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
) -> torch.Tensor:
    """Accumulate gradients and return detached loss; caller handles mode and updates."""
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
    start = 0
    for microbatch_advantages in advantages.tensor_split(gradient_accumulation_steps): # 均衡切分
        end = start + microbatch_advantages.numel()
        microbatch = {key: tensor[start:end].to(device) for key, tensor in token_batch.items()} # 字典推导式
        scores = get_response_log_probs(model, microbatch["input_ids"], microbatch["labels"]) # 算 label的 log_prob
        token_losses, _ = compute_policy_gradient_loss(
            microbatch_advantages.to(device), scores["log_probs"]
        )
        loss = aggregate_loss_across_microbatch(token_losses, microbatch["response_mask"]) # 得出这个批次的loss
        weighted_loss = loss * ((end - start) / batch_size)
        weighted_loss.backward()
        total_loss += weighted_loss.detach()
        start = end
    return total_loss
