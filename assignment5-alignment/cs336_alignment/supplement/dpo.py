import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase

from cs336_alignment.policy import get_response_log_probs


def _sequence_log_prob(model: torch.nn.Module, token_ids: list[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    scores = get_response_log_probs(model, tokens[:, :-1], tokens[:, 1:])
    return scores["log_probs"].sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Return scalar DPO loss on the policy device, with gradients only through it.

    Score complete Alpaca documents: shared prompt terms cancel within each
    model's chosen-minus-rejected difference. No truncation or length averaging.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative.")
    if not all(isinstance(text, str) for text in (prompt, response_chosen, response_rejected)):
        raise TypeError("prompt and responses must be strings.")
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError("The DPO tokenizer must define an integer eos_token_id.")
    template_path = Path(__file__).resolve().parents[1] / "prompts_safety" / "alpaca_sft.prompt"
    template = template_path.read_text(encoding="utf-8").rstrip("\n")
    token_sequences = [
        tokenizer.encode(template.format(instruction=prompt, response=response), add_special_tokens=True)
        + [tokenizer.eos_token_id]
        for response in (response_chosen, response_rejected)
    ]
    was_training, ref_was_training = lm.training, lm_ref.training
    try:
        # eval disables dropout, not autograd, so the policy remains differentiable.
        lm.eval()
        lm_ref.eval()
        with torch.no_grad():
            ref_margin = _sequence_log_prob(lm_ref, token_sequences[0]) - _sequence_log_prob(lm_ref, token_sequences[1])
        policy_margin = _sequence_log_prob(lm, token_sequences[0]) - _sequence_log_prob(lm, token_sequences[1])
        return -F.logsigmoid(beta * (policy_margin - ref_margin.to(policy_margin.device)))
    finally:
        lm.train(was_training)
        lm_ref.train(ref_was_training)
