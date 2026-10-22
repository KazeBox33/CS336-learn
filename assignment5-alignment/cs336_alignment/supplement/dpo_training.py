import math
from collections.abc import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.supplement.dpo import compute_per_instance_dpo_loss
from cs336_alignment.supplement.hh import HHPreferenceExample


def dpo_train_step(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[HHPreferenceExample],
    beta: float = 0.1,
    max_grad_norm: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Average per-pair DPO losses, accumulate gradients, and update policy once.

    Permanently freeze reference parameters and leave it in eval mode. Only the
    policy may belong to optimizer. The caller groups up to its accumulation
    count of examples; a partial group is averaged by its actual size.
    """
    if not examples:
        raise ValueError("Provide at least one DPO preference example.")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and non-negative, or None.")
    policy_parameters = {id(parameter) for parameter in lm.parameters()}
    if any(id(parameter) in policy_parameters for parameter in lm_ref.parameters()):
        raise ValueError("Policy and reference must not share Parameter objects.")
    optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not optimizer_parameters <= policy_parameters:
        raise ValueError("The optimizer must contain only policy parameters.")
    if any(parameter.requires_grad and id(parameter) not in optimizer_parameters for parameter in lm.parameters()):
        raise ValueError("The optimizer must include all trainable policy parameters.")

    lm_ref.requires_grad_(False)
    lm_ref.eval()
    lm.train()
    num_examples = len(examples)
    total_loss = torch.zeros((), device=next(lm.parameters()).device)
    optimizer.zero_grad(set_to_none=True)
    try:
        for example in examples:
            loss = compute_per_instance_dpo_loss(
                lm, lm_ref, tokenizer, beta,
                example.instruction, example.chosen, example.rejected,
            )
            weighted_loss = loss / num_examples
            weighted_loss.backward()
            total_loss += weighted_loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lm.parameters(), max_grad_norm if max_grad_norm is not None else math.inf,
            error_if_nonfinite=True,
        )
        optimizer.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    return total_loss, {"grad_norm": grad_norm.detach(), "num_examples": num_examples}
