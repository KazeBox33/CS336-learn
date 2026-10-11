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
