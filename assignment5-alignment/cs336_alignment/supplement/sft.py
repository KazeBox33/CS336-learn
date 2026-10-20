import torch
from torch.nn import functional as F


def compute_sft_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean next-token loss over a packed batch whose labels are already shifted."""
    input_ids, labels = batch["input_ids"], batch["labels"]
    if input_ids.ndim != 2 or input_ids.shape != labels.shape or input_ids.numel() == 0:
        raise ValueError("input_ids and labels must share a non-empty (batch_size, sequence_length) shape.")

    device = next(model.parameters()).device
    # Do not pass labels to the model: causal-LM loss helpers may shift them again.
    logits = model(input_ids=input_ids.to(device)).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), # 前面的维度合并
        labels.to(device).reshape(-1),
    )
