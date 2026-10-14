import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def save_grpo_checkpoint(model, tokenizer, optimizer, step: int, output_dir: str | Path, *, rng=None) -> Path:
    """Publish a complete HF model plus training state without replacing an existing checkpoint."""
    if step < 0:
        raise ValueError("Checkpoint step must be non-negative.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so failed writes never look like usable checkpoints.
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        torch.save({
            "step": step,
            "optimizer": optimizer.state_dict(),
            "rollout_rng": rng.getstate() if rng is not None else None,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, staging / "training_state.pt")
        staging.rename(output_dir)
    return output_dir
