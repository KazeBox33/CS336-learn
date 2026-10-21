from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def save_sft_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
) -> Path:
    """Export HF model and tokenizer for evaluation/DPO, not training resumption."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"SFT model output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after both model and tokenizer have been saved successfully.
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        staging.rename(output_dir)
    return output_dir
