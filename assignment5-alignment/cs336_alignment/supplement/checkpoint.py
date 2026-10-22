import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DPOBestCheckpoint:
    accuracy: float
    update: int
    path: Path


def save_best_dpo_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    accuracy: float,
    update: int,
    output_dir: str | Path,
    best: DPOBestCheckpoint | None = None,
) -> DPOBestCheckpoint:
    """Export only on strict accuracy improvement, retaining earlier tied models.

    The caller keeps the returned record between validations. Existing exports
    remain untouched; this selects inference weights, not resumable train state.
    """
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Validation accuracy must be finite and in [0, 1].")
    if not isinstance(update, int) or update < 0:
        raise ValueError("update must be a non-negative integer.")
    if best is not None and accuracy <= best.accuracy: # 核心判断
        return best
    path = Path(output_dir) / f"step_{update:06d}"
    # HF serialization is identical for SFT and DPO; reuse the safe exporter.
    saved_path = save_sft_model(model, tokenizer, path)
    return DPOBestCheckpoint(accuracy, update, saved_path)
