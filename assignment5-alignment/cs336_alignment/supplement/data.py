import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def load_sft_documents(dataset_path: str | Path) -> list[str]:
    """Read JSONL prompt-response pairs and format complete SFT documents."""
    template_path = Path(__file__).resolve().parents[1] / "prompts_safety" / "alpaca_sft.prompt"
    template = template_path.read_text(encoding="utf-8").rstrip("\n")
    documents = []
    path = Path(dataset_path)
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            example = json.loads(line)
            if not isinstance(example, dict) or not all(
                isinstance(example.get(key), str) for key in ("prompt", "response")
            ):
                raise ValueError(f"{path}: line {line_number}: prompt and response must be strings.")
            documents.append(template.format(instruction=example["prompt"], response=example["response"])) # 填充模版
    return documents


def tokenize_sft_documents(documents: list[str], tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Encode SFT documents with default special tokens, appending EOS after each."""
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int):
        raise TypeError("The SFT tokenizer must define an integer eos_token_id.")
    token_ids = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document, add_special_tokens=True))
        token_ids.append(eos_token_id)
    return token_ids


class PackedSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose consecutive fixed-length input/next-token pairs from SFT documents."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ):
        if seq_length <= 0:
            raise ValueError("seq_length must be positive.")
        documents = load_sft_documents(dataset_path)
        if shuffle:
            random.shuffle(documents)
        self._token_ids = torch.tensor(tokenize_sft_documents(documents, tokenizer), dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self) -> int:
        # Reserve one extra token for the final input position's label.
        return max(0, (self._token_ids.numel() - 1) // self.seq_length) # - 1 是为了对其

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError("SFT dataset index out of range.")
        start = i * self.seq_length
        end = start + self.seq_length
        return {
            "input_ids": self._token_ids[start:end],
            "labels": self._token_ids[start + 1 : end + 1],
        }


def iterate_batches(
    dataset: Dataset[dict[str, torch.Tensor]], batch_size: int, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    """Batch every dataset item once per epoch, retaining an incomplete final batch."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
