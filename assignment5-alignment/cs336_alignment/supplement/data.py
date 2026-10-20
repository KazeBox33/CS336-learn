import json
from pathlib import Path

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
