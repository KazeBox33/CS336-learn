import json
from pathlib import Path


def load_sft_documents(dataset_path: str | Path) -> list[str]:
    """Read JSONL prompt-response pairs and format complete SFT documents."""
    template_path = Path(__file__).resolve().parents[1] / "prompts_safety" / "alpaca_sft.prompt"
    template = template_path.read_text(encoding="utf-8")
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
