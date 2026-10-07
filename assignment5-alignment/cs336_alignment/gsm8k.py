import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    ground_truth: str


def load_gsm8k(path: str | Path) -> list[GSM8KExample]:
    """Read GSM8K JSONL records and extract their final reference answers."""
    examples = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip(): # 空白就跳过
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{line_number}: question must be a non-empty string")
            if not isinstance(answer, str):
                raise ValueError(f"{path}:{line_number}: answer must be a string")
            _, separator, ground_truth = answer.rpartition("####")
            ground_truth = ground_truth.strip()
            if not separator or not ground_truth:
                raise ValueError(f"{path}:{line_number}: missing final answer after ####")
            examples.append(GSM8KExample(question=question.strip(), ground_truth=ground_truth))
    return examples
