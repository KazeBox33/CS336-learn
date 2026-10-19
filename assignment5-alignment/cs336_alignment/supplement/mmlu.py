import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class MMLUExample:
    subject: str
    question: str
    options: tuple[str, str, str, str]
    answer: str


def load_mmlu(data_dir: str | Path, *, split: Literal["dev", "val", "test"] = "test") -> list[MMLUExample]:
    """Read one bundled MMLU split, ordered by subject filename and CSV row."""
    if split not in ("dev", "val", "test"):
        raise ValueError("MMLU split must be dev, val, or test.")
    split_dir = Path(data_dir) / split
    files = sorted(split_dir.glob(f"*_{split}.csv"))
    if not files:
        raise FileNotFoundError(f"No MMLU {split} CSV files found in {split_dir}.")

    examples = []
    for path in files:
        subject = path.stem.removesuffix(f"_{split}")
        with path.open(encoding="utf-8", newline="") as file:
            for row_number, row in enumerate(csv.reader(file), start=1):
                if not row:
                    continue
                if len(row) != 6:
                    raise ValueError(f"{path}: row {row_number}: expected six CSV columns.")
                question, a, b, c, d, answer = (cell.strip() for cell in row)
                if not all((question, a, b, c, d)) or answer not in ("A", "B", "C", "D"):
                    raise ValueError(f"{path}: row {row_number}: expected non-empty text and an A-D answer.")
                examples.append(MMLUExample(subject, question, (a, b, c, d), answer))
    if not examples:
        raise ValueError(f"No MMLU examples found in {split_dir}.")
    return examples
