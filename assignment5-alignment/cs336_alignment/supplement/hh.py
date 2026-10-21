import gzip
import json
from dataclasses import dataclass
from pathlib import Path


HH_FILENAMES = (
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
)
HUMAN_MARKER = "\n\nHuman:"
ASSISTANT_MARKER = "\n\nAssistant:"


@dataclass(frozen=True)
class HHPreferenceExample:
    instruction: str
    chosen: str
    rejected: str
    source: str


def _split_hh_conversation(conversation: str) -> tuple[str, str]:
    if not conversation.startswith(HUMAN_MARKER):
        raise ValueError("HH conversations must start with a Human role marker.")
    instruction, separator, response = conversation[len(HUMAN_MARKER):].partition(ASSISTANT_MARKER)
    if not separator:
        raise ValueError("HH conversations must contain an Assistant role marker.")
    return instruction.strip(), response.strip()


def load_hh_preferences(data_dir: str | Path) -> list[HHPreferenceExample]:
    """Combine four HH training files, dropping pairs with multiple human turns.

    Strip boundary whitespace, retain response text after the first Assistant
    marker, and keep empty responses. Malformed single-turn pairs raise errors.
    """
    examples = []
    for filename in HH_FILENAMES:
        path = Path(data_dir) / filename
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    pair = json.loads(line)
                    if not isinstance(pair, dict) or not all(
                        isinstance(pair.get(key), str) for key in ("chosen", "rejected")
                    ):
                        raise ValueError("chosen and rejected must be strings.")
                    chosen, rejected = pair["chosen"], pair["rejected"]
                    if chosen.count(HUMAN_MARKER) > 1 or rejected.count(HUMAN_MARKER) > 1: # 多轮回答就pass
                        continue
                    instruction, chosen_response = _split_hh_conversation(chosen)
                    rejected_instruction, rejected_response = _split_hh_conversation(rejected)
                    if instruction != rejected_instruction:
                        raise ValueError("chosen and rejected must share the same instruction.")
                except ValueError as error:
                    raise ValueError(f"{path}: line {line_number}: {error}") from error
                examples.append(HHPreferenceExample(instruction, chosen_response, rejected_response, filename))
    return examples
