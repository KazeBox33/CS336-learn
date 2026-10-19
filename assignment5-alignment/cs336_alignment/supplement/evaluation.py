import re

_MMLU_ANSWER_PATTERN = re.compile(
    r"\bthe\s+correct\s+answer\s+is\s+([A-D])(?=\s*(?:[.!?`]|$))",
    re.IGNORECASE,
)


def parse_mmlu_response(model_output: str) -> str | None:
    """Extract the first template-formatted answer, or None if none is present."""
    match = _MMLU_ANSWER_PATTERN.search(model_output)
    return match.group(1).upper() if match is not None else None
