from collections.abc import Iterable
from pathlib import Path

from cs336_alignment.supplement.mmlu import MMLUExample


def build_mmlu_prompts(examples: Iterable[MMLUExample]) -> list[str]:
    """Wrap each task prompt in the supplement's zero-shot system template."""
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts_safety"
    task_template = (prompt_dir / "mmlu_zero_shot.prompt").read_text(encoding="utf-8")
    system_template = (prompt_dir / "zero_shot_system_prompt.prompt").read_text(encoding="utf-8")
    prompts = []
    for example in examples:
        instruction = task_template.format(
            subject=example.subject.replace("_", " "), question=example.question, options=example.options,
        )
        prompts.append(system_template.format(instruction=instruction))
    return prompts
