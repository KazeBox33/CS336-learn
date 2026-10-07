from collections.abc import Iterable
from pathlib import Path

from cs336_alignment.gsm8k import GSM8KExample


PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}


def build_prompts(examples: Iterable[GSM8KExample], prompt_name: str) -> list[str]:
    """Fill a bundled prompting-baseline template with each question."""
    if prompt_name not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt name: {prompt_name!r}")
    template_path = Path(__file__).resolve().parent / "prompts" / PROMPT_FILES[prompt_name]
    template = template_path.read_text(encoding="utf-8")
    if prompt_name == "question_only":
        # The bundled text file contains an escaped LaTeX backslash.
        template = template.replace(r"\\boxed", r"\boxed")
    return [template.format(question=example.question) for example in examples]


def build_sampling_params(prompt_name: str, *, seed: int = 0) -> dict:
    """Return the assignment's generation settings for one prompt style."""
    if prompt_name not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt name: {prompt_name!r}")
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": seed,
    }
    if prompt_name in ("r1_zero", "r1_zero_three_shot"):
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True
    return sampling_params
