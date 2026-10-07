from collections.abc import Iterable

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.prompting import PROMPT_FILES


def score_response(
    response: str,
    ground_truth: str,
    prompt_name: str,
) -> dict[str, float]:
    """Grade one generated response using the assignment's existing parser."""
    if prompt_name not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt name: {prompt_name!r}")
    if prompt_name == "question_only":
        reward_fn = question_only_reward_fn
    else:
        reward_fn = r1_zero_reward_fn
    return reward_fn(response, ground_truth)


def summarize_scores(scores: Iterable[dict[str, float]]) -> dict[str, int | float]:
    """Summarize binary scores returned by the assignment's reward functions."""
    num_correct = 0
    num_format_only = 0
    num_unformatted = 0
    for score in scores:
        if score["answer_reward"] == 1.0:
            num_correct += 1
        elif score["format_reward"] == 1.0:
            num_format_only += 1
        else:
            num_unformatted += 1

    num_responses = num_correct + num_format_only + num_unformatted
    if num_responses == 0:
        raise ValueError("Cannot summarize an empty set of scores")

    return {
        "num_responses": num_responses,
        "num_correct": num_correct,
        "num_format_only": num_format_only,
        "num_unformatted": num_unformatted,
        "accuracy": num_correct / num_responses,
        "format_rate": (num_correct + num_format_only) / num_responses,
    }
