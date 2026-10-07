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
