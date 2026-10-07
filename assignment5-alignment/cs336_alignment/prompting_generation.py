from collections.abc import Iterable

from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.prompting import build_prompts, build_sampling_params
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


def generate_responses(
    server: VLLMServer,
    examples: Iterable[GSM8KExample],
    prompt_name: str,
    *,
    seed: int = 0,
    batch_size: int | None = 32,
) -> list[VLLMCompletion]:
    """Generate one answer per question using an already running vLLM server."""
    prompts = build_prompts(examples, prompt_name)
    sampling_params = build_sampling_params(prompt_name, seed=seed)
    if not prompts:
        return []
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} responses, received {len(completions)}"
        )
    return completions
