import math
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, set_seed

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.gsm8k import GSM8KExample, load_gsm8k
from cs336_alignment.prompting import build_prompts, build_sampling_params
from cs336_alignment.training import grpo_train_step
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gsm8k"


@dataclass(frozen=True)
class GRPOExperimentConfig:
    """Standard on-policy experiment settings, without loading models or data."""

    model_id: str = "allenai/OLMo-2-0425-1B"
    train_path: Path = DATA_DIR / "train.jsonl"
    val_path: Path = DATA_DIR / "test.jsonl"
    prompt_name: str = "r1_zero"
    seed: int = 0
    n_train_examples: int = 6400
    n_val_examples: int = 1024
    num_rollout_steps: int = 200
    rollout_batch_size: int = 256
    group_size: int = 8
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-5
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 512
    max_grad_norm: float | None = 1.0
    eval_interval: int = 10
    rollout_log_interval: int = 40

    def __post_init__(self) -> None: # 在生成前会自动调用他
        for name in (
            "n_train_examples", "n_val_examples", "num_rollout_steps",
            "rollout_batch_size", "group_size", "gradient_accumulation_steps",
            "sampling_max_tokens", "eval_interval", "rollout_log_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.rollout_batch_size % self.group_size != 0:
            raise ValueError("rollout_batch_size must be divisible by group_size.")
        if self.gradient_accumulation_steps > self.rollout_batch_size:
            raise ValueError("gradient_accumulation_steps cannot exceed the response batch size.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature < 0:
            raise ValueError("sampling_temperature must be finite and non-negative.")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0
        ):
            raise ValueError("max_grad_norm must be finite and non-negative, or None.")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty.")
        if self.prompt_name != "r1_zero":
            raise ValueError("The standard experiment uses the zero-shot r1_zero prompt.")

    @property
    def prompts_per_rollout_batch(self) -> int: #  取出每一次rollout
        return self.rollout_batch_size // self.group_size

    @property
    def train_batch_size(self) -> int:
        return self.rollout_batch_size


@dataclass(frozen=True)
class GRPOExperimentData:
    train_examples: list[GSM8KExample]
    val_examples: list[GSM8KExample]
    train_prompts: list[str]
    val_prompts: list[str]


def load_experiment_data(config: GRPOExperimentConfig) -> GRPOExperimentData:
    """Load fixed dataset subsets and aligned prompts without starting inference."""
    if Path(config.train_path).resolve() == Path(config.val_path).resolve():
        raise ValueError("Training and validation must use different dataset files.")

    train_examples = load_gsm8k(config.train_path)
    val_examples = load_gsm8k(config.val_path)
    for name, examples, requested in (
        ("training", train_examples, config.n_train_examples),
        ("validation", val_examples, config.n_val_examples),
    ):
        if len(examples) < requested:
            raise ValueError(
                f"Requested {requested} {name} examples, but the dataset contains {len(examples)}."
            )

    # Keep subsets fixed across seeds; training batches will be sampled later.
    train_examples = train_examples[:config.n_train_examples]
    val_examples = val_examples[:config.n_val_examples]
    return GRPOExperimentData(
        train_examples=train_examples,
        val_examples=val_examples,
        train_prompts=build_prompts(train_examples, config.prompt_name),
        val_prompts=build_prompts(val_examples, config.prompt_name),
    )


def initialize_training_components(
    config: GRPOExperimentConfig,
    device: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.optim.AdamW]:
    """Load the seeded training model and create the handout's AdamW optimizer."""
    set_seed(config.seed)
    model, tokenizer = get_model_and_tokenizer(config.model_id, device)
    # Training scores complete sequences; generation caches belong to vLLM.
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    return model, tokenizer, optimizer


@contextmanager
def open_rollout_server(
    config: GRPOExperimentConfig,
    policy: torch.nn.Module,
    *,
    inference_gpu: int = 1,
) -> Iterator[VLLMServer]:
    """Own a vLLM server with initialized weight transfer and current policy weights."""
    policy_device = next(policy.parameters()).device
    if policy_device.type != "cuda":
        raise ValueError("Weight synchronization requires a CUDA training model.")
    if policy_device.index == inference_gpu:
        raise ValueError("Training and inference must use different GPUs.")

    server = VLLMServer(
        model_id=config.model_id,
        gpu=inference_gpu,
        seed=config.seed,
    )
    try:
        server.start()
        server.init_weight_sync(str(policy_device))
        server.sync_policy_weights(policy)
        yield server
    finally:
        server.stop()


def generate_training_rollouts(
    config: GRPOExperimentConfig,
    data: GRPOExperimentData,
    server: VLLMServer,
    policy: torch.nn.Module,
    *,
    rng: random.Random,
) -> tuple[list[str], list[VLLMCompletion], list[str]]:
    """Sample questions and return grouped prompts, completions, and reference answers."""
    if not data.train_examples or len(data.train_examples) != len(data.train_prompts):
        raise ValueError("Training examples and prompts must be non-empty and aligned.")

    indices = rng.choices(
        range(len(data.train_examples)), k=config.prompts_per_rollout_batch
    )
    prompts = [data.train_prompts[index] for index in indices]
    ground_truths = [data.train_examples[index].ground_truth for index in indices]
    sampling_params = build_sampling_params( # 生成配置
        config.prompt_name, seed=rng.randrange(2**31)
    )
    sampling_params.update(
        temperature=config.sampling_temperature,
        max_tokens=config.sampling_max_tokens,
        n=config.group_size,
    )

    # Every generation batch must use the current training weights.
    server.sync_policy_weights(policy)
    completions = server.generate_completions(
        prompts=prompts, sampling_params=sampling_params
    )
    if len(completions) != config.rollout_batch_size:
        raise RuntimeError(
            f"Expected {config.rollout_batch_size} responses, received {len(completions)}."
        )

    repeated_prompts = [prompt for prompt in prompts for _ in range(config.group_size)] # 每个prompt和答案重复group_size次
    repeated_ground_truths = [truth for truth in ground_truths for _ in range(config.group_size)]
    return repeated_prompts, completions, repeated_ground_truths


def run_grpo_iteration(
    config: GRPOExperimentConfig,
    data: GRPOExperimentData,
    server: VLLMServer,
    policy: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    rng: random.Random,
) -> tuple[dict[str, torch.Tensor | float], list[str], list[VLLMCompletion]]:
    """Generate a fresh on-policy batch, update once, and retain samples for logging."""
    repeated_prompts, completions, repeated_ground_truths = generate_training_rollouts(
        config, data, server, policy, rng=rng
    )
    rollout_responses = [completion.text for completion in completions]
    loss, metadata = grpo_train_step(
        model=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        reward_fn=reward_fn,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=config.group_size,
    )
    return {"loss": loss, **metadata}, repeated_prompts, completions
