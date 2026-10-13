import math
from dataclasses import dataclass
from pathlib import Path


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
