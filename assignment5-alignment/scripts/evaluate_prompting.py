"""Run the GSM8K prompting-baselines experiment without updating model weights."""

import argparse
import json
import logging
from pathlib import Path

from cs336_alignment.gsm8k import load_gsm8k
from cs336_alignment.prompting import PROMPT_FILES

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        required=True,
        help="OLMo-2-0425-1B model directory or Hugging Face ID",
    )
    parser.add_argument(
        "--data-path", type=Path, default=ASSIGNMENT_DIR / "data/gsm8k/test.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new or empty result directory"
    )
    parser.add_argument(
        "--prompts", nargs="+", choices=list(PROMPT_FILES), default=list(PROMPT_FILES)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N questions for a smoke test"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Use an existing server without launching or stopping it",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be positive")
    if (
        args.gpu < 0
        or not 1 <= args.port <= 65535
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error("Invalid GPU index, port, or GPU memory utilization")
    if len(set(args.prompts)) != len(args.prompts):
        parser.error("--prompts must not contain duplicates")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("--output-dir must be a new or empty directory")
    examples = load_gsm8k(args.data_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        parser.error("The dataset contains no questions")

    # Keep --help and input validation usable without importing the GPU helper.
    from cs336_alignment.prompting_baselines import run_prompting_baselines
    from cs336_alignment.vllm_utils import VLLMServer

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        host=args.host,
        port=args.port,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        launch_server=not args.connect_only,
    )
    try:
        server.start()
        summary = run_prompting_baselines(
            server,
            examples,
            args.output_dir,
            prompt_names=args.prompts,
            seed=args.seed,
            batch_size=args.batch_size,
            data_path=args.data_path,
        )
    finally:
        if not args.connect_only:
            server.stop()
    print(
        json.dumps(
            {name: result["metrics"] for name, result in summary["prompts"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
