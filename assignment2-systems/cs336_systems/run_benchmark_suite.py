import argparse
import subprocess
import sys
from pathlib import Path

from cs336_systems.benchmark import MODEL_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark experiment matrix.")

    parser.add_argument(
        "--model-sizes",
        nargs="+",
        choices=MODEL_CONFIGS,
        default=["small"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forward", "forward_backward", "full"),
        default=["forward", "forward_backward", "full"],
    )
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[5])
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("float32", "bfloat16"),
        default=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("eager", "compiled"),
        default=["eager", "compiled"],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/end_to_end"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for model_size in args.model_sizes:
        for mode in args.modes:
            for warmup_steps in args.warmup_steps:
                for precision in args.precisions:
                    for implementation in args.implementations:
                        precision_tag = "bf16" if precision == "bfloat16" else "fp32"

                        output_path = args.output_dir / (
                            f"{model_size}_{mode}_{precision_tag}_"
                            f"{implementation}_b{args.batch_size}_"
                            f"l{args.context_length}_w{warmup_steps}.json"
                        )
                        command = [
                            sys.executable,  # 返回当前Python解释器的绝对路径
                            "-m",  # 按照python模块运行
                            "cs336_systems.benchmark",
                            "--model-size",
                            model_size,
                            "--batch-size",
                            str(args.batch_size),
                            "--context-length",
                            str(args.context_length),
                            "--mode",
                            mode,
                            "--warmup-steps",
                            str(warmup_steps),
                            "--measurement-steps",
                            str(args.measurement_steps),
                            "--device",
                            args.device,
                            "--output-path",
                            str(output_path),
                        ]
                        if precision == "bfloat16":
                            command.append("--mixed-precision")
                        if implementation == "compiled":
                            command.append("--compile-model")
                        print("Running:", " ".join(command), flush=True)
                        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
