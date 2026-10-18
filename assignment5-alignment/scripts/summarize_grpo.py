"""Plot metrics across completed GRPO runs and generate a report scaffold."""

import argparse
from pathlib import Path

from cs336_alignment.grpo_reporting import X_AXES, write_grpo_comparison, write_grpo_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-methods", action="store_true")
    parser.add_argument("--x-axis", choices=list(X_AXES), default="rollout_batches")
    args = parser.parse_args(argv)
    writer = write_grpo_comparison if args.compare_methods else write_grpo_report
    print(writer(args.run_dirs, args.output_dir, x_axis=args.x_axis))


if __name__ == "__main__":
    main()
