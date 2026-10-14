"""Plot metrics across completed GRPO runs and generate a report scaffold."""

import argparse
from pathlib import Path

from cs336_alignment.grpo_reporting import write_grpo_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_grpo_report(args.run_dirs, args.output_dir))


if __name__ == "__main__":
    main()
