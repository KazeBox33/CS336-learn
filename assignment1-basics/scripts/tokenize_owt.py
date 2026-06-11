from __future__ import annotations

import argparse
import pickle
from array import array
from pathlib import Path
from time import perf_counter

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_pickle(path: str | Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize OpenWebText text into a numpy token array.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("../outputs/owt_bpe/vocab.pkl"))
    parser.add_argument("--merges-path", type=Path, default=Path("../outputs/owt_bpe/merges.pkl"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-size", type=int, default=1_000_000)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = perf_counter()

    vocab = load_pickle(args.vocab_path)
    merges = load_pickle(args.merges_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"], cache_size=args.cache_size)

    token_ids = array("H")
    with args.input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            token_ids.extend(tokenizer.encode(line))
            if line_number % args.progress_interval == 0:
                elapsed_seconds = perf_counter() - start
                print(
                    f"lines {line_number:,}: {len(token_ids):,} tokens "
                    f"in {elapsed_seconds:.2f} seconds",
                    flush=True,
                )

    tokens = np.frombuffer(token_ids, dtype=np.uint16)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, tokens)

    elapsed_seconds = perf_counter() - start
    print(
        f"saved {len(tokens)} tokens to {args.output_path} "
        f"in {elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
