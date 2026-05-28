from __future__ import annotations

import argparse
import cProfile
import json
import pickle
import pstats
import resource
from pathlib import Path
from time import perf_counter

from cs336_basics.bpe import train_bpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument("--input-path", type=Path, default=Path("TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinystories_bpe"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--profile-top-n", type=int, default=30)
    return parser.parse_args()


def dump_merges_text(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for left, right in merges:
            file.write(f"{left!r}\t{right!r}\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()
    vocab, merges = train_bpe(
        args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        num_processes=args.num_processes,
    )
    profiler.disable()
    elapsed_seconds = perf_counter() - start

    longest_token = max(vocab.values(), key=len)
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    vocab_path = args.output_dir / "vocab.pkl"
    merges_path = args.output_dir / "merges.pkl"
    merges_text_path = args.output_dir / "merges.txt"
    metrics_path = args.output_dir / "metrics.json"
    profile_path = args.output_dir / "profile.prof"
    profile_text_path = args.output_dir / "profile.txt"

    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file)
    with merges_path.open("wb") as file:
        pickle.dump(merges, file)
    dump_merges_text(merges_text_path, merges)
    profiler.dump_stats(profile_path)

    with profile_text_path.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file).sort_stats("cumulative")
        stats.print_stats(args.profile_top_n)

    metrics = {
        "input_path": str(args.input_path),
        "vocab_size": len(vocab),
        "requested_vocab_size": args.vocab_size,
        "num_merges": len(merges),
        "special_tokens": args.special_token,
        "num_processes": args.num_processes,
        "elapsed_seconds": elapsed_seconds,
        "max_rss_kb": max_rss_kb,
        "longest_token_len": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "merges_text_path": str(merges_text_path),
        "profile_path": str(profile_path),
        "profile_text_path": str(profile_text_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nProfile summary written to {profile_text_path}")


if __name__ == "__main__":
    main()
