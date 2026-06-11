from __future__ import annotations

import argparse
import pickle
import shutil
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def load_pickle(path: str | Path):
    with open(path, "rb") as file:
        return pickle.load(file)


@dataclass(frozen=True)
class ChunkResult:
    chunk_index: int
    line_count: int
    token_count: int
    elapsed_seconds: float
    temp_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize OpenWebText text into a numpy token array.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("../outputs/owt_bpe/vocab.pkl"))
    parser.add_argument("--merges-path", type=Path, default=Path("../outputs/owt_bpe/merges.pkl"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-size", type=int, default=1_000_000)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--flush-tokens", type=int, default=5_000_000)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    return parser.parse_args()


def find_line_boundaries(path: Path, num_chunks: int) -> list[int]:
    file_size = path.stat().st_size
    if num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    boundaries = [0]
    chunk_size = file_size // num_chunks
    with path.open("rb") as file:
        for chunk_index in range(1, num_chunks):
            file.seek(chunk_index * chunk_size)
            file.readline()
            boundaries.append(file.tell())

    boundaries.append(file_size)
    return sorted(set(boundaries))


def flush_buffer(buffer: array, file) -> int:
    if not buffer:
        return 0
    count = len(buffer)
    buffer.tofile(file)
    del buffer[:]
    return count


def tokenize_range(
    *,
    chunk_index: int,
    input_path: Path,
    start_byte: int,
    end_byte: int,
    vocab_path: Path,
    merges_path: Path,
    temp_path: Path,
    cache_size: int,
    flush_tokens: int,
    progress_interval: int,
) -> ChunkResult:
    start = perf_counter()

    vocab = load_pickle(vocab_path)
    merges = load_pickle(merges_path)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"], cache_size=cache_size)

    line_count = 0
    token_count = 0
    buffer = array("H")
    with input_path.open("rb") as input_file, temp_path.open("wb") as output_file:
        input_file.seek(start_byte)
        while input_file.tell() < end_byte:
            line = input_file.readline()
            if not line:
                break
            line_count += 1
            buffer.extend(tokenizer.encode(line.decode("utf-8", errors="replace")))
            if len(buffer) >= flush_tokens:
                token_count += flush_buffer(buffer, output_file)
            if progress_interval > 0 and line_count % progress_interval == 0:
                elapsed_seconds = perf_counter() - start
                print(
                    f"chunk {chunk_index:02d}: lines {line_count:,}, "
                    f"tokens {token_count + len(buffer):,}, {elapsed_seconds:.2f}s",
                    flush=True,
                )

        token_count += flush_buffer(buffer, output_file)

    elapsed_seconds = perf_counter() - start
    return ChunkResult(
        chunk_index=chunk_index,
        line_count=line_count,
        token_count=token_count,
        elapsed_seconds=elapsed_seconds,
        temp_path=str(temp_path),
    )


def copy_part_to_memmap(part_path: Path, output: np.memmap, start: int, chunk_tokens: int = 20_000_000) -> int:
    offset = start
    with part_path.open("rb") as file:
        while True:
            tokens = np.fromfile(file, dtype=np.uint16, count=chunk_tokens)
            if len(tokens) == 0:
                break
            output[offset : offset + len(tokens)] = tokens
            offset += len(tokens)
    return offset


def tokenize_single_process(args: argparse.Namespace) -> int:
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
    return len(tokens)


def tokenize_parallel(args: argparse.Namespace) -> int:
    start = perf_counter()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_path.with_suffix(args.output_path.suffix + ".parts")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    boundaries = find_line_boundaries(args.input_path, args.num_processes)
    chunk_specs = [
        (
            chunk_index,
            start_byte,
            end_byte,
            temp_dir / f"part_{chunk_index:04d}.bin",
        )
        for chunk_index, (start_byte, end_byte) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        if end_byte > start_byte
    ]

    results = []
    with ProcessPoolExecutor(max_workers=min(args.num_processes, len(chunk_specs))) as executor:
        futures = [
            executor.submit(
                tokenize_range,
                chunk_index=chunk_index,
                input_path=args.input_path,
                start_byte=start_byte,
                end_byte=end_byte,
                vocab_path=args.vocab_path,
                merges_path=args.merges_path,
                temp_path=temp_path,
                cache_size=args.cache_size,
                flush_tokens=args.flush_tokens,
                progress_interval=args.progress_interval,
            )
            for chunk_index, start_byte, end_byte, temp_path in chunk_specs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"finished chunk {result.chunk_index:02d}: "
                f"{result.line_count:,} lines, {result.token_count:,} tokens, "
                f"{result.elapsed_seconds:.2f}s",
                flush=True,
            )

    results.sort(key=lambda result: result.chunk_index)
    total_tokens = sum(result.token_count for result in results)
    output = np.lib.format.open_memmap(args.output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))
    offset = 0
    for result in results:
        offset = copy_part_to_memmap(Path(result.temp_path), output, offset)
    output.flush()
    shutil.rmtree(temp_dir)

    elapsed_seconds = perf_counter() - start
    total_lines = sum(result.line_count for result in results)
    print(
        f"merged {total_tokens:,} tokens from {total_lines:,} lines "
        f"in {elapsed_seconds:.2f} seconds",
        flush=True,
    )
    return total_tokens


def main() -> None:
    args = parse_args()
    start = perf_counter()

    if args.num_processes <= 1:
        token_count = tokenize_single_process(args)
    else:
        token_count = tokenize_parallel(args)

    elapsed_seconds = perf_counter() - start
    print(
        f"saved {token_count} tokens to {args.output_path} "
        f"in {elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
