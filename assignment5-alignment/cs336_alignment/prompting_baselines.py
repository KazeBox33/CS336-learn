import json
import logging
from pathlib import Path

from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.prompting import PROMPT_FILES, build_prompts, build_sampling_params
from cs336_alignment.prompting_evaluation import score_response, summarize_scores
from cs336_alignment.prompting_generation import generate_responses
from cs336_alignment.vllm_utils import VLLMServer

logger = logging.getLogger(__name__)


def evaluate_prompt(
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_name: str,
    *,
    seed: int = 0,
    batch_size: int = 32,
) -> tuple[list[dict], dict[str, int | float]]:
    """Generate and score one response per question for a single prompt style."""
    if not examples:
        raise ValueError("Evaluation requires at least one question")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompts = build_prompts(examples, prompt_name)
    completions = generate_responses(
        server, examples, prompt_name, seed=seed, batch_size=batch_size
    )
    records = []
    for index, (example, prompt, completion) in enumerate(
        zip(examples, prompts, completions, strict=True)
    ):
        scores = score_response(completion.text, example.ground_truth, prompt_name)
        if scores["answer_reward"] == 1.0:
            category = 1
        elif scores["format_reward"] == 1.0:
            category = 2
        else:
            category = 3
        records.append(
            {
                "example_index": index,
                "question": example.question,
                "ground_truth": example.ground_truth,
                "prompt": prompt,
                "response": completion.text,
                "token_ids": completion.token_ids,
                "finish_reason": completion.finish_reason,
                "scores": scores,
                "category": category,
            }
        )
    metrics = summarize_scores(record["scores"] for record in records)
    metrics["num_length_limited"] = sum( # 统计因限制长度而结束的数量
        record["finish_reason"] == "length" for record in records
    )
    return records, metrics


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Prompting Baselines",
        "",
        f"Model: `{summary['model_id']}`. Seed: {summary['seed']}. Completed: {summary['completed']}.",
        "",
        "| Prompt | Responses | Accuracy | Format Rate | Category 1 | Category 2 | Category 3 | Length Limited |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["prompts"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['num_responses']} | {metrics['accuracy']:.2%} | "
            f"{metrics['format_rate']:.2%} | {metrics['num_correct']} | "
            f"{metrics['num_format_only']} | {metrics['num_unformatted']} | {metrics['num_length_limited']} |"
        )
    lines.extend(
        [
            "",
            "## Manual Analysis (Pending)",
            "",
            "Category 1: format and correctness rewards are both 1. Category 2: format 1, correctness 0.",
            "Category 3: format and correctness rewards are both 0. These are parser-based judgments.",
            "",
            "For each prompt, inspect at least 10 category-2 and category-3 examples when available.",
            "The review JSONL includes the first examples in dataset order, not a random sample.",
            "If a category has fewer than 10 responses, inspect all and report that limitation.",
            "",
            "- (a) Record how many inspected responses are actually correct but missed by the parser.",
            "- (b) Compare task answering, reasoning, formatting, and unrelated continuation behavior.",
            "- Cite example_index values and include representative prompts and responses from the JSONL files.",
            "- The script does not infer human correctness or generate experimental conclusions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_prompting_baselines(
    server: VLLMServer,
    examples: list[GSM8KExample],
    output_dir: str | Path,
    *,
    prompt_names: list[str] | None = None,
    seed: int = 0,
    batch_size: int = 32,
    data_path: str | Path | None = None,
) -> dict:
    """Evaluate selected prompts on a running server and save reproducible audit files."""
    names = list(PROMPT_FILES) if prompt_names is None else list(prompt_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in PROMPT_FILES for name in names)
    ):
        raise ValueError("Select distinct, supported prompt names")
    if not examples or batch_size <= 0:
        raise ValueError("Evaluation requires questions and a positive batch_size")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_id": server.model_id,
        "data_path": str(Path(data_path).resolve()) if data_path is not None else None,
        "seed": seed,
        "batch_size": batch_size,
        "num_examples": len(examples),
        "requested_prompts": names,
        "completed": False,
        "prompts": {},
    }
    _write_json(output_dir / "summary.json", summary)
    for name in names:
        logger.info("Evaluating %s on %d questions", name, len(examples))
        records, metrics = evaluate_prompt( # 返回全部道题的各种情况 和 正确率等
            server, examples, name, seed=seed, batch_size=batch_size
        )
        _write_jsonl(output_dir / f"{name}.responses.jsonl", records)
        review = []
        review_counts = {}
        for category, limit in ((1, 3), (2, 10), (3, 10)):
            selected = [record for record in records if record["category"] == category][
                :limit
            ]
            review_counts[str(category)] = len(selected)
            review.extend(
                {
                    **record,
                    "human_answer_correct": None,
                    "parser_missed_correct_answer": None,
                    "behavior_notes": "",
                }
                for record in selected
            )
        _write_jsonl(output_dir / f"{name}.review.jsonl", review)
        summary["prompts"][name] = {
            "sampling_params": build_sampling_params(name, seed=seed),
            "metrics": metrics,
            "review_counts": review_counts,
            "responses_file": f"{name}.responses.jsonl",
            "review_file": f"{name}.review.jsonl",
        }
        _write_json(output_dir / "summary.json", summary)
        _write_report(output_dir / "report.md", summary)
        logger.info(
            "%s: accuracy %.2f%%, format %.2f%%",
            name,
            metrics["accuracy"] * 100,
            metrics["format_rate"] * 100,
        )
    summary["completed"] = True
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary
