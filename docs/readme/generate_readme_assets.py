from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "readme_metrics.json"

COLORS = {
    "navy": "#17324D",
    "blue": "#2878B5",
    "green": "#3A8D5D",
    "orange": "#E08A33",
    "gray": "#B8C0C8",
    "light_gray": "#E8EDF2",
    "text": "#17212B",
}


def load_metrics() -> dict:
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / name, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tokenizer_throughput(metrics: dict) -> None:
    rows = metrics["tokenizer_throughput"]
    names = [row["name"] for row in rows]
    throughput = [row["tokens_per_second_millions"] for row in rows]
    speedups = [row["speedup"] for row in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    bars = ax.barh(
        names,
        throughput,
        color=[COLORS["gray"], COLORS["blue"], COLORS["green"]],
        height=0.58,
    )
    ax.invert_yaxis()
    ax.set_title("OpenWebText Tokenization Throughput", loc="left", fontweight="bold", color=COLORS["text"])
    ax.set_xlabel("Million tokens / second")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value, speedup in zip(bars, throughput, speedups, strict=True):
        ax.text(
            value + 0.16,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}M tok/s  |  {speedup:.2f}x",
            va="center",
            color=COLORS["text"],
            fontweight="bold",
        )

    ax.set_xlim(0, max(throughput) * 1.35)
    fig.tight_layout()
    save_figure(fig, "tokenizer-throughput.svg")


def plot_owt_loss(metrics: dict) -> None:
    rows = metrics["owt_validation_loss"]
    names = [row["name"] for row in rows]
    losses = [row["validation_loss"] for row in rows]
    tokens = [row["tokens_seen_millions"] for row in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    bars = ax.bar(
        names,
        losses,
        color=[COLORS["orange"], COLORS["blue"], COLORS["green"]],
        width=0.62,
    )
    ax.set_title("OpenWebText Validation Loss", loc="left", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Validation loss (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, loss, token_count in zip(bars, losses, tokens, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            loss + 0.08,
            f"{loss:.4f}\n{token_count:.2f}M tokens",
            ha="center",
            va="bottom",
            color=COLORS["text"],
            fontweight="bold",
        )

    ax.set_ylim(0, max(losses) * 1.22)
    fig.tight_layout()
    save_figure(fig, "owt-validation-loss.svg")


def plot_roadmap(metrics: dict) -> None:
    rows = metrics["roadmap"]
    status_colors = {
        "complete": COLORS["green"],
        "in_progress": COLORS["orange"],
        "planned": COLORS["gray"],
    }

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(-0.45, 1.05)
    ax.axis("off")
    ax.plot(range(len(rows)), [0.52] * len(rows), color=COLORS["light_gray"], linewidth=5, zorder=0)

    for index, row in enumerate(rows):
        color = status_colors[row["status"]]
        circle = plt.Circle((index, 0.52), 0.105, facecolor=color, edgecolor="white", linewidth=2.5, zorder=2)
        ax.add_patch(circle)

        label = row["name"].replace(" ", "\n", 1)
        ax.text(index, 0.2, label, ha="center", va="top", color=COLORS["text"], fontsize=9)

        status_label = row["status"].replace("_", " ").upper()
        badge = FancyBboxPatch(
            (index - 0.31, 0.72),
            0.62,
            0.17,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(badge)
        ax.text(index, 0.805, status_label, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

    ax.set_title("CS336 Learning Roadmap", loc="left", fontweight="bold", color=COLORS["text"], pad=8)
    fig.tight_layout()
    save_figure(fig, "learning-roadmap.svg")


def main() -> None:
    metrics = load_metrics()
    plot_roadmap(metrics)
    plot_tokenizer_throughput(metrics)
    plot_owt_loss(metrics)


if __name__ == "__main__":
    main()
