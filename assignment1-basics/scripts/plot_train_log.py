from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/valid loss and learning rate from a JSONL train log.")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--title", default="Training curves")
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Plot all records. By default, only the last run is plotted when step numbers reset.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_last_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records

    start = 0
    previous_step = records[0]["step"]
    for index, record in enumerate(records[1:], start=1):
        step = record["step"]
        if step <= previous_step:
            start = index
        previous_step = step
    return records[start:]


def scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value == min_value:
        return (start + end) / 2
    ratio = (value - min_value) / (max_value - min_value)
    return start + ratio * (end - start)


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{width}" />'


def draw_panel(
    *,
    records: list[dict[str, Any]],
    x_key: str,
    y_keys: list[tuple[str, str]],
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    left = x + 70
    right = x + width - 30
    top = y + 40
    bottom = y + height - 55

    steps = [float(record[x_key]) for record in records]
    y_values = [float(record[key]) for record in records for key, _ in y_keys if key in record]

    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(y_values), max(y_values)
    y_padding = (y_max - y_min) * 0.08 if y_max != y_min else 1.0
    y_min -= y_padding
    y_max += y_padding

    elements = [
        f'<text x="{x + width / 2:.2f}" y="{y + 24:.2f}" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />',
        f'<text x="{(left + right) / 2:.2f}" y="{y + height - 15:.2f}" text-anchor="middle" class="label">step</text>',
        f'<text x="{left - 12:.2f}" y="{top + 4:.2f}" text-anchor="end" class="tick">{y_max:.4g}</text>',
        f'<text x="{left - 12:.2f}" y="{bottom + 4:.2f}" text-anchor="end" class="tick">{y_min:.4g}</text>',
        f'<text x="{left:.2f}" y="{bottom + 20:.2f}" text-anchor="middle" class="tick">{x_min:.0f}</text>',
        f'<text x="{right:.2f}" y="{bottom + 20:.2f}" text-anchor="middle" class="tick">{x_max:.0f}</text>',
    ]

    for key, color in y_keys:
        points = [
            (
                scale(float(record[x_key]), x_min, x_max, left, right),
                scale(float(record[key]), y_min, y_max, bottom, top),
            )
            for record in records
            if key in record
        ]
        elements.append(polyline(points, color))

    legend_x = right - 150
    legend_y = top + 12
    for offset, (key, color) in enumerate(y_keys):
        current_y = legend_y + offset * 20
        elements.append(f'<line x1="{legend_x}" y1="{current_y}" x2="{legend_x + 22}" y2="{current_y}" stroke="{color}" stroke-width="3" />')
        elements.append(f'<text x="{legend_x + 30}" y="{current_y + 4}" class="legend">{html.escape(key)}</text>')

    return "\n".join(elements)


def write_svg(records: list[dict[str, Any]], output_path: Path, title: str) -> None:
    width = 1000
    height = 700
    escaped_title = html.escape(title)

    loss_panel = draw_panel(
        records=records,
        x_key="step",
        y_keys=[("train_loss", "#2563eb"), ("valid_loss", "#dc2626")],
        title="Loss",
        x=30,
        y=70,
        width=940,
        height=290,
    )
    lr_panel = draw_panel(
        records=records,
        x_key="step",
        y_keys=[("lr", "#16a34a")],
        title="Learning rate",
        x=30,
        y=380,
        width=940,
        height=290,
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .headline {{ font-size: 24px; font-weight: 700; }}
  .title {{ font-size: 18px; font-weight: 700; }}
  .label, .legend {{ font-size: 13px; }}
  .tick {{ font-size: 12px; fill: #4b5563; }}
  .axis {{ stroke: #374151; stroke-width: 1.2; }}
  .panel {{ fill: #ffffff; stroke: #e5e7eb; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#f9fafb" />
<text x="500" y="38" text-anchor="middle" class="headline">{escaped_title}</text>
<rect x="30" y="70" width="940" height="290" rx="6" class="panel" />
<rect x="30" y="380" width="940" height="290" rx="6" class="panel" />
{loss_panel}
{lr_panel}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(args.log_path)
    if not args.include_all_runs:
        records = select_last_run(records)

    if not records:
        raise ValueError(f"No records found in {args.log_path}")

    write_svg(records, args.output_path, args.title)
    print(f"wrote {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
