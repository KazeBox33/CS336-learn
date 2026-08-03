import json
from pathlib import Path

import pandas as pd

from cs336_systems.benchmark import MODEL_CONFIGS


def load_results(input_dir: Path) -> pd.DataFrame:
    records = []

    for result_path in sorted(input_dir.glob("*.json")):  # 找到所有以.json结尾的文件 转化为对应的Path对象
        result = json.loads(result_path.read_text(encoding="utf-8"))
        records.append(result)

    return pd.DataFrame(records)


def build_phase_table(
    results: pd.DataFrame,
    batch_size: int,
    context_length: int,
    warmup_steps: int,
) -> pd.DataFrame:
    filtered = results[(results["batch_size"] == batch_size) & (results["context_length"] == context_length) & (results["warmup_steps"] == warmup_steps)]

    cumulative = filtered.pivot(  # 转换成新的表格
        index="model_size",
        columns="mode",
        values="mean_ms",
    )

    phases = pd.DataFrame(index=cumulative.index)  # 建立一张新表，并沿用模型名称作为行索引
    phases["forward"] = cumulative["forward"]
    phases["backward"] = cumulative["forward_backward"] - cumulative["forward"]
    phases["optimizer"] = cumulative["full"] - cumulative["forward_backward"]

    model_order = [model_size for model_size in MODEL_CONFIGS if model_size in phases.index]

    return phases.loc[model_order]
