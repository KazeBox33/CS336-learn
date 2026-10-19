# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

A supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF is included at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## 当前进度

| 部分 | 状态 | 说明 |
| --- | --- | --- |
| 主线 prompting baseline | 代码已实现 | GSM8K 提示、生成、评分与日志；真实 GPU 测量及分析待做 |
| GRPO 核心与 on-policy 变体 | 代码已实现 | 奖励与 advantage、序列/常数归一化、零 advantage 剪枝、梯度累积和训练步 |
| Off-policy GRPO / GSPO | 核心与 6.4 配套代码已实现 | 固定旧评分、每批 rollout 多次更新、clip fraction、耗时、实验计划和方法对比报告 |
| 本地主线回归 | 518 项通过 | CPU 小模型与合成日志测试，不代表实际 GPU 训练效果；不含附加作业测试 |
| 真实 GPU 实验与报告分析 | 待完成 | 尚未运行正式 OLMo/vLLM 实验，不宣称准确率提升、速度提升或达到 25% |
| 其他主线交付 | 待完成 | 训练 prompt 消融与 on-policy 变体的 CLI 配置、自定义 estimator、理论讨论及实验分析整理 |
| 附加作业 SFT / DPO | 已开始学习，算法待实现 | 已阅读概览和工具说明并整理目录；答案解析、数据加载、SFT/DPO 训练及评测尚未实现 |

当前学习位置：附加作业概览，接下来按讲义进入 zero-shot evaluation。
可选 CISPO、常数归一化 GSPO 扩展和 checkpoint 恢复训练入口也尚未实现，
不把这些可选扩展与必做内容混为一谈。

## 主线 / 附加作业导航

本仓库同时包含两份作业。已有主线文件不搬迁；新增附加作业代码放入独立子包，
不在主线的训练和评测模块中继续堆叠 SFT/DPO 实现。

| 分类 | 文件或目录 | 用途 |
| --- | --- | --- |
| 主线：数学 RL | [grpo_experiment.py](./cs336_alignment/grpo_experiment.py)、[training.py](./cs336_alignment/training.py)、[policy.py](./cs336_alignment/policy.py)、[rewards.py](./cs336_alignment/rewards.py) | GRPO 及变体、训练与统计 |
| 主线：提示评测 | [prompting_baselines.py](./cs336_alignment/prompting_baselines.py)、`prompting*.py`、[evaluate_prompting.py](./scripts/evaluate_prompting.py) | 主线 GSM8K prompting baseline |
| 主线：运行入口 | [train_grpo.py](./scripts/train_grpo.py)、[run_grpo_off_policy.py](./scripts/run_grpo_off_policy.py)、[summarize_grpo.py](./scripts/summarize_grpo.py) | 训练、实验计划与报告 |
| 主线：提示模板 | [prompts/](./cs336_alignment/prompts/) | 数学 RL 模板 |
| 附加：新增实现 | [supplement/](./cs336_alignment/supplement/) | 后续评测、SFT 数据处理、SFT 和 DPO 实现的独立位置 |
| 附加：新增运行入口 | [scripts/supplement/](./scripts/supplement/) | 后续附加作业脚本，与现有主线入口分开 |
| 附加：已有材料 | [prompts_safety/](./cs336_alignment/prompts_safety/)、[evaluate_safety.py](./scripts/evaluate_safety.py)、[modal_utils_safety.py](./cs336_alignment/modal_utils_safety.py)、[AlpacaEval 评审配置](./scripts/alpaca_eval_vllm_llama3_3_70b_fn/) | 官方提供的模板和工具，保持原路径 |
| 附加：官方测试 | [test_data.py](./tests/test_data.py)、[test_metrics.py](./tests/test_metrics.py)、[test_dpo.py](./tests/test_dpo.py) | 数据加载、答案解析和 DPO loss；保留在原目录 |
| 共享接口与资源 | [adapters.py](./tests/adapters.py)、[conftest.py](./tests/conftest.py)、[data/](./data/) | 测试适配、测试工具和原始数据，不复制两份 |

附加作业的详细文件归属、后续命名约定和测试命令见
[附加作业目录说明](./cs336_alignment/supplement/README.md)。
已有实验结果不搬迁；后续附加作业的输出统一使用 `results/supplement/`。
目录已分类不代表附加作业算法已实现，真实 GPU 实验仍待完成。

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the main assignment's required adapter tests:

``` sh
uv run --frozen pytest tests/test_grpo.py
```

The main GRPO adapters in [./tests/adapters.py](./tests/adapters.py) are connected
to the implementations and pass the local regression. The optional supplement's
MMLU response parser is connected; its other adapters remain unimplemented and
their failures should not be mistaken for
main-track regressions. Separate main/supplement test commands are documented in
[the supplement guide](./cs336_alignment/supplement/README.md).

## Prompting Baselines

The inference-only GSM8K prompting experiment has a runnable entry point:
`uv run python -m scripts.evaluate_prompting --help`.
See [PROMPTING_BASELINES.md](./PROMPTING_BASELINES.md) for full-run commands,
output formats, and the required manual analysis. GPU measurements and written
conclusions remain pending; this does not implement the later GRPO assignment.

## Off-Policy GRPO Experiments

See [OFF_POLICY_EXPERIMENT_GUIDE.md](./OFF_POLICY_EXPERIMENT_GUIDE.md) for the
four-method experiment plan, clip-fraction definitions, timing, and comparison
reports. The planner prints commands by default; real GPU experiments remain pending.
