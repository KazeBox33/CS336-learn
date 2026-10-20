# 附加作业：Instruction Tuning、Safety 和 DPO

这是 A5 附加作业新增实现的独立目录，不是 GRPO 主线目录。
当前已实现 MMLU 标准格式答案解析、CSV 数据读取和 zero-shot 双层 prompt 构造；
SFT 数据准备已完成 JSONL 问答读取、Alpaca 模板填充、分词与文档结束标记拼接、定长数据集；
完整评测脚本、SFT 批处理、SFT 训练和 DPO 尚未实现。不加载模型、不启动 GPU。

## 文件分类

| 内容 | 位置 | 当前处理方式 |
| --- | --- | --- |
| 主线讲义 | [alignment.pdf](../../cs336_spring2026_assignment5_alignment.pdf) | 保持原位 |
| 附加讲义 | [supplement_safety_rlhf.pdf](../../cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | 保持原位，按讲义顺序学习 |
| 附加作业实现 | 当前 `cs336_alignment/supplement/` 子包 | 后续真实实现写在这里 |
| 附加作业运行入口 | [scripts/supplement/](../../scripts/supplement/) | 后续新增入口写在这里 |
| 附加提示模板 | [prompts_safety/](../prompts_safety/) | 沿用官方路径，不混用主线 `prompts/` |
| 安全评测工具 | [evaluate_safety.py](../../scripts/evaluate_safety.py) | 官方已有脚本，保持原位 |
| 自动评审配置 | [AlpacaEval annotator](../../scripts/alpaca_eval_vllm_llama3_3_70b_fn/) | 保持原位，不破坏外部工具的路径约定 |
| 附加 Modal 工具 | [modal_utils_safety.py](../modal_utils_safety.py) | 保持原位，之后使用前再检查环境配置 |
| 官方适配接口 | [adapters.py](../../tests/adapters.py) | 后续只导入并转接这里的真实实现 |
| 官方附加测试 | [test_metrics.py](../../tests/test_metrics.py)、[test_data.py](../../tests/test_data.py)、[test_dpo.py](../../tests/test_dpo.py) | 不搬迁、不修改测试要求 |

## 后续代码命名

新模块按实际学习步骤创建，不预先放入未实现的函数或空算法文件。
MMLU 答案解析已放入 `supplement/evaluation.py`，只从模型输出提取选项，不读取标准答案。
MMLU 数据读取位于 [mmlu.py](./mmlu.py)，prompt 构造位于 [prompting.py](./prompting.py)。
默认读取 `data/mmlu/test/*_test.csv`，按文件名和文件内行顺序返回题目；
标准答案保留用于后续评分，不填入模型输入。模板仍使用原位的 `prompts_safety/`。
SFT 数据处理位于 [data.py](./data.py)，`load_sft_documents` 返回完整文档字符串列表，
只移除模板文件自带的尾部换行，不裁剪问答字段，以便在回答后直接追加 EOS。
`tokenize_sft_documents` 使用分词器默认特殊标记编码每篇文档，再追加 EOS 并拼接为 token ID 列表。
官方 Llama 测试数据包含每篇文档的 BOS 和 EOS，不沿用主线 GRPO 的无特殊标记分词规则。
`PackedSFTDataset` 可在拼接前打乱文档，按需切出定长输入和错开一位的标签。
长度为 `N` 的 token 流可提供 `max(0, (N - 1) // seq_length)` 条完整数据，输入和标签都为整数张量。
批处理尚未实现。SFT 训练和 DPO 后续分别使用 `supplement/sft.py`、
`supplement/dpo.py`；这两个文件尚未创建。

实现的导入路径使用 `cs336_alignment.supplement.<模块名>`。
新增脚本放入 `scripts/supplement/`，从 A5 根目录用
`uv run --frozen python -m scripts.supplement.<入口名>` 运行。
这只是入口命名约定，不是现在可执行的训练命令。

不把附加作业实现直接写进主线 `training.py`、`policy.py`、`prompting*.py` 或测试适配器。
`vllm_utils.py` 等工具可以按需复用，但先确认行为是否适合，不为分类而移动或复制它们。
已有 `checkpoint.py` 等模块也不搬迁；其中主线专用逻辑不视为附加作业的默认实现。

## 数据和结果

- [data/gsm8k/](../../data/gsm8k/)：原始题目可共享；主线和附加作业的提示格式及答案解析规则分别实现。
- [data/mmlu/](../../data/mmlu/)、[data/alpaca_eval/](../../data/alpaca_eval/)、[data/simple_safety_tests/](../../data/simple_safety_tests/)：附加作业评测数据目录。
- [data/hh/](../../data/hh/)：附加作业的偏好数据目录。
- SFT 训练数据和大型模型权重：使用前单独核查路径与完整性，不因这些目录存在就视为已准备好。
- 新实验结果：统一使用 `results/supplement/`，在其下区分 baseline、SFT 和 DPO。

不移动已有主线实验结果，不复制 GSM8K 原始数据，也不创建虚假的模型或实验输出占位文件。
`results/` 已被 Git 忽略；训练权重和生成结果不随代码提交。

## 分开运行测试

以下命令都在 `assignment5-alignment` 根目录执行。

主线回归，包括已有绘图测试：

```bash
uv run --frozen --with matplotlib pytest -q \
  --ignore=tests/test_data.py \
  --ignore=tests/test_dpo.py \
  --ignore=tests/test_metrics.py \
  --ignore-glob='tests/test_supplement_*.py'
```

附加作业官方测试：

```bash
uv run --frozen pytest -q \
  tests/test_metrics.py tests/test_data.py tests/test_dpo.py
```

MMLU 答案解析接口已接通，可单独运行：

```bash
uv run --frozen pytest -q tests/test_metrics.py tests/test_supplement_mmlu.py -k test_parse_mmlu_response
```

MMLU 数据读取和 prompt 构造测试：

```bash
uv run --frozen pytest -q tests/test_supplement_mmlu_preparation.py
```

其余附加测试仍包含未实现接口，预期会失败。
SFT 数据准备和定长数据集可用 `uv run --locked --extra plots pytest -q tests/test_supplement_sft_data.py tests/test_data.py -k 'not test_iterate_batches'` 单独验证；
`get_packed_sft_dataset` 已接通，`run_iterate_batches` 批处理适配器尚未接通。
后续按讲义顺序逐个实现、逐个运行对应测试，不把这些失败误认为 GRPO 主线回归。
新增附加测试统一使用 `tests/test_supplement_*.py` 命名，主线命令已预留相应排除规则。
附加回归时应把已新增的这些测试文件一起传给 pytest。

## 学习顺序

Zero-shot evaluation -> SFT 数据与训练 -> SFT 后评测与安全检查 -> HH 偏好数据 -> DPO。
这里不提前实现后面的算法，也不启动真实 GPU 实验。
