# 附加作业：Instruction Tuning、Safety 和 DPO

这是 A5 附加作业新增实现的独立目录，不是 GRPO 主线目录。
当前已实现 MMLU 标准格式答案解析、CSV 数据读取和 zero-shot 双层 prompt 构造；
SFT 数据准备已完成 JSONL 问答读取、Alpaca 模板填充、分词与文档结束标记拼接、定长数据集和批处理；
SFT 已实现前向、next-token 交叉熵 loss、多 microbatch 累积更新和单 epoch 外层循环，使用 CPU 小模型测试；
完整评测脚本、SFT 训练入口（含调度、验证、保存）和 DPO 尚未实现，不启动 8B 训练或 GPU 实验。

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
`iterate_batches` 使用 PyTorch `DataLoader` 按字段堆叠张量，可在每轮打乱数据块顺序，
保留最后不足一批的数据，使一轮遍历覆盖每条数据恰好一次。
SFT 前向与 loss 位于 [sft.py](./sft.py)，`compute_sft_loss` 接收模型与 batch 字典，
将 `(B, T, V)` logits 展平为 `(B*T, V)`，与已经移位的 `(B*T,)` 标签直接计算平均交叉熵。
不再次移位、不套用 GRPO 的回答掩码，保留梯度供后续训练循环使用。
`sft_train_step` 接收一组 microbatch，按实际标签 token 数加权 loss，再逐批 backward。
等大小 microbatch 时等价于各自 loss 除以组内批次数；最后批次或最后一组不足额时，仍按实际 token 总数平均。
累积完成后只裁剪一次梯度、更新一次参数，返回 detached 平均 loss、裁剪前 grad norm、token 数与批次数。
函数清除进入前的旧梯度，也在返回或计算异常时清理本组梯度；它不会自行分组或遍历整个 epoch。
`run_sft_epoch` 每次从批次迭代器收集最多 `gradient_accumulation_steps` 批数据，调用一次 `sft_train_step`。
最后不足一组的数据也会更新；空输入不更新。每次更新后 yield 从 1 开始的 epoch 内更新编号和普通数值指标。
它是惰性生成器，调用本身不训练，调用方必须迭代消费；只收集当前组，不提前把整个 epoch 的数据读入列表。
当前不含学习率调度、验证和保存，也不把各组 loss 的简单平均当作整个 epoch 的 token 平均 loss。
DPO 后续使用 `supplement/dpo.py`，该文件尚未创建。

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
SFT 数据准备、定长数据集与批处理可用 `uv run --locked --extra plots pytest -q tests/test_supplement_sft_data.py tests/test_data.py` 单独验证；
`get_packed_sft_dataset` 与 `run_iterate_batches` 均已接通。
SFT 前向、loss 与梯度测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_loss.py`，
只使用本地 CPU 小模型，不加载 Llama 8B 权重。
SFT 单次梯度累积更新测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_step.py`。
SFT 单 epoch 分组、尾组更新与惰性消费测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_epoch.py`。
后续按讲义顺序逐个实现、逐个运行对应测试，不把这些失败误认为 GRPO 主线回归。
新增附加测试统一使用 `tests/test_supplement_*.py` 命名，主线命令已预留相应排除规则。
附加回归时应把已新增的这些测试文件一起传给 pytest。

## 学习顺序

Zero-shot evaluation -> SFT 数据与训练 -> SFT 后评测与安全检查 -> HH 偏好数据 -> DPO。
这里不提前实现后面的算法，也不启动真实 GPU 实验。
