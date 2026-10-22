# 附加作业：Instruction Tuning、Safety 和 DPO

这是 A5 附加作业新增实现的独立目录，不是 GRPO 主线目录。
当前已实现 MMLU 标准格式答案解析、CSV 数据读取和 zero-shot 双层 prompt 构造；
SFT 数据准备已完成 JSONL 问答读取、Alpaca 模板填充、分词与文档结束标记拼接、定长数据集和批处理；
SFT 已实现前向、next-token 交叉熵 loss、多 microbatch 累积更新、单 epoch 循环、验证集 loss 和学习率调度，使用 CPU 小模型测试；
已支持模型与 tokenizer 导出并用本地小模型验证重新加载；
SFT 训练入口已接通配置、日志、定期验证和最终保存，并通过本地 CPU 小模型的命令行端到端测试；
HH 偏好数据已实现四文件合并、单轮过滤、问题/回答拆分与来源保留；
单样本 DPO loss、单组梯度累积更新、单 epoch 循环和验证 loss/准确率已实现并通过 CPU 小模型测试；
DPO 最佳 checkpoint 的判断与导出已实现；完整评测脚本和 DPO 训练入口尚未实现，真实 8B 训练与 GPU 实验仍未运行。

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
支持普通 `.jsonl` 与讲义所用的 `.jsonl.gz` 压缩文件。
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
可选 scheduler 仅在成功完成参数更新后推进一次，返回的 `learning_rate` 是本次更新实际使用的第一参数组学习率。
循环不自行验证和保存，也不把各组 loss 的简单平均当作整个 epoch 的 token 平均 loss。
`create_sft_lr_scheduler` 使用 PyTorch `LambdaLR`，将 optimizer 初始学习率作为峰值，先线性 warmup 再 cosine 衰减至零。
`total_updates` 和 `warmup_updates` 都按 optimizer 更新计数，不按 microbatch 计数；总更新数应包含每个 epoch 的不足额尾组。
有 warmup 时第一个更新的学习率为零；无 warmup 时从峰值开始。最后一次更新后，为下一次设置的学习率为零。
scheduler 在整个训练开始时只创建一次，跨 epoch 复用；训练入口将 warmup_ratio 乘总更新数并向上取整。
warmup 最多为 `total_updates - 1`，因此只有一次更新的 CPU 小测试会禁用 warmup，保留实际参数更新。
`evaluate_sft` 独立遍历验证批次，在 eval 模式和 no_grad 下按实际 token 数汇总 `val_loss`。
它不 backward、不更新参数、不清理已有梯度；正常结束或计算异常时均恢复模型进入前的 train/eval 模式。
空验证集会报错，不返回虚假的零 loss；训练入口负责在训练前、指定更新间隔及最后一次更新后验证。
SFT 导出位于 [checkpoint.py](./checkpoint.py)，`save_sft_model` 将模型权重、配置和 tokenizer 保存为 HF 格式。
模型与 tokenizer 先写入同级临时目录，全部成功后再发布为目标目录；写入失败会清理临时数据，不覆盖已有目标。
该导出用于评测和后续 DPO 加载，不保存 optimizer、scheduler、随机状态或数据进度，因此不是完整断点续训 checkpoint。
函数本身不训练、不改变模型 train/eval 模式；训练入口在全部训练和最终验证完成后调用。
DPO 单样本 loss 位于 [dpo.py](./dpo.py)，使用相同 Alpaca 模板组织 chosen/rejected 文档，按分词器规则编码后追加 EOS token ID。
复用主线的 token log-probability 函数，以错开一位的输入/标签计算完整序列分数并求和，不取平均、不截断、不 padding。
每个模型内部的 chosen-minus-rejected 差值会抵消共同提示前缀的分数；实际 tokenizer 应保持两条文档的共同提示 token 前缀一致。
参考模型在 no_grad 下评分，分数移到当前模型设备后计算稳定的 `-logsigmoid(beta * (policy_margin - ref_margin))`。
评分时临时将两个模型设为 eval 以关闭 dropout，正常返回或报错后恢复进入时的 train/eval 模式；当前模型仍保留计算图。
函数不 backward、不更新参数、不清理已有梯度，也不永久修改 requires_grad；DPO 训练循环后续再负责冻结参考参数和优化当前模型。
`compute_dpo_loss_and_margins` 复用这段评分并返回三个标量：loss、当前模型 chosen-minus-rejected 分数差、参考模型分数差。
原有 `compute_per_instance_dpo_loss` 仍只返回 loss，官方 adapter 接口不变；验证可直接复用两个 margin，不额外运行前向。

实现的导入路径使用 `cs336_alignment.supplement.<模块名>`。
新增脚本放入 `scripts/supplement/`，当前 SFT 入口为 [train_sft.py](../../scripts/supplement/train_sft.py)。
从 A5 根目录运行 `uv run --frozen python -m scripts.supplement.train_sft --help` 可查看参数，不会训练。

不把附加作业实现直接写进主线 `training.py`、`policy.py`、`prompting*.py` 或测试适配器。
`vllm_utils.py` 等工具可以按需复用，但先确认行为是否适合，不为分类而移动或复制它们。
已有 `checkpoint.py` 等模块也不搬迁；其中主线专用逻辑不视为附加作业的默认实现。

## SFT 训练入口

编排逻辑位于 [sft_training.py](./sft_training.py)：`SFTTrainingConfig` 校验参数，`run_sft_training` 接收已经加载的单设备模型与 tokenizer。
默认设置为 1 epoch、context 512、microbatch size 1、梯度累积 32 次（单设备有效 batch size 32）、
AdamW 学习率 2e-5、weight decay 0.1、3% warmup、cosine decay、梯度裁剪 1.0。
默认每 10 次参数更新记录训练指标、每 100 次更新验证；最终更新总会记录并验证。
训练记录的 loss 是该次更新所用数据的 token 平均值，不是日志间隔内所有更新的平均值。
CLI 要求显式提供 `--model-id`、`--train-path`、`--val-path` 和 `--output-dir`，不默认下载 8B 模型。
默认 `--device cpu --dtype float32 --attn-implementation eager`；将来使用 CUDA 时可显式选择 bf16 和 FlashAttention 2（需对应依赖）。
可用 `--local-files-only` 禁止下载，用 `--gradient-checkpointing` 启用模型提供的激活重计算。
没有使用 Hugging Face Trainer；当前为单设备完整参数微调，不包含分布式训练、量化或 LoRA。

输出目录必须不存在，内容为：

- `run.json`：配置、数据规模、计划更新次数、warmup 次数和最终验证 loss；只在最终模型保存成功后标记 `completed=true`。
- `metrics.jsonl`：带全局更新编号和 epoch 编号的训练/验证记录，同时输出到控制台。
- `final/`：最终模型和 tokenizer，可供后续评测或 DPO 加载，不含断点续训状态。

每个 epoch 重新遍历并打乱训练数据块；文档顺序只在初次打包时按 seed 打乱，验证数据不打乱。
CPU 小模型测试会实际走完命令行入口；这不代表已完成 Llama 8B 实验，也不提供虚构的验证结果或学习曲线。

## HH 偏好数据

`look_at_hh` 的 (a) 位于 [hh.py](./hh.py)：`load_hh_preferences(data_dir)` 按固定顺序读取四个官方 `.jsonl.gz` 文件，
返回 `HHPreferenceExample` 列表，每条包含 `instruction`、`chosen`、`rejected` 和来源文件名 `source`。
任一分支包含多条 `\n\nHuman:` 消息就过滤整对；单轮样本按第一个 `\n\nAssistant:` 拆分，保留回答中后续的 Assistant 标记及内部换行。
只去除拆分字段边界的空白，不额外过滤空回答；格式错误或两边问题不同会报告文件及行号，缺文件也会报错。
`source` 是分析用元数据，不是模型输入。该步骤不分词、不计算 loss、不训练模型，不需 GPU，也不需要 adapters 转接。
当前本地四文件共 160800 对，过滤 111364 对多轮样本后保留 49436 对；保留样本中 15 条 chosen、33 条 rejected 为空回答。
按文件的保留数量为：harmless-base 12253、helpful-base 12843、helpful-online 7578、helpful-rejection-sampled 16762。
这些是当前本地文件的实际读取统计，不是训练或模型评测结果。
本步完成 (a)；(b) 的 3 条 helpful 和 3 条 harmless 随机样本及标签讨论留到下一小步。

## DPO 训练（分步）

单次更新位于 [dpo_training.py](./dpo_training.py)：`dpo_train_step` 接收当前模型、参考模型、tokenizer、optimizer 和一组 HH 样本。
每次只处理一对 chosen/rejected，使用已有 DPO loss，除以本组实际样本数后立即 backward，最后更新一次参数。
这里按偏好样本对平均，不像 SFT 按 token 数加权；一条偏好样本内部的序列 log-probability 仍然求和，不做长度归一化。
调用方以后再按 gradient accumulation steps 分组；本函数天然支持最后不足一组的实际样本数。
函数会永久冻结参考参数并将参考模型设为 eval；optimizer 只能含当前模型参数，并且必须覆盖全部可训练的当前参数。
当前模型与参考模型不能共享 Parameter 对象。当前模型设为 train，但复用的评分函数会临时关闭 dropout，同时保留当前模型的计算图。
返回 detached 组平均 loss、裁剪前 grad norm 和样本对数；清除进入前的 optimizer 梯度，以及本组结束或计算异常时的梯度。
`max_grad_norm=None` 默认只测量梯度范数、不裁剪；显式传入阈值时在全部 backward 后裁剪一次。非有限梯度会阻止更新。
optimizer 由调用方传入，CPU 测试采用讲义建议的 RMSprop；正式入口后续再接学习率 1e-6、beta 0.1、有效 batch size 64 等配置。
`dpo_train_step` 只处理一组，不计算验证准确率、不保存最佳 checkpoint；optimizer.step 自身失败时不承诺回滚参数。
`run_dpo_epoch` 接收 HH 样本的 iterable，每次最多收集 `gradient_accumulation_steps` 对，调用一次单组更新；不足额尾组也更新。
循环按输入顺序处理，不自动打乱或划分验证集；只收集当前组，不提前读取整个 iterable，也不需要将文本拼成 padding batch。
它是惰性生成器，必须迭代消费才训练；每次更新完成后 yield 从 1 开始的 epoch 内更新编号，以及组平均 loss、grad norm、样本数和第一参数组学习率。
空输入不更新、不冻结参考模型；一组失败时不 yield 成功指标，也不继续读取下一组。不把各组 loss 的简单平均当成整个 epoch 的样本平均 loss。
`evaluate_dpo` 在 no_grad 下逐对验证，按样本对数平均 `val_loss`，返回 `val_accuracy` 和 `num_examples`。
准确率按讲义第 17 页的明确口径：当前模型的 chosen 序列 log-probability 严格高于 rejected 才计为正确，相等算错误。
这个口径对应 `policy_margin > 0`，不是 `policy_margin - ref_margin > 0`；后者是参考校正后的隐式奖励偏好，两者不混用。
验证仍需参考模型计算 DPO loss；不永久冻结参数、不清理已有梯度，评分时临时 eval 并恢复模式，空验证集报错。
本函数只执行一次独立验证，不自行安排验证间隔或保存模型。
最佳模型选择位于 [checkpoint.py](./checkpoint.py)：`save_best_dpo_checkpoint` 接收验证准确率、全局更新编号、输出根目录和上次最佳记录。
首次验证（即使准确率为 0）会保存；以后仅准确率严格提高时保存，相同准确率保留较早的模型，不以 loss 打破平局。
返回不可变的 `DPOBestCheckpoint(accuracy, update, path)`；调用方在后续验证时传回它，不提升时直接返回原记录且不写磁盘。
模型与 tokenizer 保存到 `step_000001` 等独立目录，复用已有安全 HF 导出；新记录只在保存成功后返回，旧目录不覆盖、不删除。
这里只保存当前模型与 tokenizer，不保存参考模型、optimizer 或数据进度；best 记录目前由调用方保存在内存，完整入口后续再写入运行摘要。
当前尚未把验证间隔、最佳保存和 epoch 循环接成完整训练入口，也没有真实 GPU 训练结果。

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
SFT 验证集加权 loss、模式恢复与无梯度测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_evaluation.py`。
SFT warmup/cosine 学习率与按更新推进的测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_schedule.py`。
SFT 模型/tokenizer 导出、重新加载与失败清理测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_checkpoint.py`。
SFT 训练编排和真实 CPU 命令行端到端测试使用 `uv run --frozen pytest -q tests/test_supplement_sft_training.py`。
HH 数据合并、过滤、拆分与来源测试使用 `uv run --frozen pytest -q tests/test_supplement_hh.py`。
DPO 官方 loss 接口使用 `uv run --frozen pytest -q -k test_per_instance_dpo_loss`；
补充的数值、梯度、EOS 和模式恢复测试使用 `uv run --frozen pytest -q tests/test_supplement_dpo_loss.py`。
双 CUDA 设备测试仅在存在两张 CUDA GPU 时运行；CPU 测试通过不代表已验证真实跨 GPU 训练。
DPO 单次 RMSprop 更新、样本对平均与参考冻结测试使用 `uv run --frozen pytest -q tests/test_supplement_dpo_step.py`。
DPO 单 epoch 分组、尾组、惰性执行及顺序更新测试使用 `uv run --frozen pytest -q tests/test_supplement_dpo_epoch.py`。
DPO 验证 loss、讲义准确率口径、无梯度和模式恢复测试使用 `uv run --frozen pytest -q tests/test_supplement_dpo_evaluation.py`。
DPO 最佳模型判断、平局规则、导出失败与重新加载测试使用 `uv run --frozen pytest -q tests/test_supplement_dpo_checkpoint.py`。
后续按讲义顺序逐个实现、逐个运行对应测试，不把这些失败误认为 GRPO 主线回归。
新增附加测试统一使用 `tests/test_supplement_*.py` 命名，主线命令已预留相应排除规则。
附加回归时应把已新增的这些测试文件一起传给 pytest。

## 学习顺序

Zero-shot evaluation -> SFT 数据与训练 -> SFT 后评测与安全检查 -> HH 偏好数据 -> DPO。
这里不提前实现后面的算法，也不启动真实 GPU 实验。
