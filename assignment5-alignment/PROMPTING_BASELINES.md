# Prompting Baselines 实验

这份说明对应主作业第 3.4 节 `prompting_baselines`，不是整个 A5 的 GRPO 实现。
代码已经连接数据加载、三种提示、vLLM 生成、官方评分器、指标统计和结果保存。
这里仅做推理测评，不执行反向传播，不更新模型参数。
真实 GPU 测评及人工分析尚未完成；测试中的模拟回答不能当作实验结果。

## 运行环境

在 `assignment5-alignment` 目录运行。真实推理需要能够运行官方 vLLM helper 的 CUDA 环境及 OLMo-2-0425-1B **base model** 权重。
使用已有的 A5 GPU 环境，或者按 `pyproject.toml` 的 GPU extra 安装依赖：

```bash
uv sync --extra gpu --no-install-package flash-attn
uv sync --extra gpu
```

`--model-id` 可以是模型所在目录，也可以是服务使用的 Hugging Face 模型 ID。
下列 `/path/to/OLMo-2-0425-1B` 需要改为真实模型目录；连接已有服务时，必须与服务接受的 model ID 一致。
不要改用 chat/instruct 模型，否则测评对象与题目要求不同。

## 先做小规模检查

```bash
uv run python -m scripts.evaluate_prompting \
  --model-id /path/to/OLMo-2-0425-1B \
  --gpu 0 \
  --limit 16 \
  --batch-size 8 \
  --output-dir results/prompting-smoke-seed0
```

默认数据是仓库内 `data/gsm8k/test.jsonl`。`--limit` 只截取前 N 道题，适合检查流程，不适合报告完整测试集结果。
`--gpu` 传给官方 helper 的 `CUDA_VISIBLE_DEVICES`；若运行平台已经做了 GPU 映射，请使用平台允许的索引。
自动启动模式会使用官方 helper 管理指定端口上的 vLLM 进程，请选择没有其他任务使用的端口。

## 完整测评

```bash
uv run python -m scripts.evaluate_prompting \
  --model-id /path/to/OLMo-2-0425-1B \
  --gpu 0 \
  --seed 0 \
  --batch-size 32 \
  --output-dir results/prompting-full-seed0
```

不传 `--limit` 就评测整个输入文件；默认依次运行 `question_only`、`r1_zero`、`r1_zero_three_shot`。
可用 `--prompts r1_zero` 单独运行一种。输出目录必须是新目录或空目录，每次实验请使用不同的目录。
服务只启动一次，所有提示共享同一个 base model；无论生成成功还是失败，脚本都会清理自己启动的服务。

如果服务已经启动，则连接它，不启动或关闭服务：

```bash
uv run python -m scripts.evaluate_prompting \
  --model-id /path/to/OLMo-2-0425-1B \
  --connect-only --host 127.0.0.1 --port 8000 \
  --output-dir results/prompting-existing-server
```

## 生成与评分规则

- 固定设置为 temperature=1.0、top-p=1.0、max_tokens=512、n=1；seed 默认 0。
- 官方 helper 没有转发 `top_p`，本实现也没有修改它，因此 top-p 依赖服务默认值；连接已有服务时请确保默认 top-p 为 1.0。
- 只有两种 r1 提示在 `</answer>` 处停止，并把停止字符串保留在回答中。
- `question_only` 使用 boxed-answer grader；两种 r1 提示使用 r1 grader。
- 使用官方严格格式规则，不额外修补输出或宽松匹配。
- 正确率是官方 grader 的答案分均值；格式分只用于记录，不加到总奖励中。

## 输出文件

每个 prompt 输出两份 JSONL；所有 prompt 共享汇总文件和报告：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 模型、数据路径、seed、batch size、生成参数、各项指标、检查样例数量及完成标记 |
| `<prompt>.responses.jsonl` | 全部题目的原始 prompt、回答、标准答案、token IDs、结束原因、评分及类别 |
| `<prompt>.review.jsonl` | 类别 1 最多 3 条，类别 2 和 3 各最多 10 条，以及待填写的人工检查字段 |
| `report.md` | 自动测评指标表及待完成的人工分析要求 |

`example_index` 是当前评测数据中的零基索引。样例按数据顺序选取，不是随机抽样。
三类分别是：1=格式正确且判对，2=格式正确但未判对，3=格式和答案分均为 0。
`num_length_limited` 统计结束原因为 `length` 的回答，帮助检查生成截断。
`completed=true` 只表示自动测评完成，不表示人工分析完成。
如果中途失败，已完成提示的文件及 `completed=false` 的汇总仍会保留；再次运行需选择新目录。
`results/` 已忽略，不会把大量生成结果或尚未审核的实验结论误提交到 Git。

## 作业书面部分

1. 对每种 prompt 报告正确率、格式通过率和三类数量，标明数据规模、模型、seed 及生成设置。
2. 人工检查类别 2 和 3 各至少 10 条，填写 review 中的 `human_answer_correct`、`parser_missed_correct_answer` 和 `behavior_notes`。
3. 若某类少于 10 条，则检查全部，并明确说明样本不足；不要虚构额外样例。
4. 记录检查样例中“实际正确但解析失败”的数量，保留自动指标，另行解释解析局限。
5. 比较三种提示下是否直接回答、是否输出推理、是否遵守格式、是否出现无关续写；用原始 prompt 和回答支持结论。

最终将实际实验结果及人工分析整理到作业的 `writeup.pdf`。脚本不会代替人工判断，也不会凭测试数据生成实验结论。

## 本地验证

在配置好 A5 依赖的环境中，仅运行本题新增测试：

```bash
uv run python -m unittest \
  tests.test_gsm8k tests.test_prompting tests.test_prompting_generation \
  tests.test_prompting_evaluation tests.test_prompting_baselines -v
```

测试覆盖真实官方 grader、指标统计、三种 prompt 的文件输出、错误处理、服务生命周期及本地模拟 HTTP 流程。
不启动真实 vLLM，不下载 OLMo 权重，不需要 GPU。它们是自加测试，不是官方 GRPO 测试，也不能证明真实模型精度。
