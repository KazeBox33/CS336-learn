# 标准 On-Policy GRPO 实验说明

## 当前状态

本题的训练链路和实验配套代码已实现，并用小模型 CPU 测试与模拟推理服务验证。
真实 OLMo + vLLM 双 GPU 训练、50 步调试证据、四个种子实验与报告中的人工分析仍待完成。
没有真实实验数据时，不宣称准确率提升或达到 25%。
本说明不包含下一题的 RFT、Dr. GRPO、MaxRL 或 off-policy 变体。

## 一次更新的流程

1. 同步训练模型的权重到 vLLM。
2. 抽取 32 道题，每题生成 8 个回答，得到 256 个训练回答。
3. 逐个评分，在每道题的 8 个回答内计算均值、标准差与 advantage。
4. tokenize 并构造 response mask，只对回答部分计算 loss。
5. 将 256 个回答拆成 32 个 microbatch，累积梯度，裁剪梯度，更新一次参数。
6. 按间隔验证、保存回答和 checkpoint，再 yield 当步指标给日志写入器。

注意：第 n 步的训练回答来自更新 n-1 次后的模型，验证回答来自更新 n 次后的模型。

## Checkpoint 的内容和用途

默认每 40 次更新保存一次，最后一步也必定保存。可以通过 --checkpoint-interval 修改。
例如第 200 步的目录结构：

```text
results/grpo/seed_0/
  run.json
  metrics.jsonl
  rollouts/
  checkpoints/
    step_0200/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      training_state.pt
```

大模型的权重也可能分成多个 safetensors 文件，具体文件名由 Transformers 决定。

- model.save_pretrained 保存训练后的权重和模型配置。
- tokenizer.save_pretrained 保存对应 tokenizer，方便以后独立加载这个模型。
- training_state.pt 包含更新步数、优化器状态、抽题 RNG、Python RNG、Torch CPU/CUDA RNG。
- AdamW 状态包括动量等信息；只有模型权重不能完整表示训练状态。
- 先在同一文件系统内写临时目录，成功后重命名发布，避免失败保存留下看起来完整的 checkpoint。
- 不覆盖已有 checkpoint。保存失败时不将 run.json 标成 completed。
- run.json 的 last_checkpoint 指向最新成功保存的位置。

这些状态为以后实现恢复训练提供基础，但当前没有 --resume 入口。
加载保存后的模型进行推理不等于恢复优化器和训练循环；不要混淆两者。
周期 checkpoint 很占磁盘空间。GPU 实验前确认空间，或增大间隔，仅在最后保存。

## 在 Mac 上检查配置，不训练

在 assignment5-alignment 目录执行：

```bash
uv run --frozen python scripts/train_grpo.py \
  --output-dir results/grpo/preview \
  --dry-run
```

只检查数据与参数并打印预览，不加载模型、不启动 CUDA/vLLM、不创建结果目录。

## GPU 准备好之后的实验

以下命令是操作说明，本次没有执行。需要已准备好的 Linux 双 CUDA GPU 环境及 gpu 依赖。
GPU 编号是当前进程可见的逻辑编号。默认训练使用 cuda:0，推理使用 GPU 1。

先做约 50 步的调试运行：

```bash
uv run --frozen --extra gpu python scripts/train_grpo.py \
  --output-dir results/grpo/smoke_seed_0 \
  --num-rollout-steps 50 --seed 0
```

检查训练/验证奖励、回答样例、服务权重同步、显存和 checkpoint。
奖励开始接近 0 并不自动表示代码错误，但不能仅凭测试通过就认定实际训练有效。

再按默认参数进行四次正式实验，输出目录必须是新的：

```bash
for seed in 0 1 2 3
do
  uv run --frozen --extra gpu python scripts/train_grpo.py \
    --output-dir "results/grpo/seed_$seed" --seed "$seed"
done
```

默认模型 allenai/OLMo-2-0425-1B，数据 GSM8K，zero-shot r1_zero prompt，
训练 200 次更新，每批 256 个回答，group_size=8，学习率 1e-5。
只修改 seed，其他配置和数据子集保持一致。50 步 smoke run 不混入正式四次运行。

## 汇总和绘图

```bash
uv run --frozen --with matplotlib python scripts/summarize_grpo.py \
  --run-dirs results/grpo/seed_0 results/grpo/seed_1 \
             results/grpo/seed_2 results/grpo/seed_3 \
  --output-dir results/grpo/report
```

Matplotlib 只用于绘图，通过 --with 提供，不改训练依赖和锁文件。
report 目录也必须是新的。产生 summary.json、九张指标图和 report.md。

汇总器会检查：
- 每次运行 completed=True。
- 随机种子不重复，模型、数据路径和训练配置除 seed 外一致。
- 步数记录完整且顺序正确，必要的训练/验证指标存在，数值有限。
- 验证准确率在 0 到 1 之间。

同一个 step、同一个指标才会汇总。验证稀疏记录不会被补成 0。
每个指标图画出各个种子的原始曲线、均值，以及均值的正态近似 95% 区间。
对于 n 次运行的值 z_1,...,z_n：

```text
mean = sum(z_i) / n
sample_std = sqrt(sum((z_i - mean)^2) / (n - 1))
margin = 1.96 * sample_std / sqrt(n)
interval = [mean - margin, mean + margin]
```

样本标准差描述各次运行的波动；区间描述均值估计的不确定性，不是单次运行的范围。
n=4 时这个区间仍是正态近似，不是严格的小样本置信区间。
只有一个运行时不画区间，也不把标准差假装成 0。
只有恰好四个完整种子时才判断平均最终验证准确率是否达到 25%，其他情况为 null。
代码只读取真实日志统计，不保证达标，也不自动声称推理能力变好。

## 最后需要人工补充的交付物

- 50 步调试运行的曲线和合理回答样例，说明为什么相信训练脚本正确。
- 四个种子的九个指标图，以及每个指标的趋势和跨运行波动分析。
- 从 rollouts/val_step_0000.jsonl 和最终验证快照选取真实的训练前后回答。
- 固定验证集顺序不变，可以按 response_index 对比同一题。
- 最终平均验证准确率与是否达到 25%。
- report.md 里的 Analysis pending 是待填写项，不是自动生成的实验结论。

## 本地测试

```bash
uv run --frozen --with matplotlib pytest \
  tests/test_grpo_checkpoint.py tests/test_grpo_reporting.py \
  tests/test_grpo_experiment_cli.py tests/test_grpo_experiment_loop.py
```

绘图测试使用合成日志，只验证统计和文件输出；不构成 OLMo 的真实实验结果。
Checkpoint 测试用小 GPT-2 在 CPU 上实际更新、保存和重新加载权重及优化器状态。
