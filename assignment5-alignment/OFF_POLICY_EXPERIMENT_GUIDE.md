# 6.4 Off-Policy 实验说明

## 当前状态

四种方法的训练入口、旧评分缓存、32 次独立更新、指标记录、checkpoint、
多种子统计和跨方法对比报告均已接通，并通过 CPU 小模型及合成日志测试。
尚未进行真实 GPU 实验；没有速度、稳定性或准确率方面的实测结论。
本说明仅覆盖 6.4，不代表整个 A5 的实验与书面交付已经完成。

## 配置和更新顺序

| 方法 | importance_reweighting_method | cliprange |
| --- | --- | --- |
| offpolicy_naive | none | 不使用 |
| offpolicy_noclip | noclip | 不使用 |
| offpolicy_clip | grpo | 0.2 |
| offpolicy_gspo | gspo | 3e-4 |

四种方法统一使用 rollout_batch_size=256、train_batch_size=8、gradient_accumulation_steps=1。
一次生成 256 个回答；先在完整的原始回答组内计算 advantage，再切成 32 批，每批更新一次。
需要 importance reweighting 时，在任何更新之前缓存一次 old_log_probs，之后不刷新。
这不是对 256 个回答重复训练 32 个 epoch，也不是累积 32 次梯度后只更新一次。
默认不指定 train_batch_size 时，它等于 rollout_batch_size，原来的 on-policy 路径不变。

## 指标口径

- step：已经完成的 rollout 批次数；验证、回答快照和 checkpoint 间隔都按这个计数。
- optimizer_updates：实际累计参数更新次数。32x off-policy 时等于 step * 32。
- train_loss、train_grad_norm：本批 rollout 内各次更新的均值；grad_norm 是裁剪前的值。
- train_grad_norm_max：本批 rollout 内最大的裁剪前梯度范数。
- train_mean_token_entropy：仅保留非零 advantage 回答，按有效回答 token 数加权。
- train_clip_fraction：实际进入截断平坦区的有效单位数 / 有效单位总数。
- train_clipped_unit_count、train_clip_unit_count：上述分子、分母，跨 microbatch 和更新先求和再相除。

GRPO 的裁剪单位是回答 token；GSPO 的裁剪单位是整条回答，而不是广播后的 token 数。
只统计非零 advantage 且至少有有效回答 token 的数据，不包含 prompt/padding。
正 advantage 只有 ratio > 1+epsilon 才计为裁剪；负 advantage 只有 ratio < 1-epsilon 才计为裁剪。
反方向越界仍有梯度，不计入这个指标；边界相等不计入严格越界。
全零 advantage 批次的分子、分母和比例都记录为 0，结合分母判断是否没有有效训练信号。
none/noclip 不记录 clip fraction，避免把“不使用裁剪”误当成“使用裁剪但从未触发”。
该口径与部分实现使用的“所有 ratio 越界比例”不同，不应直接混用。

## 时间口径

- train_iteration_seconds：一次完整的生成和训练迭代，含权重同步、生成、准备数据、旧评分计算及更新。
- val_seconds：对应那次验证的时间，不包含写回答快照。
- elapsed_seconds：累计循环内部耗时，含初始验证、周期验证、回答快照和 checkpoint。

计时边界会同步训练模型的 CUDA 工作，避免只测到异步任务提交时间。
elapsed_seconds 不包含模型/服务初始化，也不包含 yield 后日志消费者写盘或用户暂停的时间，
因此它是累计循环耗时，不是整个程序从启动到退出的端到端时间。
不同方法的比较应保持设备、验证和 checkpoint 间隔等设置一致；保存大模型的耗时也会影响比较。

## 现在只生成实验计划，不训练

在 assignment5-alignment 目录执行：

```bash
uv run --frozen python -m scripts.run_grpo_off_policy \
  --output-root results/off_policy
```

默认只打印四种方法、各四个种子的 16 条命令，不创建结果目录、不加载模型、不启动 GPU。
如果还没有正式 on-policy baseline，可以加 --include-on-policy，得到 20 条命令。
共享训练参数放在 -- 后，例如 -- --learning-rate 1e-5 --num-rollout-steps 200。
统一选择之前调好的学习率，不能给每种方法随意更改其他参数。

GPU 环境准备好后才运行：

```bash
uv run --frozen python -m scripts.run_grpo_off_policy \
  --output-root results/off_policy --execute
```

这会逐个启动真实训练；一个运行失败即停止，不覆盖已有结果目录。
重试时使用新的输出根目录，或手动选择尚未运行的命令。
建议先另外做少步 smoke run；不要把它混入正式四种子实验。
批量入口会按训练脚本的默认 checkpoint 设置运行，需要提前检查磁盘空间。

## 多方法汇总与绘图

完成真实实验后：

```bash
uv run --frozen --with matplotlib python -m scripts.summarize_grpo \
  --compare-methods \
  --run-dirs results/off_policy/offpolicy_*/seed_* \
  --output-dir results/off_policy/report_batches \
  --x-axis rollout_batches
```

如果要加入之前的 baseline，在 --run-dirs 后同时加入它的四个 seed 目录。
省略 --compare-methods 仍可汇总同一种方法的多个种子。
换用新的 output-dir，再分别指定 --x-axis optimizer_updates 或 --x-axis elapsed_seconds，
即可比较相同参数更新次数或相同累计循环时间下的表现。
旧版日志没有计时数据时，仍可按批次/更新次数绘图，但不能生成时间横轴图。

报告先按方法分组，再在组内统计种子的均值、样本标准差和正态近似 95% 均值区间。
图中细线是各个种子的真实曲线；不同方法不会混在一起平均。
时间横轴的均值曲线使用同一 rollout step 的平均时间与平均指标，不插值到统一时间网格。
只有一个种子时不画区间；少于四个种子的报告只能作为预览。
汇总器拒绝未完成运行、重复种子、缺失指标以及不应改变的超参数不一致。
不同 loss 定义的数值不能直接当作模型质量比较，主要观察奖励、准确率和稳定性。
GRPO/GSPO 的 clip fraction 图会注明 token/response 单位，不能忽略这个口径差异。

## GPU 实验之后补充

- 每种方法四个种子的正式日志和曲线，以及与 on-policy baseline 的对比。
- 对奖励、稳定性、跨种子方差、clip fraction 和训练耗时的分析。
- 真实训练前后的回答样例，以及是否需要继续调参的判断。
- report.md 中 Analysis pending 的人工结论，不用合成测试数据冒充实验结果。

## 本地验证

```bash
uv run --frozen --with matplotlib pytest -q \
  tests/test_clip_metrics.py tests/test_grpo_off_policy_reporting.py \
  tests/test_grpo_reporting.py tests/test_grpo_experiment_loop.py \
  tests/test_grpo_experiment_off_policy.py tests/test_rollout_batch_training.py
```

这些测试不启动真实 GPU 实验。绘图使用明确构造的合成日志，只验证统计和图像输出。
