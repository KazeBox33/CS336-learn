# Assignment 5 实验路线

更新：2026-10-24。依据本仓库 Spring 2026 主线讲义第 3.4、4.3、5.4、6.4、7 节及当前实际入口整理。
这是实验计划，不是已完成结果；本次只运行 CPU 测试与 dry-run，不启动付费 GPU 任务。
先完成 OLMo-2-0425-1B + GSM8K 主线，再独立决定是否做可选 Llama 8B SFT / DPO。

## 0. 硬件与运行前检查

- 推理 baseline 可先用单卡。当前 GRPO 入口默认训练 cuda:0、vLLM GPU 1，不是 DDP/FSDP 多卡训练。
- 双 RTX 5090 是主线的优先试跑候选：各 32 GB，一张训练、一张推理；是否满足默认 microbatch 需要实测，不能把两张卡当作一块 64 GB 显存。
- 若只有单张 RTX PRO 6000 Blackwell Workstation Edition（96 GB），显存虽大，但训练和推理共卡的调度及 vLLM 显存预算尚未接通验证；不要只把两个 GPU 参数都改为 0 就直接批量运行。
- “RTX 6000 Pro”需确认完整型号；不要把旧款 RTX 6000 与 96 GB Blackwell 型号混同。
- 双大显存卡也可沿用一训一推；更多卡可用独立卡对并行不同 seed，不应与单次训练的数据并行混为一谈。当前批量 planner 是顺序启动，不自动分配多组 GPU。
- GPU 规格来源：[RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)、[RTX PRO 6000 Blackwell](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/)。硬件选择是待 smoke run 验证的工程判断，不是吞吐保证。

在实际 Linux GPU 机器上核查：驱动、可见 GPU/显存、Python 3.12、锁定的 Torch/vLLM/FlashAttention 组合、Blackwell 内核支持、模型下载权限、网络、系统内存和磁盘空间。
记录 commit、完整命令、依赖版本、GPU 型号与数量、模型版本、数据子集/seed 和运行开始结束时间。
先验证模型加载、一次真实生成、一次真实 backward、权重同步和 checkpoint 重载，不能用 CPU 测试替代 CUDA 兼容性检查。
不要在看到依赖版本号后就假定 GPU 扩展可运行；以目标机器实际导入及小规模执行为准。

## 1. Prompting baseline：先知道没有训练时的表现

对应主线第 3.4 节，入口和人工检查说明见 [PROMPTING_BASELINES.md](./PROMPTING_BASELINES.md)。

1. 单卡对 16 题做 smoke run，检查生成、停止标记、评分、解析和结果文件。
2. 对完整 GSM8K test 集评测 question_only、r1_zero、r1_zero_three_shot。
3. 固定 temperature=1、top-p=1、max_tokens=512；只给 r1 提示设置保留在输出中的 `</answer>` 停止标记。
4. 报告准确率、格式通过率和三类数量；每种提示人工检查格式对/答案错、格式错/答案错两类各至少 10 条，不足时检查全部并说明。

验收：不仅有 summary，还能解释解析失败、截断和无关续写；这些是真实 base model 输出，不是测试构造的回答。
产物：原始 responses/review JSONL、指标及人工分析。此阶段不更新参数。

## 2. 标准 on-policy GRPO：先调通，再固定对照组

对应第 4.3 节，操作说明见 [GRPO_EXPERIMENT_GUIDE.md](./GRPO_EXPERIMENT_GUIDE.md)。

- 先 1-2 个 rollout batch 检查 CUDA、权重同步、显存、梯度和保存，然后单独完成约 50 步调试运行。
- 正式配置：6400 训练题、至少 1024 验证题、200 rollout batch、每批 32 题 x 8 回答=256 回答。
- train_batch_size=256，gradient_accumulation_steps=32，默认 microbatch 8；整批只更新一次，不是 32 次更新。
- 默认学习率 1e-5、AdamW betas=(0.9,0.95)、weight_decay=0、gradient clipping=1；生成温度 1、最长 512 token。
- 每 10 批验证、每 40 批保存回答；保存频率与磁盘空间一起规划。
- 确认调试证据合理后，用 seed 0/1/2/3 各跑一次正式 200 步；每次从同一 base 权重开始，不接着上个 seed 训练。

验收：四条完整曲线、均值与跨种子波动、训练前后同题回答；讲义要求最终验证准确率跨 seed 平均至少 25%，没有达标应如实记录并排查，不能保证一次运行就达标。
记录 loss、裁剪前 grad norm、token entropy、训练/验证 total 与 format reward、验证回答长度。
奖励接近零不自动表示 bug；持续全零时先查原始回答、评分器、截断和同步，不直接开启大规模 sweep。

## 3. 学习率 sweep：选一个后续共同配置

对应第 4.3 节 grpo_learning_rate。当前 CLI 已支持学习率，无需重写核心算法。

- 至少覆盖默认值两侧，例如 3e-6、1e-5、3e-5；这是实验候选，不是已调好的最优值。
- 先用少量 seed 筛查发散，再依据标准 GRPO 的种子方差补重复；单 seed 筛选不作为正式优劣结论。
- 正式比较仍跑相同 200 步，固定其他参数，画最终验证奖励 vs 学习率；发散运行也记录，不挑选性删除。
- 若采用新的学习率，后续方法都用它，并补齐该学习率下的四 seed 标准 GRPO 对照；相同配置已完成的运行可复用。

## 4. Prompt 训练消融：当前还缺实验配置接线

对应第 4.3 节 grpo_prompt_ablation：增加 question_only、r1_zero_three_shot 训练，与 zero-shot r1_zero 对比，使用若干 seed。
当前训练 CLI/config 只允许 r1_zero；先补提示选择、对应 reward_fn、停止条件和配置记录的接线及测试，再跑实验。
不要把“prompting 推理 baseline 能运行三种提示”误认为“GRPO 训练已支持三种提示”。
固定选定学习率和其他超参数，建议复用 seed 0/1/2/3，比较均值、方差、格式、熵和长度，并说明 parser 差异。

## 5. On-policy 算法变体：当前核心已有，完整实验入口待补

对应第 5.4 节。四种方法各 4 seed，与同学习率标准 GRPO 对照：

| 方法 | baseline | advantage_normalizer | loss_normalization |
| --- | --- | --- | --- |
| 标准 GRPO | mean | std | sequence |
| GRPO_constant | mean | std | constant |
| Dr. GRPO | mean | none | constant |
| RFT | none | none | constant |
| MaxRL | mean | mean | constant |

常数 normalizer 固定为训练回答批大小 x max_tokens，默认 256 x 512；不随零 advantage 剪枝后的有效样本数改变。
需先把上述参数及 normalizer 接入完整 config/CLI、日志和实验汇总兼容性检查，不只修改底层 train_step。
比较方法时不更换模型、数据、prompt 或 seed 集；同学习率下某法较差不证明充分调参后也较差。

## 6. Off-policy：当前已有四方法实验入口

对应第 6.4 节，见 [OFF_POLICY_EXPERIMENT_GUIDE.md](./OFF_POLICY_EXPERIMENT_GUIDE.md)。

| 方法 | importance_reweighting_method | cliprange | 正式 seed 数 |
| --- | --- | --- | --- |
| offpolicy_naive | none | 不使用 | 4 |
| offpolicy_noclip | noclip | 不使用 | 4 |
| offpolicy_clip | grpo | 0.2 | 4 |
| offpolicy_gspo | gspo | 3e-4 | 4 |

统一 rollout=256、train_batch=8、accumulation=1：每批 rollout 切为 32 批，各更新一次。
需要重加权时，任何更新前缓存 old_log_probs，32 次更新内不刷新；不是把同批数据反复训练 32 个 epoch。
每法先少步 smoke，再跑正式 200 个 rollout batch；每次正式运行累计 6400 次 optimizer update。
其他参数与选定的 on-policy 对照固定。四法共 16 次正式运行，可复用已有 4 次 on-policy 对照。
分别按 rollout 批数、optimizer updates、累计循环耗时画图；额外记录进程端到端时间，不能把循环耗时称作全部运行时间。
比较奖励、稳定性、跨 seed 方差与 clip fraction；GRPO 是 token 单位，GSPO 是 response 单位，比例不能不加说明直接等同。
none/noclip 没有裁剪，不应将缺失 clip 指标解释成零裁剪率。

## 7. 自定义 estimator 与最后交付

第 7 节是主线交付，不是可选 CISPO 扩展。只改变已有方法的一处，先写直觉/推导，再实现测试并与对应对照多 seed 比较。
具体选择等标准实验和消融结果后再定；本路线不预先声称任何新方法有效。
理论讨论、人工样例分析、图表与方法限制需要整理到 writeup.pdf；按提交脚本生成 code.zip。
所有自动 report 中的 Analysis pending 都需人工补齐，不用合成日志填实验结论。

## 8. 可选 SFT / DPO：独立第二条路线

不依赖主线 GRPO 实验结果，也不阻塞主线。当前训练入口可 CPU 端到端运行，真实 Llama 8B/GPU 尚未验证。

1. 补齐完整下游评测入口及附加 GSM8K parser，跑 base model 的 MMLU/GSM8K/AlpacaEval/Safety 对照，固定模板和评审配置。
2. SFT：单 epoch、context 512、有效 batch 32，默认 lr=2e-5、3% warmup/cosine、weight_decay=0.1、grad clip=1，保存模型/tokenizer，记录验证 loss 和曲线。
3. 用同一套下游评测评 SFT，再从 SFT 权重初始化 DPO 当前/冻结参考模型。
4. DPO：HH 单轮数据、固定验证划分，单 epoch、RMSprop lr=1e-6、beta=0.1、有效 batch 64，保存最佳验证准确率 checkpoint。
5. 比较 base/SFT/DPO：AlpacaEval winrate 和 length-controlled winrate、SimpleSafetyTests、MMLU/GSM8K 能力变化。

8B 全参数训练不能只按约 16 GB 的 BF16 权重预算，还需梯度、优化器状态、激活、logits；DPO 另有冻结参考模型。
单 5090 不作为当前全参数 8B 代码的默认方案；96 GB 级设备是后续实测候选，不保证长 HH 样本一定可容纳。
当前 SFT/DPO 未接 FSDP，多张小卡不会自动汇总显存；LoRA/量化/分片需要另行实现验证并注明不同于原全参方案。
70B 自动评审只在这些可选下游评测阶段安排，不是主线数学 RL 或 DPO loss 训练的依赖。

## 执行优先级与预算记录

第一轮只做：环境验收 -> 16 题推理 -> 完整 prompting baseline -> 50 步 GRPO 调试 -> 4 seed 标准 GRPO。
之后按讲义补学习率、prompt 消融、on-policy 四变体、off-policy 四法、自定义 estimator。
已有可运行的 off-policy 入口可以提前做小规模技术检查，但不能因此跳过尚未完成的主线消融交付。
仅标准 GRPO + on-policy 四变体 + off-policy 四法就有 4 + 16 + 16 = 36 次正式训练；还不含 sweep、prompt、自定义方法和失败重跑。
讲义标注的 2/4/4/8/8 B200 hrs 对应标准、学习率、prompt、on-policy 变体、off-policy 的参考预算，合计 26；不包含自定义方法等，不能直接换算成 5090 小时或租用价格。
以目标机器 smoke run 的生成、训练、验证、保存分项时间估算，再加初始化与重跑余量；先测后排程，不承诺未知硬件速度。
权重、原始回答和大日志放 results/（不进 Git）；提交可追溯的小型汇总、图表和分析。

本次验证：753 passed, 1 skipped（排除 test_metrics.py），另 MMLU 2 passed；GRPO dry-run 与 16 条 off-policy 命令预览成功。
跳过的双 CUDA 测试和未实现的附加 GSM8K parser 不冒充通过；GPU 有效性、速度与准确率仍需真实测量。
