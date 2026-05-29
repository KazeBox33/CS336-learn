# CS336-learn

This repository tracks my self-study of Stanford CS336: Language Modeling from Scratch.

本仓库用于记录我自学 Stanford CS336「从零实现语言模型」的过程，重点是通过 Assignment 1 打通从原始文本到小型 Transformer LM 训练的完整链路。

## Goals / 学习目标

- Understand the end-to-end language modeling pipeline: raw text -> tokenizer -> token IDs -> Transformer LM -> training -> evaluation -> generation.
- 从工程角度理解大语言模型训练流程：原始语料、Tokenizer、数据序列化、Transformer 模块、训练循环、评估与生成。
- Implement core components by hand instead of treating them as black boxes.
- 通过手写核心组件来理解实现细节，而不是只调用高层框架 API。
- Practice debugging, testing, profiling, and performance-aware implementation.
- 练习测试、调试、性能分析和面向效率的实现方式。

## Current Progress / 当前进度

### Completed / 已完成

- Set up the CS336 Assignment 1 workspace under `assignment1-basics/`.
- 已完成 `assignment1-basics/` 的 A1 环境配置。
- Implemented byte-level BPE tokenizer training with special-token-aware pre-tokenization.
- 已实现 byte-level BPE tokenizer 训练，包括 special token 分割与 GPT-2 regex pre-tokenization。
- Added multiprocessing pre-tokenization and incremental pair-stat updates for faster BPE training.
- 已加入多进程预分词和增量 pair 统计更新，提升 BPE 训练效率。
- Trained a TinyStories BPE tokenizer with vocab size 10,000 and `<|endoftext|>`.
- 已在 TinyStories 上训练 10K 词表 BPE tokenizer，并保留 `<|endoftext|>` special token。
- Implemented BPE `Tokenizer` with `encode`, `decode`, special-token handling, and `encode_iterable`.
- 已实现 BPE `Tokenizer`，支持 `encode`、`decode`、special token 保留和流式 `encode_iterable`。
- Passed tokenizer and BPE training tests.
- 已通过 tokenizer 与 BPE training 相关测试。

Validation / 验证：

```bash
uv run pytest tests/test_train_bpe.py -q
uv run pytest tests/test_tokenizer.py -q
```

### In Progress / 进行中

- Starting PyTorch model components for A1, beginning with a custom `Linear` module.
- 正在进入 A1 的 PyTorch 模型组件实现，当前从自定义 `Linear` 层开始。
- Current focus: understanding `nn.Module`, `nn.Parameter`, weight initialization, and matrix multiplication semantics.
- 当前重点：理解 `nn.Module`、`nn.Parameter`、权重初始化和矩阵乘法语义。

### Next / 下一步

- Finish and test `Linear`.
- 完成并测试 `Linear` 层。
- Implement and test `Embedding`, `RMSNorm`, `SwiGLU`, attention, Transformer block, and Transformer LM.
- 继续实现并测试 `Embedding`、`RMSNorm`、`SwiGLU`、Attention、Transformer Block 和 Transformer LM。
- Return to tokenizer experiments when needed: compression ratio, throughput, OpenWebText tokenizer, and token ID serialization.
- 后续回到 tokenizer 实验：压缩率、吞吐量、OpenWebText tokenizer，以及 token id 序列化。

## Notes / 说明

- This repository is for self-directed learning and implementation practice.
- 本仓库用于自学和实现练习。
- Large local datasets and generated outputs are ignored by git.
- 大型本地数据集和训练输出不会提交到 git。
