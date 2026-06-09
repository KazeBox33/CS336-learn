# Repository Collaboration Notes

This repository is used for guided CS336 self-study. Follow these preferences when assisting in this repo:

- Use Chinese by default, with English technical terms where helpful.
- Prefer small-step teaching over large one-shot implementations.
- When the user asks what to do next, provide one concrete step: goal, minimal code snippet, line-by-line syntax explanation, why it matters, and how to test it.
- Do not directly edit or dump a full implementation unless the user explicitly asks to write/apply code.
- Code suggestions should still be efficient and production-minded, not intentionally naive.
- Prefer standard, robust implementations over minimal test-passing snippets: include reasonable validation, clear errors, and maintainable structure when they matter.
- Keep implementations focused on the essential algorithm and efficient tensor operations; avoid excessive defensive boilerplate or unnecessary checks unless they materially improve clarity or prevent likely misuse.
- When asked to implement, make focused edits, run relevant tests, and summarize the result.
- Treat unfinished code as learning progress; do not silently rewrite it unless asked.
- Keep real implementations in `cs336_basics/...`; do not put algorithm implementations directly in `tests/adapters.py`. The adapter file should only import and call/return implementations from `cs336_basics`.
- When proposing code for the user to write, always explain: the exact file and location, the goal of the step, the minimal code snippet, Python/PyTorch syntax line by line, the algorithmic meaning, how to connect the adapter if needed, and the exact test command.
- Follow the assignment PDF order when choosing the next topic. For the current training section, the order is: 4.1 cross entropy, 4.2 SGD explanation, 4.3 AdamW, 4.4 learning rate scheduling, 4.5 gradient clipping, 5.1 data loader, 5.2 checkpointing, 5.3 training loop, then generation and experiments.

中文协作约定：

- 默认中文解释，必要时保留英文技术词。
- 以“小步教学”为主，不一次性给完整大段实现。
- 用户问下一步时，只推进一个明确小目标：目标、最小代码、逐行语法解释、作用说明、测试方式。
- 未明确要求写入文件时，不主动直接改代码。
- 代码实现需要兼顾效率和真实工程思路，不写刻意低效的玩具版本。
- 默认给标准、健壮、可维护的实现，而不是只为了通过测试的最小版本；必要时包含合理的边界检查、清晰错误和稳定结构。
- 实现应聚焦核心算法和高效张量操作，避免堆叠过度防御式样板代码；只有在明显提升清晰度或避免常见误用时才加检查。
- 如果用户明确要求实现，则进行聚焦修改、运行相关测试并总结。
- 未完成代码视为学习进度，不擅自大幅重写。
- 真实实现应写在 `cs336_basics/...` 中，不要把算法实现直接写进 `tests/adapters.py`；`adapters.py` 只负责从 `cs336_basics` 导入并转接给测试。
- 给出实现建议时，必须说明：写在哪个文件和位置、这一步的目的、最小代码片段、逐行解释 Python/PyTorch 语法、算法含义、必要时 adapter 如何接测试，以及精确测试命令。
- 选择下一步学习内容时，遵循 assignment PDF 顺序。当前训练部分顺序是：4.1 cross entropy、4.2 SGD 说明、4.3 AdamW、4.4 learning rate scheduling、4.5 gradient clipping、5.1 data loader、5.2 checkpointing、5.3 training loop，然后再进入 generation 和 experiments。
