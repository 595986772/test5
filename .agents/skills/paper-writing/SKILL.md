---
name: paper-writing
description: 论文写作总索引。当用户提到"写论文 / 改论文 / 润色 / 翻译 / 投稿 / 图表 / 实验分析 / Reviewer 视角"等论文写作相关请求时,先读本 skill 选择合适的子 skill。本项目专攻 FDEdge (delay-energy-accuracy 三目标 + ω-自适应分配 + GMORL baseline),所有写作需保持术语一致。
---

# Paper Writing Skill Suite (FDEdge 项目)

源参考: github.com/Leey21/awesome-ai-research-writing
适用论文: FDEdge — Generalizable Pareto-Optimal Edge Offloading with ω-adaptive Allocation
目标会议级别: NeurIPS / ICML / ICLR / INFOCOM / TMC

## 项目固定术语 (写作时必须统一,不允许混用)

| 中文 | English (固定写法) |
|---|---|
| ω-自适应分配 | ω-adaptive allocation |
| 偏好向量 | preference vector ω |
| 多目标强化学习 | multi-objective reinforcement learning (MORL) |
| Pareto 前沿 | Pareto frontier (不写 Pareto front / Pareto-front 混用) |
| 超体积 | hypervolume (HV) |
| 三目标 | three-objective {delay, energy, accuracy} |
| 时延 / 能耗 / 精度 | delay / energy / accuracy (小写,无连字符) |
| 边缘卸载 | edge offloading |
| 非平稳环境 | non-stationary environment |
| Test-time adaptation | test-time adaptation (TTA, 不写 test time) |
| 经验回放缓冲区 | replay buffer |
| 物理下界 | physical lower bound |
| 漂移恢复 | drift recovery |
| GMORL | GMORL (baseline,首次出现需展开 Generalizable MORL) |

## 子 skill 触发对照表

| 用户意图 | 调用 skill |
|---|---|
| 把中文段落写成 LaTeX 英文 | `zh-to-en-latex` |
| 把已有英文段润色到顶会水准 | `en-polish` |
| 段落"AI 味"重,需自然化 | `dehumanize` |
| 字数超 / 不够,要缩写或扩写 | `compress-expand` |
| 投稿前最后逻辑/术语一致性扫一遍 | `logic-check` |
| 给已有图/表生成 caption | `caption-gen` |
| 不知道实验数据该用什么图表 | `fig-recommend` |
| 把原始实验数据写成分析段 | `exp-analysis` |
| 整篇 / 整节用 Reviewer 视角审稿 | `reviewer-sim` |

## 写作通用红线 (所有子 skill 必须遵守)

1. **不编造数字**: 实验数字必须来自用户提供的原始文件/日志,不允许凭印象写 (参见 `[[feedback_data_rigor]]`)。引用 HV/delay/energy 等具体数值前先 Read 文件核对。
2. **不堆术语**: 拒绝 "leverage / delve / showcase / paradigm shift" 等大模型口癖词,改用 plain 词。
3. **不列点**: 论文正文段落不允许 markdown 列表 / 无序号要点。若原文是列表,改写为连贯段落。
4. **不堆格式**: 不主动加 `\textbf`、emoji、`✅` 等装饰;`\paragraph{}` 只用于 实验分析章节。
5. **被动 → 主动**: 优先主动语态,被动仅用于强调对象。
6. **属格替换**: 写 "performance of METHOD" 而非 "METHOD's performance"。
7. **特殊字符 LaTeX 转义**: `95%` → `95\%`,`model_v1` → `model\_v1`,`R&D` → `R\&D`。
8. **时态**: 方法描述统一 present tense;只有历史工作引用用 past tense。
9. **缩写规范**: 首次出现展开 (e.g., "hypervolume (HV)"),之后用缩写。
10. **数字与单位**: 阿拉伯数字 + 单位之间有空格 (`24.92\,\text{s}`),量级用 SI 制。

## 推荐工作流

- **从中文草稿到投稿**: `zh-to-en-latex` → `en-polish` → `dehumanize` → `logic-check`
- **新写一节方法**: `exp-analysis` (如含实验) + `en-polish`
- **画图配文**: `fig-recommend` → 出图 → `caption-gen`
- **投稿前**: `reviewer-sim` 整篇过一遍,按报告修改后再 `logic-check`
