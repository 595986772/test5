---
name: en-polish
description: 已有英文段落润色到 NeurIPS/ICML/ICLR 级语言质量。修语法、改用词、改句式、改时态;不改变事实和结论。当用户说"润色 / polish / 改英文 / 把这段写得更学术 / refine"时调用。
---

# English Refinement (顶会语言质量)

## 修订门槛 (重要)

- **高门槛**: 若原文已经清晰、地道、无错,只在必要处微调。不要为了显示工作量而强改。
- 若原文已达水准,**输出原文** + 一行 "Already at conference quality; no substantive change needed."

## 必改项 (硬性)

1. 语法、拼写、标点错误
2. 时态错乱 (方法描述非 present tense)
3. 主谓不一致、悬垂修饰、指代不清
4. 中式英语句式 (e.g., "Because...so..." 双连接)
5. 属格滥用 (METHOD's X → X of METHOD)
6. 缩约 (it's, don't, won't) → 全展开
7. 列表化的正文 → 改为连贯段落
8. LaTeX 特殊字符未转义

## 应改项 (建议修)

1. 大模型口癖词 → plain 词 (leverage, delve, tapestry, intricate, robustly, seamlessly, paradigm shift, pivotal, showcase)
2. 冗余词 ("in order to" → "to"; "due to the fact that" → "because"; "a number of" → "several / many")
3. 弱动词 + 名词 (e.g., "perform an analysis" → "analyze")
4. 长复合句拆分,或短句合并形成节奏

## 不改项

1. 作者特意的修辞 / 强调 (即使略 informal)
2. 已有的格式选择 (`\textbf`、italics)
3. 公式 / 数字 / 实验结果 — 数值一字不动
4. 已经合理的学术行话

## 输出格式

```
[LaTeX]
<润色后的英文,LaTeX 就位>

[中文回译]
<逐句意译回中文供审校>

[修改日志]
- 原: "<原句>"
  改: "<改后>"
  原因: <语法 / 用词 / 时态 / 句式>
- (列出所有非琐碎修改; 琐碎的拼写错误可合并为 "拼写: typo×N")
```

## FDEdge 项目术语锁定

严格使用 `paper-writing` skill 中的术语表。润色过程中若发现术语用错 (e.g., "Pareto front" 而非 "Pareto frontier", "test time adaptation" 而非 "test-time adaptation"),必须列入修改日志统一。
