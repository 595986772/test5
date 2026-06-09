---
name: logic-check
description: 投稿前最终一遍逻辑/术语/一致性扫描。只报致命问题:逻辑矛盾、术语漂移、严重 Chinglish/语病。当用户说"逻辑检查 / 投稿前过一遍 / 一致性检查 / final pass" 时调用。
---

# Logic Check (投稿前最后一道)

## 检查门槛 (重要)

- **假设原文已多轮修订**,容忍度高,只标 "fatal" 与 "non-fatal but obvious" 问题。
- **不做润色**: 看到不够地道但不致命的句子放过,留给 `en-polish`。
- **不做去 AI 味**: 看到模板化但不矛盾的句子放过,留给 `dehumanize`。
- 若整篇 / 整节扫完无实质问题,直接输出: `[检测通过,无实质性问题]`

## 必报项 (Fatal — 必须修)

1. **逻辑矛盾**:
   - 同一变量两处定义不同
   - 摘要说 "+5% accuracy",实验表实际是 "-2%"
   - 方法 A 比 B 好,但讨论又说 B 比 A 好
2. **术语漂移**:
   - "Pareto frontier" 与 "Pareto front" 混用
   - "ω" 写成 "omega" 与 "w" 混用
   - "GMORL" 与 "G-MORL" 混用
3. **citation 错配**: `\cite{Smith2020}` 引到的工作不是讨论的那个方法
4. **数字对不上**: 文中数字与表中数字、表中数字与图中数字不一致
5. **未定义符号**: 公式里出现 ε / λ / τ 但前后文未定义
6. **复数 / 时态混乱**: 同一方法忽 present 忽 past
7. **致命 Chinglish**: 句子读不懂或歧义

## 非必报项 (Skip)

- 风格偏好 ("we" vs "the authors")
- 略 informal 但语义清楚的句子
- AI 味痕迹但不矛盾
- 标点偏好 (Oxford comma 等)

## 输出格式

### 情形 A — 无问题
```
[检测通过,无实质性问题]
```

### 情形 B — 有问题
```
[逻辑检查报告]

Fatal:
1. Section 3.2 定义 ω ∈ [0,1]^3 且 Σ=1; 但 4.1 公式 (5) 用 ω ∈ R^3 无归一化约束 → 矛盾
2. Abstract 报告 "HV +8.4%"; Table 2 实际为 "+8.1%" → 数字不一致
3. "Pareto frontier" / "Pareto front" 在 Intro/Method/Exp 三处混用 → 统一为 "Pareto frontier"

Non-fatal but worth fixing:
4. 公式 (3) 出现 τ 但未定义 (推测是 temperature, 需在前文给出)
5. Section 4.3 时态从 present 切到 past 再切回 → 统一 present

(不修也能投, 但 reviewer 大概率会问)
```

## 输入要求

- 用户应粘贴整节或整篇 LaTeX 源码,或给出文件路径用 Read 工具读取
- 如用户只贴一段,提示对方:"逻辑检查通常需要整节或整篇上下文,只贴一段我只能做局部一致性扫描"
