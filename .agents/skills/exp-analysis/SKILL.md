---
name: exp-analysis
description: 把原始实验数字 / 表 / 日志写成可直接放进论文 Experiments 章节的分析段落。当用户给一张表 / 一组数据并说"帮我写分析段 / 写讨论 / experimental analysis / 解读这表"时调用。
---

# Experiment Analysis (实验数据 → 分析段落)

## 铁律 (违反即作废)

1. **数字必须严格来自输入**: 任何具体百分比、HV 值、delay/energy 数字,必须能在用户提供的表/数据/文件里逐一找到。**不允许凭印象生造**。参见 `[[feedback_data_rigor]]`。
2. **不夸大**: 报告的就是"X 比 Y 高 8.4%",不要写"显著超越 / 大幅领先"(除非有显著性检验支撑)。
3. **不只是 restate**: 不能只把表里的数字念一遍 (reviewer 会扣分),必须有 trend / comparison / explanation。
4. **结论先行**: `\paragraph{<Core Finding>}` 一句话,然后展开证据。

## 分析段落标准结构

```latex
\paragraph{<核心结论, 一短句>.}
<证据 1: 量化对比, 引数字>. <证据 2: 趋势 / 跨设置一致性>.
<解释: 为什么会这样, 与方法设计的哪条 design choice 对应>.
<可选: 例外 / 边界, 提升可信度>.
```

### 一段好分析的四要素

1. **结论 (Claim)** — 一句话, 不超过 20 词
2. **证据 (Evidence)** — 引用 ≥2 个数字, 跨 ≥2 个 setting/method
3. **解释 (Explanation)** — 把现象归因到方法机制 (e.g., "due to ω-conditioned critic separating objectives")
4. **节制 (Limitation)** — 承认例外或边界 (e.g., "the gain diminishes when ω is highly skewed toward energy")

## 写法对照

### ❌ 差: 只 restate
> "Our method achieves 78.3 HV, GMORL achieves 72.1, and the random baseline achieves 65.0. This shows our method is better."

### ✓ 好: claim + evidence + explanation
> "\paragraph{ω-adaptive Allocation Improves Hypervolume.} FDEdge attains 78.3 HV, exceeding the GMORL baseline (72.1) by 8.6\% and the random scheduler (65.0) by 20.5\%. The advantage holds across all five evaluated preference vectors (Table 2, rows 1--5), with the largest gap (+11.2\%) observed under balanced ω = (1/3, 1/3, 1/3). We attribute this to the preference-conditioned critic that decouples the three objectives during value estimation, allowing the policy to specialize without sharing gradients across orthogonal trade-offs."

## 项目专属分析模板

### 模板 A — Pareto / HV 对比

> \paragraph{Pareto Quality across Preference Vectors.} FDEdge's hypervolume is `X.X`, higher than GMORL (`Y.Y`) and `Baseline-Z` (`Z.Z`) on the {delay, energy, accuracy} space. The improvement is consistent across `N` ω samples drawn uniformly from the simplex (Table T, Fig. F). We attribute the gain to the `<mechanism>`, which `<causal sentence>`. The gap shrinks to `<small number>` when ω is concentrated on the accuracy axis, where the lower bound is dominated by model capacity rather than allocation policy.

### 模板 B — 漂移恢复 (drift recovery)

> \paragraph{Recovery from ω-shift.} After an abrupt ω-shift at step `T`, FDEdge re-reaches `X\%` of its pre-shift HV within `K` episodes, compared with `M` episodes for GMORL. The replay buffer with test-time adaptation absorbs the new preference within `<bounded time>`, whereas GMORL must re-train its scalarization head. The asymptotic HV remains `<gap>` below pre-shift, reflecting `<honest explanation of limitation>`.

### 模板 C — 消融

> \paragraph{Effect of <Component>.} Removing `<component>` reduces HV from `X` to `Y` (\,$-Z\%$\,), with the loss concentrated on the `<objective>` axis. This confirms that `<component>` contributes primarily to `<the objective>`, consistent with the design intention stated in Section `<S>`.

## 输出格式

```
[LaTeX]
<分析段落, 含 \paragraph{} 起首>

[中文回译]
<...>

[数字核对]
- 引用 78.3 HV → 来源: <用户提供的文件/表的位置>
- 引用 +8.6% → 来源: <...>
- (列出每个具体数字的来源, 用户可逐项验证)
```

**如果输入数据不足以支撑某个结论 (e.g., 没有 5 个 ω 的数据却让你写 "across all 5"), 直接告诉用户哪里数据不够, 不要补全。**
