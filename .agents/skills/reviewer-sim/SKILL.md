---
name: reviewer-sim
description: 以顶会 Reviewer 视角审稿,产出 Summary / Strengths / Weaknesses / Rating 报告 + 修改策略建议。当用户说"用 reviewer 视角审一下 / 模拟审稿 / 提前 rebuttal / 找问题" 时调用。
---

# Reviewer Simulation (顶会审稿模拟)

## 审稿人画像 (默认)

- 来自 NeurIPS / ICML / ICLR / INFOCOM area
- 对 MORL / edge computing / Pareto optimization 熟悉
- 严苛但 fair, 区分 **fatal** vs **fixable**
- 关注 community contribution, 不为 incremental work 给高分

## 审稿维度 (依次评估)

### 1. Contribution
- 这个工作给社区**新**的什么? (新方法 / 新视角 / 新基准 / 新经验结论)
- 是否仅是 incremental? 与 GMORL / 既有 MORL 工作的 delta 是什么?
- 卖点 (ω-自适应分配 + buffer 驱动泛化/漂移鲁棒) 是否在 paper 里被清楚论证?

### 2. Soundness
- 实验设计是否回答了 claim?
- 多 seed 数 / 显著性检验 / baselines 是否充足?
- Pareto / HV / 漂移恢复指标定义是否严格?
- ω 已知 vs unknown 的假设是否一致?

### 3. Clarity
- Abstract / Intro 能否让 outsider 一句话明白 contribution?
- 方法图、符号表是否清晰?
- 实验描述是否可复现?

### 4. Significance
- 解决的问题在 community 里有多重要?
- 提升幅度是否值得 publish? (e.g., +0.5% HV vs +8% HV 处理方式不同)

### 5. 致命问题 Checklist
- ω 假设不一致 (方法说 known, 实验偷偷用 oracle)?
- baseline 没调超参,造成不公平?
- 数字 / 图 / 文字三者矛盾?
- 物理下界 / Opt baseline 是否对照过?
- 实验是否覆盖论文主张的所有 setting?

## 输出格式

```
[The Review Report]

Summary
<200 词以内, 客观复述论文做了什么, 不带评价>

Strengths
S1. <具体优点, 引用 Section / Fig / Table 编号>
S2. ...
S3. ...

Weaknesses (Critical)
W1. <最严重的问题, 引用具体位置>
    Severity: Fatal | Major | Minor
    Justification: <为什么这是问题>
    Suggested fix: <作者能在 rebuttal 周期内做的修改>
W2. ...
W3. ...

Questions to Authors (Q1, Q2, ...)
<rebuttal 阶段会问的具体问题, 鼓励作者准备答案>

Rating
- Soundness: <1-5>
- Presentation: <1-5>
- Contribution: <1-5>
- Overall: <Strong Reject / Reject / Borderline / Accept / Strong Accept>
- Confidence: <1-5>

Rationale for Overall
<一段话解释为什么给这个分数>

---

[Strategic Advice for Revision]

P0 (必修, 否则必拒):
- <action>
- <action>

P1 (强烈建议改):
- <...>

P2 (锦上添花):
- <...>

时间预算建议:
- 距 deadline X 天的情况下, 优先级 P0 → P1 → P2, 不要花时间在 P2
```

## FDEdge 项目特定审稿关注点

基于项目 memory 的当前状态,这些是 reviewer **极可能问** 的问题,作者要先想清楚:

1. **ω 假设**: "ω 已知" 是方法核心假设 — paper 里要明确,实验里也要严格 (`[[project_fdedge_paper_direction]]`)
2. **GMORL vs FDEdge delta**: 必须用一段话清楚说明 algorithmic 差异 (`[[project_fdedge_v2_env]]`)
3. **物理下界**: Opt 21 点数据建立了 delay/energy 下界 — paper 必须在主表对照 (`[[project_fdedge_opt_baseline]]`)
4. **B-OPE 否决**: 如果论文里出现任何 B-OPE 类讨论, 注意已知是循环论证 (`[[project_fdedge_bope_premise]]`)
5. **奖励通道归一化**: 实验细节要交代 αT/αE 配比 (`[[project_fdedge_reward_scaling]]`)
6. **C1 验收线**: 主表数字必须满足 C1 四条硬指标 (`[[project_fdedge_c1_acceptance]]`),否则审稿人一眼能看出 cherry-pick

## 反例

- 写"我认为这是 strong accept",没具体证据 — ❌
- 全是 Strengths 没 Weaknesses — ❌ (reviewer 不会这么写)
- Weakness 模糊 "presentation could be improved" — ❌, 要具体到段落
- 评分不一致 (Soundness 5 但说有 fatal flaw) — ❌
