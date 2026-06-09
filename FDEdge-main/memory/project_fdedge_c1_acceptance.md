---
name: project-fdedge-c1-acceptance
description: "C1 (ω-自适应分配) 卖点的验收线, 是判断\"修复是否成功\"的唯一硬指标"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

C1 卖点 ("系统按 ω 自适应分配延迟和能耗") 的验收线. 完整训练跑完后**必须**对照这张表判定是否通过.

## 4 条验收指标

| # | 指标 | 通过线 | broken 时实测 | 物理上限 |
|---|---|---|---|---|
| 1 | Spearman ρ(ω_E, delay) | ≥ +0.70 | 估计 < +0.3 | +1.0 (Opt) |
| 2 | Spearman ρ(ω_E, energy) | ≤ -0.70 | 估计 > -0.3 | -1.0 (Opt) |
| 3 | `spread_energy / mean_energy` | ≥ 0.30 | 0.26 (1.00J / 3.83J) | 0.84 (Opt: 3.22/3.85) |
| 4 | 训练末段 30 epoch energy 变异系数 | ≤ 0.05 | 单调爬升 (broken) | — |

## 如何计算 (基于已有产出的 csv)

读 `<results_dir>/mofd_pareto_aggregated.csv` (21 行 × 2 列, delay/energy), 然后用 `build_preference_set(21)` 反推每行对应的 ω_T (line 0=0.0, line 20=1.0).

```python
from scipy.stats import spearmanr
import numpy as np
pareto = np.loadtxt('omega_resp_seed0.csv')
omega_T = np.linspace(0, 1, 21)
omega_E = 1 - omega_T
r1 = spearmanr(omega_E, pareto[:, 0]).correlation  # delay vs ω_E
r2 = spearmanr(omega_E, pareto[:, 1]).correlation  # energy vs ω_E
spread_E = pareto[:, 1].max() - pareto[:, 1].min()
mean_E = pareto[:, 1].mean()
print(f"ρ_delay={r1:.3f}, ρ_energy={r2:.3f}, spread_E/mean_E={spread_E/mean_E:.3f}")
```

## Why

之前我 (Claude) 一度提过 "spread_energy / mean_energy ≥ 0.25" 验收线, 但用户指出 broken 时已经是 0.26 — 旧验收线太宽. **新阈值 0.30 是基于 Opt 物理上限 0.84 的中间点**.

C1 验收**必须有 Opt 对照**才有意义. 不要只看 MOFD 自己的绝对值, 一定要算 "MOFD spread / Opt spread" 这个比例.

## How to apply

- 用户跑完 50 epoch 完整训练后, 拿到 `mofd_pareto_aggregated.csv`, **第一时间跑上面这段代码**, 报告 4 个指标;
- 通过/未通过的判定:
  - 4 条全过 → C1 成立, 推进 C2 (drift/shift);
  - 1-2 条未过 → 可能需要再调 target_entropy (0.5 → 0.7) 或 alpha_lr (3e-4 → 1e-4);
  - 3+ 条未过 → 通道归一化常数 (delay_scale/energy_scale) 不够好, 考虑用 EMA 自动归一化 (mofd_v5 里已经有 PopArt 但只作用于 critic loss, 没作用于 actor 的 _scalarize_q);
- 不要凭印象判定 "看起来不错" — **必须四个数都打印出来**.

相关引用: [[project-fdedge-current-state]], [[project-fdedge-opt-baseline]], [[feedback-data-rigor]].
