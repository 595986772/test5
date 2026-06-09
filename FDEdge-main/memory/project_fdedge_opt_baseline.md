---
name: project-fdedge-opt-baseline
description: "Opt 穷举最优 baseline 给出的物理下界数据, 用于判断 MOFD 离最优有多远"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

`baselines/Opt/opt_mofd_main.py` 已经跑出 21 个 ω 点的物理下界. 数据在 `results/opt_pareto_aggregated.csv` + `results/opt_summary.txt`.

## 关键数字 (基于 csv 实际读取, 不是估算)

**Opt Pareto 前沿** (21 行, 按 ω_T 从 0 到 1 排列):
- 行 1 (ω≈(0,1)): delay=454.91s, energy=**0.90J** ← 能耗最优极端点
- 行 11 (中点): delay=24.93s, energy=3.95J
- 行 21 (ω=(1,0)): delay=**24.93s**, energy=4.12J ← 延迟最优极端点

**关键比例**:
- Opt delay 最低值 = 24.92s (后 11 个点都扎堆在 24.91-24.93, **跨度 0.02s**)
- Opt energy 最低值 = 0.90J, 最高值 = 4.12J, **跨度 3.22J**
- Opt 自己也扎堆 — 证明延迟侧扎堆是**物理瓶颈**, 不是任何算法的 bug

## MOFD v5 (broken training 时) 对比

- MOFD delay 最低 ~27.5s, **离 Opt 仅 10%** ✓ 延迟侧已接近下界
- MOFD energy 最低 = 2.97J, **离 Opt 230%** ✗ 能耗侧巨大空间
- MOFD energy 跨度 1.00J / Opt 跨度 3.22J = **覆盖率 31%**

**Why:** 用户问"是否要提算力解决延迟扎堆", 我用 Opt 数据证明: 延迟侧扎堆是物理瓶颈 (Opt 都扎堆), 提算力没意义; 真正的问题在**能耗维度覆盖率只有 31%**, 这是训练 bug (αE=0.25 + α 崩塌) 造成的, 不是环境问题. 修复后预期能耗覆盖率拉到 60%+ 才算 C1 成立.

**How to apply:**
- 当用户讨论"延迟为什么这么高"时, 用 Opt = 24.92s 这个下界回答: 物理上没有任何算法能更低;
- 当讨论"是否要改环境参数 (f_range, num_tasks_max)"时, 用 Opt 数据反驳: MOFD 已经接近延迟下界, 应该集中精力把能耗侧从 31% 拉到 60%+;
- 任何论文叙事里写"MOFD 接近 oracle 性能" 都要带上这两个数字 (delay 28→25 vs Opt 24.92; energy 1.00 J 跨度 vs Opt 3.22 J 跨度);
- 如果修复完整训练后能耗跨度从 1.00J 提到 ≥ 2.0J (覆盖率 60%+), C1 就立得住; 如果还在 1.5J 以下, 说明 actor 仍然没学到能耗侧策略.

**Opt 跑的 cfg**: Emax=6, num_tasks_max=50, bit_range=(10,40), time_slots=100, f_range=(10,40), 21 个 ω × 3 episode 取均值. **跟 MOFD 评估完全一致**, 所以可直接对照.
