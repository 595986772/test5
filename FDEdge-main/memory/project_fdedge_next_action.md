---
name: project-fdedge-next-action
description: "用户当前正在等的事 — 跑完整训练后给我 csv, 我做诊断"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

## 用户当前在做什么

跑 mofd_main.py 完整训练 (50 epoch, 单 seed, ~4-5 小时). 这是应用了 3 项修复 ([[project-fdedge-current-state]]) 后的第一次完整训练.

## 用户回来时, 我要立刻做的事

用户拿到 `<results_dir>/` 后会贴给我以下 3 个文件 (或路径):
1. `mofd_monitor_seed0.csv` — 6 条监控曲线
2. `mofd_pareto_aggregated.csv` — 21 个 ω 点的 (delay, energy)
3. `mofd_training_curves.png` / `mofd_monitor_curves.png` (可选, 看图直观)

**优先级顺序**:

### 步骤 1: 检查 α 是否稳住 (最关键)
读 `mofd_monitor_seed0.csv` 的 `alpha` 列, 看 50 epoch 是否稳定在 [0.03, 0.10] 区间.
- 如果 α 又崩到 < 0.001 → target_entropy 0.5 还是太低, 改成 0.7 重跑
- 如果 α 稳住 → 进步骤 2

### 步骤 2: C1 验收
按 [[project-fdedge-c1-acceptance]] 算 4 个指标. **必须打印实际数字, 不要凭印象**.
- 4 条全过 → 触发 task #5 完成, 推进 drift/shift
- 1-2 条边缘 → 微调 (alpha_lr 或 cor_lambda) 再跑
- 全部未过 → 通道归一化常数不够, 考虑 EMA 自动归一化

### 步骤 3: 与 Opt 对照
按 [[project-fdedge-opt-baseline]] 的物理下界对比:
- 算 `MOFD_spread_E / Opt_spread_E` 比例 (目标 ≥ 0.60)
- 算 `(MOFD_min_E - Opt_min_E) / Opt_min_E` (现在 230%, 目标 < 80%)

## 现存的待定 task

`TaskList` 里会看到:
- #5 完整训练 + C1 验收 (pending → in_progress 当用户给数据时)
- #7 调参循环 (条件触发, 视 C1 验收结果)

其他 task (#6 #8 #9 #10) 都已 completed.

## 重要警告

- **不要凭印象答数据**. 任何"19 个点扎堆"、"跨度 0.02J"这种具体数字必须先 Read 原始 csv. 详见 [[feedback-data-rigor]].
- **不要直接建议改环境参数 (f_range, num_tasks_max)**. 物理瓶颈论已用 Opt 数据排除, 真正的修复方向是训练侧.
- **不要去重新讨论 Set-Transformer (C3)** — 用户已明确删除, 不再是卖点. 现在 contribution 骨架是 C1/C2/C4/C5. 详见 [[project-fdedge-core-appeal]].
