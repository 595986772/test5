---
name: project-fdedge-current-state
description: "FDEdge/MOFD v5 训练 pipeline 当前已应用的 3 项修复 + 监控日志, 是后续讨论的起点"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

截至 2026-05-15, FDEdge-main (D:/python_project/实验版5/FDEdge-main) 的 v5 训练 pipeline 已经应用了 3 项耦合修复, 是后续讨论的默认起点:

## 修复 1: 奖励通道归一化 (env 内)
`mofd_environment.py:33-34, 52-53, 197-198` 加了 `delay_scale=0.05, energy_scale=0.25` 参数, 在 `step()` 里 `r_T = -delay * delay_scale`, `r_E = -energy * energy_scale`. 量级估计: delay≈20s×0.05≈1.0, energy≈4J×0.25≈1.0, 两通道对称. **delay/energy 返回原值不变, 不影响 Pareto / HV 计算**, 只影响 reward 信号. 详见 [[project-fdedge-reward-scaling]].

## 修复 2: 删 αT/αE, 走真凸组合
`mofd_main.py:1001-1007` cfg 默认 `alpha_T=1.0, alpha_E=1.0` (旧值 0.25 会让能耗权重被打 4 折). `mofd_omega_drift_main.py:224` 和 `mofd_omega_shift_main.py:58` 同步对齐. `mofd_v5.py` 里的 `alpha_vec=[1,1]` 退化为乘 1, 等价于真凸组合 `r = ω·r_T + ω·r_E`. **回滚方式**: 把 alpha_E 改回 0.25 即可复现旧行为做消融.

## 修复 3: target_entropy -1.0 → 0.5
旧值 -1.0 是不可达负数 (离散动作 H ∈ [0, log(6)≈1.79]), 导致 SAC 永远在压熵 → α 单调指数下降, epoch 6-7 崩到 0, 之后 actor 完全失去探索, C0/C2-passive/C2-aware 三种方法 drift 评测几乎完全重合 (delay 差距 ±1.5s 噪声级). 修复后 5 处: `mofd_main.py:1076`, `mofd_omega_drift_main.py:230`, `mofd_omega_shift_main.py:64`, `smoke_test_v5.py:46`, `mofd_v5.py:152`. 修复后 smoke (5 epoch) 验证 α 先升后降找稳态 (0.078→0.186→0.228→0.193→0.119), SAC 自调节恢复正常.

## 加了监控日志
`mofd_main.py` 训练循环现在每 epoch 记录 6 条曲线: `c_loss, a_loss, alpha (= exp(log_alpha)), H, sigma_T, sigma_E`. 保存为 `<results_dir>/mofd_monitor_seed<N>.csv` + `mofd_monitor_curves.png` (2×3 子图). 这是后续判断训练是否健康的**主要诊断工具**.

**Why:** 之前 drift 实验在 envelope 时代验证过 C2-aware 在 cyclic 早期漂移有 85% 优势, 但用 v5 重跑后三方法几乎重合 (`results_omega_drift/latest/comparison.txt`). 根因是上面 3 个 bug 叠加, 不是 ω-buffer 机制本身失效.

**How to apply:**
- 用户下一步会跑完整训练 (~4-5 小时单 seed), 跑完给 `mofd_monitor_seed0.csv` + `mofd_pareto_aggregated.csv`;
- 拿到数据后首先检查 α 是否稳定在 [0.03, 0.10] (而不是崩到 0), H 是否稳定在 ~0.5;
- 然后按 [[project-fdedge-opt-baseline]] 里的物理下界对照, 算 energy 覆盖率从 31% (broken) 提升到目标 60%+;
- C1 验收线见 [[project-fdedge-c1-acceptance]].

**关键 ckpt 兼容性**: 新 ckpt (αE=1.0, target_entropy=0.5, 通道归一化 0.05/0.25) **不能与旧代码跨用**. drift_main / shift_main 已经全部对齐到新 cfg, 用新 ckpt 跑这两个实验自洽.

**仍未做的改造 (备用)**:
- A1 信道升级 (Shannon + path loss + shadowing) — 5 行公式, 但会改变实验数值, 需要重跑;
- A2 cloud 模块解耦 — ~3 天, 与 mofd_v5 Emax→action_dim 分离强耦合;
- A3 任务到达 ✅ 已解耦在 `task_generator.py` (random / poisson / trace).
