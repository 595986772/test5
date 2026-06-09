---
name: project-fdedge-core-appeal
description: FDEdge/MOFD 项目去掉 Set-Transformer 后的核心诉求与 contribution 骨架
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

FDEdge-main / MOFD 项目（mofd_v5 为当前基线）的核心诉求：**ω-条件化反馈扩散策略 + ω-buffer 模块**，同时支撑三个场景——

1. **静态多 ω**：系统按用户不同 ω 自适应分配延迟/能耗，铺出有意义的 Pareto 前沿；
2. **ω 泛化**：训练见过的 ω ≠ 部署 ω，模型仍稳定；
3. **ω 漂移**：部署中 ω 沿时间变化（sudden/gradual/cyclic），buffer 让策略快速 warm-start。

**用户已明确：Set-Transformer (原 C3) 已删除，不再是卖点。** 当前 contribution 骨架是：
- C1: CMO-MDP + ω-自适应分配
- C2: ω-buffer 反馈扩散同时支撑泛化和漂移
- C4: H-MCSS (feedback / prior / random 三源候选 + Critic 选优)
- C5: 严格评测协议（固定 ref HV + IGD + 偏好一致性 + 零样本外推）

**Why:** 用户 2026-05-14 明确纠正：之前 introduction.md 里的 C3 已经不算了；他要的是"多 omega 环境 + buffer 驱动的泛化&漂移鲁棒"，不是结构创新。

**How to apply:**
- 讨论改进时优先围绕 C1/C2/C4/C5，别再提 Set-Transformer / 主干结构；
- 通道归一化（αT=1.0/αE=0.25 vs 原始 r_T/r_E）不是工程修复——它直接卡死 C1 的"按 ω 自适应分配"论点（前沿塌成横线，能耗跨度只有 0.02 J）；详见 [[project-fdedge-reward-scaling]]；
- 漂移卖点（C2）和通道量级正交，可独立验证；
- 关键 ablation 是 w/o buffer（`use_omega_buffer=False`），证明 buffer 是泛化&漂移收益的真实来源。

相关文件: `mofd_v5.py` (当前主模型), `mofd_environment.py` (CMO-MDP), `mofd_omega_drift_main.py` (漂移实验入口), `introduction.md` 里 C3 段落已过时。
