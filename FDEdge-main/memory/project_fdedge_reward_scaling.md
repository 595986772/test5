---
name: project-fdedge-reward-scaling
description: FDEdge v5 项目中 alpha_T/alpha_E 与 r_T/r_E 通道量级不匹配是已识别的核心问题
metadata: 
  node_type: memory
  type: project
  originSessionId: ec0489a0-b84f-404c-a0d1-80d34e21d436
---

FDEdge-main (mofd_v5) 项目的延迟/能耗权衡当前设置 `alpha_T=1.0, alpha_E=0.25` 配合**原始** `r_T=-delay (秒)`, `r_E=-energy (焦耳)`，在 ω=(0.5,0.5) 时延迟通道贡献 ≈10、能耗通道 ≈0.5——延迟在 reward 里大 20 倍，actor 梯度被延迟主导，训练曲线表现为能耗单调爬升、HV 后期回落、Pareto 前沿塌成一团。

参考项目 `Generalizable-Pareto-Optimal-Offloading-with-Reinforcement-Learning-in-Mobile-Edge-Computing-main` 的做法是**在 env 内做"通道归一化常数"**：`reward_dt = -delay * 0.01; reward_de = -energy * 5`，让两通道都落在 O(1) 量级，再用单标量 `w∈[0,1]` 做真凸组合 `w*r_t + (1-w)*r_e`，**没有独立的 αT/αE**。同时 64 个并行 env 在 [0,1] 上等距取 w，preference 还作为 2-d 向量喂进网络 `bais_network1` 做条件化。

**Why:** 这条诊断不是凭代码风格猜的——`results/mofd_20260513_220111` 的训练曲线 (HV 峰值 392→末段 260, energy 3.4→4.6 J 单调爬升, Pareto 前沿 19/21 个点堆在 (27.5–28, 3.95–3.97)) 和 `mofd_summary.txt` 中 final HV 仅 64.16 是直接证据。

**How to apply:** 后续讨论 FDEdge 训练问题时，**优先怀疑通道未归一化**而不是先调 lr/网络结构；改造路径已经讨论过——A) env 里加 DELAY_SCALE=0.05, ENERGY_SCALE=0.25；B) 删掉 αT/αE 用真凸组合；C) ω 用 uniform 网格代替 Dirichlet。PopArt 在 critic loss 里有归一化但 actor 的 `_scalarize_q` 仍用原始 `omega*alpha_vec`，所以 PopArt 救不了 actor。

相关文件: `mofd_environment.py:163-191` (step/reward 定义)，`mofd_main.py:452` (r_scal 标量化)，`mofd_v5.py:219-228, 350, 382` (αT/αE 使用点)。
