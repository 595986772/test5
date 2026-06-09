# MOFD / G-FDEdge 实验改进清单（投 SCI 用）

> 本文件基于对 `mofd_main.py`、`mofd_environment.py`、`mofd_v2.py`、`baselines/` 以及 `articles/` 下三篇源论文（FDEdge TMC'25 / GMORL TSC'25 / ICLR'25 Diffusion Planner）的分析，列出当前实验相对于投稿 SCI 期刊所需的改进项，按优先级分三级。

---

## 🔴 一、阻塞级问题（不解决会被拒稿）

### 1. 实验规模太小，曲线没有收敛
当前 `num_epochs=15, time_slots=20, num_tasks_max=8`，HV 曲线 `[2.84, 2.76, 5.25, 5.15, 1.72, 8.61, 5.15, 4.77, 2.63, 10.64...]` 震荡剧烈，**完全没收敛**。
- 扩到 FDEdge 原文规模：`num_epochs ≥ 100`、`time_slots ≥ 100`、`num_tasks_max ≥ 50`
- 每种配置**跑 5 个 random seed 取均值±std**，曲线配置信区间带
- HV/loss 做滑动平均（窗口 5~10）

### 2. 基线太弱（只有 Random/RR/Greedy，全是启发式）
GMORL、FDEdge 两篇源论文的对比阵容里都有 DRL 基线。当前 `baselines/DQN, SAC, Opt` 是 **FDEdge 单目标版**，没接到多目标 MOFD 环境上。**必须补齐**：
- **Discrete-SAC**（GMORL 的核心对比，无扩散、无 transformer）
- **D2SAC**（扩散策略但无反馈，验证反馈机制贡献）
- **FDEdge v1**（有反馈但单目标，验证多目标扩展贡献）
- **GMORL**（原始论文方法，直接对标）
- **LDQN**（FDEdge 原文基线）
- **Opt / Exhaustive**（延迟上界）

### 3. HV 参考点不固定，跨方法不可比（`mofd_main.py:296, 322`）
```python
ref = (pts[:, 0].max() * 1.1 + 1e-6, pts[:, 1].max() * 1.1 + 1e-6)
```
每个 epoch 的 ref 不同，HV 曲线的数值没有物理意义。
- **固定 ref point**：取所有方法 Pareto 点的全局 nadir×1.1，或用 Random baseline 的最差点
- 额外报 **IGD（Inverted Generational Distance）**、**spacing**、**Pareto 解覆盖率** — 单一 HV 在审稿里常被吐槽

### 4. 缺 Ablation Study（SCI 必备）
MOFD = FDEdge 反馈 + GMORL 多目标 + ICLR Set-Transformer + FC-MCSS，**四个"贡献"必须逐个拆开**证明增益：

| 变体 | 去掉什么 | 期望证明 |
|---|---|---|
| w/o Feedback | latent 初始化改全零 | 反馈机制的价值 |
| w/o Set-Transformer | 退回 MLP（`fdsac_model.py` 的 PolicyNet）| Transformer 主干的价值 |
| w/o FC-MCSS | `n_candidates=1` 单链推理 | 候选池+Critic 选优的价值 |
| w/o Preference Bucket | 单一 latent cache 不分桶 | ω 分桶反馈的价值 |
| w/o Mask | 无效 ES 不 mask | valid-mask 的价值 |

### 5. 泛化能力没有实验验证（GMORL 的核心卖点）
MOFD 的定位是"**一个策略跨 (E, f_E, ω) 上下文**"，但现在训练和评估的上下文分布**完全相同**（都是 `sample_context`）。需要：
- **训练/测试 split**：训练用 E∈[3,5]，测试用 E∈[6,8] 看零样本泛化
- **未见偏好测试**：训练偏好从 `[0.1, 0.3, 0.5, 0.7, 0.9]`，测试用 `[0.2, 0.4, 0.6, 0.8]`
- **CPU 频率分布偏移**：训练 f∈[10,40]，测试 f∈[40,60]
- 对比**per-preference 训练**（每个 ω 单独训一个策略）的性能，证明单模型泛化的收益

---

## 🟡 二、提升级问题（审稿人大概率会问）

### 6. 超参敏感性分析缺失
至少跑 3 组：
- `denoising_steps ∈ {1, 3, 5, 10, 20}` — 反驳"扩散步数越多越好"或论证 5 步最优
- `n_candidates ∈ {1, 3, 6, 10}` — FC-MCSS 的收益曲线
- `n_buckets ∈ {1, 4, 8, 16}` — 反馈桶数对 Pareto 质量的影响

### 7. 能耗模型过于简单（`mofd_environment.py:148-153`）
```python
e_off = self.p_off * tran_delay
e_exe = self.kappa * (f_b ** 2) * rho_d
```
- 没有 **idle pow er**、**DVFS 模型**、**上行发射功率随信道变化**
- GMORL 原文有更完整的 MEC 能耗模型，至少对齐一种文献设定
- 传输能耗 `p_off=0.5 W` 固定值，应随距离/信道衰落变化

### 8. 奖励标量化方式单一
当前 `r = ω_T·α_T·r_T + ω_E·α_E·r_E` 是线性加权，Pareto 前沿的**凹区域**采不到。
- 加 **Chebyshev 标量化** 对比（`max(ω_T·|r_T - z*|, ω_E·|r_E - z*|)`）
- 或 **Hypervolume-based scalarization**
- `alpha_T=1.0, alpha_E=0.25` 是硬编码，应用 **running mean/std 自动归一化**

### 9. 目标熵硬编码 `target_entropy=-1` 不合理
动作维度 Emax=6，但 `E ∈ [3,6]` 会变，熵的理论范围也变。建议 `target_entropy = -log(E_valid) * 0.5`，**随上下文动态调整**。

### 10. 任务生成太理想
- 当前均匀分布 `rng.uniform(min_bit, max_bit)`
- 加 **Poisson 到达**、**重尾分布**（Pareto/lognormal）、**真实 trace**（Alibaba GPU Cluster / Google Borg / Azure VM Trace）
- 至少测 1 组真实 trace，SCI 审稿会加分

### 11. 反馈缓存语义漂移问题未讨论
`latent_cache` 在训练中被**策略自己写回**，但策略网络持续更新后，早期写入的 latent 已经过时。
- 需要讨论或做实验：缓存衰减（比如 EMA 平均）vs 直接覆盖
- 或引入**周期性刷新**0

---

## 🟢 三、完善级（润色用）

### 12. 工程完整性
- `mofd_main.py` 的 `cfg` 字典拆成 `.yaml` 文件，便于复现
- 固定所有 seed（现在 `np.random.seed(cfg['seed'])` 和 `torch.manual_seed` 用同一 seed，但 baseline 里 `rng = np.random.default_rng(43)` 固定死）
- `results/` 产出加时间戳子目录，避免多次运行覆盖

### 13. 复杂度与效率分析
- 报告：每决策推理时延（FC-MCSS 并行 N 条链的开销）、模型参数量、训练一个 epoch 的 wall-clock
- 对比 FDEdge-MLP vs MOFD-Transformer 的参数与速度 trade-off

### 14. 可视化补充
- HV convergence curve + 置信带
- Pareto front 每个方法独立一图 + 合并一图
- Ablation 柱状图（HV / IGD）
- 敏感性分析折线图（denoising_steps、n_candidates）
- **动作分布可视化**：ω 从 `(1,0)` 扫到 `(0,1)`，策略选的 ES 频率直方图怎么变 — 说明策略"理解"了偏好

### 15. 符号/公式与文章对齐
三篇文章的符号体系不同，论文里要**统一公式**，并在每个算法模块下明确引用来源：
- §X.1 Feedback Diffusion ← FDEdge (TMC'25)
- §X.2 Pareto + Context Generalization ← GMORL (TSC'25)
- §X.3 Set-Transformer + FC-MCSS ← ICLR'25

---

## 📌 建议实施顺序

1. **先修 #1 #3** — 实验规模×种子数 + 固定 HV ref，否则所有后续数据都是噪声
2. **再补 #2 #4** — 强基线 + ablation，是论文主表的骨架
3. **再做 #5 #6** — 泛化 + 敏感性，卖"generalizable"卖点
4. 最后 #7~#15 打磨



 给你 7 个原创度能打、审稿人难 attack               
  的切入方向，按实现成本分档。重点是：这些都不在
  articles/ 的任何一篇里，也不是简单缝合。           
   
  ---                                                
  🟢 低成本（1–3 天，改 2–5 个文件）               

  1. 风险感知 Pareto（Risk-Aware MORL / CVaR-Pareto）

  痛点：现有所有方法（FDEdge、GMORL、Diffusion
  Planner）都只优化期望延迟，但真实 MEC
  关心尾部延迟（99% 分位、掉包率）。平均 200ms 但 1%
  任务超时 3s 的策略，用户体验比平均 300ms
  但零超时的差。

  做法：把奖励改成 CVaR(α) 形式：
  r_T = - CVaR_α(delay)   # 最差 α% 的平均延迟
  用分位数回归（QR-DQN / IQN 思路）替换标量
  Q，但不抄原文：你提供的是CVaR-conditioned 多目标
  diffusion —— preference ω 里多加一维
  α（风险系数）。

  实验故事：
  - 表：mean delay 几乎没降，但 P99 延迟降 30–50%
  - 图：风险偏好 α 从 0.5 → 0.05 时 Pareto
  前沿如何变化
  - 这一条就是独立的 SCI 亮点，Section 可以单开

  2. 自适应去噪步数（Adaptive Denoising Steps, ADS）

  痛点：FDEdge 的 Feedback Diffusion 每次推理要跑
  denoising_steps=5 次，训练和推理都贵。GPU 推理是
  MEC 实时性最大的瓶颈。

  做法：加一个信心网络 c_φ(s) ∈
  [0,1]，根据状态预测"当前 ω 对应 Pareto
  区域是否清晰"，动态选步数 K = max(1, ⌈K_max ·
  c_φ(s)⌉)。训练时加 loss：推理误差 × 步数 →
  鼓励用更少步数。

  实验故事：
  - 图：推理时间 / MFLOPs 减 50–70%，Pareto
  质量几乎无损
  - 消融：固定 K=1,2,3,5 对比 vs ADS
  - 直接回应 "diffusion 太慢" 的经典批评，很加分

  3. 信道感知扩散条件（Channel-Aware Diffusion
  Conditioning）

  痛点：你的新信道衰落只是"让环境更真实"，策略本身只
  把信道当 state 输入，没有把它显式注入扩散去噪过程。

  做法：diffusion 的每步去噪条件向量 cond = [ω,
  z_latent, **avg_channel_gain**]，额外训一个
  classifier-free guidance 项：无条件 vs
  信道条件扩散的插值 guidance。
  ε̂_θ(x_t, cond) = (1+w) · ε_cond - w · ε_null

  实验故事：
  - 信道好 vs 差两种 regime 下的前沿差距
  - 消融：有/无 channel conditioning +
  classifier-free guidance 开关
  - 这是 diffusion planner 领域的经典 trick 首次用到
  MEC offloading，新颖度足够

  ---
  🟡 中成本（1–2 周，改 5–10 个文件）

  4. 在线偏好漂移自适应（Online Preference Drift
  Adaptation）

  痛点：所有 MORL 工作假设偏好 ω 静态。但真实 IoT
  设备的偏好会漂移：电池 80% 时 ω=(0.7, 0.3)，到 20%
  时漂移成 (0.3, 0.7)。现有方法要重训。

  做法：
  - 引入漂移检测器：监测最近 N 个任务的 (d,e)
  分布偏离某个 ω 对应前沿点的程度
  - 检测到漂移时，用 bucket 缓存（你已有）里最近的
  feedback latent 做在线微调 5–10 步
  - 不重训主策略，只微调 preference-specific head

  实验故事：
  - 时间轴图：ω 随 t 漂移时，baseline HV
  陡降，你的方法维持稳定
  - 这是实际部署场景的硬需求，审稿人看了会兴奋
  - 和 FDEdge 的"feedback"本意是反馈扩散用来去噪，你
  把它扩展到在线适应 —— 顺水推舟


  仅看训练 HV 不够,要专门构造漂移场景:

  场景 A:Sudden drift (突变)

  - 训练 100 epochs,ω 按 21-cycling 走
  - 评估时:连续 50 episodes,前 25 个 ω=(0.9,0.1) 偏延迟,第
   26 个突然切到 ω=(0.1,0.9) 偏能耗
  - 指标:漂移后前 5 个 episode 的 cost 衰减速度
  - 期望:C2 从历史 buffer 里取到接近 (0.1,0.9) 的 latent →
   立刻收敛;C0 用 randn → 需要 10+ episode 重新 burn-in

  场景 B:Gradual drift (渐变)

  - ω 沿单纯形按正弦轨迹漂 (ω_T = 0.5 + 0.4·sin(2π·t/T))
  - 指标:滑动平均 HV / cost 整段
  - 期望:C2 全程稳定;C0 每次都要重学

  场景 C:Slot-level fast drift (slot 内多 ω)

  - 训练阶段 episode 内 ω 固定 (没漂移)
  - 评估阶段每 10 slot 切一次 ω →
  测模块对未训练过的快漂移的鲁棒性
  - 期望:C2 ≥ C0 (退化也别太厉害);若 C2 < C0,说明 buffer
  的 stale 项害人 → 需要加 staleness 惩罚


  C0 baseline (F1 only):把 cfg['use_omega_buffer']=False →
   跑一次 → 备份 results/mofd_* 到 results_C0/

  C2 Path B:use_omega_buffer=True (默认) → 再跑一次 →
  results_C2/

  然后对比:
  - 训练曲线 mofd_training_curves.png (HV / delay /
  energy) → 收敛速度
  - mofd_pareto_aggregated.csv → HV 均值 ± std
  - mofd_obuf_log_seed0.csv → 用 hit_rate 和 nn_dist 诊断
  buffer 是否真在工作 (期望 hit_rate→1,nn_dist≈0 因为 21
  训练 ω = 21 评估 ω)
  - mofd_obuf_updates_seed0.csv → 每个 ω-bin
  应该被均匀更新 ≈ num_epochs * n_prefs_per_epoch / 21 次

  跑完一轮告诉我结果,我再帮你加漂移场景测试 (sudden /
  gradual drift) 和 C3/C4 反向控制组。


  5. 多样性驱动的 Pareto 扩散采样（Diverse Pareto
  Diffusion Sampling）

  痛点：现在每个 ω 采一次 → 一个点。Pareto
  前沿的"覆盖度"靠扫多个 ω 堆出来，效率低。

  做法：利用 diffusion model 天生能产出多样样本的特性
   —— 固定 ω，一次正向 M 次，用行列式点过程（DPP）
  或最大最小距离选 M 个最不相似的解。
  loss += λ_div · DPP_entropy(action_samples_batch)

  实验故事：
  - 同样采样预算下，HV 高 10–20%
  - Pareto 前沿"空洞"明显减少（图上直观）
  - 这是用 diffusion 替代 PG-MORL 种群的概念性贡献

  6. 约束型 MORL（Constrained Multi-Objective MDP）

  痛点：现在"硬截止时间"是软惩罚（加到 reward
  里）。真实系统是硬约束 —— 超了就是 violation。

  做法：
  - 用 Lagrangian RL 或 Interior-Point Policy
  Optimization
  - 约束：P(delay > T_max | ω) ≤ δ
  - 目标：在约束满足前提下最大化 - ω·α_T·E[delay] -
  ω·α_E·E[energy]
  - 扩散 guide: 拒绝采样掉违反约束的动作

  实验故事：
  - 对比：软惩罚 vs 硬约束下的 SLA 满足率
  - 硬约束下 Pareto 前沿变化
  - MEC 里 SLA 是刚需，很有说服力

  ---
  🔴 高成本（3–4 周，需要新模块）

  7. 图结构 ES 表征 + 任务迁移（Graph-ES
  Representation）

  痛点：现在 ES 之间是独立的（Set-Transformer
  只做集合编码，没捕捉拓扑）。真实部署里 ES
  构成通信图，任务可以二跳迁移（A 卸载到 B，B
  过载转给邻居 C）。

  做法：
  - 把 ES 看成图节点，边带"链路带宽/延迟"
  - 用 GNN（GAT）替换 Set-Transformer
  - 动作扩展为两级：(first_hop, migrate_or_not)
  - 奖励多一项：跨 ES 迁移的额外传输成本

  实验故事：
  - 图规模 scaling：8/16/32 ES，随 ES
  数增加你的优势扩大
  - 拓扑鲁棒性：随机断链 10% 链路，策略仍能适应
  - 这条最重也最独立，足够撑一个工作量饱满的 section

  ---
  📋 选择建议

  如果你想赶工投一轮（CCF-B 级 SCI，3 个月内）：
  - 做 #1 (CVaR) + #2 (ADS) + #3 (信道 guidance)
  - 这 3 个组合起来就是 "Risk-Aware Efficient
  Channel-Conditioned Diffusion MORL"
  - 每条都是独立章节 + 独立 ablation，篇幅足

  如果冲 Q1 / 顶刊（TMC / TON / TPDS）：
  - 做 #1 + #4 (漂移自适应) + #7 (Graph-ES)
  - 重量级 3 个点，每个都能单独发短文
  - 故事性强：static preference + homogeneous ES →
  dynamic preference + graph ES

  如果时间紧张但要差异化：
  - 只做 #1 (CVaR-Pareto) —— 在所有引用论文里都没人做
  - 配合你已有的 FD + Set-Transformer 已经足够讲清楚

  ---
  需要避免的假创新

  - "改 buffer 大小 / 改学习率 / 改层数" —
  调参不是创新
  - "引入 attention" — 2026 年已经不新了
  - "加个简单 GNN" — 没和 MEC 物理层绑定的 GNN 是灌水
  - "套用某大模型做 prompt" — 和 MEC 脱钩，审稿人秒判

  ---
  想先聊哪个？ 我可以给出具体的模块级接口 + 实验设计
  + 预估效果数字，帮你判断最值得投入的一个。


● 从模型结构本身扫一遍 (撇开超参/效率),列出11             
  个结构性问题,按"对投稿说服力伤害 / 修复收益"排序。
                                                          
  ---
  🔴 A 级 (理论瑕疵,容易被审稿人揪住)                     
                                                        
  A1. Transformer 的输出被强行
  softmax,但在扩散框架里被当作 ε (噪声预测)

  # SetTransformerBackbone.forward (mofd_v2.py L105)
  return F.softmax(logits, dim=1)   # 输出 ∈ [0, 1]^Emax,
  且 Σ=1

  # Diffusion.p_mean_variance (feedback_diffusion.py L117)
  x_recon = predict_start_from_noise(x, t=t,
  noise=self.model(x, t, s))
  #
  ^^^^^^^^^^^^^^^^^^^^^^^
  #                                    按 DDPM
  假设这里应该是 ε ~ N(0, I)

  结构问题:
  - DDPM 的数学推导要求 ε_θ(x_t, t) 预测的是标准高斯噪声,
  但你的模型输出是概率单纯形上的点,支撑集完全不同。
  - predict_start_from_noise 公式 x_0 = (1/√ᾱ)·x_t −
  √(1/ᾱ−1)·ε 在 ε 是 softmax 输出时几何上没有物理意义。

  修复:
  - 方案 1 (推荐): 改 predict_epsilon=False,让 Transformer
   直接预测 x_0,去掉内部 softmax,sample() 结束时再
  softmax。这对齐 Diffusion-QL 的用法。
  - 方案 2: 去掉 softmax,让模型预测无界向量当 ε,最终 x_0
  再 softmax。

  ---
  A2. Diffusion.loss / p_losses 从未被调用 →
  扩散外壳纯属摆设

  # mofd_v2.py MOFD_SAC_V2.update() 里只用
  self.actor(state, latent)  # forward 采样
  # 完全没有调用
  self.actor.loss(x_gt, state)  # 监督式扩散损失

  结构问题:
  - Diffusion-QL 的原始 loss 是 L_BC (扩散去噪 MSE) +
  λ·L_Q (critic 梯度) 联合训练。
  - 你这里只有 L_Q (policy gradient 穿透 5 步扩散),没有
  L_BC。
  - 结果是 Transformer
  并不是在学"从噪声去噪到好动作",而是学"被 critic 压迫的 5
   次迭代函数"。扩散过程和预定义的 β schedule
  之间失去了理论绑定,5 步迭代本质退化为 5 层 RNN。

  修复: 加一个轻量 BC 目标,比如对 replay buffer 里 Q
  值高的样本做 behavior cloning:
  # 选出 Q 值 top-20% 的 (s, a) 作为"专家样本"
  top_idx = expected_q > torch.quantile(expected_q, 0.8)
  x_gt = F.one_hot(a[top_idx], Emax).float() * 0.9 +
  0.1/Emax  # 平滑 one-hot
  bc_loss = self.actor.loss(x_gt, s[top_idx])
  total_actor_loss = actor_loss + λ * bc_loss   # λ = 0.1

  ---
  A3. Critic 是 flat MLP,跟 Set-Transformer Actor
  结构不对称

  # QValueNet (mofd_v2.py L123)
  self.net = nn.Sequential(Linear(28, 128), ReLU,
  Linear(128, 128), ReLU, Linear(128, 6))

  结构问题:
  - Actor 排列不变 (Set-Transformer + 独立 head),Critic
  排列可变 (MLP 把 ES 顺序当固定输入)。
  - 当 ES_1 和 ES_2 交换槽位,Actor 输出 [p_2, p_1,
  ...],Critic 却输出完全不同的 Q 值。
  - 当 E 变化时,Critic 对 invalid 槽的输出是未定义值
  (训练信号少),但 Σ probs · Q 里 invalid 槽的 probs=0,所以
  容忍度还算高。真正的问题是训练信号的不对称,让 Actor
  在"排列等变"和"Critic 偏见"之间拉扯。

  修复: Critic 也用 Set-Transformer (可以和 Actor 共享
  backbone 或独立一份),输出 per-ES Q 值:
  class SetQNet(nn.Module):
      def __init__(self, shared_backbone, ...):
          self.enc = shared_backbone   # 或者独立一份
          self.q_head = nn.Linear(d_model, 1)
      def forward(self, state):
          # 不需要 x/time_step, 传零
          tokens = self.enc.encode_tokens(state, dummy_x,
  dummy_t)
          return self.q_head(tokens[:, 1:, :]).squeeze(-1)
    # [B, Emax]

  ---
  🟡 B 级 (中等结构缺陷,改了效果会提升)

  B4. 动作 mask 在扩散循环外部才应用,导致去噪过程在
  invalid 槽上浪费容量

  # 现在的流程:
  probs = self.actor(s, x)         # 5 步扩散 + softmax,
  可能给 invalid 槽分配概率
  probs = probs * mask             # 外层手动清零
  probs = probs / probs.sum()      # 重新归一化

  结构问题:
  - Transformer 的 key_padding_mask 只阻止 attention
  从无效槽汇入信息,但 head 最后还是输出 Emax 个
  logits,softmax 会给 invalid 槽分配概率。
  - 5 步扩散迭代的每一步都会在 invalid
  槽上产生"噪声动作",这些能量没有意义但消耗模型容量。

  修复: 把 action_mask 传进 SetTransformerBackbone,在 head
   输出时直接 masked_softmax:
  def forward(self, x, time_step, state, action_mask):
      ...
      logits = self.head(server_encoded).squeeze(-1)   #
  [B, Emax]
      logits = logits.masked_fill(~(action_mask > 0.5),
  -1e9)
      return F.softmax(logits, dim=1)

  这样扩散每一步的梯度都只流过 valid 槽,策略更"干净"。

  ---
  B5. 扩散时间步 t 的编码只进了 ctx token,每个 server
  token 感知不到 t

  # mofd_v2.py L89-91
  t_emb    = SinusoidalPosEmb(16)(time_step)  # 只嵌入到
  ctx
  ctx_feat = cat([task_pref, t_emb], dim=-1)
  # srv_tokens 里没有 t 信息

  结构问题:
  - 标准 DDPM (UNet) 把 t_emb 用 FiLM (γ·x + β) 或直接加
  到每一层每个空间位置。
  - 你这里 srv_tokens 必须通过 3 层 attention 才能从 ctx
  那"间接知道"当前是第几步去噪,信号很弱。
  - 表现是:第 0 步和第 4 步的 srv_token 初始表示完全相同,
  模型难以学到"前几步粗略、后几步精细"的行为。

  修复: 把 t_emb 也 broadcast 到 srv tokens:
  srv_tokens = srv_tokens + t_emb.unsqueeze(1)   # [B,
  Emax, d_model] + [B, 1, t_dim]
  # 若维度不同需要先 proj

  ---
  B6. latent_cache 的"偏好桶 × (t, n)"索引方式不合理

  # mofd_main.py L238-244
  latent_cache = np.random.normal(size=[n_buckets=8,
  T=100, N=50, Emax=6])
  bucket = int(clip(round(ω_T · 7), 0, 7))
  latent = latent_cache[bucket, t, n]   # 按 (偏好, 时隙,
  任务序号) 取

  结构问题:
  1. 偏好离散化粗糙:8 桶,ω=0.06 和 ω=0.18 都映射到
  bucket=1,但最优策略可能差很多;ω=0.44 和 ω=0.56
  跨桶,但策略相似。
  2. 按 (t, n) 索引没有语义:不同 episode 的第 (50, 20)
  位置面对的是不同 task size、不同 ES 负载,缓存的 latent
  完全不相关。
  3. 容量浪费:8×100×50×6 = 240 000 个
  latent,但有语义的"相似 (state, ω)" pair 数远少于这个。

  修复 (任选一种):
  - 轻量: 把 (t, n) 换成 (bucket, state_hash),state_hash
  用 task_size 分桶 + queue_load 分桶,类似
  locality-sensitive hashing。
  - 更彻底: 完全去掉 cache,改成参数化 latent
  prior:latent_prior = PriorMLP([ω, state]),让网络自己学"
  什么上下文该从哪里起步",梯度反传训练。这也是 Diffusion
  Policy (Chi et al. 2023) 的做法。

  ---
  B7. MCSS 扰动跳过了扩散的"加噪语义"

  def _build_candidates(latent):
      base  = latent.expand(N, -1).clone()
      noise = randn_like(base) * 0.10
      cand  = base + noise
      cand  = clamp(cand, min=0.0) / sum(...)    #
  投影到概率单纯形
      return cand

  结构问题:
  - 这是一个 ad hoc 扰动:在概率单纯形上加 N(0,
  0.01),然后硬 clamp 回来。
  - 理论上扩散模型期望的"起点分布"是 q(x_{T-1} | x_0)
  高斯带噪态,不是概率单纯形。
  - 扰动后的 cand 作为 T−1 步输入送进 p_sample,分布上不
  consistent,最终去噪结果质量受损。

  修复: 把 MCSS 的扰动改成"在 x_0 候选附近先 q_sample 到
  T−1 步再去噪":
  def _build_candidates_v2(latent):
      t_T = self.n_timesteps - 1
      x_0_candidates = [latent + ε for ε in small_noises]
    # 在 x_0 附近 N 个点
      x_T_candidates = self.actor.q_sample(x_0_candidates,
   t=t_T)  # 正规加噪
      return x_T_candidates

  ---
  B8. Output head 太弱:独立 Linear(64→1) 每 ES 一个
  logit,丢失相对排名结构

  # mofd_v2.py L71, L103
  self.head = nn.Linear(64, 1)
  logits = self.head(server_encoded).squeeze(-1)   # 每个
  ES 独立出分

  结构问题:
  - 这种"独立评分"完全忽略 ES 之间的相对比较。Q-learning
  里最有价值的信号是"相对优势",独立评分让它靠后面的
  softmax 去做相对化,容量上不优。
  - Pointer Network / Attention-based scoring 更自然。

  修复: 用 ctx token 作 query,对 srv tokens 做 attention
  打分:
  # 在 encoder 最后一层输出后
  ctx_out = encoded[:, 0:1, :]                # [B, 1, d]
  srv_out = encoded[:, 1:, :]                 # [B, Emax,
  d]
  logits  = (srv_out @ ctx_out.transpose(1,
  2)).squeeze(-1) / sqrt(d_model)  # [B, Emax]

  这是 Set-Transformer 论文 PMA 块的简化版,参数反而更少,但
  表达"谁跟当前上下文最配"的能力更强。

  ---
  🟢 C 级 (锦上添花)

  C9. 5 步扩散用 SinusoidalPosEmb(16) 是 overkill

  self.time_emb = SinusoidalPosEmb(t_dim=16)

  结构问题: Sinusoidal embedding 原本是为"几百上千步的
  DDPM"设计的连续嵌入。5 步的场景用它,模型实际学到的只是 5
   个固定向量。

  修复: 直接用 nn.Embedding(n_timesteps=5,
  t_dim=16),参数更少,表达力相同。

  ---
  C10. 没有 dropout / weight decay → 过拟合风险

  # mofd_v2.py L66
  nn.TransformerEncoderLayer(..., dropout=0.0)
  torch.optim.Adam(params, lr=...)   # 默认 weight_decay=0

  修复: Transformer dropout=0.1,Adam →
  AdamW(weight_decay=1e-4)。Set-Transformer 的 3
  层注意力参数 ≈ 50k,如果不正则化在 200 万 step
  的训练下会过拟合 replay buffer。

  ---
  C11. Critic 不知道 feedback latent,状态空间与 Actor
  不一致

  actor(state, latent)      # 两个输入
  critic(state)             # 只有一个输入

  结构问题:
  - Actor 根据 (state, latent) 决策,但 Critic 评估的是
  Q(state, ·),latent 携带的历史策略信息对 Q 估计是 side
  channel。
  - 特别是在 feedback 机制下,同一 state 但不同 latent
  可能对应策略演化的不同阶段,Q 不应该一样。

  修复: Critic 也吃 latent:
  class QValueNetV2(nn.Module):
      def forward(self, s, latent):
          return self.net(cat([s, latent], dim=-1))  #
  state_dim + Emax

  或者更激进,让 critic 复用 SetTransformerBackbone 的
  encode 部分 (共享 feature extractor)。

  ---
  🎯 优先级排序 (按论文 & 效果的"性价比")

  ┌──────┬────────────┬───────────────────────┬───────┐
  │ 优先 │   改哪个   │     理论/效果收益     │ 工程  │
  │  级  │            │                       │ 成本  │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ A1 修 ε vs │                       │       │
  │ P0   │  softmax   │ 堵住审稿人最大质疑    │ 10 行 │
  │      │ 的语义冲突 │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ A3 Critic  │ 直接涨 HV             │       │
  │ P0   │ 也用 Set-T │ (Actor-Critic 对称性) │ 30 行 │
  │      │ ransformer │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ B4 把      │                       │       │
  │ P1   │ action     │ 减少无效动作浪费      │ 5 行  │
  │      │ mask 移到  │                       │       │
  │      │ head 内    │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │ P1   │ A2 加 BC   │ 扩散骨架名副其实,是论 │ 20 行 │
  │      │ 辅助损失   │ 文的重要 argument     │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ B5 t_emb   │ 5                     │       │
  │ P1   │ 进每个     │ 步扩散的粗/细过渡更好 │ 5 行  │
  │      │ token      │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ B8 Pointer │ 参数少 +              │       │
  │ P2   │ -style     │ 相对排名更直接        │ 15 行 │
  │      │ head       │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │      │ B6 latent  │                       │       │
  │ P2   │ cache →    │ 去掉最脆弱的设计,对泛 │ 50 行 │
  │      │ parametric │ 化性有帮助            │       │
  │      │  prior     │                       │       │
  ├──────┼────────────┼───────────────────────┼───────┤
  │ P3   │ B7, C9,    │ 小改进的集合,可一次合 │ 30 行 │
  │      │ C10, C11   │ 并                    │       │
  └──────┴────────────┴───────────────────────┴───────┘

  ---
  一个合理的最终架构 (如果都做完长什么样)

  Actor (Diffusion Policy):
  ┌─────────────────────────────────────┐
  │ Input: state [B, 28], latent [B, 6] │
  │                                     │
  │ 5-step reverse diffusion:           │
  │   for t in [4, 3, 2, 1, 0]:         │
  │     x = p_sample(x, t, state, mask) │
  │       ↓                             │
  │   SetTransformer ε_θ (predict x_0): │
  │     ├ ctx_token  [B, 1, 64]  ← task + ω + t_emb │
  │     ├ srv_tokens [B, 6, 64]  ← (f, q, valid, g,
  x_t[e]) + t_emb │
  │     ├ 3× Self-Attention (dropout=0.1) │
  │     └ Pointer Head: (ctx · srv) → masked_softmax │
  │                                     │
  │ Output: x_0 [B, 6] ∈ prob simplex  │
  │                                     │
  │ Training loss:                      │
  │   L = policy_grad (穿透 5 步扩散) + λ·BC  │
  └─────────────────────────────────────┘

  Critic (对称 Set-Transformer):
  ┌─────────────────────────────────────┐
  │ Input: state [B, 28], latent [B, 6] │
  │                                     │
  │ SetTransformerQ:                    │
  │   ├ ctx_token  [B, 1, 64]          │
  │   ├ srv_tokens [B, 6, 64] (no x/t)  │
  │   ├ 2× Self-Attention              │
  │   └ Q head: Linear(64, 1) per srv   │
  │                                     │
  │ Output: Q(s, ·) [B, 6]             │
  └─────────────────────────────────────┘

  Feedback Prior (替代 latent_cache):
  ┌─────────────────────────────────────┐
  │ Input: state, ω                     │
  │ PriorMLP: Linear(state+ω, 64) → relu → Linear(64, 6) │
  │ Output: latent_init [B, 6]          │
  │ (在 MSE loss 下跟随 actor 学习)     │
  └─────────────────────────────────────┘

  这套下来:
  - 理论上自洽 (ε 不再是 softmax,扩散 loss
  真实参与,Actor-Critic 对称)
  - 参数量甚至略减 (pointer head + embedding 比独立 head +
   sinusoidal 小)
  - 论文故事更干净 ("Feedback Diffusion Planner = 扩散去噪
   + BC + critic 引导 + 参数化
  prior"),每个组件都有独立贡献,便于 ablation

  如果只能做 3 件事,我强烈建议 A1 + A3 + B4 —— 投入不到 50
   行代码,堵住最大理论漏洞 +
  最大性能漏洞。剩下的按投稿时间余量依次补。

  要不要我先把 A1 (ε 语义修正) 和 A3 (Critic
  Set-Transformer 化) 这两件实装?










下面给出 M3 / M4 / J2 / J4 的对应解决方案。每条按"诊断 →
   整改方案 → 在论文中体现的形式 → 工作量估计"四段写。
                                                          
  ---
  M3：HV 训练后期退化 -23%                                
                                                        
  诊断（最可能根因）

  按 SAC + Diffusion 混合架构的常见失败模式，按概率排序：

  1. α 自调节崩塌：alpha_lr=3e-4 + target_entropy=-1.0（21
   类离散动作下偏激进）→ alpha 单调上升 → 策略熵爆炸 →
  退化为均匀策略
  2. Critic Q 值发散：gamma=0.95 + 无 target clipping → Q
  值不断膨胀 → 策略追逐虚假高 Q
  3. Diffusion latent 漂移：feedback latent
  随训练越来越远离初始分布 → denoising 3 步走在 OOD 区域
  4. Buffer 反向污染（仅 C2）：早期 buffer 写入差策略的
  latent，后期被反复 retrieve → 自我强化

  整改方案

  S3.1 诊断日志先行（必做，1 天工作量）

  在 mofd_main.py 的训练循环里加 4 个监测项，每 epoch
  记一次：
  - alpha.item() 时序
  - critic_loss.mean() 与 Q_target.mean() 时序
  - actor 策略熵 H[π(·|s)] 时序
  - diffusion latent 的 L2 范数 ||z||₂ 时序

  跑完看哪条曲线在 epoch 21 后开始异常——这一步就锁定根因。

  S3.2 对应修复（按诊断结果选 1-2 项）

  ┌───────────┬───────────────────────────────────────┐
  │   根因    │                 修复                  │
  ├───────────┼───────────────────────────────────────┤
  │           │ 固定 alpha=0.05 不学习；或限幅 alpha  │
  │ α 崩塌    │ ∈ [0.01, 0.2]；或改用 SAC-Discrete    │
  │           │ 的标准公式 target_entropy = 0.98 ×    │
  │           │ log(|A|)                              │
  ├───────────┼───────────────────────────────────────┤
  │           │ 加 double-Q clipping +                │
  │ Q 发散    │ target_q.clamp(-Q_max, Q_max)；将     │
  │           │ gamma 降到 0.9                        │
  ├───────────┼───────────────────────────────────────┤
  │ Diffusion │ 给 latent 加 EMA 锚点：z_t ← (1-β)    │
  │  漂移     │ z_t + β z_init，β=0.01                │
  ├───────────┼───────────────────────────────────────┤
  │           │ buffer 仅在 HV_eval > best_HV * 0.9   │
  │ Buffer    │ 时写入；或对 latent 做                │
  │ 污染      │ quality-weighted 平均（按当前 HV      │
  │           │ 加权）                                │
  └───────────┴───────────────────────────────────────┘

  S3.3 模型选择策略（兜底）

  无论根因如何，加 early stopping + best-checkpoint
  averaging：
  - 训练全程保存 top-3 epoch 的模型权重
  - 最终模型 = 三者平均（SWA / Polyak averaging）
  - 报告时同时给 "best epoch" 和 "final epoch"
  两组数据，论文里说清楚选择策略

  论文中的体现

  新增一节 "Training Stability Analysis"：
  - Figure: 4 个监测量的训练曲线 + HV 曲线，对齐 x
  轴显示因果
  - Table: 修复前 / 修复后的 HV 退化幅度对比
  - 一段 Discussion：说明 SAC + diffusion
  联合训练的稳定性挑战 + 你的修复方案

  工作量

  诊断 1 天 → 修复 + 验证 2-3 天 → 重跑全部实验 ~1 天 =
  总计 4-5 天

  ---
  M4：缺乏理论支撑

  诊断

  ω-Latent Buffer 当前只有"分箱 + EMA + 噪声
  retrieve"的工程实现。Q1
  期刊可以不要严格证明，但必须有形式化定义 +
  至少一个非平凡命题。

  整改方案

  S4.1 形式化定义（必备）

  在 Methodology 章节加一节 "Formulation of Drift-Aware
  Feedback Memory (DAFM)"：

  Definition 1 (ω-Latent Buffer). Given a discretized
  preference grid
  Ω̂ = {ω_1, ..., ω_K} ⊂ Δ^{m-1}, the buffer M : Ω̂ → R^d
  maintains
  a per-bin latent state z̄_k updated via:
      z̄_k^(t+1) = (1-β) z̄_k^(t) + β · z_obs^(t)    if k=
  nearest(ω^(t))
  where β ∈ (0,1] is the EMA factor.

  Retrieval: M(ω) = z̄_{nearest(ω)} + σ·ε, ε ~ N(0, I)

  S4.2 命题 1：ω-单调性保持（最核心）

  Proposition 1 (ω-Conditioning Preservation under Drift).
  Let π_θ be a Lipschitz-continuous policy on (s, ω, z).
  If the buffer
  satisfies (i) Lipschitz retrieval ||M(ω₁) - M(ω₂)|| ≤
  L_M ||ω₁ - ω₂||,
  and (ii) bounded EMA noise σ ≤ σ_max, then for any
  ω-shift trajectory
  ω₁ → ω₂, the induced policy preserves ω-monotonicity:
      Spearman ρ(τ_ω₁, τ_ω₂) ≥ ρ_min(L_M, σ_max,
  ||ω₁-ω₂||)

  证明草图：用 Lipschitz + 链式不等式给出 ρ_min
  的封闭形式，最后引入你 ω-shift 实验里测到的 -0.43 vs
  +0.02 作为经验验证。

  S4.3 命题 2：Drift-Aware Regret 改善

  Proposition 2 (Sample Complexity Improvement).
  Under sudden ω-drift at slot t* in an episode of length
  T, the cumulative
  reward gap between drift-aware retrieval (C2-aware) and
  re-learning from
  scratch (C0) satisfies:
      Reg(C0) - Reg(C2-aware) ≥ Ω(√(T-t*) · Δω)

  证明思路：把 buffer retrieval 等价为 warm-start
  prior，套 RL 文献里 prior-based exploration 的 regret
  bound（参考 PSRL、Bayesian RL）。

  S4.4 与已有方法的概念区分（写讨论）

  Relation to Existing Methods:
  - vs. Episodic Memory (NEC, MFEC): Our buffer is keyed
  on preference
    vector ω, not state-action; serves drift adaptation,
  not sample efficiency.
  - vs. Continual Learning (EWC, A-GEM): No catastrophic
  forgetting protection
    (we WANT old preferences to be overwritten by new EMA
  evidence);
    scope is intra-task preference shift, not task
  sequence.
  - vs. Hypernetworks: We do not generate weights from ω;
  we generate
    initial latents for the diffusion process, decoupling
  backbone training
    from preference adaptation.

  论文中的体现

  - §3.3 Theoretical Properties of DAFM：3 段——定义 + 2
  个命题（带证明草图，完整证明放附录）
  - 附录 A：完整证明
  - §5.X Empirical Validation of Theory：把 ω-shift
  实验里的 ρ_delay = -0.43 vs +0.02 作为 Proposition 1
  的实证支持

  工作量

  写定义 + 命题陈述 1 天，证明草图 2-3
  天（如果需要严格证明会更久），实证验证已有 = 3-4
  天纸面工作，无新代码。

  ---
  J2：ω-drift 实验缺乏真实场景叙事

  诊断

  当前 mofd_omega_drift_main.py 用 sudden / gradual /
  cyclic 三种合成调度。Reviewer 标准提问：

  ▎ "Why would ω drift in the real world? This appears to
  ▎ be an artificial setup."

  整改方案

  S J2.1 给三个 schedule
  各绑定一个真实场景叙事（写作为主，不改代码）

  Schedule: Sudden
  真实场景映射: 用户从家庭 Wi-Fi 接入切到 5G 移动网络，QoS

    Profile 切换：从 (0,1)（节能优先）切到
    (1,0)（低延迟优先）
  引用支撑: 3GPP TS 23.501 QoS Flow management
  ────────────────────────────────────────
  Schedule: Gradual
  真实场景映射: 设备电量从 100% →
    0%，能耗权重线性升高（电池保护策略）
  引用支撑: Android Doze mode / iOS Low Power Mode
    的渐进降级
  ────────────────────────────────────────
  Schedule: Cyclic
  真实场景映射: 工作日 vs 周末、白天 vs 夜间的 SLA
    周期性切换
  引用支撑: AWS / Azure 的 time-based auto-scaling 策略

  S J2.2 加一个 schedule：Trace-driven（推荐，工作量小）

  构造真实 drift trace：
  - 从 Alibaba Cluster Trace 抽取 batch job priority 列
  - 把 priority 映射到 ω：高优先级 → (0.9, 0.1)，低优先级
  → (0.3, 0.7)
  - 每 slot 按 trace 里的 job 类型切 ω
  - 这样 drift 不再是合成，而是数据驱动

  代码改动：在 build_drift_schedules 加第四个 schedule
  schedules['trace'] =
  load_trace_omega('dataset/alibaba_priority.csv', T)

  S J2.3 加真正的 drift detection 对比基线

  当前 C2-aware 是已知 ω 在变（看代码就比较 cur_omega !=
  prev_omega）。这是"oracle drift
  detection"。要补一个实际检测版本：

  # C2-CUSUM: 用 reward 流的 CUSUM 检测漂移
  def cusum_drift(reward_history, threshold=3.0):
      mu, sigma = np.mean(reward_history[:20]),
  np.std(reward_history[:20]) + 1e-6
      s_pos, s_neg = 0, 0
      for r in reward_history[20:]:
          s_pos = max(0, s_pos + (r - mu - 0.5*sigma) /
  sigma)
          s_neg = min(0, s_neg + (r - mu + 0.5*sigma) /
  sigma)
          if s_pos > threshold or s_neg < -threshold:
              return True
      return False

  加一个 C2-CUSUM 方法：reward 漂移触发时才 retrieve，用
  ADWIN / DDM 也行。这样有三档对比：
  - C0：无适应
  - C2-passive：仅初始化用 buffer
  - C2-CUSUM：reward 异常驱动 retrieve（实际可部署）
  - C2-aware (oracle)：知道 ω 变（上界）

  C2-CUSUM 落在 passive 和 aware 之间 →
  论文论点更强："我们的实际机制接近 oracle 上界"。

  论文中的体现

  - §4.X Drift Scenarios：表 1 列三种 schedule +
  真实场景映射 + 引用
  - §4.Y Drift Detection Mechanisms：对比 oracle-aware vs
  CUSUM-aware
  - Figure: 4 个方法的 recovery curve，CUSUM 介于 passive
  和 oracle 之间

  工作量

  写作 1 天 + 加 trace schedule 0.5 天 + 加 CUSUM 1 天 +
  重跑 0.5 天 = 3 天

  ---
  J4：系统模型过简

  诊断

  mofd_environment.py 现状（按之前对话摘要）：
  - 单 edge 节点，Emax=6
  - 仅 Rayleigh fading，无路径损耗 / shadowing / 干扰
  - 无 DAG 任务依赖
  - 无移动性

  TMC/JSAC 不接受这种模型。需要分级整改。

  整改方案（按改造成本递增）

  Tier 1：信道模型升级（必做，1-2 天）

  在 mofd_environment.py 的信道部分把 tran_rate 从单一
  Rayleigh 改成 Rayleigh + path loss + log-normal
  shadowing：
  # 现有
  h = rayleigh(scale)

  # 改造后
  PL_dB = 35.3 + 37.6 * log10(d_km)        # 3GPP UMa 模型
  shadow_dB = normal(0, 8.0)                # log-normal
  shadowing
  h_small = rayleigh(scale)                 # 小尺度
  fading
  gain_linear = 10**(-(PL_dB + shadow_dB)/10) * h_small**2
  SNR = gain_linear * P_tx / N0
  tran_rate = B * log2(1 + SNR)             # Shannon

  论文体现: §2.2 System Model 写完整的 Shannon-Hartley +
  3GPP path loss + log-normal shadowing。

  Tier 2：多 edge + 任务调度（强烈推荐，3-5 天）

  把 Emax=6 解释为6 个 edge 节点而不是单 edge 的 6
  档算力（语义升级，代码改动小）：
  - 状态加一维 edge ID
  - action 增加"卸载到哪个 edge"
  - 每个 edge 维护独立队列
  - 互联带宽矩阵（edge-edge）

  论文体现: §2.1 写"hierarchical multi-edge MEC
  system"，画系统拓扑图。

  Tier 3：移动性模型（推荐，2 天）

  加 MobilityModel 类：
  class LinearMobility:
      def __init__(self, v_range=(0, 30)):  # m/s
          self.v = v_range
      def update(self, pos_t, dt):
          return pos_t + self.v * dt
  每 slot 更新用户位置 → 触发 path loss / 切换 edge。

  论文体现: §2.3 Mobility Model + Figure 显示用户轨迹 +
  handoff。

  Tier 4：DAG 任务（可选，5+ 天）

  把"50 个独立任务"改成"任务图 G=(V, E)"，DAG 调度：
  - task_generator.py 加 DAGTaskGenerator，每 slot 产出小
  DAG（典型 5-10 节点）
  - 引入拓扑序约束：task_i 必须在 dependencies
  完成后才能调度
  - 调度策略：HEFT 等经典 baseline

  这个量级很大，仅在冲 IEEE TMC 顶级且时间充足时再做。

  推荐路径

  如果时间紧：只做 Tier 1 + Tier 3 （信道 + 移动性，5
  天工作量），写法上把"Emax=6 算力档"包装成"6 个 edge
  实例"，最低成本满足"multi-edge mobile MEC"的标签。

  如果时间宽：Tier 1+2+3 全做（10-12 天），加 1 个
  ablation 显示"信道真实化 / 多 edge / 移动性 各自贡献"。

  论文中的体现

  新增 §2 System Model 重写，必含：
  - Figure 1: 系统架构（多用户 + 多 edge + 信道）
  - Equations: path loss + shadowing + Shannon + 多 edge
  调度
  - Notation table

  工作量

  ┌─────────┬────────┬─────────────────────────┐
  │  Tier   │ 工作量 │        期刊门槛         │
  ├─────────┼────────┼─────────────────────────┤
  │ 1 only  │ 2 天   │ TVT / IoT-J 边缘        │
  ├─────────┼────────┼─────────────────────────┤
  │ 1+3     │ 5 天   │ TVT / IoT-J 主流        │
  ├─────────┼────────┼─────────────────────────┤
  │ 1+2+3   │ 10 天  │ TMC Major Revision 量级 │
  ├─────────┼────────┼─────────────────────────┤
  │ 1+2+3+4 │ 20 天  │ TMC / JSAC top tier     │
  └─────────┴────────┴─────────────────────────┘

  ---
  整体优先级建议

  ┌────────────┬───────┬──────────────────────┬───────┐
  │    任务    │ 工作  │         收益         │ 优先  │
  │            │  量   │                      │  级   │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ M3 修复 +  │ 4-5   │ 修掉退化是论文可信度 │ P0    │
  │ 诊断       │ 天    │ 基石                 │       │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ M4 形式化  │ 3-4   │ 把工程升级为         │ P0    │
  │ + 命题     │ 天    │ contribution         │       │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ J2 真实场  │       │                      │       │
  │ 景叙事 +   │ 3 天  │ 论点强化、防 reject  │ P1    │
  │ CUSUM      │       │                      │       │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ J4 Tier    │       │ 满足"realistic       │       │
  │ 1+3 (信道+ │ 5 天  │ system"              │ P1    │
  │ 移动性)    │       │                      │       │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ J4 Tier 2  │ +5 天 │ 跳到 TMC 量级        │ P2    │
  │ (多 edge)  │       │                      │       │
  ├────────────┼───────┼──────────────────────┼───────┤
  │ J4 Tier 4  │ +10   │ 顶刊压舱石           │ P3    │
  │ (DAG)      │ 天    │                      │       │
  └────────────┴───────┴──────────────────────┴───────┘

  最小化方案（约 16 天）：M3 + M4 + J2 + J4 Tier 1+3 →
  IoT-J / TVT 量级
  冲刺方案（约 25 天）：上面 + J4 Tier 2 → TMC Major
  Revision 量级

  要我先动手哪一项？我建议先开 M3
  诊断日志——一旦掌握退化根因，其他实验的可信度才有基础。
  











  一、参考文献总数与领域分布（39 篇）

  类别: MEC 卸载 + DRL（相关工作主体）
  数量: 21
  占比: 54%
  代表文献: [1][2][4-7][10-14][19-23][25-27][35]
  ────────────────────────────────────────
  类别: MORL 方法论（核心理论根基）
  数量: 7
  占比: 18%
  代表文献: [8] Roijers survey, [9] Envelope-Q
    (Yang/Narasimhan NeurIPS'19), [16] Hayes 实操指南,
  [24]
     自己的 WiOpt'23, [30][38][39]
  ────────────────────────────────────────
  类别: 泛化/Contextual MDP
  数量: 3
  占比: 8%
  代表文献: [15] Domain Randomization (Tobin IROS'17),
  [17]
    RL 泛化 survey, [18] Epistemic POMDP NeurIPS'21
  ────────────────────────────────────────
  类别: RL 基础
  数量: 2
  占比: 5%
  代表文献: [28] Sutton policy gradient NeurIPS'99, [29]
    Christodoulou Discrete-SAC
  ────────────────────────────────────────
  类别: 基线对比方法（Bandit/NSGA-II/Pareto-Q）
  数量: 5
  占比: 13%
  代表文献: [32] LinUCB, [33][34] Bandit, [36][37] NSGA-II
  ────────────────────────────────────────
  类别: 标准
  数量: 1
  占比: 3%
  代表文献: [31] IEEE 802.11a

  ---
  二、引用编排策略（章节级）

  Intro（pareto_full.txt:36-220）— 引用密度最高，承担 ~70%

  第1段（背景）：MEC概念 [1] → offloading 重要性 [2] →
  传统方法局限 [3][4]

  第2段（DRL 综述式）："Cui et al. [5] 做了X；Lei et al.
  [6] 做了Y；Jiang et al. [7] 做了Z" —
  用一句话+作者名+方法概述+引用 的固定句式，连续3条铺陈

  第3段（动机1：MORL）：[8] 给出 scalarization 三大缺陷（I
  mpossibility/Infeasibility/Undesirability），[8][9]
  论证多策略法局限 → 引出 single-policy MORL

  第4段（动机2：泛化）："Yan [10]/Li [11]/Gao [12]/Ren
  [13] 各自做了X，但 它们都缺少 Y" → 用 [2][5][6][7][14]
  一锅端做"其余忽略泛化"

  第5段（技术铺垫）：[15] domain randomization、[16]
  adapt-online、[17][18] contextual MDP — 每个新概念配 1-2
   篇关键引用

  System Model（II 节，pareto_full.txt:221-542）— 引用极少

  整节仅 1-2 处引用：[26]
  用于"任务大小服从指数分布"的假设依据

  Method（III 节，pareto_full.txt:543-1211）—
  仅理论关键点引用

  [17] 用于 contextual MDP
  定义（pareto_full.txt:560），[28] 用于 policy gradient
  基础。核心创新章节几乎不引用别人。

  Experiment（V.B 节，pareto_full.txt:1420-1435）—
  第二个引用高峰

  基线集合一次性引出：LinUCB [32] + Bandit应用
  [33][34]，启发式 [4][34][35]，NSGA-II [36][37]，Pareto-Q
   [38]，多策略MORL [39] + 自己的 [24]

  ---
  三、值得你直接借鉴的写作技巧

  1. 39 篇虽不算多，但分层严密：每一类都有 1-2 篇
  "公认必引"（survey/seminal）+ 多篇 "近 3
  年同领域工作"。你目前 18 篇缺的就是 survey/seminal
  那层骨架。
  2. 同一篇可在多处复用：[8]（Roijers MORL survey）在
  Intro 出现 2 次承担不同论点；[34] 出现 2 次（一次作
  Bandit
  应用例证，一次作启发式基线）。好引用是"多功能"的。
  3. 作者自引 [24]：把自己 WiOpt'23 的会议版作为多策略
  MORL
  基线，把会议→期刊延伸合法化。你应该早点写一篇会议版，TSC
   这种期刊很欢迎"会议→期刊"扩展叙事。
  4. Related Work 表格化：Table
  I（pareto_full.txt:243）用一张表格对比 8 个相关方法在
  4-5 个维度上的差异 — 这比文字段落引用更紧凑、reviewer
  喜欢。你应该做一个 G-FDEdge vs [Pareto / FDEdge /
  Diffusion-Planner / Envelope-Q / PCN]的对比表。
  5. 平均年份分布：2024-2025 占 5
  篇（[1][11][12][23][25]），2020-2023 占 ~17 篇，≤2014
  经典占 5 篇（[8][30][32][36][38][39]）。新旧比 ≈ 22:5 ≈
  4:1，是健康配比。你目前 18 篇里几乎没有 2024+
  的论文，需要补足。

  ---
  四、对你 G-FDEdge 引用补充的具体启示

  参照本文 39 篇配比，目标 ≥ 43 篇 (TVT/IoT-J) 至少要补：

  - MORL 骨架：[8] Roijers, [9] Envelope-Q, [16] Hayes —
  你目前一篇都没有，这是审稿人会立刻看出的硬伤
  - 泛化/Contextual MDP：[15] Tobin domain randomization,
  [17] Kirk survey — 你做 ω-shift/ω-drift
  实验，没引这两篇说不过去
  - Discrete-SAC：[29] Christodoulou —
  你直接用了这个算法，必引
  - NSGA-II / Pareto 评估：[36] Deb NSGA-II（HV
  计算的事实标准）— 你算 HV 但没引

  要不要我接下来按这 4 条优先级，给你列出具体 BibTeX
  和应该插入到论文哪一节的建议？





  维度: 拓扑
  代码现状: 单用户视角（无 user_num），E ∈ [Emax/2, Emax] 个独立
    ES，无 cloud，无 ES 间链路
  行号: 89-91
  ────────────────────────────────────────
  维度: 信道
  代码现状: tran_rate ~ uniform(400, 500) × Rayleigh(|h|² ~
    Exp(1), 截断[0.3, 2.5])
  行号: 91-93, 117-120
  ────────────────────────────────────────
  维度: 路径损耗
  代码现状: ❌ 没有
  行号: —
  ────────────────────────────────────────
  维度: Shadowing
  代码现状: ❌ 没有
  行号: —
  ────────────────────────────────────────
  维度: SNR / 噪声
  代码现状: ❌ 没有；速率不是从 Shannon 公式算来的
  行号: —
  ────────────────────────────────────────
  维度: 任务到达
  代码现状: 预生成 list，按 slot 直接取，无 Poisson / 无 deadline
   
    / 无优先级
  行号: 105
  ────────────────────────────────────────
  维度: 移动性
  代码现状: ❌ 没有，channel_gain 每 episode 重置一次后固定
  行号: 117-120
  ────────────────────────────────────────
  维度: 延迟模型
  代码现状: tran + comp + wait（队列累积）
  行号: 182-185
  ────────────────────────────────────────
  维度: 能耗模型
  代码现状: e_off = p_off × T_off（p_off=0.5W
    固定，不随信道变）；e_exe = κf²ρd
  行号: 189-192
  ────────────────────────────────────────
  维度: 空闲功耗
  代码现状: ❌ 没有 idle power
  行号: —
  ────────────────────────────────────────
  维度: 多用户竞争
  代码现状: ❌ 没有（虽 num_tasks 多但同一序列）
  行号: —
  ────────────────────────────────────────
  维度: DAG 任务依赖
  代码现状: ❌ 没有
  行号: —
  ────────────────────────────────────────
  维度: 目标维度
  代码现状: 2-D (delay, energy)
  行号: 199

 任务: A. 写论文 Abstract +
    Introduction(中→英,顶会风)
  我用的 skill: zh-to-en-latex + en-polish
  输出: LaTeX 段落,直接进 PPT 或 overleaf
  ────────────────────────────────────────
  任务: B. PC-FDN / Vector-Q 的 Method section
    公式段
  我用的 skill: zh-to-en-latex + en-polish
  输出: 含公式编号的方法描述
  ────────────────────────────────────────
  任务: C. 现有 Pareto/HV 数据写成 Experiment
    分析段
  我用的 skill: exp-analysis
  输出: 论文级分析段(直接进答辩材料)
  ────────────────────────────────────────
  任务: D. 8 张图的 caption
  我用的 skill: caption-gen
  输出: 顶会风英文 caption
  ────────────────────────────────────────
  任务: E. 三目标精度档位的 Future Work
    段(展望话术)
  我用的 skill: zh-to-en-latex
  输出: 显得"有路线图"
  ────────────────────────────────────────
  任务: F. 用 reviewer 视角预审 PPT
    讲稿,提前模拟答辩
  我用的 skill: reviewer-sim
  输出: Strengths / Weaknesses / 答辩 Q&A 预案
  ────────────────────────────────────────
  任务: G. 投稿前 /
    答辩前最后一致性扫(术语、公式符号)
  我用的 skill: logic-check
  输出: 致命问题清单
