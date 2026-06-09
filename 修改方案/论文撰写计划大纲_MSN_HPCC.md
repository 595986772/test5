# FDEdge / PC-FDN 论文撰写计划大纲（目标会议：MSN / HPCC）

> 生成日期 2026-06-09。本大纲基于对本项目代码、`修改方案/` 全部方案、`创新点.docx`、
> README、以及记忆里历次方向决策的通读，并对 MSN / HPCC 既往相关论文做了检索定位。
> **红线**：本大纲区分「已验证真实数据 / 已删除作废数据 / 尚未验证的假设」三类，
> 写论文时只许引用第一类。任何具体 HV/delay/energy 数字写进论文前必须 Read 原始文件核对。

---

## 0. 数据真实性总账（动笔前必须认清的现状）

| 类别 | 内容 | 能否写进论文 |
|---|---|---|
| ✅ **已验证真实** | ① PC-FDN V1 边缘前沿 `results/pc-fdn_pareto_aggregated.csv`（21 点，单 seed）：delay∈[27.69,106.62]s，energy∈[2.68,3.99]J，ρ(ω_E,delay)=+0.884，ρ(ω_E,energy)=−0.949，E-spread/mean=0.362。<br>② V8 H-MCSS 源消融 `results/ablation_hmcss_v8_summary.txt`：src_prior 459.88 > src_feedback 407.25 > src_random 380.19 > src_full3 369.79（共享 ref，单 seed）。<br>③ 校准卷子重测 `run_recheck_testset.py`：oracle-greedy(hard) HV=950.755（K=20，固定 ref）；4 个训练模型 HV 跑完补。<br>④ 机制诊断探针：D1 内容探针 +50pp、D2 起点方差→尖锐度、D3 full3 净负（均单 seed）。 | 可用，但都要标「single seed」并补多 seed |
| ❌ **作废/已删（中期造假）** | 所有 baseline（SA/LinUCB/NSGA-II/Random/RR/Discrete-SAC/GenMOSAC/LDQN，V1+V2）、PC-FDN 的 V2 版本、**Opt 物理下界**。旧 HV 排名（V1≈294.8、V2≈10.81 等）全部作废。 | **禁止引用，必须重跑** |
| ⚠️ **尚未验证（假设）** | PGW 端到端大幅胜过纯贪心/纯扩散；去噪器「纠正」而非「照抄」贪心；适中温度最优；三目标 accuracy 通道（代码 N_OBJ=2，未实现）。 | 跑出来前**禁止当结论写** |

**一句话现状**：手里只有 1 条真前沿 + 1 张真消融表 + 几个真诊断探针，全是单 seed；
**没有任何真 baseline**。论文的实验章节本质上是「待建」，这是最大瓶颈，下面 §6/§7/§8 重点解决。

---

## 1. 投稿定位与故事线

### 1.1 venue 适配判断（MSN / HPCC）

- **MSN**（Int’l Conf. on Mobility, Sensing and Networking，CCF-C/EI）与 **HPCC**（IEEE Int’l Conf.
  on High Performance Computing and Communications，CCF-C/EI）都有成熟的「边缘计算 + 任务卸载 + DRL/调度」赛道。
  既往可对标：MSN’21 *Distributed task offloading based on multi-agent DRL*、MSN’20
  *Game-theoretic joint offloading & resource allocation*；HPCC’19 *Task Scheduling in MEC with
  Stochastic Requests and M/M/1 Servers*、HPCC’19 *Energy-Efficient Cooperative Edge Computing*。
- **录用标准画像**：清晰系统模型 + 合理 DRL 方法 + **可信的 baseline 对比（HV/delay/energy）** +
  适度 novelty。**不要求** NeurIPS 级理论或纯 mechanism-finding。这对本项目是好消息——
  当前成熟度（一个能跑、有真前沿的系统）正好卡在这一档，门槛可达。
- **策略含义**：把论文写成「**有效的系统方法 + 干净的多目标实验**」，而不是赌「correct-vs-copy
  的微妙机制发现」。机制分析作为加分小节，不作唯一卖点。

### 1.2 两个候选故事线 + 推荐

**Framing A（推荐，稳）——系统方法主线**
> *PC-FDN: Preference-Conditioned Feedback-Diffusion for Multi-Objective Edge Offloading.*
> 单策略按用户偏好 ω 自适应分配 delay/energy，铺出 Pareto frontier；并用 **PGW（偏好贪心暖启动）**
> 把去噪起点从无信息升级为偏好正确的物理先验，由学习去噪器纠正其拥塞近视。
> 主卖点 = ω-自适应分配（已有真前沿支撑）+ PGW 增强 + 干净多目标评测协议。

**Framing B（激进，险）——机制发现主线**
> 以「学好的去噪器到底纠正还是照抄贪心先验？」为核心 finding。
> 风险：对 MSN/HPCC 偏 subtle，依赖 negative-result 叙事，且严重依赖 PGW 端到端结果（现未验）。

**推荐：以 A 为骨架，把 B 的「correct-vs-copy」做成 Discussion 里的机制分析小节。**
这样即使 PGW 端到端只是「打平稍胜」，论文仍因「系统 + 多目标前沿 + 评测协议」站得住；
若 PGW 大赢，则机制小节顺势升级为亮点。**双保险，不把全部赌注压在一个未验证假设上。**

### 1.3 论文一句话主张（draft）
> *We present PC-FDN, a preference-conditioned feedback-diffusion policy that lets a single model
> trace the delay–energy Pareto frontier of an edge-offloading system, and PGW, a
> preference-conditioned greedy warm-start that replaces the uninformative denoising start with an
> online physical prior refined by the learned denoiser, improving both objectives at 1× inference cost.*

---

## 2. 整篇结构大纲（逐节：写什么 / 现有素材 / 还缺什么）

> 目标 ~10–12 页双栏（MSN/HPCC 常见篇幅）。

1. **Abstract**（~200 词）
   - 写：MEC 多目标卸载的偏好未知/多变痛点 → 单策略 ω-自适应 → PGW 暖启动 → 主结果（HV/delay/energy）。
   - 缺：最终主结果数字（等真 baseline + 多 seed）。

2. **Introduction**
   - 写：①卸载的 delay-energy 冲突 + 偏好因用户/场景而异；②现有单目标/固定权重方法的不足；
     ③本文贡献（C1 ω-自适应单策略前沿；C2 PGW 暖启动；C3 干净评测协议 + 真 baseline 对比）。
   - 素材：`创新点.docx` 缺陷/动机；GMORL/WA-D3QN 痛点表述。
   - 缺：贡献点要**砍到能兑现的**——删掉「三目标 accuracy」「漂移鲁棒」等未实现/已否决项。

3. **Related Work**（详见 §3）
   - 三簇 + contrast。素材充足（`创新点.docx` 38 篇 + 检索到的 MSN/HPCC 例子）。

4. **System Model & Problem Formulation**（详见 §4）
   - 写：MEC 架构、任务/信道/队列模型、delay/energy 物理式、CMO-MDP、偏好 ω、scalarization、Pareto/HV 定义。
   - 素材：`mofd_environment.py` 的 step/reward、`greedy_warmstart.py` 的 cost 公式、`fixed_testset.py` 的 ref/HV。
   - 缺：把代码物理**形式化成 LaTeX 方程 + notation 表**（目前散在代码里）。

5. **Method: PC-FDN + PGW**（详见 §5）
   - 写：feedback-diffusion actor（引用 backbone）、vector-Q critic（V8 干净版）、SAC-α、COR、PopArt-lite、
     ω 条件化、PGW（贪心物理先验 + 温度 + 去噪精修）。
   - 缺：**架构图、PGW 流程图、训练/推理伪代码**（全无现成图）。

6. **Experiments**（详见 §6 —— 最关键、最缺）
   - 写：setup、主表（HV/delay/energy vs baselines）、Pareto 前沿图、PGW 三臂+温度扫、组件消融、
     偏好一致性（Spearman）、泛化（未见 ω / 变 Emax）、计算成本。
   - 缺：**几乎全部真实结果**（见 §0/§7）。

7. **Discussion**（机制分析 = Framing B 内容）
   - 写：correct-vs-copy 的证据（温度扫 + 三臂）、H-MCSS 净负的教训、与物理下界的差距。

8. **Conclusion + Limitations**
   - 写：诚实写单 seed→多 seed、平稳环境、离散动作、未做三目标等边界（反而加分）。

---

## 3. 相关工作（Related Work）写法

按「**三簇 + 每簇 contrast + 留 gap**」组织，结尾一句话收束本文定位。避免流水账。

**Cluster 1 — DRL-based task offloading in MEC（建制方法）**
- 代表：DQN/Double-DQN+LSTM（Tang & Wong, TMC’22）、SAC、D2SAC/AGOD（Du et al., TMC’24，扩散）、
  DRL-OS（D3QN）、MSN’21 multi-agent DRL、HPCC’19 MEC scheduling。
- contrast：多为**单目标或固定权重**；偏好变化要重训。→ gap：缺单策略覆盖整条偏好谱。

**Cluster 2 — MORL / preference-conditioned / Pareto offloading（直接竞品）**
- 代表：**GMORL**（arXiv:2509.10474，单策略泛化、+121% HV，本文头号对标/baseline）、
  **WA-D3QN**（Automotive Innovation’24，偏好权重自适应、+23.6% HV over NSGA-II）、
  PGMORL、Envelope Q-Learning、Pareto Conditioned Networks、**PCRL/C-MORL**（2026 强基线，至少讨论）。
- contrast：GMORL 藏 ω 估偏好；本文 ω 已知静态输入，专注**去噪起点这一被忽视的设计杠杆**。
  → gap：没人研究「扩散卸载策略的偏好条件化暖启动」。

**Cluster 3 — Diffusion policies for decision/offloading（方法骨干所在）**
- 代表：FDN/feedback-diffusion（本文 backbone，TMC’25，delay-only，**明确引用、不主张为创新**）、
  D2SAC、DIPO/DPPO（on-policy 训扩散，可在 Limitations 提为未来升级）。
- contrast：原 FDN 无 ω、起点是「上一步动作」；本文是**偏好条件化 + 物理贪心起点 + 拥塞纠正**。
  → gap：多目标设定下特有的暖启动设计旋钮。

**收束句模板**：
> Unlike prior MORL offloading that learns to infer preferences or aggregates fixed weights, and
> unlike single-objective diffusion offloading, we treat the *denoising start* of a
> preference-conditioned diffusion policy as a design lever and inject an online physical greedy prior.

---

## 4. 系统建模（要补齐的方程，按写作顺序）

> 目标：把现有代码物理「翻译」成论文级 LaTeX。以下每条都对应已有代码实现，**不是新建模**。

1. **MEC 架构与 notation 表**：master node + Emax 台 edge server（可变 E），时隙 t∈[0,T)，
   每时隙到达任务集 N_t，任务 n 比特量 d_n、计算密度 ρ。**先做一张 notation 表**（目前缺）。
2. **信道 / 传输速率**：v = tran_rate[e]·channel_gain[t,n,e]（现为标量增益；可在 Limitations 说明
   未建 Rician/path-loss，或按 `创新点.docx` 2.1-4 升级为 Shannon 式——升级会改数值，谨慎）。
3. **延迟模型**：delay_e = tran_d + comp_d + wait_d，其中
   tran_d=d_n/v，comp_d=ρ_d/f_E[e]，wait_d=(queue_len+queue_bef)/f_E[e]（队列等待是拥塞来源）。
4. **能耗模型**：energy_e = p_off·tran_d + κ·f_E[e]²·ρ_d（计算能耗随频率平方）。
5. **CMO-MDP**：state s_{t,n}（含归一化信道/队列/任务 + ω 两维）、action a∈{1..E}、
   向量奖励 r_vec=(−delay·delay_scale, −energy·energy_scale)（delay_scale=0.05, energy_scale=0.25，
   通道归一化，要在论文里**显式说明并给消融**，否则前沿塌）。
6. **偏好与标量化**：ω=(ω_T,ω_E)，ω_T+ω_E=1；scalarized return = ω_T·Q_delay+ω_E·Q_energy。
   偏好集 = [0,1] 均匀网格 21 点。
7. **Pareto frontier 与 HV**：定义支配关系、Pareto 前沿、hypervolume(HV) 与固定参考点 ref
   （`fixed_testset.py`：随机策略 nadir×1.1，方法无关、存盘共享——这是评测协议贡献 C3 的核心，要写清）。
8. **问题陈述**：单策略 π_θ(·|s,ω) 使「在整条 ω 谱上的期望 HV 最大」，等价于覆盖 Pareto 前沿。

---

## 5. 方法章节（PC-FDN + PGW）

**5.1 整体框架**（需画图 Fig.1）：SAC 框架，actor=feedback-diffusion(FDN)，critic=vector-Q。
**5.2 偏好条件化反馈扩散 actor**：DDPM T=3 步去噪，输入(时间步 I，起点概率 x_{t,n,I}，状态 s)，
  输出动作分布；ω 进 state 做条件化。**要给步数消融**（回应「3 步是否太浅」质疑，`创新点.docx` 缺陷6）。
**5.3 向量 critic（V8 干净版）**：纯逐目标回报 target（去熵）+ 等权 critic loss（仅 PopArt-lite 1/σ²），
  避免极端偏好下能耗通道零梯度。actor 仍用 _scalarize_q(ω) 注入偏好。
**5.4 PGW —— 偏好贪心暖启动**（核心贡献，需画图 Fig.2 + 伪代码 Alg.1）：
  - 逐决策现算 `greedy_omega_prior`：对每台有效 server 算 myopic cost_e（复刻 env 物理），
    prior = softmax(−cost_e/τ) over valid servers；
  - 用 prior 作去噪起点，**单源、无每步 Critic 选**（因 D3：full3 净负）；
  - 温度 τ 是「纠正 vs 照抄」旋钮：τ→0 照抄贪心，τ 适中留出摊负载空间；
  - **可部署性说明**（防 reviewer 质疑信息泄漏）：prior 由编排器侧已知系统信息现算，只用当前量、不用未来。
**5.5 COR / PopArt-lite**：简述，作为组件消融对象。

> 缺：Fig.1 框架图、Fig.2 PGW 暖启动示意、Alg.1 训练伪代码、Alg.2 PGW 推理伪代码——**全部要新画**。

---

## 6. 实验设计（最关键 —— 当前最大缺口）

### 6.1 Setup（要写死并固定）
- env：Emax=6，num_tasks_max=50，time_slots=100，bit∈[10,40]，f∈[10,40]，delay_scale=0.05/energy_scale=0.25。
- 评测：**固定卷子**（21 ω × K 场景，钉死随机种子，共享 ref）——这套协议本身是贡献 C3，先在 setup 讲清。
- seed：**3–5 个**（现全 1 seed，审稿第一硬伤，必补）。

### 6.2 主实验表（HV / delay_min / energy_min / E-spread）—— vs 真 baseline
必跑 baseline（按强度排，**全部要真跑**）：
1. **Opt**（穷举物理下界，上界参照）—— 旧的已删，**重跑**；
2. **GMORL**（头号 MORL 竞品，arXiv:2509.10474，有开源参考）；
3. WA-D3QN（偏好自适应 MORL，可选）；
4. Discrete-SAC / D2SAC（扩散 RL）/ LDQN / DQN（建制 DRL）；
5. Greedy-myopic（=PGW 的 arm A，本身是强启发式基线）、Random、Round-Robin。
> 计算预算对齐：SA 类每偏好搜上万次不公平（`创新点.docx` 缺陷5），若用须报 wall-time/FLOPs 对齐。

### 6.3 核心图
- **Fig.A Pareto 前沿对比**（21 点 delay-energy 散点 + 前沿线，PC-FDN vs baselines vs Opt）——
  现有真 V1 前沿可直接进图（标 single seed）。
- **Fig.B HV 收敛曲线**（vs episode，证明收敛快，对标 GMORL/FDN）。
- **Fig.C 偏好响应曲线**（delay/energy vs ω_T，证 ω-自适应单调）——用 ρ=+0.884/−0.949 那组真数据。

### 6.4 PGW 三臂 + 温度扫（贡献 C2 的判据，go/no-go）
| 臂 | 起点 | 判据 |
|---|---|---|
| A Greedy-myopic | 纯贪心无去噪 | — |
| B Diffusion-uninformative | 零/均匀起点（现状） | — |
| C PGW | 贪心物理先验起点 | **C≫max(A,B)→真贡献；C≈A→照抄枪毙；C≈B→暖启动无效枪毙** |
- 温度 τ∈{→0, 0.5, 1.0, 2.0} 扫，回答 correct-vs-copy。
- **本对话正在跑的校准重测**正是这条线的前置证据：① greedy vs 训练模型（greedy 碾压→坐实 PGW 动机）；
  ② src_prior > src_feedback +13%（informed-prior > learned-feedback，支持「有信息起点」论点）。

### 6.5 组件消融（贡献验证，`创新点.docx` 缺陷2）
PC-FDN / w/o COR / w/o PopArt-lite / w/o ω-conditioning(单目标) / 扩散步数 T∈{1,3,5,10}，各 3 seed，看 HV 各掉多少。

### 6.6 泛化实验（支撑 “generalizable” 宣称）
- 未见 ω 外推（训练用稀疏 ω，测试用密 ω）；
- 变 Emax∈{3,8,10} 看 HV 衰减（拓扑泛化）。

### 6.7 计算成本
PC-FDN/PGW 训练+推理时间、PGW 贪心先验开销（O(Emax) 解析，远小于一次去噪前向）。

---

## 7. Gap 总清单（论文还缺什么，按优先级）

### 🔴 致命（不补无法投稿）
1. **真 baseline 实验全缺**：至少 Opt（重跑）+ GMORL + Discrete-SAC + Greedy + Random/RR。现状无任何真对比。
2. **多 seed**：所有真数据都是 single seed，必须 3–5 seed 报 mean±std。
3. **PGW 端到端结果未验**：三臂 + 温度扫要真跑出 go/no-go（现仅假设 + 诊断探针）。
4. **贡献点与实现对齐**：删掉论文里所有未实现/已否决的宣称（accuracy 三目标 N_OBJ=2 未做、ω-buffer 漂移已否决、
   Set-Transformer 已删、H-MCSS 多源已知净负）。中期把三目标写成第三大贡献——**必须撤掉或降级为 future work**。

### 🟠 重要（决定论文档次）
5. **系统建模 LaTeX 化 + notation 表 + 问题形式化**（§4，现散在代码）。
6. **方法图 + 伪代码**（Fig.1/Fig.2/Alg.1/Alg.2，全无）。
7. **组件消融 + 扩散步数扫**（证明每个组件有用、3 步非随意）。
8. **评测协议 C3 的论文化**：固定卷子 + 共享 ref + 确定性，写成方法贡献而非附录细节。

### 🟢 锦上添花（有时间再做）
9. 泛化实验（未见 ω / 变 Emax）；10. 计算成本表；11. PCRL/C-MORL 等 2026 强基线的讨论段；
12. 真实 trace（Alibaba cluster）重放替代纯采样分布。

---

## 8. 执行路线图（从现在到投稿，有序）

> 原则：先把「能不能成论文」的 go/no-go 跑完，再写正文，最后补图润色。**不在未验证假设上提前写结论。**

**阶段 1｜锁定核心结论（1–2 周）**
- [ ] 等当前**校准重测**跑完 → 得「greedy vs 4 训练模型」+「prior vs feedback」校准排名（回答 PGW 动机是否成立）。
- [ ] 实现 `greedy_warmstart.py` + `run_pgw_ab.py`（§5.4），单 seed 跑 PGW 三臂 + 温度扫 → **go/no-go**。
- [ ] go：进阶段 2；no-go（C 照抄/无效）：按红线如实降级 PGW，回退 Framing 到「PC-FDN 系统 + 多目标前沿」主线。

**阶段 2｜补齐真 baseline + 多 seed（2–4 周，算力密集）**
- [ ] 重跑 Opt 物理下界（旧已删）；跑 GMORL（开源参考）+ Discrete-SAC + Greedy + Random/RR；
- [ ] PC-FDN / PGW / 关键 baseline 各 3–5 seed，固定卷子统一 ref；
- [ ] 产出主表 + Fig.A/B/C + 组件消融表 + 步数扫。

**阶段 3｜成文（2–3 周）**
- [ ] 系统建模 LaTeX 化（§4）→ 调 `zh-to-en-latex`；
- [ ] 方法图 + 伪代码；Related Work（§3）；
- [ ] 实验分析段 → 调 `exp-analysis`；英文润色 → `en-polish` → 去 AI 味 `dehumanize`；
- [ ] 配图 caption → `caption-gen`；投稿前 `reviewer-sim` 全篇审 + `logic-check` 一致性扫。

**里程碑判据**：阶段 1 的 go/no-go 决定主卖点；阶段 2 的主表决定能否投；阶段 3 决定档次。

---

## 9. 风险与红线

| # | 风险 | 对策 |
|---|---|---|
| R1 | PGW 端到端只打平 | 双保险 Framing A：系统 + 多目标前沿仍成立，PGW 降级为「暖启动加速/不劣」 |
| R2 | 重跑 baseline 算力/时间不够 | 优先 Opt+GMORL+Greedy+Random 四个最关键的；其余按时间补 |
| R3 | reviewer 质疑 PGW 用了状态外的 tran_rate=作弊 | §5.4 明写「编排器侧在线已知量、不用未来」，类比 Greedy-Min-Delay 基线同样用 |
| R4 | 环境过简（平稳、单跳、无衰落） | 写进 Limitations，引 `创新点.docx` 2.1 的升级路径作 future work |
| R5 | 再次踩造假 | **红线**：任何数字进论文前 Read 原始文件核对；single seed 必标注；已删数据永不复活 |

---

### 附：可直接复用的写作素材位置
- 真前沿数据：`FDEdge-main/results/pc-fdn_pareto_aggregated.csv`
- 真消融表：`FDEdge-main/results/ablation_hmcss_v8_summary.txt`
- PGW 完整方法/故事线/诚实边界：`修改方案/Preference_Greedy_Warmstart_Diffusion_完整修改方案.md`
- 文献地图（38 篇 + 2026 前沿 17 篇）：`FDEdge-main/创新点.docx`
- 评测协议实现：`FDEdge-main/fixed_testset.py`、`run_recheck_testset.py`
- 物理公式来源：`FDEdge-main/mofd_environment.py`（step/reward）、`greedy_warmstart.py`（cost）
- 术语固定表 + 写作红线：`.claude/skills/paper-writing`
