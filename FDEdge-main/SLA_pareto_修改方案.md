# SLA-约束下能耗-延迟 Pareto 权衡：环境 → 模型 → 评估 完整修改方案 (v2)

> 本文给 reviewer(GPT)审阅。目标:把当前"无约束 ω 加权 MORL"重构为"**延迟 SLA 约束下、p95-延迟与能耗的 Pareto 权衡**"。
> 标注规则:【现状】= 已验证的现有代码事实(含 file:line);【改】= 提议改动;【开放】= 需 reviewer 判断的决策点。
> v2 已把第一轮 review 的 4 条修订**就地并入正文**(旧错误说法已删),修订记录见末尾 §8。

---

## 0. 背景 / 为什么要改

【现状】系统是边缘任务卸载的双目标(delay, energy)MORL。环境 `MOFDEnvironment`(`mofd_environment.py`),主方法 `MOFD_SAC_V5_HMCSS`(`ablation_agents.py:83`,默认 `MCSS_MODE='full3'`)。**命名注意:类名带 "V5" 是遗留误名,实际继承链为 `MOFD_SAC_V5_HMCSS → MOFD_SAC_V8(mofd_v8.py:24)→ MOFD_SAC_V5`,即主方法是 V8(干净版 critic)**;训练结果 `abl_mcss_v8_src_*` 的 "v8" 前缀即指此。vector critic + ω-加权 PopArt 的 `update()` 主体在 `mofd_v5.py:313-368`(继承),但 critic 的 `_critic_target`/`_critic_weight` 被 **`mofd_v8.py:27/31` 覆盖**(去熵干净版)= 实际生效版本。训练 reward 通道 `r_T=-delay·0.05`、`r_E=-energy·0.25`(`mofd_environment.py:197-198`),标量化在各标量-reward baseline 为 `ω_T·α_T·r_T + ω_E·α_E·r_E`(`mofd_main.py:460`)。

【现状】诊断发现三件事:
1. **HV 评估有 bug**(已修 `helpers.hypervolume_2d`):不裁 ref 框外的点,给"高 delay/低 energy"极端点凭空灌水。**口径说明**:把已存的各方法 21 点前沿(`results/testset_*_pareto.csv`)在**同一固定 ref `(39.79, 4.68)`** 下用修复后的 HV 离线重算,greedy_omega 由 ~950 降到 ~32、greedy_energy ~291→0、ldqn ~32→5(数量级示意,确切值以重算脚本输出为准)。**注意**:线上的 `results/testset_compare.csv` 仍是**修复前**的旧数(尚未重刷,正在跑的 baseline 还按缓存旧函数 append),最终须统一重算后替换——勿直接引用该表当前数字。
2. **主方法不是逐点支配,而是用延迟换能耗**:实测 ω_T=0(纯能耗偏好)时 delay 冲到 ~200s(对照 greedy_delay ~22s);即便 ω_T=1 延迟侧仍略差于 greedy_delay。"延迟和能耗两条曲线都在 baseline 下面"**不成立**(权衡问题里本就难成立)。
3. **ref/盒子无原则**(随机 nadir×1.1 太小,全局 nadir 又被退化策略顶到量纲失衡)。

【改】重构思路:**引入延迟 SLA 约束**。超过 SLA 的解视为违约/不可行。于是:
- SLA 截止期 `D_sla` 成为**有领域依据的 ref_delay**,解决"3";
- 把模型约束在可行域内(不再冲极端延迟),目标从"无约束逐点支配"改为"**SLA 可行域内、p95-延迟与能耗上支配可部署 baseline**"(合法、可验证,但**不保证**,要靠模型真更优);
- 卖点(ω-自适应 + buffer 漂移鲁棒)保留,叙事更贴现实。

【开放】Q1:这个 reframe 是否构成"移动球门/回避弱点"?论据是 SLA 在边缘卸载里是标准设定且 `D_sla` 由数据派生(§1),并报 SLA 敏感性。

---

## 1. SLA 数值的设定与标定

【现状】env 物理量级粗算:task bits 10–40 Mbit,`tran_rate` 400–500(Mbps),`f_E` 10–40(GHz),`comp_density` 0.1–0.3(Gcycles/Mbit)。单任务 `delay ≈ bits/rate + density·bits/f + 队列项`,约 0.05–1.3 s,**大值(数十~200s)来自时隙内队列累积**。即 delay≈秒级。

【改】**路线 1(轻,先做,不重训不标定)——相对 SLA**:
- **主口径(预注册,与策略无关)**:对每个任务在所有可用服务器上算无队列 service time `tran_delay+comp_delay`,**取服务器维 min_e**(最优可服务基准),`D_sla = k × median_over_tasks(min_e service_time)`,`k ∈ {1.5,2,3}`。**主口径锁 min_e**(避免 min/median 含糊)。
- **敏感性(不与主口径并列)**:服务器维改 `median_e` 只作对照;随机策略 P50/P75 仅作量级 sanity(依赖策略、不可作主标定)。
- **校准集隔离(必写)**:`D_sla` **在独立 calibration set / 训练分布采样集上标定,标定后冻结**,再用于 fixed testset 与所有方法;**绝不用 testset 任务算 D_sla**(test leakage)。calibration set 与 testset 用不同种子从同分布采样、不相交。
- 用 3GPP/应用分级做**一句交叉印证**,说明该 slack 对应哪类业务。

【改】**路线 2(重,投稿更硬)——物理标定**:
- 经 `task_generator.py` 的 `trace` 模式(`TraceTaskGenerator`,读 `dataset/tasks_trace.csv`,格式 `slot,bits(Mbit)`)灌真实负载;把 `bit_range/f_range/tran_rate_range` 设成真实单位 → delay 落真实秒/ms → 直接套 ms 级 SLA。代价:数值全变、要重训重测。

【改】SLA 数值来源(可引用,**引用前须回原文核准确值**):
- **3GPP TS 22.261 / TR 22.804 / TS 23.501**;**ITU-T IMT-2020**(URLLC ~1ms、eMBB 等分级)。
- 应用分级典型量级:URLLC ~1ms;AR/VR motion-to-photon ~20ms;V2X ~10–100ms;云游戏 ~50–100ms;交互 Web ~100ms(Nielsen)。
- 真实 trace:**Alibaba Cluster Trace** / **Google Borg Trace** / **Azure Functions Dataset**。

【开放】Q2:delay≈秒级仿真下,直接套 ms 级 3GPP 数会量纲冲突;是否必须走路线 2 标定才能引 3GPP,还是路线 1 相对 SLA + 定性印证已足够?

---

## 2. 环境修改(`mofd_environment.py`)

### 2.1 可预测延迟 + SLA 准入掩码(orchestrator 安全层)【改】
- 新增 `predict_delay(t, n, e)`:复用 `step()` 同一公式(`:182-185`)给出"任务 (t,n) 派给 e 的预测延迟",决策前可算(`proc_queue_bef` 时隙内随分配累积,顺序相关但决策时点可得;逻辑已存在于 `eval_baselines_on_testset.py:gd_action_fn`)。
- 新增 `get_sla_mask(t, n, D_sla)`:在 `get_valid_mask()`(`:165`)基础上把 `predict_delay > D_sla` 的服务器置 0。
- **定位**:此掩码是 **orchestrator 侧的 admission-control 安全层**(系统组件,基于在线系统遥测 tran_rate / 队列负载),**对所有方法统一施加**;不是某个策略的私有信息(见 §5 措辞与 §2.3)。
- **可行性兜底**:某任务所有服务器都违约时不能空 mask → 回退到"预测延迟最小"(最小违约)的服务器,并由 `step()` 审计记一次违约(§2.2)。`step()` 的越界回退(`:173-174`)保留。

### 2.2 SLA 违约成本(独立通道,**不进 r_T**)+ step() 自审计【改】
- **不再把惩罚写进 r_T**(旧 v1 错误)。`step()` 计算 `viol_cost = max(0, delay − D_sla)`,作为**独立成本量**返回/累计,由训练侧以 ω-无关方式消费(§3.3);**`delay, energy` 原值不变**(`:199`),保证 Pareto/HV 不被污染。
- **step() 自审计(终审,与 mask 的事前预防分离)**:算出 `delay` 后,若 `D_sla is not None and delay > D_sla` → env 内累计 `self.sla_violations / self.sla_total`,暴露 `violation_rate()`(并可置 `self.last_viol_cost` 供训练循环取)。**不改返回 arity**(避免破坏 `r_T,r_E,d,e,real_a = env.step(...)` 的解包),违约统计走 env 属性。
- **violation_rate 以 step 审计为 ground truth**,不以 mask 为准 → 即使某路径漏接 `get_sla_mask` 也会被记录,不静默通过。
- **关键接线(训练侧)**:`run_episode` 在每次 `env.step(...)` 之后**立即读取 `env.last_viol_cost`**,构造三维奖励 `r_vec = (r_T, r_E, −last_viol_cost)` 存入 V8 replay buffer(`ReplayBufferV5` 的 reward 字段从 2 维扩 3 维,见 §3.3)。dsac/genmosac 走标量路径则直接 `r -= λ_sla·last_viol_cost`。`env.last_viol_cost` 每步在 `step()` 内刷新(无违约=0)。

### 2.3 SLA 可观测特征(可选,强化"非特权"叙事)【改/开放】
- 由于准入掩码用到 state 里没有的 `tran_rate` 本体与 `proc_queue_bef`(`:143-153` state 仅 `f_e/q_len_e/valid_e/gain_e`),若想让**策略本身**内化 SLA(而非只依赖外部安全层),在 per-server 特征(`per_server_dim=4`,`:67`)加 `effective_rate_e`(或 tran_rate_e)、`proc_queue_bef_e`、`slack=clip((D_sla−predict_delay_e)/D_sla,-1,1)`。
- **代价**:`state_dim`(`:68`)变 → 旧 ckpt 全废、必须重训。
- 【开放】Q3:加这些特征是否值回重训?A 路线(掩码=共享安全层,§5 据此措辞)已能合法地不写"策略无特权信息";B 路线只在"要论证策略自身 SLA-aware/对不完美掩码鲁棒"时才需要。两者是**独立 claim**。

### 2.4 新增 env 参数【改】
- `__init__` 增 `D_sla`(默认 None=关闭,向后兼容,V1 可复现)。

---

## 3. 模型 / 训练修改

### 3.1 准入掩码接入(最小改动)【改】
【现状】agent 已是 `take_action(state, mask, ...)`(dsac/ldqn/genmosac/主方法一致)。
【改】训练与评估取动作处把 `env.get_valid_mask()` 换成 `env.get_sla_mask(t, n, D_sla)`。网络几乎不动,改的是环境+循环;掩码对所有方法统一。

### 3.2 SLA 成本接线(ω-无关)【改】
- 关键:SLA 惩罚是 **ω-无关项**。若塞进 r_T,在 ω_T=0(纯能耗偏好)时会被 `ω_T·r_T` 整条吃掉、完全失效(`mofd_main.py:460`)。故**必须独立于 ω 权重**。
- dsac/**ldqn**/genmosac(标量 reward 路径 `r = ω[0]·αT·r_T + ω[1]·αE·r_E`):直接 `r -= λ_sla · viol_cost`(ω 之外;三者统一,勿漏 ldqn)。

### 3.3 主方法(V8)的实现:第三成本通道(v1 推荐)【改】
【现状】主方法是 **V8(vector critic)**:`update()` 主体在 `mofd_v5.py`(继承)——`r_vec` `[B,2]`(`mofd_v5.py:320`)、critic 输出 `[B,A,2]`(`:360`)、target_vec `[B,2]`(`:340`)、逐目标加权 loss `(w_main*err).sum(dim=1)`(`:357-368`)、PopArt σ 逐目标(`:344`);但 **critic hook `_critic_target`/`_critic_weight` 由 `mofd_v8.py:27/31` 覆盖**(去熵干净版)= 实际生效。**没有标量 target 可供"减 penalty"**,故"scalar target 后减 λ·viol_cost"在 V8 行不通。
【改,v1 推荐】**加第三个成本通道**,把 `N_OBJ`(`mofd_v5.py:52`)2→3:
- `r_vec`:`(r_T, r_E)` → `(r_T, r_E, r_C)`,`r_C = −viol_cost`(由 `run_episode` 从 `env.last_viol_cost` 取,见 §2.2 接线);
- critic / target 网络末维 2→3,输出 `Q=(Q_T,Q_E,Q_C)`;
- **critic loss 加权与 actor 标量化必须分开写**(关键!V8 的核心恰恰是 critic loss 去掉 ω,`mofd_v8.py:31` 现为 `ones_like(ω)/σ²` 等权;若 critic 又按 ω 加权 = 退回 V5 的病,极端偏好下某通道梯度变弱):
  - **critic loss 权重**(`mofd_v8.py:31` `_critic_weight`)扩为 **`[1, 1, λ_c] / σ²`** —— **不含 ω**,等权哲学保留;`λ_c` = 成本通道的拟合相对权重(默认 1,可调);
  - **actor / 动作打分标量化**(`mofd_v5.py:222-231` `_scalarize_q`,V8 未覆盖、共享生效)扩 `weights=[ω_T·α_T, ω_E·α_E, λ_sla]`,即 `Q_eff = ω_T·Q_T + ω_E·Q_E + λ_sla·Q_C` —— **ω 与约束强度 λ_sla 只在这里进**;
  - `_critic_target`(`mofd_v8.py:27`)第三通道用 `r_C + γ·V_C`(同 V8 干净版,不含熵);`alpha_vec`(`mofd_v5.py:326`)扩 `[α_T,α_E,1]`;PopArt σ 扩 `[3]` 自然泛化。
- **两个 λ 要分清**:`λ_c` = critic 拟合 Q_C 的相对权重(默认 1);`λ_sla` = 策略避违约的约束强度(真正旋钮,只在 actor 标量化)。
- 注意 V5 基类的 `_critic_target`/`_critic_weight`(`mofd_v5.py:234/239`)是非活跃版,改了不生效,**别改错文件**。
- 好处:成本作为独立价值通道被 bootstrap(不止即时惩罚),且 ω_T=0 时仍生效;critic 仍是 V8 的 ω-无关等权拟合,不退回 V5。
【改,增强版 B】**CMDP/Lagrangian**:把 `E[viol] ≤ δ` 写成约束,对偶变量 λ 自动调,与 ω 标量化解耦。新颖性更高、实现更重。
- 【开放】Q4:v1 三通道是否够投稿,还是直接上 Lagrangian?

### 3.4 其它【现状】
- `target_entropy` 已从 -1.0 修正为 0.5(离散 |A|=6);奖励通道归一化 0.05/0.25 已在。

---

## 4. 评估修改(`fixed_testset.py` / `helpers.py` / 新脚本)

### 4.1 task-level SLA(**核心:不能裁平均 Pareto 点**)【改】
【现状】当前前沿点是 **episode 平均 delay**(`mofd_main.py:486→500`;`fixed_testset.py:126/151/189` 只留 `mean(d_all)`)。平均 delay ≤ SLA **不代表每个任务满足 SLA** → 按平均点裁 `delay>D_sla` 是错的。
【改】
1. **采集 per-task 延迟**:三处 episode 循环(`mofd_main.run_episode`、`fixed_testset.eval_agent_on_testset / eval_policy_on_testset / eval_greedy_on_testset`)把每个 `env.step` 的 `delay` 收进 `delays_list`,不只累加。
2. **per-ω(逐 ω 点)统计**:每个 ω 点单独算 `violation_rate_ω = mean(delay_i > D_sla)`、`p95_ω` / `p99_ω`、`energy_ω`(违约以 env 审计 §2.2 为准)。**可行性按 per-ω point 判,而非整方法判**——一个方法可能某些 ω 可行、某些不可行。
3. **不可行 ω 点不进前沿**;额外报 **feasible preference coverage = #(可行 ω) / 21**(避免"只保住几个点也被当 Pareto 赢家"的误读)。
4. **延迟轴用尾延迟,且与 δ 对齐**(关键:p95 不能代表 δ=1% 的约束):
   - **δ=5% → 用 p95-delay frontier**;**δ=1% → 用 p99-delay frontier**。主文选一档(建议 δ=5%/p95),另一档作敏感性。
   - 前沿点 = `(pXX_delay_ω, energy_ω)`,SLA 线画在该尾延迟轴的 `D_sla`。
5. **HV 仅在可行点/可行方法间比**:HV 用 `ref=(D_sla, E_ref)`,只对该 δ 下可行的 ω 点计入;`violation_rate_ω > δ` 的点排除(等价于落框外)。`E_ref` = 所有方法能耗上界×1.05。靠冲爆 SLA 换低能耗的点不计分。

### 4.2 归一化 HV【改】
- 盒子 = SLA 可行域内、可部署方法 `(p95_delay, energy)` 前沿并集的 [min,max];两轴归一到 [0,1],ref=(1.1,1.1)。量纲平衡(详见量纲讨论)。

### 4.3 必报指标【改】
- **SLA 违约率**(env 审计)+ **p95/p99 delay**;
- **可行域归一化 HV**(仅 viol≤δ 方法,(p95_delay, energy) 平面);
- **能耗 under SLA**(给定 δ 下的 energy,作主表的一列);
- **per-ω 响应曲线**(p95_delay↑ / energy↓ 随 ω,展示 ω-自适应);
- **SLA-constrained Pareto 前沿图**((p95_delay, energy),限可行域)——主图。

### 4.4 SLA 敏感性【改】
- 在 `D_sla ∈ {k=1.5,2,3}`(或对应 ms 档)上重复对比,**结论须跨档稳健**。
- **训练 vs 评估口径(W3,别混)**:默认路线下 **`D_sla_mid` 是主表**(完整重训模型),tight/loose 是该模型的**跨阈值 eval robustness**(不重训);只有选"三档全重训"时才有"每档各一张主表"。统计口径见 §6(≥3 seeds / K=40 / mean±std / bootstrap CI)。

### 4.5 公平性铁律【改】
- **分清两类 baseline,措辞不能含糊**(启发式没有训练成本通道):
  - **RL baselines(dsac / ldqn / genmosac)**:训练用**相同的 SLA penalty/cost**(标量路径 `r -= λ_sla·viol_cost`)+ 相同准入掩码;
  - **heuristic baselines(greedy_delay / round_robin / rand / greedy_energy)**:免训练,只能在**评估时用相同的 admission mask 和同一 feasibility/violation gate**;
  - 三者共享:同一固定卷子、同一 `D_sla`、同一 ref、同一 per-ω 可行性判定。
- **λ 共享 tuning budget(W2,防 baseline 调弱质疑)**:`λ_sla` 在独立 validation/calibration set(不碰 testset)上用**同一小网格**(如 {0.5,1,2,5})选,**所有 RL 方法共享同一网格与选择准则**(违约率≤δ 前提下能耗最低);`λ_c` 默认 1 只作敏感性。
- **oracle 口径(W6)**:greedy_omega **也在同一 SLA mask + 同一 feasibility gate 下报**,作"特权信息上界"单列,**不进可部署方法主排名**(不是"少了约束的作弊版")。

---

## 5. 实验矩阵与验收线 + 叙事口径

| 维度 | 内容 |
|---|---|
| 方法 | 主方法(prior / feedback / full3 三源都报)+ 可部署 baseline(dsac / ldqn / greedy_delay / round_robin / rand)+ oracle greedy_omega(单列上界) |
| 标尺 | 每个 D_sla 档:可行域归一化 HV((p95_delay, energy))+ SLA 违约率 + energy under SLA |
| 静态验收 | **先过覆盖门槛 feasible coverage ≥ 80%(≥18/21)**,再要求可行域内(viol≤δ)主方法在 **(p95_delay, energy) 上归一化 HV > 全部可部署 baseline**,且跨 D_sla 稳健。覆盖率不达标的方法只作失败案例报告,不进主排名 |
| 真正卖点 | HV 之外的**漂移/非平稳 + ω-自适应**轴(oracle 的"已知精确模型 + 平稳"前提在此失效) |

**5.1 归因消融(必做,否则贡献无法闭合)**【改】
SLA 满足很大程度来自 §2.1 的 orchestrator 准入掩码;第三成本通道(§3.3)也合理。Reviewer 必问:**到底是 mask 带来的,还是 V8 三通道真学到了更好的可行域内策略?** 用 2×2 消融闭合归因:

| 变体 | 准入 mask | 成本通道 Q_C | 作用 |
|---|---|---|---|
| no-mask + no-Q_C | ✗ | ✗ | 原始无约束(极端延迟基线,参照点) |
| **mask only**(no Q_C) | ✓ | ✗ | 可行性全靠 mask;策略=普通 V8 ω-MORL |
| **mask + Q_C**(full,本文) | ✓ | ✓ | 完整提议 |
| no-mask + Q_C(可选) | ✗ | ✓ | 成本通道单独能否维持可行(策略是否内化 SLA) |

- **mask-only vs mask+Q_C**(隔离**成本通道对可行域内策略质量**的贡献)。若 mask+Q_C 能耗更低/HV 更高,说明三通道真学到更好策略,而非 mask 的功劳。
- **mask+Q_C vs no-mask+Q_C**(比 violation_rate)→ 隔离 **mask 对可行性**的贡献。若 no-mask+Q_C 仍能压住违约率 → 策略内化了 SLA;若违约暴涨 → 诚实承认可行性主要由 mask 提供。
- **比较口径(W5,关键)**:两变体可行的 ω 点可能不同,直接比 HV 会混入 coverage 差异。消融表**同时报两套**:① all-feasible-points HV + coverage(各自可行集);② **common-feasible-ω set 上的 paired energy/HV**(只在两者都可行的 ω 上配对)。后者才是干净的成本通道归因。

**主叙事口径(不要自我削弱 Pareto 故事)**:
- 论文主线写 **"SLA-constrained Pareto frontier over p95 delay and energy"**;`energy under SLA` 只作主表的一列指标,**不**把整个故事降级成"可行域内最小化能耗"(否则标题里的 Pareto 权衡被自己削掉)。ω-自适应仍是核心贡献:同一策略随 ω 在 (p95_delay, energy) 可行前沿上滑动。
- **"非特权信息"措辞收紧**:不写"policy uses no privileged information";改写 **"all methods share the same orchestrator-side admission-control layer based on online system telemetry"**(掩码用了 state 外的 tran_rate/proc_queue_bef,只能这么说才诚实)。若另做了 §2.3 的 B 路线,才可补一句"策略自身亦观测 SLA slack"。

【开放】Q5:静态验收过线后是否仍不足以支撑"优越性"?(我方:静态是入场券,优越性主张靠非平稳轴 —— 叙事重心放哪?)

---

## 6. 实施顺序与风险

**顺序(先不重训的都先做):**
1. §1 路线 1:从现有数据派生 D_sla 候选(只读)。
2. §4 评估改 + **重跑固定 testset 评估(不重训)**已有 checkpoint/启发式,存 **per-task delay log** 算 p95/violation/coverage(**不能裁现有平均前沿**:现存只有 episode-mean `fixed_testset.py:126/189`,且 mask 改动作→改队列 `mofd_environment.py:195`,事后裁近似不了 masked rollout)。两模式:无 mask / eval 时套 mask。看主方法离"可行域支配"多远。
3. §2 env(掩码 + step 审计 + 可选特征)+ §3 接线(三通道)。
4. 重训主方法(prior/feedback/full3)+ 全 baseline,**同协议同卷同掩码**。
5. 出最终表/图 + SLA 敏感性。

**风险:**
- **SLA 过紧** → 可行解过少/退化;过松 → 约束无意义。靠敏感性 + 违约率监控。
- **掩码改动作空间** → baseline 必须用**同一掩码**(尤其 greedy_delay 会天然受益),否则不公平。
- **§2.3 特征 / §3.3 三通道 → ckpt 不兼容**,要全重训。
- **cherry-pick 风险** → D_sla 由领域/数据定,报敏感性。
- **可行域内未必真支配 baseline**(数据显示 ω=1 延迟侧仍略输 greedy_delay)→ 掩码只挡极端,**延迟侧行为得真改善**,是经验问题,非改约束就自动成立。
- **p95 估计稳定性 / 统计口径** → 最终主表 **≥3 seeds**、固定 testset **K=40**(不是 20);p95/HV 报 **mean±std**,关键对比加 **bootstrap 95% CI**(p95/HV 对抽样敏感,reviewer 必问)。
- **D_sla 训练矩阵别混** → mask/penalty 都依赖 D_sla:**主结果选单一 `D_sla_mid` 完整重训**;跨档要么三档全重训(强 claim,×3 成本),要么"固定 D_sla_mid 训练、跨档仅 eval 鲁棒性"并在文中写明,不要两种混着讲。
- **mask 全 0 兜底** → 某任务所有服务器都违约时 mask 不能为空 → 保留预测延迟最小的服务器,step 审计仍记违约(`get_sla_mask` 与训练/评估共用此兜底)。

---

## 7. 给 reviewer 的核心待决问题汇总
- **Q1** reframe 到 SLA 是否被视为回避弱点?措辞怎样最稳?(§0)
- **Q2** delay≈秒级仿真:SLA 用相对值(路线1)还是必须物理标定(路线2)?(§1)
- **Q3** 加 SLA 可观测特征(§2.3-B)是否值回重训?还是掩码=共享安全层(§2.1)已够?
- **Q4** 约束机制:V5 三通道(§3.3 v1)够不够,还是上 Lagrangian(增强版)?
- **Q5** 静态验收过线后,优越性叙事重心放静态 HV 还是非平稳轴?(§5)
- **Q6** headline 方法是否该从 full3 换成更强单源(prior/feedback)?(贯穿)
- **Q7** 前沿延迟轴用 p95-delay(§4.1)是否标准,还是另有更被接受的尾延迟口径(如违约率曲线)?

---

## 8. 修订记录(changelog)
- **v7(本轮)**:并入第七轮 review(公平性/隔离类硬化):
  1. **校准集隔离(W1)**:D_sla 在独立 calibration set 标定后冻结,不碰 testset(§1)。
  2. **λ 共享 tuning(W2)**:λ_sla 同小网格、所有 RL 方法同 budget;λ_c 默认1(§4.5)。
  3. **D_sla 训练/评估口径(W3)**:默认 D_sla_mid 主表 + tight/loose 跨阈值 eval robustness(§4.4)。
  4. **step 审计时序(W4)**:阶段 2b 违约由 evaluator 的 per-task log 判,不依赖阶段3的 env step 审计(TODO)。
  5. **消融 common-feasible-ω(W5)**:同时报 all-feasible+coverage 与 common-ω paired energy/HV(§5.1)。
  6. **oracle 口径(W6)**:greedy_omega 也在同一 mask+gate 下报、单列上界(§4.5)。
- **v6**:执行顺序(predict_delay/get_sla_mask 提前到 2b)+ 全0兜底 + D_sla 训练矩阵 + 统计口径 + min_e 主口径。
  1. **执行顺序(W1)**:轻量 `predict_delay`/`get_sla_mask` 提前到阶段 2b(eval 套 mask 要用),阶段 3 只剩 step 审计 + 训练侧;
  2. **全 0 兜底(W2)**:mask 全 0 时保留预测延迟最小服务器、step 仍记违约,写进 §6 风险;
  3. **D_sla 训练矩阵(W3)**:主结果单一 D_sla_mid 完整重训 / 三档全训 / 单档训跨档评,三选一别混;
  4. **统计口径(W4)**:≥3 seeds、K=40、mean±std、bootstrap CI;
  5. **D_sla 主口径(W6)**:锁 `min_e` 无队列 service time(median_e 仅敏感性)。
- **v5**:并入第五轮 review:
  1. **§6 step2 / TODO 阶段2**:删"在现有前沿上模拟 SLA"(平均点算不出 p95/violation,且 mask 改队列),改为**重跑 eval 存 per-task log**(无 mask / eval 套 mask 两模式)。
  2. **D_sla 主口径**:改为**物理无队列 service time × k**(方法无关);随机策略 P50/P75 降为 sanity/敏感性。
  3. **§5 加硬门槛**:feasible coverage ≥ 80%(≥18/21)才进主排名,否则只作失败案例。
  4. **§3.2 补 ldqn**(dsac/ldqn/genmosac 统一走标量 penalty)。
  5. **归因消融优先级**:mask-only vs mask+Q_C 必做;no-mask+Q_C 仅在 claim "策略内化 SLA" 时才训。
- **v4**:新增 §5.1 归因消融(2×2)+ §4.5 公平性措辞分 RL/启发式两类。
  1. **新增 §5.1 归因消融**(2×2:mask only / mask+Q_C / no-mask+Q_C / 原始),闭合"可行性是 mask 带来的还是三通道学到的"这一归因问题。
  2. **§4.5 公平性措辞精确化**:RL baselines 用相同 SLA penalty/cost;heuristic baselines 只在评估时用相同 admission mask + feasibility gate(启发式无训练成本通道)。
- **v3**:命名(澄清主方法实为 V8)+ critic 加权逻辑(critic loss 无 ω `[1,1,λ_c]/σ²`、actor 标量化才含 ω+λ_sla)+ per-ω 可行性粒度 + p95↔δ 对齐 + line14 HV 口径 + 训练接线。
- **v2**:把第一轮 review 的 4 条**并入正文并删除旧错误说法**:
  1. **命名**:澄清主方法实为 **V8**(`MOFD_SAC_V5_HMCSS → MOFD_SAC_V8`),"V5" 是遗留误名;§0/§3.3 文件引用改为活跃覆盖版 `mofd_v8.py:27/31`。
  2. **§3.3 critic 加权逻辑冲突**:V8 的 `_critic_weight`(`mofd_v8.py:31`)本就是 ω-无关等权 `ones/σ²`。改为**分开写**:critic loss = `[1,1,λ_c]/σ²`(无 ω);actor 标量化(`_scalarize_q` `mofd_v5.py:222-231`)= `ω_T·Q_T+ω_E·Q_E+λ_sla·Q_C`。区分 `λ_c`(拟合权重)与 `λ_sla`(约束强度)。
  3. **§4.1 可行性粒度**:改为 **per-ω point** 判可行,不可行点不进前沿,加报 **feasible preference coverage = #可行ω/21**。
  4. **§4.1 p95↔δ 对齐**:**δ=5%→p95 frontier,δ=1%→p99 frontier**,主文选一档另一档敏感性。
  5. **§0 line14 HV 数字口径**:"950→32" 改为"同一固定 ref 下离线重算的数量级示意",并注明 `testset_compare.csv` 仍是旧数、勿直接引用。
  6. **§2.2/§3.3 接线**:补明 `run_episode` 在 `env.step()` 后读 `env.last_viol_cost` → 构 `r_vec=(r_T,r_E,−last_viol_cost)` 存 V8 buffer。
- **v2**:把第一轮 review 的 4 条**并入正文并删除旧错误说法**:
  1. §2.2/§4.1 旧"惩罚进 r_T""按平均 delay 裁点"已删 → 改为独立成本通道 + task-level(p95 / violation_rate);
  2. §3.3 给出 V5 **第三成本通道**(N_OBJ 2→3)的具体实现,替代 v1 不可行的"scalar target 减 penalty"(V5 是 vector critic,`mofd_v5.py:357-368`);
  3. §5 主叙事固定为 **"SLA-constrained Pareto frontier over p95 delay and energy"**(不降级成"最小化能耗");
  4. §5 "非特权信息"收紧为 **"shared orchestrator-side admission-control based on online system telemetry"**;§2.1 明确掩码=共享安全层。
- **v1**:初版(环境→模型→评估三段式)。
