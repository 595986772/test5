# SLA-约束 Pareto:通俗版实验 TODO 清单

> 配套 `SLA_pareto_修改方案.md`(技术细节看那份)。本清单按"**先零成本探路 → 过关再重训**"排序。
> 原则:每个"重训"前都有一个"只读预演"关卡,避免白训几小时。
> 标记:🟢=**不重训/低成本**(可能要改代码或重跑评估,但不训练) 🔴=要重训(贵,数小时) 🚪=go/no-go 决策关卡

---

## 阶段 0 —— 把现在在跑的事收尾(🟢 已在进行)

**做什么**:等后台 dsac / genmosac(target_entropy 修复版,**无 SLA**)跑完。
**怎么做**:跑完后 `python rescale_hv.py`,用修好的 HV 把所有方法的"**无约束诚实分数**"重算一遍。
**意义**:① 确认 target_entropy 修复后 baseline 不再熵崩(α 不奔 0)、不是 strawman;② 留一张"**上 SLA 之前**"的对照快照,后面好说明 SLA 带来了什么。
**产物**:`testset_compare_fixed.*`(无约束、修正 HV 版)。
**注意**:这批是**无约束**模型,SLA 对比还要在阶段 5 重训;但它们不浪费(验证修复 + 无约束对照)。

---

## 阶段 1 —— 把 SLA 这把"尺"定出来(🟢 不重训)

**做什么**:派生 3 个 `D_sla` 候选(松 / 中 / 紧)。
**主口径(预注册,唯一,与策略无关)**:对每个任务,在所有可用服务器上算无队列 service time `tran_delay + comp_delay`,**取服务器维度的 min_e**(= 最优可服务基准,即"理论上最快能服务多久"),得到每任务一个值;`D_sla = k × median_over_tasks(min_e service_time)`,`k ∈ {1.5, 2, 3}`(松/中/紧)。只依赖物理参数、不依赖任何策略 → 不可被挑。
**敏感性(不与主口径并列)**:服务器维度改用 `median_e`(而非 min_e)只作敏感性对照;随机策略可达延迟 P50/P75 仅作量级 sanity check。**主口径锁死 min_e,避免 min/median 含糊。**
**校准集隔离(W1,必写)**:`D_sla` **在独立 calibration set(或训练分布的采样集)上标定,标定后冻结**,再用于 fixed testset 与所有方法。**绝不能用最终 fixed testset 的任务来算 D_sla**(否则=用测试集定阈值,test leakage / cherry-pick)。calibration set 与 testset 用不同随机种子从同一环境分布采样、互不相交。
**意义**:SLA 不能拍脑袋,**标定既不能依赖策略、也不能碰测试集**。物理 min_e service time + 校准集隔离,挡得住"阈值是不是挑出来的 / 是不是偷看了测试集"。配一句 3GPP/应用分级印证。
**产物**:3 个**冻结的** `D_sla` 数值 + 物理标定公式(明确 min_e + calibration set 来源)+ 一句出处。

---

## 阶段 2 —— 改 task-level 评估 + 重跑(不重训)存 per-task log(🟢 不重训但要重跑)🚪

**做什么**:**不重训**,但必须**重新跑固定 testset 评估**(已有 checkpoint + 启发式),拿到 task-level 数据,看主方法在可行域内离"赢 baseline"还差多少。
**为什么不能只裁现有前沿**:现有评估只存了 **episode 平均** delay/energy(`fixed_testset.py:126 / :189`),**算不出 violation_rate_ω / p95 / p99**;而且 SLA mask 会改变动作 → 改变后续队列累积(`mofd_environment.py:195` `proc_queue_bef += rho_d`),所以**事后裁平均点近似不了 masked rollout**。必须真的重跑。
**怎么做**:先改评估代码(逐任务收延迟,落 **per-task rollout log**;算每 ω 点 `violation_rate_ω`/`p95_ω`/`p99_ω`/`energy_ω`、`feasible coverage=#可行ω/21`、可行点上的修正 HV),再分两小步重跑:
- **2a(无 mask,零新依赖)**:现有 checkpoint/启发式直接重评 → 看**无约束模型 task-level 违约多严重**、p95 前沿在哪。只需 per-task 日志。
- **2b(eval 套 admission mask)**:策略网络不变、只在评估时限制可选服务器。
  - **依赖修正(W1)**:2b 要用 `get_sla_mask`,所以**把轻量 `predict_delay` + `get_sla_mask` 从阶段 3 提前到这里实现**(阶段 3 只剩 step 审计 + 训练侧改造);
  - **全 0 兜底(W2,必写)**:若某任务所有服务器都 `predict_delay > D_sla`,mask 会全 0 → agent/启发式拿到空 mask 会崩。**兜底:保留预测延迟最小的那个服务器**(最小违约)。
  - **违约统计来源(W4)**:阶段 2b **由 evaluator 的 per-task log 自己判违约**(`delay_i > D_sla`),**不依赖**阶段 3 才接的 env 内 `step()` 审计。兜底选的最小延迟服务器其实际 delay 仍 > D_sla → evaluator 自然把它记为违约。env 内 step 审计是**训练侧**的 ground-truth(阶段 3 才需要)。
**意义**:**最关键的 go/no-go 关卡**,不重训就判"SLA 这条路有没有戏":可行域内接近/超过 baseline → 值得重训;差很远 → 先调方法/换 headline 源。
**产物**:**task-level rollout log**(每任务延迟)+ summary table(per-ω violation/p95/energy/coverage + 可行域 HV)。
**🚪 关卡**:① 主方法可行域 HV 是否接近/超 baseline;② **feasible coverage 是否 ≥ 80%(≥18/21)**——低于此先别投重训。

---

## 阶段 3 —— 补齐环境的训练侧改造(🔴 前置,本身不训)

**做什么**:`predict_delay` / `get_sla_mask`(含全 0 兜底)**已在阶段 2b 实现**;这里只补训练侧需要的部分。
**怎么做**(`mofd_environment.py`):
1. `step()` 自审计:实际延迟 > D_sla 就在 env 记一次违约(`sla_violations/sla_total`、`last_viol_cost`),**不改返回格式**;
2. 加 `D_sla` 参数(默认关,向后兼容);
3.(可选,仅 §2.3-B)加 SLA 可观测特征(`effective_rate_e`/`proc_queue_bef_e`/`slack`)——只在要 claim"策略自身内化 SLA"时做,会改 `state_dim` → ckpt 不兼容;
4. 训练取动作处接 `get_sla_mask`(同一全 0 兜底)。
**意义**:让训练能在 SLA 下进行。**mask = 事前预防**(策略只在可行服务器里挑),**step 审计 = 事后终审**(漏接 mask 也能抓到违约,违约率以它为准)。

---

## 阶段 4 —— 改模型:第三成本通道(🔴 前置,本身不训)

**做什么**:让策略学会"避免违约",且在纯能耗偏好(ω=0)时也生效。
**怎么做**(主方法 V8):
1. `N_OBJ` 2→3,奖励变 `(r_T, r_E, r_C)`,`r_C = −viol_cost`;
2. **critic loss 等权**(`mofd_v8.py:31`):`[1, 1, λ_c]/σ²`,**不含 ω**(保 V8 的干净 critic,别退回 V5);
3. **actor 打分**(`mofd_v5.py:222-231`):`ω_T·Q_T + ω_E·Q_E + λ_sla·Q_C`(ω 和约束强度 λ_sla 只在这进);
4. 接线:`run_episode` 在 `env.step()` 后读 `env.last_viol_cost`,构 `r_vec=(r_T,r_E,−last_viol_cost)` 存 buffer;
5. RL baseline(**dsac / ldqn / genmosac**)走标量 reward 路径,直接 `r -= λ_sla·viol_cost`(三者统一,勿漏 ldqn)。
**意义**:光靠 mask 只能"挡住"违约,**成本通道让策略主动学可行域内更省能的分配**;放外层(不乘 ω)保证 ω=0 时惩罚不被吃掉。

---

## 阶段 4.5 —— 选 λ_sla / λ_c(🔴 小网格,公平性关键,W2)

**做什么**:定 `λ_sla`(约束强度)和 `λ_c`(成本通道拟合权重)的取值规则,**且对所有 RL 方法一视同仁**。
**怎么做**:
- 在**独立 validation/calibration set**(同 §阶段1 的隔离原则,**不碰 testset**)上,用**同一个小网格**选 `λ_sla`(如 {0.5, 1, 2, 5});
- **所有 RL 方法(主方法 + dsac/ldqn/genmosac)共享同一 tuning budget**(同网格、同选择准则,如"违约率≤δ 前提下能耗最低");
- `λ_c` 默认 **1**,只做敏感性,不调。
**意义**:**防"主方法细调 λ、baseline 只给默认值"的不公平质疑**。reviewer 一定会查 baseline 是不是被故意调弱——同网格同预算才站得住。

---

## 阶段 5 —— 重训(🔴 贵,数小时×多个)

**做什么**:在 SLA 约束下重训所有要比的模型,同卷、同 mask、同协议。
**怎么做(训哪些)**:
- 主方法 3 个源:prior / feedback / full3;
- RL baseline:dsac / ldqn / genmosac(带 SLA penalty + mask);
- 启发式(greedy_delay/rr/rand):免训,评估时套**同一 admission mask + 同一 feasibility gate**;
- oracle greedy_omega:**也在同一 SLA mask + 同一 feasibility gate 下报**,作"特权信息上界"**单列**,**不进可部署方法主排名**(W6:口径写清,oracle 不是少了约束的"作弊版")。
**D_sla 实验矩阵(必须先定死,别混,W3)**:因为 mask 和 penalty 都依赖 D_sla,"每档重训" vs "单档训跨档评"结论不同、成本差几倍——三选一,写清:
- **(默认)主结果**:选一个 `D_sla_mid`,**完整重训**主方法 + RL baseline;
- **强 claim(跨档都赢)**:三档 D_sla **全部重训**主方法 + RL baseline(成本 ×3);
- **时间紧**:固定 `D_sla_mid` 训练,跨 D_sla **只在评估时**测 robustness(不重训),并在文中如实写"训练用单一 D_sla_mid,跨档仅为 eval 鲁棒性"。
**意义**:产出**SLA 约束下可直接互比**的一组模型。这是后面所有结论的来源。

---

## 阶段 6 —— 归因消融(🔴 必做,否则贡献说不清)

**做什么**:消融,证明"可行域内的提升是模型学到的,不只是 mask 的功劳"。
**怎么做(分必做/可选,别多烧训练)**:
- **必做(2 个)**:`mask only(no Q_C)` vs `mask+Q_C(完整)` —— 隔离**成本通道**的贡献(回答"三通道有没有用");`原始(无mask无Q_C)` 可复用阶段 2(a) 的无约束结果,基本不额外训。
- **可选(仅当要 claim "策略自身内化 SLA" 时才训)**:`no-mask + Q_C` —— 比违约率 → 隔离 **mask** 的贡献;若它仍压得住违约说明策略内化了 SLA。不想 claim 这点就**别训它,省一轮**。
- **比较口径(W5,关键)**:两个方法可行的 ω 点可能不同,直接比 HV 会混入 coverage 差异(一个只保住容易的 18 点、另一个保住 21 点,HV 不可比)。所以消融表**同时报两套**:① **all-feasible-points HV + coverage**(各自可行集);② **common-feasible-ω set 上的 paired energy/HV**(只在两方法都可行的那些 ω 上配对比)。后者才是干净的成本通道归因。
**意义**:reviewer 必问"是 mask 还是模型",必做的两个 + common-ω 配对就够闭合主归因;可选那个只为额外的 SLA-aware claim。

---

## 阶段 7 —— 出最终结果 + SLA 敏感性(🟢 评估,基于阶段 5/6 的模型)

**做什么**:出论文主结果。
**怎么做**:
- **主表口径(W3,与阶段5矩阵对齐,别误解)**:**默认路线**下 `D_sla_mid` 是**主表**(完整重训的模型);`tight/loose` 是"**mid-trained 模型的跨阈值 eval robustness**",**不是**三档各自重训。只有选"三档全重训"时才有"每档各一张主表"。每行报 `可行域归一化 HV`、`violation_rate`、`energy under SLA`、`feasible coverage`;
- **统计口径(必须,W4)**:主表 **≥3 seeds**、固定 testset **K=40**(不是 20);`p95` / `HV` 都报 **mean±std**,关键对比再加 **bootstrap 95% CI**。p95/HV 对 testset 抽样很敏感,reviewer 必问稳定性。
- **硬验收门槛(关键)**:只有 **feasible coverage ≥ 80%(≥18/21)** 的方法才进入"优于 baseline"的主排名;低于门槛的方法**只能作为失败案例报告**,不得参与主结论(否则"只保住几个 ω 点也赢"会被 reviewer 打)。
- **主图**:SLA-constrained Pareto 前沿(`p95_delay vs energy`,δ=5%);per-ω 响应曲线(展示 ω-自适应);
- **敏感性**:δ=1% 用 p99;多个 D_sla 跨档,结论须稳。
**意义**:这是 Experiments 章节的核心证据。叙事写 **"SLA 约束下 p95延迟–能耗的 Pareto 权衡"**,不要降级成"只最小化能耗"(否则自废 Pareto 卖点)。

---

## 阶段 8 —— 非平稳/漂移轴(🔴/🟢 真正卖点)

**做什么**:在漂移 / 未见分布上比 主方法 vs baseline vs oracle。
**怎么做**:复用现有 drift/shift 实验框架,在 SLA 口径下重做。
**意义**:静态 HV 只是**入场券**;oracle("已知精确模型 + 平稳"前提)在非平稳下会失效,**这才是你能赢 oracle、讲优越性的地方**(ω-自适应 + buffer 漂移鲁棒)。

---

## 一页速览(依赖与关卡)

```
阶段0(收尾,跑着)──┐
阶段1(定SLA)🟢 ────┤
                    ├─→ 阶段2(预演)🟢🚪 ──[有戏?]──→ 阶段3(env)→阶段4(model)→阶段5(重训)🔴
                    │                          └─[没戏]→ 回去调方法/换headline源
                    │
阶段5 ─→ 阶段6(归因消融)🔴 ─→ 阶段7(最终表/图+敏感性)🟢 ─→ 阶段8(漂移轴,真卖点)
```

**最省力路径**:先把 0/1/2 做完(全程只读/便宜),**在阶段 2 关卡看到"可行域内有戏"再投重训**。
