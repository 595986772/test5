# Preference-Conditioned Greedy Warm-start Diffusion (PGW) 完整修改方案

> 方法代号 **PGW**:把反馈扩散策略的去噪**起点**从"无信息(零/均匀/上一步)"换成
> **按当前偏好 ω 现算的贪心物理先验**,再让学好的去噪器**纠正贪心的拥塞近视**。
>
> 本文档自包含,供 GPT review。先交代现有系统与已否决组件,再给问题动机、**实测诊断依据**、
> 方法、代码改动(文件/函数级)、消融、论文故事线、诚实边界(已测 vs 假设)、风险、验收。
>
> 术语固定:preference vector ω / Pareto frontier / hypervolume (HV) / edge offloading /
> feedback diffusion / multi-objective reinforcement learning (MORL)。

---

## 0. 一句话定位

现有反馈扩散卸载策略每次决策从一个**无信息起点**去噪(评估时是零向量、训练时是均匀/缓冲区先验)。
我们实测发现:**去噪器会强烈跟随一个有信息的尖锐起点**(内容探针 +50 个百分点),
而一个**按 ω 现算的贪心最优服务器**正好是这样一个有信息、且偏好正确的起点。
PGW = 用这个贪心先验做去噪起点,让去噪器在它之上**学习式地纠正贪心忽略的队列拥塞**。

```
PGW = preference-conditioned greedy warm-start  +  learned congestion-aware refinement
```

---

## 1. 背景与现有系统(reviewer 必读)

**系统 FDEdge / MOFD-V8**:多目标、偏好条件化边缘卸载强化学习。

- 任务:每时隙到达若干任务,决定每个任务卸载到哪台边缘服务器(Emax 台,可变 E)。
- 目标:2 个 —— delay、energy;`r_vec=(-delay·delay_scale, -energy·energy_scale)`。
- 偏好:用户给 `ω=(ω_T, ω_E)`,作为状态一部分;单策略扫 ω 得 Pareto frontier。
- 构件(均为现有/借用,论文中引用):
  - actor = **feedback diffusion**(DDPM 多步去噪);**该骨干源自已发表的 delay-only 反馈扩散
    卸载工作,本文作 backbone 引用,不主张为创新**;
  - critic = **vector Q** `(Q_delay, Q_energy)`;**V8 干净版**(目标去熵、损失等权);COR、PopArt、SAC-α。

**已否决/移除(诚实交代,避免 reviewer 误解方法栈)**:
- **ω-buffer / 偏好漂移**:循环论证,已否决;
- **H-MCSS(三源候选 feedback/prior/random + 每步 Critic 选优)**:消融实测**净负**
  (full3 最差),已不作卖点 —— 这恰是 PGW 的反面教材(见 §3)。

**因此本文方法贡献是 PGW。** 它不依赖 buffer、不依赖漂移、不堆候选源,是单源 1× 推理。

---

## 2. PGW 要解决的真实问题

反馈扩散策略的去噪**起点本身是策略的一部分**:起点不同,去噪出的动作分布不同。
但现状的起点是**无信息**的:

- 评估时起点 = **零向量**(`evaluate_pareto` 传 `latent=np.zeros`);
- 训练时起点 = 均匀 / 缓冲区池化先验(低方差、无明确指向)。

**这浪费了去噪器最强的能力**:我们实测它能强烈跟随一个有信息的起点(§3)。
与此同时,每个决策下"哪台服务器在当前 ω 下即时最优"是一个**可现算的物理量**——
把它喂进去当起点,等于给去噪器一个偏好正确、且尖锐的"直觉",而不是从噪声里凭空学。

> 注意这跟 delay-only 反馈扩散原文不同:原文无 ω,起点是"上一步动作";
> PGW 的起点是**偏好条件化的物理贪心解**,是多目标设定下特有的设计旋钮。

---

## 3. 实测诊断依据(本方案的地基,全部为本项目已跑出的真实结果)

> 红线:以下为**已测**结果(单 seed 诊断,文件可溯源);PGW 的"大效应"是由它们**推断**的,
> 尚未端到端验证(见 §9)。

**D1 — 去噪器强烈跟随有信息的尖锐起点(内容探针,已测)。**
在训好的网络上,喂一个 one-hot 指向"当前 ω 下即时最优服务器"的起点,动作落在该服务器的比例
从基线(均匀起点)的 **23–25%** 升到 **76–82%(+50 个百分点)**;且实得即时代价更低
(good 起点 cost ≈ 0.78–0.82 vs bad 起点 1.23–1.37)。
→ **起点的内容能穿过去噪、强力左右最终动作。** 这是 PGW 的核心前提,已被预验证。

**D2 — 尖锐且 ω-正确的起点能同时撬动两个目标(机制诊断,已测)。**
起点方差↑ → 策略尖锐度↑ → delay 大幅↓(44→29);而能耗角由训练盆地决定。
贪心先验既尖锐、又按 ω 选(ω_E 大时选低频省能服务器),因此**有望同时压低 delay 与触及 energy 角**,
而不像随机起点只动 delay 轴。

**D3 — 多源 + 每步 Critic 选优是负贡献(H-MCSS 消融,已测)。**
`full3`(feedback+prior+random + 每步 Critic 选)HV=219,**低于任意单源**(feedback 252)。
→ 教训不是"候选不够好",而是**每步让不可靠的 Q 瞎选源会主动选错**。
PGW 据此**只用单一物理源、不做每步 Critic 选**,绕开这个坑。

> 注:上述数字为 V5 时代单 seed,且我们已发现旧 H-MCSS 的 feedback/prior 训练同物;
> PGW 的对照将在 V8 + 真反馈、并最终多 seed 下重测(见 §11)。这里只用它们作**机制方向**的依据。

---

## 4. 方法设计

### 4.1 贪心物理先验 `greedy_omega_prior`

对当前决策 `(t, n)` 与偏好 ω,对每台**有效**服务器 e 现算即时(myopic)代价,完全复刻 env 的
reward 物理(与 `prior_content_probe.server_costs` 同式):

```
v        = tran_rate[e] · channel_gain[t,n,e]          # 有效传输速率
tran_d   = d_n / max(v, 1e-6)
comp_d   = rho_d / f_E[e]
wait_d   = (proc_queue_len[t,e] + proc_queue_bef[t,e]) / f_E[e]
delay_e  = tran_d + comp_d + wait_d
energy_e = p_off · tran_d + kappa · f_E[e]^2 · rho_d
cost_e   = ω_T · α_T · (delay_e · delay_scale) + ω_E · α_E · (energy_e · energy_scale)
```

先验 = 对 `-cost_e` 在**有效服务器**上做温度 softmax:

```
prior[e] = softmax(-cost_e / τ)   over valid e   (无效槽位置 0 后重新归一化)
```

返回 `[Emax]` 的合法概率向量,直接作为去噪起点。

### 4.2 温度 τ —— 关键设计旋钮(决定"纠正还是照抄")

- `τ → 0`:先验趋近 one-hot(贪心硬选)→ 去噪器**容易照抄** → PGW≈贪心(可能继承拥塞)。
- `τ` 适中/偏大:先验是**软分布**,给去噪器留出**重新分配负载**的空间 → 才可能学到"摊开避免拥塞"。

τ 是 PGW 能否"纠正"而非"照抄"的核心,必须扫(见 §7)。第一版建议从 `τ=1.0` 起。

### 4.3 与扩散的组合(单源,无 Critic 选)

- 去噪起点 = `greedy_omega_prior(t,n,ω,τ)`(逐决策现算);
- actor = feedback diffusion,**单源**:就从该起点去噪(等价 H-MCSS 的 `mode='feedback'`,
  但起点从"零/缓冲"换成贪心先验),**不 cat 三源、不做每步 Critic 选**(因 D3);
- critic = V8 干净版,不变;
- **1× 推理**(单去噪链),边缘友好。

### 4.4 可部署性(诚实关键点)

贪心先验用到 `tran_rate[e]`(base 传输速率),而它**不在策略的状态向量里**(状态只有归一化信道增益)。
因此先验**必须在 env 侧(`run_episode` 内)计算**,而不能假装"从 NN 输入算出"。

这**仍是可部署的**:真实部署里,编排器(orchestrator)本就掌握服务器频率、队列长度、链路速率
(正如 Greedy-Min-Delay 基线就用这些)。先验只用**当前时隙已知量**(f、队列、信道、任务大小),
**不使用任何未来信息**,是合法的在线 myopic 启发式。论文须如实写明"先验由编排器侧系统信息现算"。

---

## 5. 核心科学问题(也就是贡献本身)

D1 说去噪器**强烈跟随**起点(76%);而队列物理要求把负载**摊开**才能避免 `wait_delay` 爆炸。
两者构成张力:

> **学好的去噪器到底会"纠正"贪心先验的拥塞(把过度集中的负载摊开),还是只会"照抄"它?**

- 若**纠正** → PGW 是真贡献:**解析偏好直觉 + 学习式拥塞纠正**,且应在高负载下大幅胜过纯贪心与纯扩散。
- 若**照抄** → PGW≈贪心,无附加值(并反过来说明该骨干在此场景贡献有限)。

这个问题**可证伪、可消融**,答案就是论文的核心 finding。

---

## 6. 代码修改方案(文件 / 函数级,全部开关默认关,主项目不变)

### 6.1 新增 `greedy_warmstart.py`

```python
def greedy_omega_prior(env, t, n, omega, alpha_T, alpha_E,
                       delay_scale, energy_scale, temperature=1.0):
    """逐决策贪心物理先验 [Emax]: softmax(-cost/τ) over valid servers.
       cost 复刻 env reward 物理 (同 prior_content_probe.server_costs)."""
    # 1) 逐有效服务器算 delay_e/energy_e/cost_e (见 §4.1)
    # 2) 无效槽位 cost=+inf
    # 3) prior = softmax(-cost/τ); 无效位置 0; 归一化; 返回 [Emax]
```

> 直接复用 `prior_content_probe.server_costs` 的算式,避免重复实现/口径漂移。

### 6.2 修改 `mofd_main.run_episode`(加一个默认关的 hook)

给 `run_episode` 增加 `warmstart_fn=None` 参数;提供时,逐决策用它产出去噪起点:

```python
def run_episode(..., prior_latent=None, use_true_feedback=False, warmstart_fn=None):
    ...
    for t in ...:
      for n in ...:
        state = env.get_state(t, n); mask = env.get_valid_mask()
        if warmstart_fn is not None:
            latent = warmstart_fn(env, t, n, omega)          # PGW: 贪心物理先验起点
        elif use_true_feedback:
            latent = fb_start.copy()
        else:
            latent = latent_slice[t, n].copy()
        action, probs = agent.take_action(state, latent, prior_latent, mask, stochastic=stochastic)
        ...
```

- `warmstart_fn=None` → 行为与现状**完全一致**(主项目不变)。
- `evaluate_pareto` 同样透传 `warmstart_fn`,保证**训练/评估同款起点**(否则又训/评不一致)。
- `run_single_seed` 从 `cfg['warmstart']`(默认 None)构造 `warmstart_fn`(闭包绑定 α/scale/τ)并传入两处。

### 6.3 agent:复用单源去噪(无 Critic 选)

PGW 的 actor = `MOFD_SAC_V5_HMCSS(MOFD_SAC_V8)` + `MCSS_MODE='feedback'`
(cand=latent,单源,从给定起点去噪)。无需新 agent 类——起点由 §6.2 的 `warmstart_fn` 注入。

### 6.4 贪心基线(消融 arm A)

新增一个极简 agent(或复用 baselines 框架):`take_action` 直接对 `greedy_omega_prior` 采样/取 argmax,
**不经 NN**。用于"纯贪心"对照。

### 6.5 运行脚本 `run_pgw_ab.py`

monkey-patch / cfg 注入三臂(见 §7),输出共享 ref 下 HV + ρ(ω,·) + 延迟/能耗下界对照,
写 `results/pgw_ab_compare.txt`。默认单 seed=0、可 `--smoke`、`--epochs`、`--tau`。

---

## 7. 消融设计(三方对照 + 温度扫:大赢或出局)

主三臂(其余 V8 / 超参 / 训练评估协议全同,唯一变量 = 起点机制):

| 臂 | 含义 |
|---|---|
| **A. Greedy-myopic** | 纯贪心,无去噪(`greedy_omega_prior` 直接出动作) |
| **B. Diffusion-uninformative** | 单源扩散,起点 = 零/均匀(现状) |
| **C. PGW(提案)** | 单源扩散,起点 = 贪心物理先验 |

判据(硬,大赢或出局):
- **C ≫ max(A, B) 明显** → 去噪器**纠正**了贪心拥塞:解析直觉 + 学习纠正成立 → 真贡献;
- C ≈ A → 去噪器只照抄,**无附加值,枪毙**;
- C ≈ B → 暖启动没加东西,**枪毙**。

温度子消融(回答 §5 的"纠正 vs 照抄"):τ ∈ {→0(硬), 0.5, 1.0, 2.0}。
预期(假设):适中 τ 下 C 最好(留出纠正空间);τ→0 时 C 退化向 A(照抄贪心)。

测评指标:HV(共享 ref)、delay_min / energy_min(两角是否都被触及,验 D2)、
ρ(ω_E, energy) / ρ(ω_E, delay)(偏好一致性)、高负载下相对纯贪心的拥塞改善。

---

## 8. 论文故事线(negative → diagnosis → fix → mechanism)

1. **现象**:偏好条件化反馈扩散卸载策略的去噪起点是无信息的(零/均匀),浪费了去噪器。
2. **诊断**(我们的实测,§3):(D1)去噪器强烈跟随有信息的尖锐起点;(D2)尖锐且 ω-正确的起点
   能同时撬动 delay/energy 两角;(D3)朴素堆多源 + 每步 Critic 选优反而最差。
3. **洞察**:正确做法是**单个偏好正确、物理现算的起点**,由学习去噪器精修——
   既拿到 myopic 最优的直觉,又纠正它对队列拥塞的盲区。
4. **方法**:PGW —— preference-conditioned greedy warm-start(温度可调) + feedback-diffusion refinement。
5. **核心 finding**(§5):学习去噪器**纠正**而非照抄贪心(由温度消融 + 三方对照证实)。
6. **结论**:在高负载非平稳边缘卸载下,PGW 以 1× 推理同时改善 delay 与 energy,
   显著超过纯贪心(近视)与纯扩散(无信息起点)。

写法示例:
> We show that the denoising start of a preference-conditioned feedback-diffusion offloading policy is
> an underused design lever: the denoiser strongly follows an informative start, yet existing policies
> start from uninformative noise. We introduce **PGW**, a preference-conditioned greedy warm-start computed
> from online system information, which the learned denoiser refines to correct the myopic prior's
> congestion-blindness, improving both delay and energy at 1× inference cost.

---

## 9. 诚实边界(已测 vs 假设)

**已测(单 seed 诊断,文件可溯源)**:
- D1 去噪器跟随有信息起点(+50pp);D2 起点方差→尖锐度→delay;D3 H-MCSS full3 净负。

**假设(尚未端到端验证,严禁当结论写)**:
- PGW 端到端**大幅**胜过 A 与 B;
- 去噪器会**纠正**而非照抄(§5 的核心问题);
- 适中温度优于硬贪心。

**承诺**:跑出来若 C 不明显胜过 A/B、或只照抄,则**如实枪毙 PGW,不写、不粉饰**(红线:实验必须真)。
效应量的"大"是基于 D1+队列物理的**有据预测**,不是已测事实。

---

## 10. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | 去噪器照抄贪心 → C≈A,继承拥塞 | 温度扫(软起点留纠正空间);如实报告,照抄即枪毙 |
| R2 | 贪心在本环境没有大失败模式 → C≈B,效应不大 | 高负载设置(num_tasks_max 大)放大拥塞;若无大效应则降级或弃 |
| R3 | tran_rate 不在状态 → 误被当"作弊" | §4.4:先验由编排器侧系统信息现算,只用当前量,不用未来;论文写明 |
| R4 | 与 delay-only 反馈扩散骨干"撞" | 明确差异:偏好条件化 + 物理贪心起点 + 拥塞纠正 finding,均为多目标特有 |
| R5 | 单 seed 噪声 | 锁定方向后多 seed 复核(若效应大,1–2 个补充 seed 即可佐证;若小,本就该弃) |
| R6 | 贪心先验逐决策计算开销 | O(Emax) 解析式,远小于一次去噪前向;报告其可忽略 |

---

## 11. 验收标准

代码:
- `warmstart_fn=None` 时主项目行为不变;
- PGW 三臂 smoke 不报错,产出合法概率起点、合法动作分布;
- 训练/评估**同款起点**(都走 warmstart_fn)。

实验(go/no-go,单 seed 探针):
- C 的 HV **明显**高于 A 和 B(非噪声级);
- delay_min 与 energy_min **两角都不劣**(验 D2);
- 温度消融显示"软起点"区间 C 最好(验 §5 纠正)。

锁定为贡献(写论文前):
- 上述在 **V8 + 真反馈** 下成立,且**多 seed** 复核效应稳定;
- 配真实基线(纯贪心 A 本身就是强基线之一,另加 GMORL / DiscreteSAC)。

---

## 12. 推荐执行顺序

1. 写 `greedy_warmstart.py`(复用 `server_costs`)。
2. 给 `run_episode` / `evaluate_pareto` 加 `warmstart_fn`(默认 None);`run_single_seed` 从 cfg 构造。
3. 写贪心基线 agent(arm A)。
4. 写 `run_pgw_ab.py`(三臂 + 温度扫),`--smoke` 验管线。
5. 单 seed 跑 A/B/C + 温度扫 → 看 C 是否大赢(go/no-go)。
6. 大赢 → V8+真反馈下多 seed 复核 + 补基线 → 锁定为贡献,开始写;
   未大赢/照抄 → 如实枪毙。

---

## 13. 供 GPT Review 的问题(见随附 prompt)

1. PGW 相对 delay-only 反馈扩散骨干,novelty 够不够?是否只是"换了个更好的启发式先验"?
2. §5 的"纠正 vs 照抄"是否是真问题、三方+温度消融能否干净回答?
3. §4.4 的可部署性(tran_rate 在 env 侧算)会不会被 reviewer 当"信息泄漏/作弊"?如何 framing?
4. 效应量"大"的预测(基于 D1+队列物理)是否站得住?贪心在此类环境是否真有大失败模式?
5. 是否有已发表工作做了几乎相同的事(diffusion policy 的 informed/heuristic warm-start)使其贬值?
6. 消融是否足以归因(排除"只是贪心好"和"只是扩散好")?还缺什么对照?
