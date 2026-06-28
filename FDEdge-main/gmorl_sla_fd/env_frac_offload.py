"""连续分数卸载 env (路A 主体, Block 1)。

与离散 env 的根本区别: 动作 a∈Δ^N 是**连续切分**(任务负载在 N 台服务器上的占比),
不再 argmax 选一台。这是扩散表达力非空的区制(见 poc_multimodal / poc_reward_landscape)。

【任务模型假设 — 必须在论文明示 (2026-06-23 锁定)】
  本工作建模 **任意可分的数据并行负载 (arbitrarily-divisible, data-parallel workload)**:
  一个任务的数据被切成 N 个**互相独立**的子块, 各服务器**并行**处理自己那块, 彼此无先后依赖,
  任务在**最后一块完成时**结束 -> delay = makespan = max_i(completion_i)。切分后还需把各家
  结果收回合并, 由聚合开销 agg 建模。
  这正是 **可分负载理论 (Divisible Load Theory, DLT; Bharadwaj/Ghose/Robertazzi 1996)** 的
  single-installment 标准设定 ("each part executed independently by a processor"), 对批量推理/
  图像处理/OCR/大规模数据等数据并行任务**精确成立**, 故子块间**不需要时间连续性**。
  反例 (本工作不涵盖): 若任务是**按计算阶段切的流水线** (如 DNN 按层切, 后段等前段输出),
  子块间有时序依赖, delay 是关键路径而非 max -> 属 split-DNN/模型分割另一问题, 不在此 scope。
  即: 子块独立是**这一任务类的正确刻画 (有 DLT 撑腰), 非简化漏洞**, 但需声明任务类。
  (跨任务时序: 见 sequential 模式 — 队列跨任务累积, 对齐 FDEdge/GMORL 的序贯 MDP。)

物理 (用真实 env 常数, 算力主导):
  服务器 i 分到 a_i·D bit。
  off_time_i = a_i·D / rate_i           (传输, 此参数下可忽略)
  exe_time_i = a_i·D·C / f_i            (计算)
  wait_i     = q_i·C / f_i              (队列 backlog, 采样自状态)
  completion_i = wait_i + off_time_i + exe_time_i        (a_i>0 时)
  delay = makespan = max_{i: a_i>0} completion_i         (并行 -> 取 max = 尾部量)
  energy/槽 = Σ_i ( a_i·D/rate_i·P_off + a_i·D·k·C·f_i² )   (传输+计算, 动态 J)
            + Σ_i [i 本槽 cold→warm]·e_f                    (★激活能耗: 仅 OFF→ON 跃迁收一次 J, 红线①)
            + p_idle_power · Δ · |powered|                  (★静态**功率**×槽时长 Δ=arrival_dt: 开机就烧)
            + agg·(K_active − 1)                            (切分聚合开销 J)
  on_i = 本槽有新负载(a_i>ε) 或 仍有 backlog(q_i>0)。
  powered = (keep_alive=0): 仅 on —— 空闲在槽边界立即睡, 不再收静态功耗;
            (keep_alive>0): on ∪ 未到期 warm —— 保温中的空闲节点**也持续收静态功率**(否则保温=免费午餐)。
  warm 进 obs 且跨时隙携带: keep_alive=0 为二元(上槽是否在跑)=Markov-clean(实验默认锁此);
            keep_alive>0 存**归一化剩余保温∈[0,1]**, 否则两个 warm=1 但剩余保温不同 → 非 Markov。
  **策略不显式开关节点**: warm set 由连续分配 a + backlog 间接推出。论文须如实表述为"策略经连续分配
            **间接塑造** warm set", 不能说显式优化开关; 显式 power-gating 需额外二元动作 u_i, 不在此 scope。
  违约 = max(0, delay − deadline);  r_C = −sla_scale·违约 (ω-无关 SLA 通道)

非凸/多峰来源 (2026-06-23 修正): 激活只在 OFF→ON 收一次 → 单槽内"已热服务器加载"是凸的;
  组合多峰**迁移到时序"养几台热机" (server right-sizing)**: 少养→idle 省但队列涨→延迟差;
  多养→延迟低但 idle 贵。延迟要并行 vs 能量要少养, 在偏能量 ω 下撕成多峰 warm-set 最优。
  (注: 旧 poc_reward_landscape 的单槽双峰建立在"每任务收激活"上, 已不直接适用, 需时序版重证。)
激活/静态能耗有文献逐字建模 (服务器激活能 e_f + 动态 right-sizing 的 idle 静态功率, Lin et al. 2011)。
bandit 模式 (sequential=False): 每任务独立 = serverless 冷启动, 保留"每任务每台收 e_f"旧解释, 无 idle。

形态: 每 episode H 个独立任务的 contextual MORL bandit (queue 是采样上下文, 不跨步累积;
       序贯队列动态留作后续扩展)。ω 每 episode 外部设定, 与离散 env 一致。
接口对齐老 env: reset()->obs dict; step(a)->(obs,r,done,info), info['r_vec']=[r_T,r_E,r_C];
       episode_sla_summary()->violation_rate/p95/p99/mean。
"""
import numpy as np

# --- 真实 env 常数 (config_sla.json multi-part) ---
F_BASE = 2.0e9        # 边缘算力基准 Hz
C = 1000.0            # cycles/bit
KCO = 5e-31           # 能量系数 k
P_OFF = 0.01          # 传输功率 W
RATE = 700e6          # 传输率 bit/s (此量级下传输可忽略, 算力主导)


def torch_scalar_reward(a, f, q, w, D, deadline, e_f, agg, delay_ref, energy_ref,
                        sla_lambda=3.0, sla_scale=1.0, tau=0.02, beta=4.0, delay_scale=1.8):
    """可微 ω-标量化奖励 (**仅 direct warm-start 原型用**, 非主算法)。a,f,q [B,N]; w [B]。
    smooth-max 近似 makespan; soft 指示 1−exp(−a/τ) 近似"是否激活"(可微 fixed-charge)。
    返回 (scalar_reward[B], delay[B], energy[B])。eval 一律用硬 env 的 _eval_alloc。
    注: 这是硬 env 的可微近似, 与硬 env 仍有 τ 平滑等残差, 故 direct 只作 warm-start。"""
    import torch
    exe = a * D * C / f
    off = a * D / RATE
    wait = q * C / f
    comp = wait + off + exe                               # [B,N]
    soft_active = 1.0 - torch.exp(-a / tau)               # ~1 若 a>>τ 否则 ~0
    # ★mask 到 active: 未激活服务器从 smooth-max 剔除, 对齐硬 env 的 max over active;
    #   否则未用但高 q 的服务器的 wait 会污染代理 makespan (硬 10s vs 旧代理 20s 的根因)。
    comp_masked = comp - (1.0 - soft_active) * 1e3
    delay = torch.logsumexp(beta * comp_masked, dim=1) / beta    # smooth max over active ≈ makespan
    K = soft_active.sum(1)
    e_cmp = a * D * KCO * C * f ** 2
    e_tx = off * P_OFF
    energy = (e_cmp + e_tx).sum(1) + e_f * soft_active.sum(1) + agg * torch.clamp(K - 1, min=0)
    viol = torch.clamp(delay - deadline, min=0)
    rT = -delay_scale * delay / delay_ref     # delay_scale 平衡两通道梯度(energy span≈delay 1.8×)
    rE = -energy / energy_ref
    rC = -sla_scale * viol / delay_ref
    return w * rT + (1 - w) * rE + sla_lambda * rC, delay, energy


class FracOffloadEnv:
    def __init__(self, n_servers=5, task_size=20e6, deadline=6.0,
                 e_f_ratio=0.10, agg_ratio=0.05, idle_ratio=0.40, sla_scale=1.0,
                 homogeneous=True, hetero_span=0.12, q_max_ratio=0.3,
                 horizon=30, w=0.5, n_obj=3, sequential=False, arrival_dt=4.0,
                 keep_alive=0):
        self.N = n_servers
        self.D = task_size
        self.deadline = deadline
        self.sla_scale = sla_scale
        self.homo = homogeneous
        self.hetero_span = hetero_span          # 近同质: f 抖动 ±span
        self.q_max = q_max_ratio * task_size    # 队列 backlog 上限 (bit), 仅 bandit 采样用
        self.H = horizon
        self.w = w
        self.n_obj = n_obj
        # sequential=True: 队列**跨任务累积**(决策影响后续状态 = 序贯 MDP, 对齐 FDEdge/GMORL);
        #   f 每 episode 固定, q 从空起、每任务加本次负载并按到达间隔 Δ 消化。
        # sequential=False(默认): contextual bandit, 每任务重采 (f,q), 决策无跨任务后果 (= DLT+LinUCB lineage)。
        self.sequential = sequential
        self.arrival_dt = arrival_dt            # 任务到达间隔 Δ (s), 控制队列累积速度
        # 参考尺度 (单任务 / 一台 / 无队列)
        self.delay_ref = self.D * C / F_BASE                  # ≈10s
        self.energy_ref = self.D * KCO * C * (F_BASE ** 2)     # ≈0.04J
        self.e_f = e_f_ratio * self.energy_ref                 # 激活能耗 (仅 OFF→ON 跃迁收一次), 单位 J
        self.agg = agg_ratio * self.energy_ref                 # 每多一台的聚合开销, 单位 J
        # 静态/空转**功率** (W) = idle_ratio × 满载计算功率(energy_ref/delay_ref)。
        # idle 能量 = p_idle_power × 槽时长(arrival_dt) -> 随到达间隔正确缩放 (不再 dt 不变)。
        self.p_idle_power = idle_ratio * self.energy_ref / self.delay_ref
        self.keep_alive = int(keep_alive)                      # 空闲后保持 warm 的槽数 (0=立即睡, Markov-clean)
        self.warm = np.zeros(self.N)                           # 节点开关状态 (跨时隙, 进 obs); bandit 恒 0
        self.warm_timer = np.zeros(self.N)                     # keep_alive>0 时的剩余保温槽数
        self._t = 0

    # ---------- 采样一个任务上下文 (服务器算力 + 队列) ----------
    def _sample_ctx(self):
        if self.homo:
            f = F_BASE * (1.0 + np.random.uniform(-self.hetero_span, self.hetero_span, self.N))
        else:
            # 异构: 含一台明显更快 (敏感性实验用, 会压低多峰)
            f = F_BASE * np.random.uniform(0.7, 1.6, self.N)
        q = np.random.uniform(0, self.q_max, self.N)
        self.f = f; self.q = q

    def _obs(self):
        # 每服务器特征 [f/F_BASE, q/D, warm]; warm=节点开关状态(跨时隙, 论文要求纳入 state); 全合法 mask
        feat = np.stack([self.f / F_BASE, self.q / self.D, self.warm], axis=1).astype(np.float32)  # [N,3]
        return {'servers': feat, 'omega': np.float32(self.w),
                'mask': np.ones(self.N, dtype=np.float32)}

    def reset(self):
        self._t = 0
        self._delays, self._energies, self._viols = [], [], []
        self._sample_ctx()
        self.warm = np.zeros(self.N); self.warm_timer = np.zeros(self.N)   # 所有节点冷启动
        if self.sequential:
            self.q = np.zeros(self.N)     # 序贯: 队列从空起, 之后跨任务累积; f 本 episode 固定
        return self._obs()

    # ---------- 核心: 给定连续分配 a, 算 (delay, base_energy, K, active) ----------
    # base_energy = 动态(计算+传输) + 聚合; **不含**激活/idle (那部分由 step 按 warm 状态算)。
    def _eval_alloc(self, a):
        a = np.asarray(a, dtype=np.float64)
        a = np.clip(a, 0, None)
        s = a.sum()
        a = a / s if s > 1e-12 else np.ones(self.N) / self.N
        active = a > 1e-4
        off_t = a * self.D / RATE
        exe_t = a * self.D * C / self.f
        wait = self.q * C / self.f
        comp = wait + off_t + exe_t
        delay = float(comp[active].max()) if active.any() else float(comp.max())
        K = int(active.sum())
        e_tx = (a * self.D / RATE) * P_OFF
        e_cmp = a * self.D * KCO * C * (self.f ** 2)
        base_energy = float((e_tx + e_cmp).sum() + self.agg * max(K - 1, 0))
        return delay, base_energy, K, active

    def step(self, a):
        delay, base_energy, K, active = self._eval_alloc(a)
        # --- 激活/静态能耗 (节点开关状态, 红线①修正版) ---
        q_before = self.q                                   # 本槽起始 backlog
        warm_b = self.warm > 1e-9                           # 槽起始是否仍温热 (powered)
        if self.sequential:
            on = active | (q_before > 1e-9)                 # 本槽干活 = 有新负载 或 仍在消化 backlog
            newly = on & (~warm_b)                          # 仅 cold→warm 跃迁才收激活
            switch_energy = self.e_f * float(newly.sum())
            if self.keep_alive > 0:
                powered = warm_b | on                       # ★未到期的 warm 也持续耗静态功率 (修 free-lunch)
            else:
                powered = on                                # keep_alive=0: 槽边界立即睡, 仅本槽干活的耗能
            idle_energy = self.p_idle_power * self.arrival_dt * float(powered.sum())  # 功率×槽时长
        else:
            on = active                                     # bandit: 每任务 serverless 冷启动
            switch_energy = self.e_f * float(active.sum())  # 旧解释: 每任务每台收一次
            idle_energy = 0.0
        energy = base_energy + switch_energy + idle_energy
        viol = max(0.0, delay - self.deadline)
        r_T = -delay / self.delay_ref
        r_E = -energy / self.energy_ref
        r_C = -self.sla_scale * viol / self.delay_ref
        self._delays.append(delay); self._energies.append(energy)
        self._viols.append(1.0 if delay > self.deadline else 0.0)
        self._t += 1
        done = self._t >= self.H
        if self.sequential:
            # 序贯队列演化 (FCFS, 对齐 FDEdge 式5): 加本任务负载, 减到达间隔 Δ 内消化的量。
            # 决策的跨任务后果在此体现: 堆给某台 -> 它队列涨 -> 后续任务在该台等更久。
            a_n = np.clip(np.asarray(a, dtype=np.float64), 0, None)
            sa = a_n.sum()
            a_n = a_n / sa if sa > 1e-12 else np.ones(self.N) / self.N   # 与 _eval_alloc 同: 零向量→均分
            drain = (self.f / C) * self.arrival_dt
            self.q = np.maximum(q_before + a_n * self.D - drain, 0.0)
            # warm 状态演化:
            backlog = self.q > 1e-9                              # 演化后仍有 backlog = 仍在跑, 必然热
            if self.keep_alive > 0:
                refresh = on | backlog                          # 本槽干活或仍有积压 -> 重置保温窗口
                self.warm_timer = np.where(refresh, self.keep_alive, np.maximum(self.warm_timer - 1, 0))
                self.warm = self.warm_timer / self.keep_alive   # 归一化剩余保温∈[0,1] 进 obs -> 保持 Markov
            else:
                # keep_alive=0 立即休眠: 仅"演化后仍有 backlog (字面仍在跑)"才保持热;
                # 本槽用过但已清空的服务器在槽边界即睡 -> 下次重用重收 e_f (修 free-hotstart)。
                self.warm = backlog.astype(np.float64)
        elif not done:
            self._sample_ctx()
        info = {'r_vec': np.array([r_T, r_E, r_C], dtype=np.float32),
                'delay': delay, 'energy': energy, 'K_active': K, 'q_mean': float(np.mean(self.q)),
                'n_warm': float((self.warm > 1e-9).sum())}
        scalar_r = self.w * r_T + (1 - self.w) * r_E + r_C
        return self._obs(), float(scalar_r), done, info

    def episode_sla_summary(self):
        d = np.array(self._delays); e = np.array(self._energies)
        return {'mean_delay': float(d.mean()), 'mean_energy': float(e.mean()),
                'violation_rate': float(np.mean(self._viols)),
                'p95_delay': float(np.percentile(d, 95)),
                'p99_delay': float(np.percentile(d, 99))}


# ---------- 形态冒烟 ----------
if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    np.random.seed(0)
    env = FracOffloadEnv(n_servers=5, w=0.5)
    idle_e = env.p_idle_power * env.arrival_dt   # 每 warm 服务器每槽 idle 能量 (功率×槽时长)
    print('delay_ref=%.2fs  energy_ref=%.4fJ  e_f=%.4fJ(%.0f%%)  idle=%.4fW×%.0fs=%.4fJ/槽  agg=%.4fJ'
          % (env.delay_ref, env.energy_ref, env.e_f, 100 * env.e_f / env.energy_ref,
             env.p_idle_power, env.arrival_dt, idle_e, env.agg))
    obs = env.reset()
    print('obs servers shape =', obs['servers'].shape, '(feat=[f,q,warm])  mask =', obs['mask'])
    # 三种分配的 base 能量对比 (不含激活/idle): 均分 / 全给最快 / 随机
    f = env.f
    fastest = int(np.argmax(f))
    for name, a in [('均分split', np.ones(5) / 5),
                    ('全给最快concentrate', np.eye(5)[fastest]),
                    ('随机', np.random.dirichlet(np.ones(5)))]:
        d, e, K, act = env._eval_alloc(a)
        print('  %-22s delay=%6.2f base_energy=%6.4f K_active=%d' % (name, d, e, K))
    # --- 序贯 warm 状态验证: 同一台连续用, 激活只收一次 ---
    print('\n[序贯 warm 验证] 全给 server0, 连续 5 槽, 看激活是否只收一次:')
    senv = FracOffloadEnv(n_servers=5, w=0.5, sequential=True, arrival_dt=4.0, horizon=5,
                          e_f_ratio=0.20, idle_ratio=0.15)
    senv.reset(); a0 = np.eye(5)[0]
    prev_e = None
    for t in range(5):
        _, _, done, info = senv.step(a0)
        tag = '←含激活e_f' if (prev_e is None or info['energy'] > prev_e + senv.e_f * 0.5) else ''
        print('  槽%d energy=%.4f n_warm=%.0f q_mean=%.3f %s'
              % (t, info['energy'], info['n_warm'], info['q_mean'], tag))
        prev_e = info['energy']
    # 跑一个 bandit episode
    obs = env.reset(); done = False
    while not done:
        obs, r, done, info = env.step(np.random.dirichlet(np.ones(5)))
    s = env.episode_sla_summary()
    print('\nbandit episode summary:', {k: round(v, 3) for k, v in s.items()})
    print('[ok] env 冒烟通过')
