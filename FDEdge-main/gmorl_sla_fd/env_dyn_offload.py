"""动态信道 + 随机任务类型 + 增强异构 的连续分数卸载 env (路A 现实化版, 不引入 DAG/跨组惩罚)。

相对 env_hetero_offload 的三处现实化改动 (2026-06-26, 用户拍板的主线):
  1. **Shannon 时变速率**: 替代固定/AR(1)-on-rate。每服务器有独立信道增益 (均值由 chan_q 决定 ->
     信道好坏与算力高低**解耦**), 增益按 AR(1) 时变 (log-正态阴影衰落)。
     rate_i = B·log2(1 + SNR_i),  SNR_i = snr_base · gain_i。
  2. **随机任务类型**: 每步从 {normal, comm, compute} 抽一类 ->
       normal  : 数据/计算适中
       comm    : 大数据 + 轻计算  -> 传输主导, 最优押**信道好**的服务器
       compute : 小数据 + 重计算  -> 计算主导, 最优押**算力高**的服务器
     => 同一 (state, ω) 下最优支撑随**任务类型离散翻转** = 非光滑 context->support 映射,
        高斯单步幅射倾向取平均=次优, 扩散多步去噪可 commit 到一个 regime (单动作扩散赢的物理来源)。
  3. **增强异构**: 算力/动态能耗/idle/冷启动 倍率谱拉宽, 并新增 chan_q (信道质量) 维度异构。

任务/信道**全部可观测**, 折进每服务器特征 (不藏信息, 对高斯同样公平):
  obs['servers'] = [N, 8] 每节点 [f/F, pe, q/D, warm, rate/RATE_NOM, off_coef, exe_coef, ecmp_coef]
    off_coef_i = (D_t / rate_i) / delay_ref        本任务全压到 i 的传输时间 (含任务大小×信道)
    exe_coef_i = (D_t·C_t / f_i) / delay_ref       本任务全压到 i 的计算时间 (含任务大小×计算密度×算力)
    ecmp_coef_i= (D_t·ecpb_i) / energy_ref         本任务全压到 i 的计算能耗 (含任务大小×每比特能耗)
  -> 任务类型 (D_t,C_t) 与信道 (rate_i) 经这三个系数进入观测, feat_dim 由 env.feat_dim 报告。

奖励尺度**固定在标称任务** (D_base,C_base) 上 -> 随机 task size 不晃动奖励量级。
接口对齐 HeteroOffloadEnv: reset()->obs dict; step(a)->(obs, scalar_r, done, info),
  info['r_vec']=[r_T,r_E,r_C]; episode_sla_summary()。
"""
import numpy as np

F_BASE = 2.0e9        # 算力基准 Hz
C_BASE = 1000.0       # cycles/bit (标称计算密度)
KCO = 5e-31           # 能耗系数基准
P_OFF = 0.1           # 传输功率 W (信道现实化后传输能耗不再可忽略)
KCO_C_F2 = KCO * C_BASE * (F_BASE ** 2)
ECPB_BASE = KCO_C_F2                       # 标称每比特计算能耗 J/bit

# Shannon 信道参数
BW = 10.0e6           # 带宽 Hz
SNR_BASE = 0.32       # 标称 SNR (chan_q=1, gain=1 时); RATE_NOM = BW·log2(1+SNR_BASE) ≈ 4.0 Mbit/s
GAIN_LO, GAIN_HI = 0.15, 3.0

# 画像 (名字, f倍率, pe每比特能耗, idle倍率, 冷启倍率, 背景强度, chan_q信道质量): 谱已拉宽 + 信道异构
#   GPU: 快/费电/高静态/高冷启;  CPU: 慢/绿/低静态/低冷启;  信道质量与算力解耦 (快算力未必好信道)
PROFILE_5 = [
    ('gpu', 1.8, 1.7, 2.0, 2.3, 0.10, 1.30),   # 快 费电  好信道
    ('gpu', 1.8, 1.7, 2.0, 2.3, 0.10, 0.70),   # 快 费电  烂信道  <- 计算密集爱它, 通信密集躲它
    ('cpu', 0.80, 0.42, 0.42, 0.32, 0.28, 1.25),  # 慢 绿   好信道  <- 通信密集爱它
    ('cpu', 0.80, 0.42, 0.42, 0.32, 0.28, 0.72),  # 慢 绿   烂信道
    ('mid', 1.10, 1.00, 1.00, 1.00, 0.16, 1.00),  # 中庸    中信道
]

# 任务类型 (名字, size_mult 数据量, c_mult 计算密度, prob)
TASK_TYPES = [
    ('normal',  1.00, 1.00, 0.40),
    ('comm',    2.20, 0.45, 0.30),   # 大数据 轻计算 -> 传输主导
    ('compute', 0.65, 2.40, 0.30),   # 小数据 重计算 -> 计算主导
]


class DynOffloadEnv:
    def __init__(self, n_servers=5, task_size=20e6, deadline=11.0,
                 e_f_ratio=0.20, agg_ratio=0.10, idle_ratio=0.40, sla_scale=1.0,
                 horizon=30, w=0.5, n_obj=3, arrival_dt=6.0,
                 keep_alive=0, chan_rho=0.7, chan_vol=0.40, bg_load=0.12,
                 dl_ratio=0.0, dl_rate_mult=2.0, coord_delay_ratio=0.0,
                 profiles=None, task_types=None):
        self.N = n_servers
        self.D_base = task_size
        self.deadline = deadline
        self.sla_scale = sla_scale
        self.H = horizon
        self.w = w
        self.n_obj = n_obj
        self.arrival_dt = arrival_dt
        self.keep_alive = int(keep_alive)
        # 信道 AR(1) on log-gain (log-正态阴影): g <- exp(rho·ln g + (1-rho)·ln chan_q + N(0,vol))
        self.chan_rho = chan_rho; self.chan_vol = chan_vol
        self.bg_load = bg_load
        prof = profiles if profiles is not None else PROFILE_5
        assert len(prof) >= n_servers, '画像数需≥节点数'
        prof = prof[:n_servers]
        self.prof_name = [p[0] for p in prof]
        self.pf = np.array([p[1] for p in prof])
        self.pe = np.array([p[2] for p in prof])
        self.pidle = np.array([p[3] for p in prof])
        self.pef = np.array([p[4] for p in prof])
        self.pbg = np.array([p[5] for p in prof])
        self.chan_q = np.array([p[6] for p in prof])      # 信道质量均值 (与算力解耦)
        self.f = F_BASE * self.pf
        self.ecpb = ECPB_BASE * self.pe
        # 参考尺度: 标称任务 (D_base, C_base) 跑基准节点, 固定 -> 随机 task size 不晃奖励
        self.delay_ref = self.D_base * C_BASE / F_BASE        # ≈10s
        self.energy_ref = self.D_base * ECPB_BASE             # ≈0.04J
        self.e_f = e_f_ratio * self.energy_ref * self.pef
        self.agg = agg_ratio * self.energy_ref
        self.p_idle = idle_ratio * self.energy_ref / self.delay_ref * self.pidle
        self.rate_nom = BW * np.log2(1.0 + SNR_BASE)          # 标称速率 ≈4Mbit/s
        # 结果回传 + scatter/merge 协调开销 (切分非免费; 默认0=旧行为)
        self.dl_ratio = dl_ratio                              # 输出/输入 比 ρ (回传数据量=ρ·分到的输入)
        self.dl_rate_mult = dl_rate_mult                      # 下行速率 = 上行 × 此倍 (下行通常更快)
        self.coord_delay_ratio = coord_delay_ratio
        self.coord_delay = coord_delay_ratio * self.delay_ref # 协调开销/多一台 (随活跃台数 K 增长)
        self.task_types = task_types if task_types is not None else TASK_TYPES
        self._tt_p = np.array([t[3] for t in self.task_types]); self._tt_p /= self._tt_p.sum()
        self.feat_dim = 8                                     # 见模块 docstring
        self._t = 0

    # ---------- 信道 / 任务采样 ----------
    def _shannon_rate(self):
        snr = SNR_BASE * self.gain                            # 每服务器 SNR
        return BW * np.log2(1.0 + snr)                        # bit/s

    def _chan_step(self):
        lg = np.log(np.clip(self.gain, 1e-6, None))
        lg = self.chan_rho * lg + (1 - self.chan_rho) * np.log(self.chan_q) + np.random.randn(self.N) * self.chan_vol
        self.gain = np.clip(np.exp(lg), GAIN_LO, GAIN_HI)
        self.rate = self._shannon_rate()

    def _draw_task(self):
        k = int(np.random.choice(len(self.task_types), p=self._tt_p))
        nm, sm, cm, _ = self.task_types[k]
        self.task_name = nm
        self.task_k = k
        self.D_t = self.D_base * sm
        self.C_t = C_BASE * cm

    # ---------- 状态 ----------
    def _obs(self):
        off_coef = (self.D_t / self.rate) / self.delay_ref            # 传输时间系数 (任务大小×信道)
        exe_coef = (self.D_t * self.C_t / self.f) / self.delay_ref    # 计算时间系数 (任务×密度×算力)
        ecmp_coef = (self.D_t * self.ecpb) / self.energy_ref          # 计算能耗系数 (任务×每比特能耗)
        feat = np.stack([self.f / F_BASE, self.pe, self.q / self.D_base, self.warm,
                         self.rate / self.rate_nom, off_coef, exe_coef, ecmp_coef], axis=1).astype(np.float32)
        return {'servers': feat, 'omega': np.float32(self.w),
                'mask': np.ones(self.N, dtype=np.float32), 'task': np.int64(self.task_k)}

    def reset(self):
        self._t = 0
        self._delays, self._energies, self._viols = [], [], []
        self.q = np.zeros(self.N)
        self.warm = np.zeros(self.N)
        self.warm_timer = np.zeros(self.N)
        self.gain = self.chan_q.copy()                # 信道增益从各自均值起
        self.rate = self._shannon_rate()
        self._bg_arrive()
        self._draw_task()
        return self._obs()

    def _bg_arrive(self):
        self.q = self.q + self.pbg * self.bg_load * self.D_base * np.random.uniform(0, 0.5, self.N)

    # ---------- 给定分配 a, 算 (delay, base_energy, K, active) ----------
    def _eval_alloc(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        s = a.sum()
        a = a / s if s > 1e-12 else np.ones(self.N) / self.N
        active = a > 1e-9
        off_t = a * self.D_t / self.rate
        exe_t = a * self.D_t * self.C_t / self.f
        dl_t = self.dl_ratio * a * self.D_t / (self.rate * self.dl_rate_mult)  # 结果回传 (走下行)
        wait = self.q * C_BASE / self.f                       # 队列用标称密度 (与当前任务类型解耦)
        comp = wait + off_t + exe_t + dl_t
        K = int(active.sum())
        makespan = float(comp[active].max()) if active.any() else float(comp.max())
        delay = makespan + self.coord_delay * max(K - 1, 0)   # +scatter/merge 协调开销 -> 切分非免费
        e_tx = (off_t + dl_t) * P_OFF
        e_cmp = a * self.D_t * self.ecpb
        base_energy = float((e_tx + e_cmp).sum() + self.agg * max(K - 1, 0))
        return delay, base_energy, K, active

    def step(self, a):
        delay, base_energy, K, active = self._eval_alloc(a)
        q_before = self.q
        warm_b = self.warm > 1e-9
        on = active | (q_before > 1e-9)
        newly = on & (~warm_b)
        switch_energy = float((self.e_f * newly).sum())
        powered = (warm_b | on) if self.keep_alive > 0 else on
        idle_energy = float((self.p_idle * powered).sum()) * self.arrival_dt
        energy = base_energy + switch_energy + idle_energy
        viol = max(0.0, delay - self.deadline)
        r_T = -delay / self.delay_ref
        r_E = -energy / self.energy_ref
        r_C = -self.sla_scale * viol / self.delay_ref
        self._delays.append(delay); self._energies.append(energy)
        self._viols.append(1.0 if delay > self.deadline else 0.0)
        self._t += 1
        done = self._t >= self.H
        # 队列演化: 加本任务分配 (按本任务 D_t) + 背景新到 - 各节点消化 (标称密度)
        a_n = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        sa = a_n.sum(); a_n = a_n / sa if sa > 1e-12 else np.ones(self.N) / self.N
        drain = (self.f / C_BASE) * self.arrival_dt
        self.q = np.maximum(q_before + a_n * self.D_t - drain, 0.0)
        if not done:
            self._bg_arrive()
            self._chan_step()
            self._draw_task()
        backlog = self.q > 1e-9
        if self.keep_alive > 0:
            refresh = on | backlog
            self.warm_timer = np.where(refresh, self.keep_alive, np.maximum(self.warm_timer - 1, 0))
            self.warm = self.warm_timer / self.keep_alive
        else:
            self.warm = backlog.astype(np.float64)
        info = {'r_vec': np.array([r_T, r_E, r_C], dtype=np.float32),
                'delay': delay, 'energy': energy, 'K_active': K, 'q_mean': float(np.mean(self.q)),
                'task': self.task_name, 'n_warm': float((self.warm > 1e-9).sum())}
        scalar_r = self.w * r_T + (1 - self.w) * r_E + r_C
        return self._obs(), float(scalar_r), done, info

    def episode_sla_summary(self):
        d = np.array(self._delays); e = np.array(self._energies)
        return {'mean_delay': float(d.mean()), 'mean_energy': float(e.mean()),
                'violation_rate': float(np.mean(self._viols)),
                'p95_delay': float(np.percentile(d, 95)),
                'p99_delay': float(np.percentile(d, 99))}


# ---------- 形态冒烟 + "支撑随任务类型/信道翻转"探针 ----------
if __name__ == '__main__':
    import sys
    from collections import Counter
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    DSCALE, SLAM = 1.5, 3.0
    cands = {
        'GPU好信道{0}':  np.array([1., 0, 0, 0, 0]),
        'GPU烂信道{1}':  np.array([0, 1., 0, 0, 0]),
        'GPU对{0,1}':    np.array([.5, .5, 0, 0, 0]),
        'CPU好信道{2}':  np.array([0, 0, 1., 0, 0]),
        'CPU对{2,3}':    np.array([0, 0, .5, .5, 0]),
        'mid{4}':        np.array([0, 0, 0, 0, 1.]),
        'GPU+CPU{0,2}':  np.array([.5, 0, .5, 0, 0]),
        '快机对{0,4}':   np.array([.5, 0, 0, 0, .5]),
        '均分5':         np.ones(5) / 5,
    }

    def env_scalar(env, a, w):
        d, be, K, active = env._eval_alloc(a)
        warm_b = env.warm > 1e-9; on = active | (env.q > 1e-9); newly = on & (~warm_b)
        energy = be + float((env.e_f * newly).sum()) + float((env.p_idle * on).sum()) * env.arrival_dt
        viol = max(0.0, d - env.deadline)
        val = w * DSCALE * (-d / env.delay_ref) + (1 - w) * (-energy / env.energy_ref) + SLAM * (-viol / env.delay_ref)
        return val, d, energy

    def best_supp(env, w):
        bv, bn = -1e18, None
        for nm, a in cands.items():
            v, _, _ = env_scalar(env, a, w)
            if v > bv: bv, bn = v, nm
        return bn

    print('=' * 84)
    print('DynOffloadEnv 冒烟 + 支撑翻转探针')
    print('=' * 84)
    np.random.seed(0)
    env = DynOffloadEnv(horizon=30)
    print('画像:', env.prof_name)
    print('f倍率=%s  chan_q=%s' % (env.pf, env.chan_q))
    print('delay_ref=%.2fs energy_ref=%.4fJ rate_nom=%.2fMbit/s feat_dim=%d'
          % (env.delay_ref, env.energy_ref, env.rate_nom / 1e6, env.feat_dim))
    obs = env.reset()
    print('obs servers shape =', obs['servers'].shape, ' task0 =', env.task_name)

    # 探针1: 固定一个状态, 三种任务类型下, w∈{0,.5,1} 各自最优支撑 -> 看是否随任务类型翻转
    print('\n[探针1] 同一信道/队列状态, 强制三种任务类型, 看最优支撑随类型翻转:')
    np.random.seed(3); env = DynOffloadEnv(); env.reset()
    base_gain = env.gain.copy(); base_q = env.q.copy()
    for w in [0.0, 0.5, 1.0]:
        row = []
        for k, (nm, sm, cm, _) in enumerate(env.task_types):
            env.gain = base_gain.copy(); env.rate = env._shannon_rate(); env.q = base_q.copy()
            env.D_t = env.D_base * sm; env.C_t = C_BASE * cm
            row.append('%-8s->%-12s' % (nm, best_supp(env, w)))
        print('  w=%.1f | %s' % (w, ' | '.join(row)))

    # 探针2: 跨槽 (信道时变) 看支撑随实现翻转 + 统计共出现几种最优支撑
    print('\n[探针2] 跨 30 槽 (信道时变+随机任务), 各 w 最优支撑分布:')
    for w in [0.0, 0.5, 1.0]:
        np.random.seed(11); env = DynOffloadEnv(horizon=30); env.reset()
        seen = Counter()
        for t in range(30):
            b = best_supp(env, w); seen[b] += 1
            env.step(cands[b])
        print('  w=%.1f | %d 种支撑: %s' % (w, len(seen), dict(seen)))

    # 探针3: regime 冲突量化 — 同一 (state, w) 下 comm-最优 与 compute-最优 支撑不同的比例
    print('\n[探针3] regime 冲突率 (comm-最优 ≠ compute-最优 支撑的状态占比, 越高=单峰越受罪):')
    for w in [0.0, 0.5, 1.0]:
        np.random.seed(21); env = DynOffloadEnv(horizon=40); env.reset()
        conflict, n = 0, 0
        for t in range(40):
            g = env.gain.copy(); q = env.q.copy()
            # comm 最优
            env.gain = g.copy(); env.rate = env._shannon_rate(); env.q = q.copy()
            env.D_t = env.D_base * 2.20; env.C_t = C_BASE * 0.45
            bc = best_supp(env, w)
            # compute 最优
            env.gain = g.copy(); env.rate = env._shannon_rate(); env.q = q.copy()
            env.D_t = env.D_base * 0.65; env.C_t = C_BASE * 2.40
            bk = best_supp(env, w)
            conflict += int(bc != bk); n += 1
            env.gain = g; env.rate = env._shannon_rate(); env.q = q
            env._draw_task()
            env.step(cands[best_supp(env, w)])
        print('  w=%.1f | regime 冲突 %d/%d = %.0f%%  (comm偏好 vs compute偏好 支撑不同)'
              % (w, conflict, n, 100.0 * conflict / n))

    # 探针4: 单峰受罪量化 — 跨 regime "平均动作" 比 "按 regime commit" 差多少 (单动作扩散的可赢空间)
    print('\n[探针4] 平均动作 vs commit 的代价 (w=1, 在 comm/compute 等概混合的状态上):')
    np.random.seed(31); env = DynOffloadEnv(horizon=30); env.reset()
    gaps = []
    for t in range(30):
        g = env.gain.copy(); q = env.q.copy()
        env.gain = g.copy(); env.rate = env._shannon_rate(); env.q = q.copy()
        env.D_t = env.D_base * 2.20; env.C_t = C_BASE * 0.45; a_comm = cands[best_supp(env, 1.0)]
        env.gain = g.copy(); env.rate = env._shannon_rate(); env.q = q.copy()
        env.D_t = env.D_base * 0.65; env.C_t = C_BASE * 2.40; a_comp = cands[best_supp(env, 1.0)]
        a_avg = 0.5 * a_comm + 0.5 * a_comp                      # 单峰只能给一个折中
        # 在两个 regime 上分别评: commit (对的那个) vs 平均
        for sm, cm, a_commit in [(2.20, 0.45, a_comm), (0.65, 2.40, a_comp)]:
            env.gain = g.copy(); env.rate = env._shannon_rate(); env.q = q.copy()
            env.D_t = env.D_base * sm; env.C_t = C_BASE * cm
            v_commit, _, _ = env_scalar(env, a_commit, 1.0)
            v_avg, _, _ = env_scalar(env, a_avg, 1.0)
            gaps.append(v_commit - v_avg)
        env.gain = g; env.rate = env._shannon_rate(); env.q = q; env._draw_task()
        env.step(cands[best_supp(env, 1.0)])
    gaps = np.array(gaps)
    print('  commit - average 标量化收益差: 均值=%+.4f 中位=%+.4f 正比例=%.0f%%'
          % (gaps.mean(), np.median(gaps), 100.0 * (gaps > 1e-6).mean()))
    print('  (>0 = 按 regime commit 优于折中平均 = 单峰策略受罪 = 单动作扩散有可赢空间)')

    print('\n[ok] dyn env 冒烟 + 探针通过')
