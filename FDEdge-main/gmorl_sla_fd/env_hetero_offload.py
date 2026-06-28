"""异构画像 + 时变链路 + 背景队列 的连续分数卸载 env。

目的: 给"最优支撑集**随状态切换**"提供土壤 —— 节点各有性格(GPU 快但费电/高冷启,
CPU 慢但绿/低冷启), 链路忽好忽坏, 背景负载忽忙忽闲。于是"此刻该用 {GPU 对} 还是
{CPU 对} 还是 {某台热机}"会随实现翻转。单峰策略只能押一个平均最优中心、跟不上翻转;
扩散采多个候选 + critic 按当前状态挑最优, 才有用武之地(配 sparsemax + M-候选选择)。

接口对齐 FracOffloadEnv: reset()->obs dict; step(a)->(obs, scalar_r, done, info),
info['r_vec']=[r_T,r_E,r_C]; episode_sla_summary()。
obs['servers'] = [N, feat_dim] 每节点特征 [f/F, k倍率, q/D, warm, rate/RATE];
feat_dim 由 env.feat_dim 报告 (管线据此推 actor/critic 输入维度, 不再硬编码 3)。

动作 a∈Δ^N 是对**一批可分负载**的条目级切分(各节点跑自己那份, 取 makespan),
配 sparsemax 时 a 自带精确零 -> z=支撑集; active = a>0。
"""
import numpy as np

F_BASE = 2.0e9        # 算力基准 Hz
C = 1000.0            # cycles/bit
KCO = 5e-31           # 能耗系数基准 (定义 ECPB_BASE 用)
P_OFF = 0.01          # 传输功率 W
RATE_BASE = 4.0e6     # 传输率基准 bit/s (调低使链路对 delay 有实质影响 -> 时变链路才有意义)
ECPB_BASE = KCO * C * (F_BASE ** 2)   # 基准每比特计算能耗 J/bit (= energy_ref/D, 全任务跑基准节点≈energy_ref)

# 画像类 (名字, f倍率[速度], pe[每比特能耗], idle功率倍率, 冷启动倍率, 背景负载强度): 2×GPU+2×CPU+1×mid
# 平衡的权衡: GPU 快(f高)但每比特能耗高(pe高)+高静态/高冷启; CPU 慢但绿; 速度与能耗解耦, 倍数温和->真竞争
PROFILE_5 = [
    ('gpu', 1.6, 1.5, 1.8, 2.0, 0.10),   # 快 / 费电 / 高静态 / 高冷启 / 背景轻
    ('gpu', 1.6, 1.5, 1.8, 2.0, 0.10),
    ('cpu', 0.85, 0.5, 0.5, 0.4, 0.25),  # 偏慢(但排得动)/ 绿 / 低静态 / 低冷启 / 背景偏重
    ('cpu', 0.85, 0.5, 0.5, 0.4, 0.25),
    ('mid', 1.1, 1.0, 1.0, 1.0, 0.15),   # 中庸
]


class HeteroOffloadEnv:
    def __init__(self, n_servers=5, task_size=20e6, deadline=11.0,
                 e_f_ratio=0.20, agg_ratio=0.10, idle_ratio=0.40, sla_scale=1.0,
                 horizon=30, w=0.5, n_obj=3, arrival_dt=6.0,
                 keep_alive=0, link_rho=0.7, link_vol=0.35, link_lo=0.25, link_hi=1.6,
                 bg_load=0.12, profiles=None):
        self.N = n_servers
        self.D = task_size
        self.deadline = deadline
        self.sla_scale = sla_scale
        self.H = horizon
        self.w = w
        self.n_obj = n_obj
        self.arrival_dt = arrival_dt
        self.keep_alive = int(keep_alive)
        # 时变链路 AR(1): rate_factor <- clip(rho·rate + (1-rho)·1 + N(0,vol), lo, hi)
        self.link_rho = link_rho; self.link_vol = link_vol
        self.link_lo = link_lo; self.link_hi = link_hi
        self.bg_load = bg_load
        # 画像 -> 每节点静态参数
        prof = profiles if profiles is not None else PROFILE_5
        assert len(prof) >= n_servers, '画像数需≥节点数'
        prof = prof[:n_servers]
        self.prof_name = [p[0] for p in prof]
        self.pf = np.array([p[1] for p in prof])      # f 倍率(速度)
        self.pe = np.array([p[2] for p in prof])      # 每比特能耗倍率(与速度解耦)
        self.pidle = np.array([p[3] for p in prof])   # idle 功率倍率
        self.pef = np.array([p[4] for p in prof])     # 冷启动倍率
        self.pbg = np.array([p[5] for p in prof])     # 背景负载强度
        self.f = F_BASE * self.pf                     # 各节点算力
        self.ecpb = ECPB_BASE * self.pe               # 各节点每比特计算能耗 J/bit
        # 参考尺度 (基准节点, 单任务无队列)
        self.delay_ref = self.D * C / F_BASE                   # ≈10s
        self.energy_ref = self.D * ECPB_BASE                   # ≈0.04J (全任务跑基准节点)
        self.e_f = e_f_ratio * self.energy_ref * self.pef      # 各节点冷启动能耗 (OFF->ON 收一次)
        self.agg = agg_ratio * self.energy_ref                 # 聚合开销/多一台
        self.p_idle = idle_ratio * self.energy_ref / self.delay_ref * self.pidle  # 各节点静态功率 W
        self.feat_dim = 5                              # 每节点 [f/F, pk, q/D, warm, rate/RATE]
        self._t = 0

    # ---------- 状态 ----------
    def _obs(self):
        feat = np.stack([self.f / F_BASE, self.pe, self.q / self.D, self.warm,
                         self.rate / RATE_BASE], axis=1).astype(np.float32)   # [N,5]=[速度,能耗,队列,热,链路]
        return {'servers': feat, 'omega': np.float32(self.w),
                'mask': np.ones(self.N, dtype=np.float32)}

    def reset(self):
        self._t = 0
        self._delays, self._energies, self._viols = [], [], []
        self.q = np.zeros(self.N)                      # 队列从空起(含背景, 之后累积)
        self.warm = np.zeros(self.N)
        self.warm_timer = np.zeros(self.N)
        self.rate = RATE_BASE * np.ones(self.N)        # 链路从基准起
        self._bg_arrive()                              # 先注入一拨背景, 让起手就有忙闲差
        return self._obs()

    def _bg_arrive(self):
        # 背景负载: 各节点按强度 pbg 注入随机背景工作量(bit), 制造忙闲随机性
        self.q = self.q + self.pbg * self.bg_load * self.D * np.random.uniform(0, 0.5, self.N)

    def _link_step(self):
        noise = np.random.randn(self.N) * self.link_vol
        fac = self.rate / RATE_BASE
        fac = self.link_rho * fac + (1 - self.link_rho) * 1.0 + noise
        self.rate = RATE_BASE * np.clip(fac, self.link_lo, self.link_hi)

    # ---------- 给定分配 a, 算 (delay, base_energy, K, active) ----------
    def _eval_alloc(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        s = a.sum()
        a = a / s if s > 1e-12 else np.ones(self.N) / self.N
        active = a > 1e-9                              # sparsemax 自带精确零 -> 精确支撑集
        off_t = a * self.D / self.rate
        exe_t = a * self.D * C / self.f
        wait = self.q * C / self.f
        comp = wait + off_t + exe_t
        delay = float(comp[active].max()) if active.any() else float(comp.max())
        K = int(active.sum())
        e_tx = off_t * P_OFF
        e_cmp = a * self.D * self.ecpb                 # 每比特能耗 × 分到的数据量 (速度/能耗已解耦)
        base_energy = float((e_tx + e_cmp).sum() + self.agg * max(K - 1, 0))
        return delay, base_energy, K, active

    def step(self, a):
        delay, base_energy, K, active = self._eval_alloc(a)
        q_before = self.q
        warm_b = self.warm > 1e-9
        on = active | (q_before > 1e-9)
        newly = on & (~warm_b)
        switch_energy = float((self.e_f * newly).sum())            # 各节点冷启动能耗
        if self.keep_alive > 0:
            powered = warm_b | on
        else:
            powered = on
        idle_energy = float((self.p_idle * powered).sum()) * self.arrival_dt   # 各节点静态功率×槽时长
        energy = base_energy + switch_energy + idle_energy
        viol = max(0.0, delay - self.deadline)
        r_T = -delay / self.delay_ref
        r_E = -energy / self.energy_ref
        r_C = -self.sla_scale * viol / self.delay_ref
        self._delays.append(delay); self._energies.append(energy)
        self._viols.append(1.0 if delay > self.deadline else 0.0)
        self._t += 1
        done = self._t >= self.H
        # --- 队列演化: 加本任务分配 + 背景新到 - 各节点消化 ---
        a_n = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        sa = a_n.sum(); a_n = a_n / sa if sa > 1e-12 else np.ones(self.N) / self.N
        drain = (self.f / C) * self.arrival_dt
        self.q = np.maximum(q_before + a_n * self.D - drain, 0.0)
        if not done:
            self._bg_arrive()                                       # 下一槽背景新到
            self._link_step()                                       # 链路演化
        # warm 演化
        backlog = self.q > 1e-9
        if self.keep_alive > 0:
            refresh = on | backlog
            self.warm_timer = np.where(refresh, self.keep_alive, np.maximum(self.warm_timer - 1, 0))
            self.warm = self.warm_timer / self.keep_alive
        else:
            self.warm = backlog.astype(np.float64)
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


# ---------- 形态冒烟 + "支撑集随状态切换"探针 ----------
if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    np.random.seed(0)
    env = HeteroOffloadEnv(horizon=20)
    print('画像:', env.prof_name)
    print('f倍率=%s\n每比特能耗pe=%s\nidle倍率=%s\n冷启倍率=%s\n背景强度=%s'
          % (env.pf, env.pe, env.pidle, env.pef, env.pbg))
    print('delay_ref=%.2fs energy_ref=%.4fJ feat_dim=%d' % (env.delay_ref, env.energy_ref, env.feat_dim))
    obs = env.reset()
    print('obs servers shape =', obs['servers'].shape, '(feat=[速度,能耗,队列,热,链路])')
    # 候选支撑在 (ω, 当前链路/队列状态) 下谁最优 -> 看是否随 ω 不同、随槽切换 (多峰必要条件)
    cands = {'GPU对{0,1}': np.array([.5, .5, 0, 0, 0]),
             'CPU对{2,3}': np.array([0, 0, .5, .5, 0]),
             '单GPU{0}': np.array([1., 0, 0, 0, 0]),
             'GPU+CPU{0,2}': np.array([.6, 0, .4, 0, 0]),
             '均分5台': np.ones(5) / 5}
    def best_supp(w):
        bv, bn = -1e18, None
        for nm, a in cands.items():
            d, be, K, _ = env._eval_alloc(a)
            val = w * (-d / env.delay_ref) + (1 - w) * (-be / env.energy_ref)
            if val > bv: bv, bn = val, nm
        return bn
    print('\n[最优支撑切换探针] 每槽看 w=0(能耗)/0.5/1(延迟) 各自最优支撑:')
    seen = set()
    for t in range(8):
        b0, b5, b1 = best_supp(0.0), best_supp(0.5), best_supp(1.0)
        seen.update([b0, b5, b1])
        print('  槽%d 链路=%s | w0=%-11s w.5=%-11s w1=%-11s'
              % (t, np.round(env.rate / RATE_BASE, 2), b0, b5, b1))
        env.step(cands[b5])
    print('-> 整段共出现 %d 种不同最优支撑: %s' % (len(seen), seen))
    print('   (>=2 = 最优支撑确实随 ω/状态切换 = 多峰土壤存在, diffusion 有戏)')
    s = env.episode_sla_summary()
    print('summary:', {k: round(v, 3) for k, v in s.items()})
    print('[ok] hetero env 冒烟通过')
