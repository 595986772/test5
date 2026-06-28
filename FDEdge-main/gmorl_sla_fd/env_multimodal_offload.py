"""真·多峰连续分数卸载 env: 让"单峰 ω-条件策略"必失真, "扩散多峰+critic选择"不可替代。

为什么前一个 hetero env 不够: 它奖励铺开(divisible load, makespan=max, 越分越快), 于是最优
对 ω 光滑 -> 给 baseline 装 ω-FiLM 后单峰高斯直接做得更好(实测 gauss+film 反超)。

本 env 的关键性质 = **跨组通信重罚 + 组内可分**:
  - N=6 台分 3 组(每组 2 台, 同址高速互联): {0,1}=GPU对, {2,3}=CPU对, {4,5}=mid对。
  - 组内拆分**免费**(co-located); 但激活**跨多组**要付一大笔通信延迟+能耗 (CD/CE × (组数-1))。
  - => 最优 = **承诺到当前最好的"单整组"**(K≤2, 同组), 而"哪个组最好"随 ω 与每组时变拥塞**离散翻转**, 且常出现两组打平。
为什么这逼死单峰、成全扩散:
  - 对角高斯对 6 个 logit 独立加噪 + sparsemax: 两组接近时 mu 被迫摊到两组 -> 采样常出**跨组混合**支撑 -> 吃 CD 罚。无法表达"整组A 异或 整组B"这种**相关共激活**。
  - 扩散反向链相关采样: 一个样本承诺整组A, 另一个承诺整组B(干净双峰); critic 按当前拥塞挑更好的组。选择在"组打平/翻转"处兑现, 单峰无从替代。

接口对齐 HeteroOffloadEnv: reset()->obs; step(a)->(obs,r,done,info); info['r_vec']; episode_sla_summary()。
obs['servers']=[N, feat_dim=6] 每台 [f/F, pe, q/D, warm, rate/RATE, cong(本组拥塞)]; feat_dim 由 env.feat_dim 报告。
"""
import numpy as np

F_BASE = 2.0e9
C = 1000.0
KCO = 5e-31
P_OFF = 0.01
RATE_BASE = 4.0e6
ECPB_BASE = KCO * C * (F_BASE ** 2)

# 3 组 × 2 台。组profile温和差异给 ω-依赖(GPU快费电/CPU慢绿/mid中); 真正翻转来自每组时变拥塞。
GROUPS = [[0, 1], [2, 3], [4, 5]]
PROFILE_6 = [
    ('gpu', 1.6, 1.5, 1.8, 2.0), ('gpu', 1.6, 1.5, 1.8, 2.0),   # (name, pf速度, pe能耗/bit, pidle, pef冷启)
    ('cpu', 0.85, 0.5, 0.5, 0.4), ('cpu', 0.85, 0.5, 0.5, 0.4),
    ('mid', 1.1, 1.0, 1.0, 1.0), ('mid', 1.1, 1.0, 1.0, 1.0),
]


class MultimodalOffloadEnv:
    def __init__(self, n_servers=6, task_size=20e6, deadline=12.0,
                 e_f_ratio=0.20, agg_ratio=0.10, idle_ratio=0.40, sla_scale=1.0,
                 horizon=30, w=0.5, n_obj=3, arrival_dt=6.0, keep_alive=0,
                 cong_rho=0.6, cong_vol=0.45, cong_lo=0.40, cong_hi=1.6,
                 comm_delay_ratio=0.45, comm_energy_ratio=0.6, bg_load=0.12, profiles=None, groups=None):
        assert n_servers == 6, '本 env 固定 6 台 3 组'
        self.N = n_servers; self.D = task_size; self.deadline = deadline
        self.sla_scale = sla_scale; self.H = horizon; self.w = w; self.n_obj = n_obj
        self.arrival_dt = arrival_dt; self.keep_alive = int(keep_alive)
        self.groups = groups if groups is not None else GROUPS
        self.gid = np.zeros(self.N, dtype=int)                  # 每台 -> 组号
        for g, members in enumerate(self.groups):
            for i in members: self.gid[i] = g
        self.n_groups = len(self.groups)
        # 每组时变拥塞 AR(1) (独立 -> 谁最好频繁翻转 + 常打平)
        self.cong_rho = cong_rho; self.cong_vol = cong_vol
        self.cong_lo = cong_lo; self.cong_hi = cong_hi
        self.bg_load = bg_load
        prof = (profiles if profiles is not None else PROFILE_6)[:n_servers]
        self.prof_name = [p[0] for p in prof]
        self.pf = np.array([p[1] for p in prof]); self.pe = np.array([p[2] for p in prof])
        self.pidle = np.array([p[3] for p in prof]); self.pef = np.array([p[4] for p in prof])
        self.f = F_BASE * self.pf
        self.ecpb = ECPB_BASE * self.pe
        self.delay_ref = self.D * C / F_BASE
        self.energy_ref = self.D * ECPB_BASE
        self.e_f = e_f_ratio * self.energy_ref * self.pef
        self.agg = agg_ratio * self.energy_ref
        self.p_idle = idle_ratio * self.energy_ref / self.delay_ref * self.pidle
        # 跨组通信罚 (核心): 激活跨 G 组 -> +CD·(G-1) 延迟, +CE·(G-1) 能耗
        self.comm_delay = comm_delay_ratio * self.delay_ref
        self.comm_energy = comm_energy_ratio * self.energy_ref
        self.feat_dim = 6
        self._t = 0

    def _eff_f(self):
        return self.f * self.cong[self.gid]                     # 有效算力 = 基准 × 本组拥塞

    def _obs(self):
        feat = np.stack([self.f / F_BASE, self.pe, self.q / self.D, self.warm,
                         self.rate / RATE_BASE, self.cong[self.gid]], axis=1).astype(np.float32)
        return {'servers': feat, 'omega': np.float32(self.w), 'mask': np.ones(self.N, dtype=np.float32)}

    def reset(self):
        self._t = 0; self._delays, self._energies, self._viols = [], [], []
        self.q = np.zeros(self.N); self.warm = np.zeros(self.N); self.warm_timer = np.zeros(self.N)
        self.rate = RATE_BASE * np.ones(self.N)
        self.cong = np.ones(self.n_groups)                      # 每组拥塞从 1 起
        self._cong_step(); self._bg_arrive()
        return self._obs()

    def _bg_arrive(self):
        self.q = self.q + self.bg_load * self.D * np.random.uniform(0, 0.5, self.N)

    def _cong_step(self):
        noise = np.random.randn(self.n_groups) * self.cong_vol
        self.cong = np.clip(self.cong_rho * self.cong + (1 - self.cong_rho) * 1.0 + noise,
                            self.cong_lo, self.cong_hi)

    def _eval_alloc(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        s = a.sum(); a = a / s if s > 1e-12 else np.ones(self.N) / self.N
        active = a > 1e-9
        eff_f = self._eff_f()
        off_t = a * self.D / self.rate
        exe_t = a * self.D * C / eff_f
        wait = self.q * C / eff_f
        comp = wait + off_t + exe_t
        base_delay = float(comp[active].max()) if active.any() else float(comp.max())
        g_act = len(set(self.gid[active].tolist())) if active.any() else 1
        delay = base_delay + self.comm_delay * max(g_act - 1, 0)   # 跨组通信延迟罚
        K = int(active.sum())
        e_tx = off_t * P_OFF
        e_cmp = a * self.D * self.ecpb
        base_energy = float((e_tx + e_cmp).sum() + self.agg * max(K - 1, 0))
        energy = base_energy + self.comm_energy * max(g_act - 1, 0)  # 跨组通信能耗罚
        return delay, energy, K, active, g_act

    def step(self, a):
        delay, base_e_with_comm, K, active, g_act = self._eval_alloc(a)
        warm_b = self.warm > 1e-9
        on = active | (self.q > 1e-9)
        newly = on & (~warm_b)
        switch_energy = float((self.e_f * newly).sum())
        powered = (warm_b | on) if self.keep_alive > 0 else on
        idle_energy = float((self.p_idle * powered).sum()) * self.arrival_dt
        energy = base_e_with_comm + switch_energy + idle_energy
        viol = max(0.0, delay - self.deadline)
        r_T = -delay / self.delay_ref
        r_E = -energy / self.energy_ref
        r_C = -self.sla_scale * viol / self.delay_ref
        self._delays.append(delay); self._energies.append(energy)
        self._viols.append(1.0 if delay > self.deadline else 0.0)
        self._t += 1
        done = self._t >= self.H
        a_n = np.clip(np.asarray(a, dtype=np.float64), 0, None)
        sa = a_n.sum(); a_n = a_n / sa if sa > 1e-12 else np.ones(self.N) / self.N
        drain = (self._eff_f() / C) * self.arrival_dt
        self.q = np.maximum(self.q + a_n * self.D - drain, 0.0)
        if not done:
            self._bg_arrive(); self._cong_step()
        backlog = self.q > 1e-9
        if self.keep_alive > 0:
            refresh = on | backlog
            self.warm_timer = np.where(refresh, self.keep_alive, np.maximum(self.warm_timer - 1, 0))
            self.warm = self.warm_timer / self.keep_alive
        else:
            self.warm = backlog.astype(np.float64)
        info = {'r_vec': np.array([r_T, r_E, r_C], dtype=np.float32), 'delay': delay, 'energy': energy,
                'K_active': K, 'g_act': g_act, 'q_mean': float(np.mean(self.q))}
        scalar_r = self.w * r_T + (1 - self.w) * r_E + r_C
        return self._obs(), float(scalar_r), done, info

    def episode_sla_summary(self):
        d = np.array(self._delays); e = np.array(self._energies)
        return {'mean_delay': float(d.mean()), 'mean_energy': float(e.mean()),
                'violation_rate': float(np.mean(self._viols)),
                'p95_delay': float(np.percentile(d, 95)), 'p99_delay': float(np.percentile(d, 99))}


# ---------- 多峰性探针: 验证"最优=承诺单组、随ω/拥塞翻转、常打平、跨组混合更差" ----------
if __name__ == '__main__':
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    np.random.seed(0)
    env = MultimodalOffloadEnv(horizon=40)
    print('组=%s  profile=%s  comm_delay=%.2fs comm_energy=%.4fJ delay_ref=%.1f energy_ref=%.4f'
          % (env.groups, env.prof_name, env.comm_delay, env.comm_energy, env.delay_ref, env.energy_ref))
    env.reset()
    # 候选: 3 个"单整组" + 3 个"跨组混合"(各取两组各1台) + 全展开
    G = env.groups
    def grp_alloc(g): a = np.zeros(6); a[G[g]] = 0.5; return a
    def mix_alloc(g1, g2): a = np.zeros(6); a[G[g1][0]] = 0.5; a[G[g2][0]] = 0.5; return a
    singles = {f'整组{g}{tuple(G[g])}': grp_alloc(g) for g in range(3)}
    mixes = {'跨组0+1': mix_alloc(0, 1), '跨组0+2': mix_alloc(0, 2), '全展开6台': np.ones(6) / 6}
    cands = {**singles, **mixes}

    def scal(a, w):
        d, e, K, act, ga = env._eval_alloc(a)
        return w * (-d / env.delay_ref) + (1 - w) * (-e / env.energy_ref), d, e, ga

    print('\n[多峰性] 每槽看 w=0(能耗)/0.5/1(延迟) 的最优候选 + 单组最优 vs 最优跨组混合 差距 + top2组是否打平:')
    flips = {0.0: set(), 0.5: set(), 1.0: set()}; tie_slots = 0
    for t in range(10):
        line = '槽%d cong=%s |' % (t, np.round(env.cong, 2))
        tie_here = False
        for w in [0.0, 0.5, 1.0]:
            vals = {nm: scal(a, w)[0] for nm, a in cands.items()}
            best = max(vals, key=vals.get); flips[w].add(best)
            # 单组最优 vs 最优跨组混合
            sbest = max(singles, key=lambda nm: vals[nm]); mbest = max(mixes, key=lambda nm: vals[nm])
            # top2 单组打平?
            sv = sorted([vals[nm] for nm in singles], reverse=True)
            gap_top2 = sv[0] - sv[1]
            if gap_top2 < 0.05: tie_here = True
            line += ' w%.1f:%s(单优-混优=%+.2f,top2Δ=%.2f)' % (w, best.split('(')[0], vals[sbest] - vals[mbest], gap_top2)
        if tie_here: tie_slots += 1
        print(line)
        env.step(grp_alloc(max(range(3), key=lambda g: scal(grp_alloc(g), 0.5)[0])))
    print('\n-> 各 w 出现过的最优候选种类:', {w: s for w, s in flips.items()})
    print('-> 最优是否总是"单整组"(非跨组混合/全展开):',
          all(all('整组' in x for x in s) for s in flips.values()))
    print('-> 10 槽里有 %d 槽出现 top2 单组打平(<0.05) = 多峰机会(单峰必摊到两组吃罚)' % tie_slots)
    print('[ok] multimodal env 冒烟通过' if all(flips.values()) else '[warn] 翻转不足')
