"""
MOFD Environment V2 (with Cloud Action)
========================================
对齐 GMORL 论文 (arXiv 2509.10474) baseline 对比 setting 的新版环境.

vs V1 (`mofd_environment.py`) 的本质差别:
  (1) 动作空间加 cloud 选项: action ∈ [0, Emax]
      - action 0     = cloud (高 f, 远距离 → 低传输速率, 同 κ)
      - action 1..E  = edge_1..edge_E (同 V1)
      action_dim = Emax + 1
  (2) cloud 物理特性:
      - cloud_f 范围 [40, 50] GHz (高于 edge 的 [10, 40])
      - cloud 传输速率 [60, 100] Mbps (远低于 edge 的 [400, 500])
      - 同 κ → cloud 执行能耗 ∝ f² 高于 edge
      → Pareto trade-off: cloud 执行快 (低 exec delay + 低 wait delay 因为有独立队列),
        但传输慢 (高 tran delay) 且执行能耗高 (高 κ·f²)
  (3) 其他保持 V1 一致 (Rayleigh 信道, 队列模型, 计算密度, 状态接口形状除 cloud 维度外)

为什么这么设计:
  GMORL 论文里 Pareto 铺得开是因为 cloud 跟 edge 物理差异大 (4 GHz vs 2 GHz, 远距离),
  baseline 在两类服务器间选才有"快但耗能 vs 慢但省能"的真冲突. 仅 edge 选择因 f 范围窄
  + 队列同源, Pareto 几乎平.

接口兼容性:
  保留 V1 的方法签名 (sample_context, reset_env, get_state, get_valid_mask, step,
  update_proc_queues, effective_rate), baseline 代码改动量最小, 主要差异:
  - self.action_dim 从 Emax 变 Emax+1
  - self.state_dim 从 4+E·4 变 4+(E+1)·4 (cloud 占第 0 槽)
  - step(t,n,action) 当 action==0 走 cloud 分支, 否则 edge[action-1]
"""
import numpy as np


class MOFDEnvironmentV2:
    """V2: action 0 = cloud, action 1..Emax = edge_1..edge_Emax."""

    def __init__(self,
                 Emax=6,
                 num_tasks_max=20,
                 bit_range=(10, 40),
                 time_slots=100,
                 # ---- edge 参数 (跟 V1 一致) ----
                 f_range=(10, 40),                 # edge CPU GHz
                 tran_rate_range=(400, 500),       # edge 传输 Mbps
                 # ---- cloud 参数 (新增, 默认值经 smoke 验证产生有意义 Pareto) ----
                 cloud_f_range=(50, 70),           # cloud CPU GHz: 中-高
                 cloud_tran_rate_range=(80, 120),  # cloud 传输 Mbps: 低 (距离远)
                 # ---- 通用 ----
                 comp_density_range=(0.1024, 0.3072),
                 p_off=0.5,
                 kappa=1e-3,            # edge κ
                 cloud_kappa=1e-4,      # cloud κ (远低于 edge: 数据中心硬件更节能 +
                                         # 静态功率分摊到多用户), 不然 f² 爆使 cloud
                                         # 永远被 edge 主导
                 channel_fading=True,
                 fading_clip=(0.3, 2.5),
                 delay_scale=0.05,
                 energy_scale=0.25,
                 seed=0):
        # ---- 上下文/动作上限 ----
        self.Emax = Emax
        self.action_dim = Emax + 1            # +1 for cloud
        self.n_tasks_max = num_tasks_max
        self.time_slots = time_slots
        self.min_bit, self.max_bit = bit_range
        self.f_min, self.f_max = f_range
        self.v_min, self.v_max = tran_rate_range
        self.cf_min, self.cf_max = cloud_f_range
        self.cv_min, self.cv_max = cloud_tran_rate_range
        self.cd_min, self.cd_max = comp_density_range
        self.duration = 1.0
        self.p_off = p_off
        self.kappa = kappa
        self.cloud_kappa = float(cloud_kappa)
        self.delay_scale = float(delay_scale)
        self.energy_scale = float(energy_scale)

        # 任务计算密度 (跟 V1 同)
        rng = np.random.default_rng(seed)
        self.comp_density = rng.uniform(self.cd_min, self.cd_max, size=[num_tasks_max])

        # ---- 信道衰落: 现在维度多 1 (cloud) ----
        self.channel_fading = bool(channel_fading)
        self.fading_lo, self.fading_hi = fading_clip
        self._channel_rng = np.random.default_rng(seed + 777)
        # shape: (T, n_max, action_dim) where slot 0 = cloud, 1..E = edge
        self.channel_gain = np.ones(
            [time_slots, num_tasks_max, self.action_dim], dtype=np.float32)

        # 状态布局:
        #   [ d_n, ρ·d_n, ω_T, ω_E,
        #     (f_cloud, q_cloud, valid_cloud, g_cloud),
        #     (f_e, q_e, valid_e, g_e) for e in Emax ]
        self.per_server_dim = 4
        self.state_dim = 4 + self.action_dim * self.per_server_dim

        # ---- 动态状态 (per episode reset) ----
        # cloud 参数:
        self.cloud_f = 0.0
        self.cloud_tran_rate = 0.0
        # edge 参数 (跟 V1 同):
        self.E = Emax
        self.f_E = np.zeros(Emax, dtype=np.float32)
        self.tran_rate = np.zeros(Emax, dtype=np.float32)
        self.omega = np.array([0.5, 0.5], dtype=np.float32)
        # valid_mask: action_dim 维, 第 0 位 cloud 总是 valid, 后 E 位看 sample_context
        self.valid_mask = np.ones(self.action_dim, dtype=np.float32)
        self.tasks_bit = []
        # 队列: shape (T+1, action_dim). 第 0 列 = cloud 队列
        self.proc_queue_len = np.zeros([time_slots + 1, self.action_dim], dtype=np.float32)
        self.proc_queue_bef = np.zeros([time_slots + 1, self.action_dim], dtype=np.float32)

    # --------------------------------------------------------------
    def sample_context(self, rng=None, fixed_omega=None):
        """采 (E, f_E, tran_rate, ω) + cloud 参数. 接口签名跟 V1 一致,
        cloud 参数缓存在 self.cloud_*, reset_env 之后生效."""
        rng = rng if rng is not None else np.random.default_rng()
        E = int(rng.integers(max(1, self.Emax // 2), self.Emax + 1))
        f_E = np.zeros(self.Emax, dtype=np.float32)
        f_E[:E] = rng.uniform(self.f_min, self.f_max, size=E).astype(np.float32)
        tran_rate = np.zeros(self.Emax, dtype=np.float32)
        tran_rate[:E] = rng.uniform(self.v_min, self.v_max, size=E).astype(np.float32)
        if fixed_omega is None:
            w_T = float(rng.uniform(0.0, 1.0))
            omega = np.array([w_T, 1.0 - w_T], dtype=np.float32)
        else:
            omega = np.asarray(fixed_omega, dtype=np.float32)
        # cloud 参数 (单值, 不是数组)
        self._next_cloud_f = float(rng.uniform(self.cf_min, self.cf_max))
        self._next_cloud_v = float(rng.uniform(self.cv_min, self.cv_max))
        return E, f_E, tran_rate, omega

    def reset_env(self, tasks_bit, E, f_E, tran_rate, omega):
        self.tasks_bit = tasks_bit
        self.E = int(E)
        self.f_E = np.asarray(f_E, dtype=np.float32).copy()
        self.tran_rate = np.asarray(tran_rate, dtype=np.float32).copy()
        self.omega = np.asarray(omega, dtype=np.float32).copy()
        # cloud 用上次 sample_context 缓存的值 (若没调过, 用范围中点)
        self.cloud_f = getattr(self, '_next_cloud_f',
                               0.5 * (self.cf_min + self.cf_max))
        self.cloud_tran_rate = getattr(self, '_next_cloud_v',
                                       0.5 * (self.cv_min + self.cv_max))
        # valid_mask: cloud (位 0) 永远 valid, edge 看 E
        self.valid_mask = np.zeros(self.action_dim, dtype=np.float32)
        self.valid_mask[0] = 1.0
        self.valid_mask[1:self.E + 1] = 1.0
        # 队列重置
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.action_dim],
                                       dtype=np.float32)
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.action_dim],
                                       dtype=np.float32)

        if self.channel_fading:
            g = self._channel_rng.exponential(
                1.0, size=(self.time_slots, self.n_tasks_max, self.action_dim))
            self.channel_gain = np.clip(g, self.fading_lo, self.fading_hi).astype(np.float32)
        else:
            self.channel_gain = np.ones(
                [self.time_slots, self.n_tasks_max, self.action_dim], dtype=np.float32)

    # --------------------------------------------------------------
    def _action_to_f_v(self, action):
        """把 action 翻译成 (f, base_tran_rate, queue_idx). action 0 = cloud."""
        if action == 0:
            return float(self.cloud_f), float(self.cloud_tran_rate), 0
        e = action - 1                          # edge 索引
        return float(self.f_E[e]), float(self.tran_rate[e]), action

    def effective_rate(self, t, n, action):
        """瞬时有效传输速率 = base_rate × Rayleigh_gain(t, n, action)."""
        if (t < 0 or t >= self.time_slots
                or n < 0 or n >= self.n_tasks_max
                or action < 0 or action >= self.action_dim):
            return 0.0
        _, base_v, _ = self._action_to_f_v(action)
        return float(base_v * self.channel_gain[t, n, action])

    # --------------------------------------------------------------
    def get_state(self, t, n):
        if t >= self.time_slots or n >= len(self.tasks_bit[t]):
            return np.zeros(self.state_dim, dtype=np.float32)

        d_n = float(self.tasks_bit[t][n])
        rho_d = float(self.comp_density[n] * d_n)

        per_srv = np.zeros([self.action_dim, self.per_server_dim], dtype=np.float32)
        # 用 cloud_f 上限做 f-norm (因为 cloud_f > edge_f)
        f_norm = max(self.cf_max, self.f_max, 1.0)
        q_norm = max(self.cf_max * self.duration * 10.0, 1.0)
        g_norm = max(self.fading_hi, 1e-6)

        # cloud (action 0)
        if self.valid_mask[0] > 0.5:
            per_srv[0, 0] = self.cloud_f / f_norm
            per_srv[0, 1] = self.proc_queue_len[t, 0] / q_norm
            per_srv[0, 2] = 1.0
            per_srv[0, 3] = float(self.channel_gain[t, n, 0]) / g_norm

        # edges (action 1..E)
        for e in range(self.Emax):
            ai = e + 1
            if self.valid_mask[ai] > 0.5:
                per_srv[ai, 0] = self.f_E[e] / f_norm
                per_srv[ai, 1] = self.proc_queue_len[t, ai] / q_norm
                per_srv[ai, 2] = 1.0
                per_srv[ai, 3] = float(self.channel_gain[t, n, ai]) / g_norm

        task_pref = np.array([
            d_n / max(self.max_bit, 1.0),
            rho_d / max(self.cd_max * self.max_bit, 1e-6),
            self.omega[0],
            self.omega[1],
        ], dtype=np.float32)

        return np.concatenate([task_pref, per_srv.flatten()])

    def get_valid_mask(self):
        return self.valid_mask.copy()

    # --------------------------------------------------------------
    def step(self, t, n, action):
        # 容错: 越界 → 退回到第一个 valid 动作
        if action >= self.action_dim or action < 0 or self.valid_mask[action] < 0.5:
            action = int(np.argmax(self.valid_mask))

        f_b, base_v, queue_idx = self._action_to_f_v(action)
        v = base_v * float(self.channel_gain[t, n, action])
        d_n = float(self.tasks_bit[t][n])
        rho_d = float(self.comp_density[n] * d_n)

        tran_delay = d_n / max(v, 1e-6)
        comp_delay = rho_d / f_b
        wait_delay = ((self.proc_queue_len[t, queue_idx]
                       + self.proc_queue_bef[t, queue_idx]) / f_b)
        delay = tran_delay + comp_delay + wait_delay

        e_off = self.p_off * tran_delay
        # cloud (action 0) 用 cloud_kappa, edge (action 1..E) 用 kappa
        k = self.cloud_kappa if action == 0 else self.kappa
        e_exe = k * (f_b ** 2) * rho_d
        energy = e_off + e_exe

        self.proc_queue_bef[t, queue_idx] += rho_d

        r_T = -delay * self.delay_scale
        r_E = -energy * self.energy_scale
        return r_T, r_E, delay, energy, action

    def update_proc_queues(self, t):
        """时间槽 t 结束时更新队列到 t+1. cloud 用 cloud_f, edge 用 f_E[e]."""
        # cloud (idx 0)
        if self.valid_mask[0] > 0.5:
            self.proc_queue_len[t + 1, 0] = max(
                self.proc_queue_len[t, 0] + self.proc_queue_bef[t, 0]
                - self.cloud_f * self.duration, 0.0)
        # edges (idx 1..E)
        for e in range(self.Emax):
            ai = e + 1
            if self.valid_mask[ai] > 0.5:
                self.proc_queue_len[t + 1, ai] = max(
                    self.proc_queue_len[t, ai] + self.proc_queue_bef[t, ai]
                    - self.f_E[e] * self.duration, 0.0)
