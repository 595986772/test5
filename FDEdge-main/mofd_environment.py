"""
MOFD Environment
================
多目标 + 上下文泛化的协作式边缘计算卸载环境。

相比原 FDEdge 环境的扩展：
  (1) 同时返回 延迟奖励 r_T 和 能耗奖励 r_E，通过偏好向量 ω 标量化
  (2) 每个 episode 随机采样上下文 c = (E, f_E, ω)：ES 数量、CPU 频率、偏好都变化
  (3) 无效 ES 槽位通过 mask 屏蔽，支持训练时在 [1, Emax] 区间内泛化
  (4) 状态向量包含偏好 ω 本身与逐 ES 特征，为策略网络提供上下文信号
  (5) 信道模型加入 Rayleigh 小尺度衰落: 每 (slot, 任务, ES) 独立采样信道增益 g
      (|h|^2 ~ Exp(1), 截断到 [fading_lo, fading_hi]), 有效速率 = tran_rate * g,
      策略可通过状态向量中的归一化瞬时增益感知信道质量 (改进点 #1)
"""
import numpy as np


class MOFDEnvironment:
    """Multi-Objective Feedback-Diffusion MEC Offloading Environment"""

    def __init__(self,
                 Emax=6,
                 num_tasks_max=10,
                 bit_range=(10, 40),
                 time_slots=20,
                 f_range=(10, 40),
                 tran_rate_range=(400, 500),
                 comp_density_range=(0.1024, 0.3072),
                 p_off=0.5,
                 kappa=1e-3,
                 channel_fading=True,
                 fading_clip=(0.3, 2.5),
                 delay_scale=0.05,
                 energy_scale=0.25,
                 seed=0):
        # 上下文/动作上限
        self.Emax = Emax
        self.action_dim = Emax
        self.n_tasks_max = num_tasks_max
        self.time_slots = time_slots
        self.min_bit, self.max_bit = bit_range
        self.f_min, self.f_max = f_range
        self.v_min, self.v_max = tran_rate_range
        self.cd_min, self.cd_max = comp_density_range
        self.duration = 1.0  # 每个时间槽时长 (s)
        self.p_off = p_off   # 卸载发射功率 (W)，用于传输能耗计算
        self.kappa = kappa   # CPU 有效电容系数，用于执行能耗计算
        # 通道归一化常数: 把 r_T (=-delay) 和 r_E (=-energy) 拉到相近量级,
        # 防止 reward 的延迟通道压倒能耗通道导致 actor 偏向延迟. 量级估计:
        # delay ≈ 20s × 0.05 ≈ 1.0;  energy ≈ 4J × 0.25 ≈ 1.0.
        # delay / energy 返回原值, 仅 r_T / r_E 被缩放, 不影响 Pareto / HV 计算.
        self.delay_scale = float(delay_scale)
        self.energy_scale = float(energy_scale)

        # 每个任务索引固定一个计算密度 (Gigacycles/Mbit)
        rng = np.random.default_rng(seed)
        self.comp_density = rng.uniform(self.cd_min, self.cd_max, size=[num_tasks_max])

        # 信道衰落: 每 episode 重置时预采 (T, n_max, Emax) 维 Rayleigh gain 网格,
        # 确保对同一 (seed, 重置顺序), 所有方法观测到的信道实现完全一致
        self.channel_fading = bool(channel_fading)
        self.fading_lo, self.fading_hi = fading_clip
        self._channel_rng = np.random.default_rng(seed + 777)
        self.channel_gain = np.ones([time_slots, num_tasks_max, Emax], dtype=np.float32)

        # 状态布局: [ d_n, rho*d_n, ω_T, ω_E, (f_e, q_len_e, valid_e, g_e) for e in Emax ]
        self.per_server_dim = 4
        self.state_dim = 4 + self.Emax * self.per_server_dim

        # 动态、逐 episode 重置的状态
        self.E = Emax
        self.f_E = np.zeros(Emax, dtype=np.float32)
        self.tran_rate = np.zeros(Emax, dtype=np.float32)
        self.omega = np.array([0.5, 0.5], dtype=np.float32)
        self.valid_mask = np.ones(Emax, dtype=np.float32)
        self.tasks_bit = []
        self.proc_queue_len = np.zeros([time_slots + 1, Emax], dtype=np.float32)
        self.proc_queue_bef = np.zeros([time_slots + 1, Emax], dtype=np.float32)

    # --------------------------------------------------------------
    # 上下文采样 (domain randomization)
    # --------------------------------------------------------------
    def sample_context(self, rng=None, fixed_omega=None):
        """
        随机采样一个上下文 c = (E, f_E, tran_rate, ω)。
        若 fixed_omega 非 None，使用给定偏好（用于评估 Pareto 前沿）。
        """
        rng = rng if rng is not None else np.random.default_rng()
        E = int(rng.integers(max(1, self.Emax // 2), self.Emax + 1))  # [Emax/2, Emax]
        f_E = np.zeros(self.Emax, dtype=np.float32)
        f_E[:E] = rng.uniform(self.f_min, self.f_max, size=E).astype(np.float32)
        tran_rate = np.zeros(self.Emax, dtype=np.float32)
        tran_rate[:E] = rng.uniform(self.v_min, self.v_max, size=E).astype(np.float32)
        if fixed_omega is None:
            w_T = float(rng.uniform(0.0, 1.0))
            omega = np.array([w_T, 1.0 - w_T], dtype=np.float32)
        else:
            omega = np.asarray(fixed_omega, dtype=np.float32)
        return E, f_E, tran_rate, omega

    # --------------------------------------------------------------
    # Episode 初始化
    # --------------------------------------------------------------
    def reset_env(self, tasks_bit, E, f_E, tran_rate, omega):
        self.tasks_bit = tasks_bit
        self.E = int(E)
        self.f_E = np.asarray(f_E, dtype=np.float32).copy()
        self.tran_rate = np.asarray(tran_rate, dtype=np.float32).copy()
        self.omega = np.asarray(omega, dtype=np.float32).copy()
        self.valid_mask = np.zeros(self.Emax, dtype=np.float32)
        self.valid_mask[:self.E] = 1.0
        self.proc_queue_len = np.zeros([self.time_slots + 1, self.Emax], dtype=np.float32)
        self.proc_queue_bef = np.zeros([self.time_slots + 1, self.Emax], dtype=np.float32)

        # Rayleigh 小尺度衰落: |h|^2 ~ Exp(mean=1), 截断避免极端速率
        if self.channel_fading:
            g = self._channel_rng.exponential(
                1.0, size=(self.time_slots, self.n_tasks_max, self.Emax)
            )
            self.channel_gain = np.clip(g, self.fading_lo, self.fading_hi).astype(np.float32)
        else:
            self.channel_gain = np.ones(
                [self.time_slots, self.n_tasks_max, self.Emax], dtype=np.float32,
            )

    def effective_rate(self, t, n, e):
        """瞬时有效传输速率 = base_tran_rate * Rayleigh_gain(t, n, e)."""
        if t < 0 or t >= self.time_slots or n < 0 or n >= self.n_tasks_max or e < 0 or e >= self.Emax:
            return float(self.tran_rate[e]) if 0 <= e < self.Emax else 0.0
        return float(self.tran_rate[e] * self.channel_gain[t, n, e])

    # --------------------------------------------------------------
    # 状态读取
    # --------------------------------------------------------------
    def get_state(self, t, n):
        if t >= self.time_slots or n >= len(self.tasks_bit[t]):
            # 边界保护：越界时用零填充
            return np.zeros(self.state_dim, dtype=np.float32)

        d_n = float(self.tasks_bit[t][n])
        rho_d = float(self.comp_density[n] * d_n)

        # 逐服务器特征，无效槽位置零
        per_srv = np.zeros([self.Emax, self.per_server_dim], dtype=np.float32)
        f_norm = max(self.f_max, 1.0)
        q_norm = max(self.f_max * self.duration * 10.0, 1.0)
        g_norm = max(self.fading_hi, 1e-6)
        for e in range(self.Emax):
            if self.valid_mask[e] > 0.5:
                per_srv[e, 0] = self.f_E[e] / f_norm
                per_srv[e, 1] = self.proc_queue_len[t, e] / q_norm
                per_srv[e, 2] = 1.0
                per_srv[e, 3] = float(self.channel_gain[t, n, e]) / g_norm
            # else: leave all zeros, valid flag = 0

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
    # 执行一步卸载：返回双目标奖励、延迟、能耗
    # --------------------------------------------------------------
    def step(self, t, n, action):
        # 容错：如果 agent 选了无效 ES，随机回退到一个有效 ES
        if action >= self.E or action < 0:
            action = int(np.argmax(self.valid_mask))

        f_b = self.f_E[action]
        v = self.effective_rate(t, n, action)
        d_n = float(self.tasks_bit[t][n])
        rho_d = float(self.comp_density[n] * d_n)

        # ----- 延迟 -----
        tran_delay = d_n / max(v, 1e-6)
        comp_delay = rho_d / f_b
        wait_delay = (self.proc_queue_len[t, action] + self.proc_queue_bef[t, action]) / f_b
        delay = tran_delay + comp_delay + wait_delay

        # ----- 能耗 -----
        # 传输能耗: E_off = p_off * T_off
        e_off = self.p_off * tran_delay
        # 执行能耗: E_exe = κ * f^2 * 工作量；使 f_b (GHz) 的平方与 rho_d (Gc) 乘积与 delay 量级相近
        e_exe = self.kappa * (f_b ** 2) * rho_d
        energy = e_off + e_exe

        # ----- 更新该 ES 本时间槽累积负载 -----
        self.proc_queue_bef[t, action] += rho_d

        r_T = -delay  * self.delay_scale
        r_E = -energy * self.energy_scale
        return r_T, r_E, delay, energy, action

    def update_proc_queues(self, t):
        """时间槽 t 结束时更新队列到 t+1"""
        for e in range(self.Emax):
            if self.valid_mask[e] > 0.5:
                self.proc_queue_len[t + 1, e] = max(
                    self.proc_queue_len[t, e] + self.proc_queue_bef[t, e]
                    - self.f_E[e] * self.duration,
                    0.0,
                )
