"""
GenMOSAC (Generalizable Multi-Objective Discrete-SAC)
=====================================================
对齐 Generalizable-Pareto-Optimal-Offloading (TSC'25) 论文的核心**状态表征**:
  * dict 状态:
      - task_pref:  (4,)   — 任务大小 + 计算量 + 偏好 ω_T/ω_E
      - servers:    (Emax, 4) — 逐 ES 特征 (f, q_len, valid, channel_gain)
      - hist:       (n_bins,) — valid ES 队列负载直方图, 对 ES 排列&数量鲁棒
      - mask:       (Emax,)  — 合法动作掩码
  * Set-Transformer-lite encoder:
      - 逐 ES MLP → mask max-pool, 对 ES 排列不变
      - 拼接任务/偏好 MLP 输出 + 负载直方图 MLP 输出
  * 其余 Discrete-SAC 训练机制 (双 Q + 自动 α) 与 baselines/DiscreteSAC 相同,
    确保两个基线的差异**仅在状态编码器**, 便于消融.
"""
import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DictReplayBuffer:
    """dict 状态 + mask/action/reward 的经验回放."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, m, r, s_next, m_next):
        self.buffer.append((s, a, m, r, s_next, m_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, m, r, sn, mn = zip(*batch)

        def stack_dict(ds):
            return {k: np.stack([d[k] for d in ds], axis=0).astype(np.float32)
                    for k in ds[0].keys()}

        return (stack_dict(s),
                np.array(a, dtype=np.int64),
                np.array(m, dtype=np.float32),
                np.array(r, dtype=np.float32),
                stack_dict(sn),
                np.array(mn, dtype=np.float32))

    def size(self):
        return len(self.buffer)


class DictEncoder(nn.Module):
    """任务/偏好 MLP + 逐 ES MLP(max-pool) + 负载直方图 MLP 三路拼接."""

    def __init__(self, task_pref_dim, per_srv_dim, n_bins, hidden, emb):
        super().__init__()
        self.srv_mlp = nn.Sequential(
            nn.Linear(per_srv_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb),
        )
        self.hist_mlp = nn.Sequential(
            nn.Linear(n_bins, emb), nn.ReLU(),
        )
        self.tp_mlp = nn.Sequential(
            nn.Linear(task_pref_dim, emb), nn.ReLU(),
        )
        self.out_dim = emb * 3

    def forward(self, state):
        """state: dict[str, Tensor], 所有 Tensor 首维为 batch."""
        task_pref = state['task_pref']        # (B, task_pref_dim)
        servers = state['servers']            # (B, Emax, per_srv_dim)
        mask = state['mask']                  # (B, Emax)
        hist = state['hist']                  # (B, n_bins)

        h_srv = self.srv_mlp(servers)         # (B, Emax, emb)
        neg_inf = torch.finfo(h_srv.dtype).min
        m3 = mask.unsqueeze(-1)
        h_srv_masked = torch.where(m3 > 0.5, h_srv, torch.full_like(h_srv, neg_inf))
        h_srv_pool = h_srv_masked.max(dim=1).values  # (B, emb)

        h_hist = self.hist_mlp(hist)
        h_tp = self.tp_mlp(task_pref)
        return torch.cat([h_tp, h_srv_pool, h_hist], dim=-1)


class GenActor(nn.Module):
    def __init__(self, enc, hidden, action_dim):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(
            nn.Linear(enc.out_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state_dict, mask_for_logits):
        h = self.enc(state_dict)
        logits = self.head(h)
        neg_inf = torch.finfo(logits.dtype).min
        mb = mask_for_logits > 0.5
        logits = logits.masked_fill(~mb, neg_inf)
        return F.softmax(logits, dim=-1)


class GenCritic(nn.Module):
    def __init__(self, enc, hidden, action_dim):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(
            nn.Linear(enc.out_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state_dict):
        h = self.enc(state_dict)
        return self.head(h)


class GenMOSAC:
    """Discrete-SAC with dict-based generalizable state encoder."""

    def __init__(self, task_pref_dim, per_srv_dim, n_bins, Emax,
                 hidden=128, emb=64,
                 actor_lr=1e-4, critic_lr=1e-3,
                 alpha=0.05, alpha_lr=3e-4,
                 target_entropy=-1.0, tau=0.005, gamma=0.95,
                 device=torch.device('cpu')):
        self.device = device
        self.action_dim = Emax
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        def mk_enc():
            return DictEncoder(task_pref_dim, per_srv_dim, n_bins, hidden, emb)

        self.actor = GenActor(mk_enc(), hidden, Emax).to(device)
        self.critic1 = GenCritic(mk_enc(), hidden, Emax).to(device)
        self.critic2 = GenCritic(mk_enc(), hidden, Emax).to(device)
        self.target1 = GenCritic(mk_enc(), hidden, Emax).to(device)
        self.target2 = GenCritic(mk_enc(), hidden, Emax).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                                      device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def _dict_to_torch(self, s, batch=False):
        out = {}
        for k, v in s.items():
            arr = v if batch else v[None]
            out[k] = torch.tensor(np.asarray(arr, dtype=np.float32),
                                  dtype=torch.float, device=self.device)
        return out

    def take_action(self, state_dict, mask_np, stochastic=True):
        s = self._dict_to_torch(state_dict, batch=False)
        m = torch.tensor(mask_np[None], dtype=torch.float, device=self.device)
        with torch.no_grad():
            probs = self.actor(s, m).cpu().numpy()[0]
        probs = np.clip(probs, 0.0, 1.0)
        if probs.sum() < 1e-8:
            valid = np.where(mask_np > 0.5)[0]
            return int(np.random.choice(valid)) if len(valid) else 0
        probs = probs / probs.sum()
        if stochastic:
            return int(np.random.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))

    def _masked_actor(self, s, m):
        probs = self.actor(s, m)
        s_sum = probs.sum(dim=1, keepdim=True)
        return probs / (s_sum + 1e-8)

    def update(self, batch):
        s, a, m, r, sn, mn = batch
        s_t = self._dict_to_torch(s, batch=True)
        sn_t = self._dict_to_torch(sn, batch=True)
        a_t = torch.tensor(a, device=self.device).view(-1, 1)
        m_t = torch.tensor(m, device=self.device)
        r_t = torch.tensor(r, device=self.device).view(-1, 1)
        mn_t = torch.tensor(mn, device=self.device)

        with torch.no_grad():
            next_probs = self._masked_actor(sn_t, mn_t)
            log_next = torch.log(next_probs + 1e-8)
            ent_next = -torch.sum(next_probs * log_next, dim=1, keepdim=True)
            tq1 = self.target1(sn_t); tq2 = self.target2(sn_t)
            min_tq = torch.min(tq1, tq2)
            v_next = torch.sum(next_probs * min_tq, dim=1, keepdim=True)
            target_q = r_t + self.gamma * (v_next + self.log_alpha.exp() * ent_next)

        q1 = self.critic1(s_t).gather(1, a_t)
        q2 = self.critic2(s_t).gather(1, a_t)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)
        self.c1_opt.zero_grad(); c1_loss.backward(); self.c1_opt.step()
        self.c2_opt.zero_grad(); c2_loss.backward(); self.c2_opt.step()

        probs = self._masked_actor(s_t, m_t)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_e = self.critic1(s_t); q2_e = self.critic2(s_t)
        v_eval = torch.sum(probs * torch.min(q1_e, q2_e), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp().detach() * entropy - v_eval)
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = torch.mean((entropy.detach() - self.target_entropy) * self.log_alpha.exp())
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        self._soft(self.critic1, self.target1)
        self._soft(self.critic2, self.target2)
        return dict(c_loss=float((c1_loss + c2_loss).detach() * 0.5),
                    a_loss=float(actor_loss.detach()),
                    alpha=float(self.log_alpha.exp().detach()),
                    H=float(entropy.mean().detach()))

    def _soft(self, net, target):
        for pt, p in zip(target.parameters(), net.parameters()):
            pt.data.copy_(pt.data * (1.0 - self.tau) + p.data * self.tau)


def build_dict_state(env, t, n, n_bins):
    """根据 MOFDEnvironment 当前状态构造 dict 状态 (含队列负载直方图)."""
    if t >= env.time_slots or n >= len(env.tasks_bit[t]):
        return {
            'task_pref': np.zeros(4, dtype=np.float32),
            'servers': np.zeros([env.Emax, env.per_server_dim], dtype=np.float32),
            'mask': env.valid_mask.copy().astype(np.float32),
            'hist': np.zeros(n_bins, dtype=np.float32),
        }

    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    task_pref = np.array([
        d_n / max(env.max_bit, 1.0),
        rho_d / max(env.cd_max * env.max_bit, 1e-6),
        env.omega[0], env.omega[1],
    ], dtype=np.float32)

    per_srv = np.zeros([env.Emax, env.per_server_dim], dtype=np.float32)
    f_norm = max(env.f_max, 1.0)
    q_norm = max(env.f_max * env.duration * 10.0, 1.0)
    g_norm = max(env.fading_hi, 1e-6)
    for e in range(env.Emax):
        if env.valid_mask[e] > 0.5:
            per_srv[e, 0] = env.f_E[e] / f_norm
            per_srv[e, 1] = env.proc_queue_len[t, e] / q_norm
            per_srv[e, 2] = 1.0
            per_srv[e, 3] = float(env.channel_gain[t, n, e]) / g_norm

    # 队列负载直方图 (只对 valid ES 统计): 对 ES 排列/数量鲁棒的全局特征
    valid_idx = np.where(env.valid_mask > 0.5)[0]
    if len(valid_idx) > 0:
        qs = env.proc_queue_len[t, valid_idx] / q_norm
        qs = np.clip(qs, 0.0, 1.0)
        hist, _ = np.histogram(qs, bins=n_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32) / max(len(valid_idx), 1)
    else:
        hist = np.zeros(n_bins, dtype=np.float32)

    return {
        'task_pref': task_pref,
        'servers': per_srv,
        'mask': env.valid_mask.copy().astype(np.float32),
        'hist': hist,
    }
