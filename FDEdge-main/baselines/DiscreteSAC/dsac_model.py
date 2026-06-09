"""
Discrete-SAC (Multi-Objective)
==============================
标准 Discrete-SAC 基线: MLP Actor + 双 MLP Critic + 自动调节 alpha.
状态已包含偏好 omega (由 MOFDEnvironment.get_state 提供), 奖励为线性标量化.
相对 MOFD 完整版: 去掉 Feedback Diffusion、去掉 Set-Transformer、去掉 FC-MCSS.
"""
import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, m, r, s_next, m_next):
        self.buffer.append((s, a, m, r, s_next, m_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, m, r, sn, mn = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(m, dtype=np.float32), np.array(r, dtype=np.float32),
                np.array(sn, dtype=np.float32), np.array(mn, dtype=np.float32))

    def size(self):
        return len(self.buffer)


class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s, mask):
        logits = self.net(s)
        neg_inf = torch.finfo(logits.dtype).min
        mask_bool = mask > 0.5
        logits = logits.masked_fill(~mask_bool, neg_inf)
        return F.softmax(logits, dim=-1)


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s):
        return self.net(s)


class DiscreteSAC:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 actor_lr=1e-4, critic_lr=1e-3,
                 alpha=0.05, alpha_lr=3e-4,
                 target_entropy=-1.0, tau=0.005, gamma=0.95,
                 device=torch.device('cpu')):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        self.actor = ActorNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic1 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target1 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target2 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                                      device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def take_action(self, state_np, mask_np, stochastic=True):
        s = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
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
        s_t = torch.tensor(s, device=self.device)
        a_t = torch.tensor(a, device=self.device).view(-1, 1)
        m_t = torch.tensor(m, device=self.device)
        r_t = torch.tensor(r, device=self.device).view(-1, 1)
        sn_t = torch.tensor(sn, device=self.device)
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
