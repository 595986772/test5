"""
MOFD Model
==========
在 FDEdge 的反馈扩散 Actor-Critic 基础上，扩展为：
  * Preference-Conditioned Feedback-Diffusion Actor (PC-FDN)：
      - 接收偏好 ω 作为状态一部分；
      - 逐 ES 共享编码器聚合各服务器特征；
      - 起始噪声用 "历史动作概率" 代替高斯噪声（修正并实现 FDEdge 原论文描述的反馈机制）。
  * Masked Softmax 处理 E ∈ [1, Emax] 的可变动作空间。
  * 标量化奖励 r_ω = ω_T α_T r_T + ω_E α_E r_E 由训练管线在外部装填。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

from helpers import SinusoidalPosEmb
from feedback_diffusion import Diffusion


# --------------------------------------------------------------
# 策略网络：偏好条件化 + 逐服务器特征编码
# --------------------------------------------------------------
class PCPolicyNet(nn.Module):
    """
    扩散模型内部的 ε-预测网络 (参照 FDEdge 的 MLP_PolicyNet，但做三点改动):
      1) 把 state 拆成 (task_pref, per_server_features)，逐 ES 过共享编码器；
      2) 偏好向量 ω 与任务特征拼接后送入上下文编码器；
      3) 聚合后与当前噪声 x_t、时间步嵌入一起送入融合 MLP，输出维度 = Emax。
    """

    def __init__(self, Emax=6, task_pref_dim=4, per_server_dim=3,
                 hidden_dim=128, t_dim=16):
        super().__init__()
        self.Emax = Emax
        self.task_pref_dim = task_pref_dim
        self.per_server_dim = per_server_dim
        self.t_dim = t_dim

        self.time_emb = SinusoidalPosEmb(t_dim)

        srv_hid = hidden_dim // 2
        self.server_enc = nn.Sequential(
            nn.Linear(per_server_dim, srv_hid),
            nn.ReLU(),
            nn.Linear(srv_hid, srv_hid),
            nn.ReLU(),
        )
        ctx_hid = hidden_dim // 2
        self.ctx_enc = nn.Sequential(
            nn.Linear(task_pref_dim, ctx_hid),
            nn.ReLU(),
        )

        # prior_latent 不再作为网络 conditioning, 仅在 inference-time H-MCSS
        # 中作为候选起点之一; 这里 fusion_in 砍掉一个 Emax 维.
        fusion_in = srv_hid * Emax + ctx_hid + t_dim + Emax  # + x
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, Emax),
        )

    def split_state(self, state):
        B = state.size(0)
        task_pref = state[:, :self.task_pref_dim]
        per_srv = state[:, self.task_pref_dim:].reshape(B, self.Emax, self.per_server_dim)
        return task_pref, per_srv

    def forward(self, x, time_step, state, prior_latent=None):
        """
        x:            [B, Emax]  当前去噪步的反馈 latent
        prior_latent: 接受但不使用 (向后兼容; 该向量改由 H-MCSS 当作候选起点之一,
                      不再作为网络内部 conditioning).
        """
        B = x.size(0)
        t_emb = self.time_emb(time_step)  # [B, t_dim]
        task_pref, per_srv = self.split_state(state)
        srv_emb = self.server_enc(per_srv.reshape(B * self.Emax, self.per_server_dim))
        srv_emb = srv_emb.reshape(B, -1)  # [B, Emax*srv_hid]
        ctx_emb = self.ctx_enc(task_pref)
        fused = torch.cat([srv_emb, ctx_emb, t_emb, x], dim=1)
        out = self.fusion(fused)
        # 与 FDEdge 原实现保持一致: ε-网络输出 softmax 化
        return F.softmax(out, dim=1)


# --------------------------------------------------------------
# 反馈式扩散：起点从"历史动作概率"开始（修正 FDEdge 原代码的细节）
# --------------------------------------------------------------
class FeedbackDiffusion(Diffusion):
    """
    p_sample_loop 的起点从 latent_action_probs（历史动作概率）出发，
    而不是 torch.randn_like。这与 FDEdge 论文 Sec IV-A.5 的描述一致。
    P2 双通道: 增加 prior 透传, 由 PCPolicyNet 内部 concat 使用.
    """

    def p_sample_loop(self, state, latent_action_probs, prior=None):
        x = latent_action_probs.clone()
        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, prior=prior)
        return x


# --------------------------------------------------------------
# 评论家网络
# --------------------------------------------------------------
class QValueNet(nn.Module):
    """输入系统状态，输出所有 Emax 个动作的 Q 值"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, s):
        return self.net(s)


# --------------------------------------------------------------
# 经验回放池
# --------------------------------------------------------------
class ReplayBuffer:
    """
    存储 (s, a, x_lat, prior, mask, r_scal, s', x_lat', mask')
    其中 r_scal 是按当前偏好 ω 标量化后的奖励，ω 已编码在 state 中。
    P2 双通道: prior 是 episode-level 动作先验, [Emax]; 同 episode 内不变,
    所以 next_prior 复用同一个 prior, 不单独存.
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, x_lat, prior, mask, r, s_next, x_lat_next, mask_next):
        self.buffer.append((s, a, x_lat, prior, mask, r, s_next, x_lat_next, mask_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, x, p, m, r, sn, xn, mn = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(x, dtype=np.float32), np.array(p, dtype=np.float32),
                np.array(m, dtype=np.float32),
                np.array(r, dtype=np.float32), np.array(sn, dtype=np.float32),
                np.array(xn, dtype=np.float32), np.array(mn, dtype=np.float32))

    def size(self):
        return len(self.buffer)


# --------------------------------------------------------------
# MOFD-SAC 主体
# --------------------------------------------------------------
class MOFD_SAC:
    """
    多目标偏好条件化反馈扩散 SAC
    """

    def __init__(self, state_dim, Emax, hidden_dim=128,
                 actor_lr=1e-4, critic_lr=1e-3,
                 alpha=0.05, alpha_lr=3e-4,
                 target_entropy=-1.0, tau=0.005, gamma=0.95,
                 denoising_steps=5, device=torch.device('cpu')):
        self.Emax = Emax
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        backbone = PCPolicyNet(Emax=Emax, task_pref_dim=4,
                               per_server_dim=(state_dim - 4) // Emax,
                               hidden_dim=hidden_dim)
        self.actor = FeedbackDiffusion(
            state_dim=state_dim, action_dim=Emax, model=backbone,
            beta_schedule='vp', denoising_steps=denoising_steps,
        ).to(device)

        self.critic1 = QValueNet(state_dim, hidden_dim, Emax).to(device)
        self.critic2 = QValueNet(state_dim, hidden_dim, Emax).to(device)
        self.target1 = QValueNet(state_dim, hidden_dim, Emax).to(device)
        self.target2 = QValueNet(state_dim, hidden_dim, Emax).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                                       device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # ----- 带 mask 的 Actor 输出 -----
    def _masked_actor(self, state, latent, prior, mask):
        probs = self.actor(state, latent, prior=prior)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    # ----- 交互：根据当前状态采样动作 -----
    def take_action(self, state_np, latent_np, prior_np, mask_np, stochastic=True):
        state = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
        latent = torch.tensor(latent_np[None], dtype=torch.float, device=self.device)
        prior = torch.tensor(prior_np[None], dtype=torch.float, device=self.device)
        mask = torch.tensor(mask_np[None], dtype=torch.float, device=self.device)
        with torch.no_grad():
            probs = self._masked_actor(state, latent, prior, mask)
        probs_np = probs.detach().cpu().numpy()[0]
        # 数值兜底
        probs_np = np.clip(probs_np, 0.0, 1.0)
        s = probs_np.sum()
        if s < 1e-8:
            # 全部被 mask 掉 (理论上不会)：均匀回退
            valid_idx = np.where(mask_np > 0.5)[0]
            action = int(np.random.choice(valid_idx)) if len(valid_idx) else 0
            probs_np = np.zeros_like(probs_np)
            probs_np[action] = 1.0
            return action, probs_np
        probs_np = probs_np / s
        if stochastic:
            action = int(np.random.choice(self.Emax, p=probs_np))
        else:
            action = int(np.argmax(probs_np))
        return action, probs_np

    # ----- 训练一步 -----
    def update(self, batch):
        s, a, x, p, m, r, sn, xn, mn = batch
        s_t = torch.tensor(s, dtype=torch.float, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        x_t = torch.tensor(x, dtype=torch.float, device=self.device)
        p_t = torch.tensor(p, dtype=torch.float, device=self.device)
        m_t = torch.tensor(m, dtype=torch.float, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float, device=self.device).view(-1, 1)
        sn_t = torch.tensor(sn, dtype=torch.float, device=self.device)
        xn_t = torch.tensor(xn, dtype=torch.float, device=self.device)
        mn_t = torch.tensor(mn, dtype=torch.float, device=self.device)

        # ----- 目标 Q -----
        with torch.no_grad():
            # 同 episode 内 prior 不变, 用同一个 p_t
            next_probs = self._masked_actor(sn_t, xn_t, p_t, mn_t)
            log_next = torch.log(next_probs + 1e-8)
            entropy_next = -torch.sum(next_probs * log_next, dim=1, keepdim=True)
            tq1 = self.target1(sn_t)
            tq2 = self.target2(sn_t)
            min_tq = torch.min(tq1, tq2)
            expected_q = torch.sum(next_probs * min_tq, dim=1, keepdim=True)
            target_q = r_t + self.gamma * (expected_q + self.log_alpha.exp() * entropy_next)

        # ----- Critic 更新 -----
        q1 = self.critic1(s_t).gather(1, a_t)
        q2 = self.critic2(s_t).gather(1, a_t)
        c1_loss = F.mse_loss(q1, target_q)
        c2_loss = F.mse_loss(q2, target_q)
        self.c1_opt.zero_grad(); c1_loss.backward(); self.c1_opt.step()
        self.c2_opt.zero_grad(); c2_loss.backward(); self.c2_opt.step()

        # ----- Actor 更新 -----
        probs = self._masked_actor(s_t, x_t, p_t, m_t)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_e = self.critic1(s_t)
        q2_e = self.critic2(s_t)
        min_q = torch.min(q1_e, q2_e)
        expected_q_eval = torch.sum(probs * min_q, dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp().detach() * entropy - expected_q_eval)
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ----- Alpha 更新 -----
        alpha_loss = torch.mean((entropy.detach() - self.target_entropy) * self.log_alpha.exp())
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # ----- 目标网络软更新 -----
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return dict(
            c_loss=float((c1_loss + c2_loss).detach() * 0.5),
            a_loss=float(actor_loss.detach()),
            alpha=float(self.log_alpha.exp().detach()),
            H=float(entropy.mean().detach()),
        )

    def _soft_update(self, net, target_net):
        for p_t, p in zip(target_net.parameters(), net.parameters()):
            p_t.data.copy_(p_t.data * (1.0 - self.tau) + p.data * self.tau)
