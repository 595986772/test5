"""
MOFD V5 Model
=============
V5 = V3 backbone (MLP + H-MCSS 三源候选) + Vector Q(s,a)∈R² + ω-weighted critic
     loss + COR (Conflict Objective Regularization, COLA NeurIPS'25) + PopArt-lite
     per-objective scale normalization.

设计目的:
  V3 critic 是标量 Q(s,a), 通过 r_scal = ω·α·r_vec 训练, 偏好被混在一锅奖励里;
  V4 用 Envelope-MORL 的 hindsight ω-relabel 改善 ω→Q 映射, 但代价是 buffer 的
  漂移自适应作用被 envelope subsume (relabel 已经穷举 ω, buffer 多余).
  V5 走中间路线: 让 Q 输出向量 (Q_T, Q_E), 用 ω 加权 critic loss 显式分离两目标,
  并通过 COR 在不 relabel 的前提下稳定 ω 间冲突. ω-buffer 的漂移角色完整保留.

与 V3 / V4 的差异:
  V3: scalar Q,  r_scal-trained,                  无 COR, 无 PopArt
  V4: scalar Q,  envelope max + hindsight relabel, 无 COR, 无 PopArt → buffer 被 subsume
  V5: vector Q,  ω-weighted per-obj loss, 无 relabel, 有 COR, 有 PopArt-lite

COR (COLA, NeurIPS 2025):
  ρ(ω_i, ω_j) = cosine(∇_θ L_Q(ω_i), ∇_θ L_Q(ω_j))      (over critic params)
  if ρ < c:  L_total += λ · weight_sample · (Q_θ(s,a) − Q_θ_old(s,a))²
  defaults (COLA paper): c = 0.0, λ = 0.1

PopArt-lite (DeepMind 2016 / ICLR 2025 revival):
  维护 σ_k 的 running RMS (来自 |target_vec[:, k]|);
  per-objective loss 除以 σ_k² 自动平衡 r_T/r_E 量级差异. 不做 Q 输出反归一,
  逻辑简单, 等价于按目标方差自适应 reward scaling.

state layout (依赖 mofd_environment.MOFDEnvironment.get_state):
  state[0]  = task data size (normalized)
  state[1]  = compute density × task size (normalized)
  state[2]  = omega[0]          ← ω_T
  state[3]  = omega[1]          ← ω_E
  state[4:] = per-server features

reward layout (依赖 mofd_environment.MOFDEnvironment.step):
  r_vec = (r_T, r_E) = (-delay, -energy)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

from feedback_diffusion import Diffusion
from mofd_model import PCPolicyNet


OMEGA_STATE_SLICE = slice(2, 4)
N_OBJ = 2  # (T, E)


# ==============================================================
# FeedbackDiffusion (与 V3 一致)
# ==============================================================
class FeedbackDiffusion(Diffusion):
    def p_sample_loop(self, state, latent_action_probs, prior=None):
        x = latent_action_probs.clone()
        for i in reversed(range(self.n_timesteps)):
            timesteps = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, prior=prior)
        return x


# ==============================================================
# Vector Q-network: 输出 [B, Emax, N_OBJ]
# ==============================================================
class QValueNetV5(nn.Module):
    """Vector Q: 每个动作输出 N_OBJ 个 Q 值 (Q_T, Q_E)."""

    def __init__(self, state_dim, hidden_dim, action_dim, n_obj=N_OBJ):
        super().__init__()
        self.action_dim = action_dim
        self.n_obj = n_obj
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_obj),
        )

    def forward(self, s):
        h = self.net(s)  # [B, action_dim * n_obj]
        return h.view(-1, self.action_dim, self.n_obj)


# ==============================================================
# ReplayBuffer (V5: 与 V4 一致, 存 r_vec=(r_T, r_E))
# ==============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, x_lat, prior, mask, r_vec, s_next, x_lat_next, mask_next):
        r_vec_arr = np.asarray(r_vec, dtype=np.float32).reshape(N_OBJ)
        self.buffer.append((s, a, x_lat, prior, mask, r_vec_arr,
                            s_next, x_lat_next, mask_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, x, p, m, r, sn, xn, mn = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(x, dtype=np.float32), np.array(p, dtype=np.float32),
                np.array(m, dtype=np.float32),
                np.array(r, dtype=np.float32),
                np.array(sn, dtype=np.float32),
                np.array(xn, dtype=np.float32),
                np.array(mn, dtype=np.float32))

    def size(self):
        return len(self.buffer)


# ==============================================================
# 工具函数
# ==============================================================
def _flat_grads(params):
    """把 params 的 .grad 拉平成一个向量. None grad 视作零."""
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros_like(p).view(-1))
        else:
            chunks.append(p.grad.detach().view(-1).clone())
    return torch.cat(chunks)


def _relabel_state_omega(state_tensor, new_omega):
    """把 state[:, 2:4] 换成 new_omega.

    new_omega: [2] (broadcast 到 batch) 或 [B, 2].
    仅用于 COR 在 relabeled state 上探测 Q 稳定性, 不影响 buffer / 训练目标.
    """
    out = state_tensor.clone()
    if new_omega.dim() == 1:
        out[..., OMEGA_STATE_SLICE] = new_omega.unsqueeze(0)
    else:
        out[..., OMEGA_STATE_SLICE] = new_omega
    return out


# ==============================================================
# MOFD_SAC_V5
# ==============================================================
class MOFD_SAC_V5:
    def __init__(self, state_dim, Emax, hidden_dim=128,
                 actor_lr=1e-4, critic_lr=1e-3,
                 alpha=0.05, alpha_lr=3e-4,
                 target_entropy=0.5, tau=0.005, gamma=0.95,
                 denoising_steps=5, t_dim=16,
                 n_candidates=3, perturb_std=0.10,
                 # ---- V5 hyper ----
                 alpha_T=1.0, alpha_E=0.25,
                 use_cor=True,
                 cor_lambda=0.1, cor_c=0.0,
                 use_popart=True, popart_beta=0.01,
                 device=torch.device('cpu')):
        self.Emax = Emax
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.alpha_T = float(alpha_T)
        self.alpha_E = float(alpha_E)
        self.use_cor = bool(use_cor)
        self.cor_lambda = float(cor_lambda)
        self.cor_c = float(cor_c)
        self.use_popart = bool(use_popart)
        self.popart_beta = float(popart_beta)
        self._omega_rng = np.random.default_rng(0)

        per_srv_dim = (state_dim - 4) // Emax
        backbone = PCPolicyNet(
            Emax=Emax, task_pref_dim=4,
            per_server_dim=per_srv_dim,
            hidden_dim=hidden_dim, t_dim=t_dim,
        )
        self.actor = FeedbackDiffusion(
            state_dim=state_dim, action_dim=Emax, model=backbone,
            beta_schedule='vp', denoising_steps=denoising_steps,
        ).to(device)

        self.critic1 = QValueNetV5(state_dim, hidden_dim, Emax).to(device)
        self.critic2 = QValueNetV5(state_dim, hidden_dim, Emax).to(device)
        self.target1 = QValueNetV5(state_dim, hidden_dim, Emax).to(device)
        self.target2 = QValueNetV5(state_dim, hidden_dim, Emax).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                                       device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # PopArt-lite: 每目标 RMS 的 EMA 估计, 用于 critic loss 缩放.
        # 惰性初始化: 首次 update 直接用 batch RMS, 避免从 1.0 爬坡
        # (σ_T 真实值 15-40, σ_E 3-25, β=0.001 爬到半程需 ~3000 步).
        self._sigma = torch.ones(N_OBJ, dtype=torch.float, device=device)
        self._sigma_init = False

        self.n_candidates = int(n_candidates)
        self.perturb_std = float(perturb_std)
        self._mcss_pick_count = [0, 0, 0]

    # --------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------
    def _masked_actor(self, state, latent, prior, mask):
        probs = self.actor(state, latent, prior=prior)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def _scalarize_q(self, q_vec, omega):
        """Q_vec[B, A, N] · (ω·α)[B, N] → Q_eff[B, A].

        ω 来自 state[:, 2:4], α 在 [alpha_T, alpha_E] 中. 此函数仅在 take_action
        与 actor 更新时使用, critic 训练直接用 vector form.
        """
        alpha_vec = torch.tensor([self.alpha_T, self.alpha_E],
                                  dtype=torch.float, device=self.device)
        weights = omega * alpha_vec  # [B, N]
        return torch.einsum('bak,bk->ba', q_vec, weights)

    # ---- critic 学法 hook (V8 干净版只覆写这两个; V5 行为不变) ----
    def _critic_target(self, r_vec_t, V_vec, entropy_next):
        """V5: 软 Q, 把 SAC 熵平摊进两个目标通道."""
        ent_term = (self.log_alpha.exp() * entropy_next) / N_OBJ
        return r_vec_t + self.gamma * (V_vec + ent_term.expand(-1, N_OBJ))

    def _critic_weight(self, omega_2d, alpha_vec, sigma_sq):
        """V5: 逐目标按 ω·α 加权再除 PopArt σ². omega_2d:[B,2]."""
        return (omega_2d * alpha_vec) / sigma_sq

    def _build_candidates(self, latent_1xE):
        N = self.n_candidates
        if N <= 1:
            return latent_1xE
        base = latent_1xE.expand(N, -1).clone()
        noise = torch.randn_like(base) * self.perturb_std
        noise[0].zero_()
        cand = base + noise
        cand = torch.clamp(cand, min=0.0)
        cand = cand / (cand.sum(dim=1, keepdim=True) + 1e-8)
        return cand

    # --------------------------------------------------------------
    # 推断: H-MCSS 三源候选 + 向量 Q 经 ω 标量化选优
    # --------------------------------------------------------------
    def take_action(self, state_np, latent_np, prior_np, mask_np, stochastic=True):
        state = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
        mask = torch.tensor(mask_np[None], dtype=torch.float, device=self.device)
        prior = torch.tensor(prior_np[None], dtype=torch.float, device=self.device)
        latent_arr = np.asarray(latent_np, dtype=np.float32)

        with torch.no_grad():
            if latent_arr.ndim == 1:
                latent = torch.tensor(latent_arr[None], dtype=torch.float, device=self.device)
                rand_init = torch.randn_like(latent)
                cand = torch.cat([latent, prior, rand_init], dim=0)
                track_pick = True
            else:
                cand = torch.tensor(latent_arr, dtype=torch.float, device=self.device)
                track_pick = False
            N = cand.size(0)
            state_rep = state.expand(N, -1)
            mask_rep = mask.expand(N, -1)
            prior_rep = prior.expand(N, -1)

            probs_batch = self.actor(state_rep, cand, prior=prior_rep)
            probs_batch = probs_batch * mask_rep
            probs_batch = probs_batch / (probs_batch.sum(dim=1, keepdim=True) + 1e-8)

            q1_vec = self.critic1(state_rep)                       # [N, A, 2]
            q2_vec = self.critic2(state_rep)
            min_q_vec = torch.min(q1_vec, q2_vec)
            # 用 state 自带的 ω 做标量化
            omega_rep = state_rep[:, OMEGA_STATE_SLICE]            # [N, 2]
            q_eff = self._scalarize_q(min_q_vec, omega_rep)        # [N, A]
            expected_q = torch.sum(probs_batch * q_eff, dim=1)
            best_idx = int(torch.argmax(expected_q).item())
            best_probs = probs_batch[best_idx]
            if track_pick and 0 <= best_idx < 3:
                self._mcss_pick_count[best_idx] += 1

        probs_np = best_probs.detach().cpu().numpy()
        probs_np = np.clip(probs_np, 0.0, 1.0)
        s = probs_np.sum()
        if s < 1e-8:
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

    # --------------------------------------------------------------
    # 训练: vector Q + ω-weighted loss + COR + PopArt-lite
    # --------------------------------------------------------------
    def update(self, batch):
        s, a, x, p, m, r_vec, sn, xn, mn = batch
        s_t = torch.tensor(s, dtype=torch.float, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device).view(-1, 1)
        x_t = torch.tensor(x, dtype=torch.float, device=self.device)
        p_t = torch.tensor(p, dtype=torch.float, device=self.device)
        m_t = torch.tensor(m, dtype=torch.float, device=self.device)
        r_vec_t = torch.tensor(r_vec, dtype=torch.float, device=self.device)  # [B, 2]
        sn_t = torch.tensor(sn, dtype=torch.float, device=self.device)
        xn_t = torch.tensor(xn, dtype=torch.float, device=self.device)
        mn_t = torch.tensor(mn, dtype=torch.float, device=self.device)

        B = s_t.size(0)
        alpha_vec = torch.tensor([self.alpha_T, self.alpha_E],
                                  dtype=torch.float, device=self.device)
        omega_b = s_t[:, OMEGA_STATE_SLICE]  # [B, 2]  batch 原 ω

        # ===== 1) 计算 vector target =====
        with torch.no_grad():
            next_probs = self._masked_actor(sn_t, xn_t, p_t, mn_t)         # [B, A]
            log_next = torch.log(next_probs + 1e-8)
            entropy_next = -torch.sum(next_probs * log_next, dim=1, keepdim=True)  # [B, 1]
            tq1_vec = self.target1(sn_t)                                    # [B, A, 2]
            tq2_vec = self.target2(sn_t)
            min_tq_vec = torch.min(tq1_vec, tq2_vec)
            V_vec = torch.einsum('ba,bak->bk', next_probs, min_tq_vec)      # [B, 2]
            # critic 目标 (hook: V5 软 Q 含熵, V8 干净版去熵)
            target_vec = self._critic_target(r_vec_t, V_vec, entropy_next)

            # PopArt-lite: 更新 σ_k = running RMS of target
            if self.use_popart:
                cur_rms = torch.sqrt((target_vec ** 2).mean(dim=0) + 1e-8)  # [2]
                if not self._sigma_init:
                    self._sigma.copy_(cur_rms)
                    self._sigma_init = True
                else:
                    self._sigma.mul_(1.0 - self.popart_beta).add_(self.popart_beta * cur_rms)

        sigma = self._sigma.detach().clamp(min=1e-3)                        # [2]
        if self.use_popart:
            sigma_sq = sigma.unsqueeze(0) ** 2                              # [1, 2]
        else:
            sigma_sq = torch.ones((1, N_OBJ), device=self.device)

        # ===== 2) 主 critic loss (vector Q, ω-weighted, PopArt-scaled) =====
        idx = a_t.unsqueeze(-1).expand(-1, 1, N_OBJ)                        # [B, 1, 2]

        q1_vec = self.critic1(s_t)
        q2_vec = self.critic2(s_t)
        q1_a = q1_vec.gather(1, idx).squeeze(1)                             # [B, 2]
        q2_a = q2_vec.gather(1, idx).squeeze(1)
        err1 = (q1_a - target_vec) ** 2
        err2 = (q2_a - target_vec) ** 2
        w_main = self._critic_weight(omega_b, alpha_vec, sigma_sq)          # [B, 2]
        c1_loss_main = (w_main * err1).sum(dim=1).mean()
        c2_loss_main = (w_main * err2).sum(dim=1).mean()

        # ===== 3) COR step =====
        rho1_val = 1.0
        rho2_val = 1.0
        cor_w1 = 0.0
        cor_w2 = 0.0
        cor_loss1 = torch.zeros((), device=self.device)
        cor_loss2 = torch.zeros((), device=self.device)

        if self.use_cor:
            # 3.1 采一个共享 ω_sample (整 batch 用同一个, 让 ρ 度量两偏好间冲突)
            omega_sample_np = self._omega_rng.dirichlet([1.0, 1.0]).astype(np.float32)
            omega_sample = torch.tensor(omega_sample_np, dtype=torch.float, device=self.device)  # [2]
            s_relabel = _relabel_state_omega(s_t, omega_sample)             # [B, state_dim]

            # 3.2 在 relabeled state 上 forward critic, 准备 sample loss + COR penalty
            #     (target_vec 沿用原 batch 的, 不重算 — 简化: COR 只测 Q 的 ω-稳定性)
            with torch.no_grad():
                q1_old_sample_vec = self.target1(s_relabel)                 # Q_old: target net
                q2_old_sample_vec = self.target2(s_relabel)
            q1_sample_vec = self.critic1(s_relabel)
            q2_sample_vec = self.critic2(s_relabel)
            q1_a_sample = q1_sample_vec.gather(1, idx).squeeze(1)           # [B, 2]
            q2_a_sample = q2_sample_vec.gather(1, idx).squeeze(1)
            q1_a_old = q1_old_sample_vec.gather(1, idx).squeeze(1)
            q2_a_old = q2_old_sample_vec.gather(1, idx).squeeze(1)

            err1_sample = (q1_a_sample - target_vec) ** 2
            err2_sample = (q2_a_sample - target_vec) ** 2
            w_sample = self._critic_weight(omega_sample.unsqueeze(0).expand(B, -1), alpha_vec, sigma_sq)
            c1_loss_sample = (w_sample * err1_sample).sum(dim=1).mean()
            c2_loss_sample = (w_sample * err2_sample).sum(dim=1).mean()

            # 3.3 ρ = cos(∇L_main, ∇L_sample) 在 critic 参数上
            c1_params = list(self.critic1.parameters())
            c2_params = list(self.critic2.parameters())

            self.c1_opt.zero_grad(set_to_none=False)
            c1_loss_main.backward(retain_graph=True)
            g1_main = _flat_grads(c1_params)
            self.c1_opt.zero_grad(set_to_none=False)
            c1_loss_sample.backward(retain_graph=True)
            g1_sample = _flat_grads(c1_params)
            self.c1_opt.zero_grad(set_to_none=False)
            denom1 = (g1_main.norm() * g1_sample.norm()).clamp(min=1e-8)
            rho1 = (g1_main * g1_sample).sum() / denom1
            rho1_val = float(rho1.detach().item())

            self.c2_opt.zero_grad(set_to_none=False)
            c2_loss_main.backward(retain_graph=True)
            g2_main = _flat_grads(c2_params)
            self.c2_opt.zero_grad(set_to_none=False)
            c2_loss_sample.backward(retain_graph=True)
            g2_sample = _flat_grads(c2_params)
            self.c2_opt.zero_grad(set_to_none=False)
            denom2 = (g2_main.norm() * g2_sample.norm()).clamp(min=1e-8)
            rho2 = (g2_main * g2_sample).sum() / denom2
            rho2_val = float(rho2.detach().item())

            # 3.4 COR penalty: max(c − ρ, 0) · λ · ω_sample-weighted (Q − Q_target_old)²
            cor_w1 = max(self.cor_c - rho1_val, 0.0) * self.cor_lambda
            cor_w2 = max(self.cor_c - rho2_val, 0.0) * self.cor_lambda
            if cor_w1 > 0:
                penalty1 = (q1_a_sample - q1_a_old) ** 2
                cor_loss1 = (w_sample * penalty1).sum(dim=1).mean() * cor_w1
            if cor_w2 > 0:
                penalty2 = (q2_a_sample - q2_a_old) ** 2
                cor_loss2 = (w_sample * penalty2).sum(dim=1).mean() * cor_w2

        # ===== 4) 综合 critic loss 反传 + step =====
        total_c1 = c1_loss_main + cor_loss1
        total_c2 = c2_loss_main + cor_loss2
        self.c1_opt.zero_grad(set_to_none=True)
        total_c1.backward()
        self.c1_opt.step()
        self.c2_opt.zero_grad(set_to_none=True)
        total_c2.backward()
        self.c2_opt.step()

        # ===== 5) actor 更新 (用 batch 自带 ω 标量化 Q_vec) =====
        probs = self._masked_actor(s_t, x_t, p_t, m_t)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_e_vec = self.critic1(s_t)
        q2_e_vec = self.critic2(s_t)
        min_q_vec = torch.min(q1_e_vec, q2_e_vec)
        q_eff = self._scalarize_q(min_q_vec, omega_b)
        expected_q_eval = torch.sum(probs * q_eff, dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp().detach() * entropy - expected_q_eval)
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = torch.mean((entropy.detach() - self.target_entropy) * self.log_alpha.exp())
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return dict(
            c_loss=float((c1_loss_main + c2_loss_main).detach() * 0.5),
            cor_loss=float((cor_loss1 + cor_loss2).detach() * 0.5),
            a_loss=float(actor_loss.detach()),
            alpha=float(self.log_alpha.exp().detach()),
            H=float(entropy.mean().detach()),
            rho1=rho1_val, rho2=rho2_val,
            cor_w1=cor_w1, cor_w2=cor_w2,
            sigma_T=float(sigma[0].item()),
            sigma_E=float(sigma[1].item()),
        )

    def _soft_update(self, net, target_net):
        for p_t, p in zip(target_net.parameters(), net.parameters()):
            p_t.data.copy_(p_t.data * (1.0 - self.tau) + p.data * self.tau)
