"""离散 SAC agent: prior-feedback 扩散 actor + 向量 critic (ω-无关 SLA 通道)。

不依赖 tianshou, 纯 PyTorch。要点:
  * actor = FDActor (set-conv 编码一次 + prior 暖启动扩散), 输出 [B, n_slots] 分配概率。
  * critic = 两个 VectorCritic, 输出 [B, n_slots, 3]; target 网络软更新。
  * 标量化: scalar_Q = w*Q_T + (1-w)*Q_E + λ*Q_C, λ 与 ω 无关 (修硬伤)。
  * target_entropy=0.5 (正值! 离散 |A| 的熵上界 log(n_valid)>0; 负值会逼 α→0 熵崩,
    这是旧 baseline 全崩的根因, 见记忆 baseline_eval_status)。
  * critic target 不含熵 (V8 风格), 熵只进 actor loss, 稳。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

from fd_actor import FDActor, MLPActor, obs_to_tensors, uniform_prior
from vector_critic import VectorCritic, N_OBJ
from set_encoder import N_SLOTS


# --------------------------------------------------------------
# Replay buffer (dict obs 感知): 存 numpy, 采样时转 tensor
# --------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def add(self, servers, pref, mask2, prior, action, r_vec,
            n_servers, n_pref, n_mask2, n_prior, done,
            act_mask=None, n_act_mask=None):
        # act_mask = mask2 ∩ 准入掩码; 不传时退回 mask2 (= 无准入)。
        if act_mask is None:
            act_mask = mask2
        if n_act_mask is None:
            n_act_mask = n_mask2
        self.buf.append((servers, pref, mask2, prior, action, r_vec,
                         n_servers, n_pref, n_mask2, n_prior, done, act_mask, n_act_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        (servers, pref, mask2, prior, action, r_vec,
         n_servers, n_pref, n_mask2, n_prior, done, act_mask, n_act_mask) = zip(*batch)
        return dict(
            servers=np.stack(servers), pref=np.stack(pref), mask2=np.stack(mask2),
            prior=np.stack(prior), action=np.asarray(action, dtype=np.int64),
            r_vec=np.stack(r_vec).astype(np.float32),
            n_servers=np.stack(n_servers), n_pref=np.stack(n_pref),
            n_mask2=np.stack(n_mask2), n_prior=np.stack(n_prior),
            done=np.asarray(done, dtype=np.float32),
            act_mask=np.stack(act_mask), n_act_mask=np.stack(n_act_mask),
        )

    def size(self):
        return len(self.buf)


# --------------------------------------------------------------
# Agent
# --------------------------------------------------------------
class FDSACAgent:
    def __init__(self, n_slots=N_SLOTS, conv_ch=256, cond_dim=256,
                 denoising_steps=3, start_mode='prior',
                 actor_lr=1e-4, critic_lr=3e-4, alpha=0.05, alpha_lr=3e-4,
                 target_entropy=0.5, sla_lambda=1.0, tau=0.005, gamma=0.95,
                 reward_scale=(0.1, 1.0, 1.0), grad_clip=10.0,
                 alpha_min=0.01, alpha_max=0.3, auto_alpha=True,
                 actor_type='diffusion', use_prior_cond=False, use_popart=False,
                 device=torch.device('cpu')):
        # auto_alpha=False -> 固定温度 (GMORL 风格), 不调 alpha。
        #   向量化下自动调温会被 Q 梯度逼垮(alpha顶钳位H仍崩), 固定 alpha 更稳。
        # alpha 钳位: 防熵控制器失控 (admission 限制动作集时, H 可能够不到 target ->
        #   alpha 单调爆炸到上千, 配合 grad_clip 把 actor 冻死。run3 教训)。
        # reward_scale: 3 通道量级平衡 (冒烟测出 r_T 约 10× r_E/r_C, 不缩放则 delay 通道
        #   主导, Pareto 前沿塌成横线 —— 见记忆 reward_scaling)。对 r_C 用 1.0, 不破坏 ω-无关。
        # grad_clip:    梯度裁剪, 防长训值尺度膨胀发散 (c_loss/a_loss 上涨的廉价保险)。
        self.device = device
        self.n_slots = n_slots
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.sla_lambda = sla_lambda
        self.auto_alpha = auto_alpha
        self.reward_scale = torch.tensor(reward_scale, dtype=torch.float32, device=device)
        self.grad_clip = grad_clip
        # PopArt: 每目标 Q-target 归一化 (统一三通道尺度, 让 delay 不被压糙、SLA 公平加权)。
        # 关掉时 ret_mean=0/ret_std=1 -> 归一化为恒等, 行为与旧版一致。
        self.use_popart = use_popart
        self.ret_mean = torch.zeros(N_OBJ, device=device)
        self.ret_meansq = torch.ones(N_OBJ, device=device)
        self.ret_std = torch.ones(N_OBJ, device=device)
        self.popart_beta = 3e-4

        self.actor_type = actor_type
        if actor_type == 'mlp':
            self.actor = MLPActor(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots).to(device)
        else:
            self.actor = FDActor(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots,
                                 denoising_steps=denoising_steps, start_mode=start_mode,
                                 use_prior_cond=use_prior_cond).to(device)
        self.critic1 = VectorCritic(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots).to(device)
        self.critic2 = VectorCritic(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots).to(device)
        self.target1 = VectorCritic(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots).to(device)
        self.target2 = VectorCritic(conv_ch=conv_ch, cond_dim=cond_dim, n_slots=n_slots).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self._log_alpha_min = float(np.log(alpha_min))
        self._log_alpha_max = float(np.log(alpha_max))

    # ----- 标量化: [B, n_slots, 3] -> [B, n_slots] (λ 与 ω 无关) -----
    def _scalarize(self, q_vec, pref):
        w = pref[:, 0:1]                 # [B,1] = omega_T
        w_e = pref[:, 1:2]               # [B,1] = omega_E = 1-w
        qT, qE, qC = q_vec[..., 0], q_vec[..., 1], q_vec[..., 2]   # 各 [B, n_slots]
        return w * qT + w_e * qE + self.sla_lambda * qC            # [B, n_slots]

    # ----- 交互: 采样动作 -----
    @torch.no_grad()
    def take_action(self, obs, prior_np, stochastic=True, act_mask_np=None):
        servers, pref, mask2 = obs_to_tensors(obs, self.device)
        prior = torch.as_tensor(prior_np[None], dtype=torch.float32, device=self.device)
        act_mask = None
        if act_mask_np is not None:
            act_mask = torch.as_tensor(act_mask_np[None], dtype=torch.float32, device=self.device)
        probs = self.actor(servers, pref, mask2, prior, act_mask=act_mask)
        p = probs.cpu().numpy()[0]
        p = np.clip(p, 0.0, 1.0)
        s = p.sum()
        if s < 1e-8:
            # 回退: 在(准入后)合法集合里均匀挑一个
            base = act_mask_np if act_mask_np is not None else np.asarray(obs['mask2'])
            valid = np.where(np.asarray(base) > 0.5)[0]
            a = int(np.random.choice(valid)) if len(valid) else 0
            oneh = np.zeros(self.n_slots, dtype=np.float32); oneh[a] = 1.0
            return a, oneh
        p = p / s
        a = int(np.random.choice(self.n_slots, p=p)) if stochastic else int(np.argmax(p))
        return a, p.astype(np.float32)

    # ----- 批量交互 (向量化训练用): N 个 env 一次前向 -----
    @torch.no_grad()
    def take_action_batch(self, obs_list, prior_np, act_mask_np=None, stochastic=True):
        servers, pref, mask2 = obs_to_tensors(obs_list, self.device)
        prior = torch.as_tensor(prior_np, dtype=torch.float32, device=self.device)
        act_mask = None
        if act_mask_np is not None:
            act_mask = torch.as_tensor(act_mask_np, dtype=torch.float32, device=self.device)
        probs = self.actor(servers, pref, mask2, prior, act_mask=act_mask)  # [N, n_slots]
        p = np.clip(probs.cpu().numpy(), 0.0, 1.0)
        N = len(obs_list)
        actions = np.zeros(N, dtype=np.int64)
        out = np.zeros_like(p)
        for i in range(N):
            s = p[i].sum()
            if s < 1e-8:
                base = act_mask_np[i] if act_mask_np is not None else np.asarray(obs_list[i]['mask2'])
                valid = np.where(np.asarray(base) > 0.5)[0]
                a = int(np.random.choice(valid)) if len(valid) else 0
                out[i, a] = 1.0; actions[i] = a
            else:
                pi = p[i] / s
                actions[i] = int(np.random.choice(self.n_slots, p=pi)) if stochastic else int(np.argmax(pi))
                out[i] = pi
        return actions, out.astype(np.float32)

    # ----- 训练一步 -----
    def update(self, batch):
        dev = self.device
        servers = torch.as_tensor(batch['servers'], dtype=torch.float32, device=dev)
        pref = torch.as_tensor(batch['pref'], dtype=torch.float32, device=dev)
        mask2 = torch.as_tensor(batch['mask2'], dtype=torch.float32, device=dev)
        prior = torch.as_tensor(batch['prior'], dtype=torch.float32, device=dev)
        action = torch.as_tensor(batch['action'], dtype=torch.long, device=dev)
        r_vec = torch.as_tensor(batch['r_vec'], dtype=torch.float32, device=dev)       # [B, 3]
        r_vec = r_vec * self.reward_scale                                              # 通道量级平衡
        n_servers = torch.as_tensor(batch['n_servers'], dtype=torch.float32, device=dev)
        n_pref = torch.as_tensor(batch['n_pref'], dtype=torch.float32, device=dev)
        n_mask2 = torch.as_tensor(batch['n_mask2'], dtype=torch.float32, device=dev)
        n_prior = torch.as_tensor(batch['n_prior'], dtype=torch.float32, device=dev)
        done = torch.as_tensor(batch['done'], dtype=torch.float32, device=dev).view(-1, 1)
        act_mask = torch.as_tensor(batch['act_mask'], dtype=torch.float32, device=dev)
        n_act_mask = torch.as_tensor(batch['n_act_mask'], dtype=torch.float32, device=dev)
        B = servers.size(0)

        # ----- 向量 critic target (每目标 Bellman, 不含熵; PopArt 归一化) -----
        with torch.no_grad():
            n_probs = self.actor(n_servers, n_pref, n_mask2, n_prior, act_mask=n_act_mask)  # [B, n_slots]
            q1n = self.target1(n_servers, n_pref, n_mask2)                   # normalized [B, n_slots, 3]
            q2n = self.target2(n_servers, n_pref, n_mask2)
            min_qn = torch.min(q1n, q2n)
            v_next_norm = (n_probs.unsqueeze(-1) * min_qn).sum(dim=1)        # [B, 3] normalized
            v_next = v_next_norm * self.ret_std + self.ret_mean              # 反归一化 -> 实际值
            target_vec = r_vec + self.gamma * (1.0 - done) * v_next          # 实际空间 Bellman
            if self.use_popart:
                self._update_popart(target_vec)
            target_norm = (target_vec - self.ret_mean) / self.ret_std        # 归一化供 critic loss

        # ----- critic 更新 (在归一化空间拟合) -----
        idx = action.view(B, 1, 1).expand(B, 1, N_OBJ)                       # gather 服务器维
        q1 = self.critic1(servers, pref, mask2).gather(1, idx).squeeze(1)    # normalized [B, 3]
        q2 = self.critic2(servers, pref, mask2).gather(1, idx).squeeze(1)
        c1_loss = F.mse_loss(q1, target_norm)
        c2_loss = F.mse_loss(q2, target_norm)
        self.c1_opt.zero_grad(); c1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip); self.c1_opt.step()
        self.c2_opt.zero_grad(); c2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip); self.c2_opt.step()

        # ----- actor 更新 (标量化, λ 与 ω 无关) -----
        probs = self.actor(servers, pref, mask2, prior, act_mask=act_mask)  # [B, n_slots]
        q1e = self.critic1(servers, pref, mask2)
        q2e = self.critic2(servers, pref, mask2)
        # actor 直接在归一化空间标量化: 各目标单位尺度 -> w/λ 公平加权; exp_q 量级 O(1) 不压熵。
        # (PopArt 关时 ret_mean=0/std=1, 归一化=恒等, 与旧版一致)
        min_qe = torch.min(q1e, q2e)                                       # normalized [B, n_slots, 3]
        scalar_q = self._scalarize(min_qe, pref)                            # [B, n_slots]
        exp_q = (probs * scalar_q).sum(dim=1)                               # [B]
        logp = torch.log(probs + 1e-8)
        entropy = -(probs * logp).sum(dim=1)                               # [B]
        alpha = self.log_alpha.exp().detach()
        actor_loss = (-exp_q - alpha * entropy).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip); self.actor_opt.step()

        # ----- alpha 更新 (推 entropy -> target_entropy); 固定 alpha 时跳过 -----
        if self.auto_alpha:
            alpha_loss = ((entropy.detach() - self.target_entropy) * self.log_alpha.exp()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            with torch.no_grad():                              # 钳位防失控
                self.log_alpha.clamp_(self._log_alpha_min, self._log_alpha_max)

        # ----- 软更新 -----
        self._soft(self.critic1, self.target1)
        self._soft(self.critic2, self.target2)

        return dict(c_loss=float((c1_loss + c2_loss).detach() * 0.5),
                    a_loss=float(actor_loss.detach()),
                    alpha=float(self.log_alpha.exp().detach()),
                    H=float(entropy.mean().detach()))

    def _update_popart(self, target_vec):
        """EMA 更新每目标 return 的均值/方差 (target_vec: [B, N_OBJ])。"""
        b = self.popart_beta
        m = target_vec.mean(dim=0)
        sq = (target_vec ** 2).mean(dim=0)
        self.ret_mean = (1 - b) * self.ret_mean + b * m
        self.ret_meansq = (1 - b) * self.ret_meansq + b * sq
        self.ret_std = torch.sqrt(torch.clamp(self.ret_meansq - self.ret_mean ** 2, min=1e-4))

    def _soft(self, net, target):
        for pt, p in zip(target.parameters(), net.parameters()):
            pt.data.copy_(pt.data * (1.0 - self.tau) + p.data * self.tau)

    # ----- checkpoint -----
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pt'))
        torch.save(self.critic1.state_dict(), os.path.join(path, 'critic1.pt'))
        torch.save(self.critic2.state_dict(), os.path.join(path, 'critic2.pt'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pt'), map_location=self.device))
        self.critic1.load_state_dict(torch.load(os.path.join(path, 'critic1.pt'), map_location=self.device))
        self.critic2.load_state_dict(torch.load(os.path.join(path, 'critic2.pt'), map_location=self.device))
        self.actor.eval()
