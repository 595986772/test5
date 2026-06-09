"""
ω-Hypernetwork (mofd_v6 新增)
================================
把"查表式 buffer.retrieve_prior(ω)"升级为连续光滑映射:

  H_φ(ω, env_ctx) → prior_latent (概率单纯形, sum=1, dim=action_dim)

设计参考: PSL-MORL (Pareto Set Learning for MORL, arXiv 2501.06773, Jan 2025)
区别: PSL-MORL 用 hypernet 直接生成 policy 参数; 这里仅用 hypernet 生成扩散
      起点 prior_latent, 改动量小, 与现有 actor 兼容.

训练: 在 mofd_v6.update() 内做 supervised regression:
  target = batch 里的历史 actor probs (即 latent x_t)
  loss   = MSE(hypernet(ω, env_ctx), x_t)
等价于 "hypernet 学到 actor 在 (ω, env_ctx) 下输出什么", 训练好后即可作为
高质量扩散起点.

env_ctx 接口允许 None: 训练阶段如果 ReplayBuffer 未存 env_ctx 则补零;
评估阶段完整传 (E, f_E, tran_rate) 归一化向量 (mofd_main.make_env_ctx 计算).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CTX_DIM_DEFAULT = 3   # 与 mofd_main.make_env_ctx 输出维度一致


class OmegaHypernet(nn.Module):
    def __init__(self, omega_dim=2, ctx_dim=CTX_DIM_DEFAULT,
                 action_dim=6, hidden=64):
        super().__init__()
        self.omega_dim = int(omega_dim)
        self.ctx_dim = int(ctx_dim)
        self.action_dim = int(action_dim)
        in_dim = self.omega_dim + self.ctx_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.action_dim),
        )

    def forward(self, omega, env_ctx=None):
        """
        omega:   [B, 2] or [2]
        env_ctx: [B, ctx_dim] or [ctx_dim] or None (None 时补零, 训练兼容)
        return:  [B, action_dim] 概率单纯形 (softmax 输出)
        """
        if omega.dim() == 1:
            omega = omega.unsqueeze(0)
        B = omega.size(0)
        if env_ctx is None:
            env_ctx = torch.zeros(B, self.ctx_dim,
                                  dtype=omega.dtype, device=omega.device)
        elif env_ctx.dim() == 1:
            env_ctx = env_ctx.unsqueeze(0).expand(B, -1)
        x = torch.cat([omega, env_ctx], dim=-1)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def predict_np(self, omega_np, env_ctx_np=None, device='cpu'):
        """numpy 接口, 评估时替代 buffer.retrieve_prior(ω)."""
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(omega_np, dtype=np.float32),
                                device=device)
            c = None
            if env_ctx_np is not None:
                c = torch.as_tensor(np.asarray(env_ctx_np, dtype=np.float32),
                                    device=device)
            p = self.forward(o, c)
            return p.squeeze(0).cpu().numpy()
