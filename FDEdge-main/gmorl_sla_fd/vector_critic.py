"""向量 Critic: 每服务器、每目标的 Q 值 [B, n_slots, N_OBJ]。

N_OBJ=3: [Q_T(延迟), Q_E(能耗), Q_C(SLA)]。
标量化在 agent 里做: w*Q_T + (1-w)*Q_E + λ*Q_C, 其中 λ (SLA 权重) 与 ω 无关。
这是修掉"SLA-进-r_T 会在 ω_T=0 被乘没"硬伤的关键: SLA 单独成通道, 不被偏好加权吞掉。

Critic 仍把 preference 作为输入 (经 SetConvEncoder 的偏好 bias): 因为它评估的是
ω-条件策略 π_ω 的每目标回报 Q^{π_ω}, 该回报本就依赖 ω。ω-无关的只是"标量化权重 λ"。
"""
import torch
import torch.nn as nn

from set_encoder import SetConvEncoder, N_SLOTS, SERVER_FEAT

N_OBJ = 3  # [delay, energy, sla]


class VectorCritic(nn.Module):
    def __init__(self, conv_ch=256, cond_dim=256, n_slots=N_SLOTS, n_obj=N_OBJ, hidden=256):
        super().__init__()
        self.n_slots = n_slots
        self.n_obj = n_obj
        self.encoder = SetConvEncoder(in_ch=SERVER_FEAT, conv_ch=conv_ch,
                                      cond_dim=cond_dim, n_slots=n_slots)
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_slots * n_obj),
        )

    def forward(self, servers, preference, mask2):
        B = servers.size(0)
        cond = self.encoder(servers, preference, mask2)
        q = self.head(cond).reshape(B, self.n_slots, self.n_obj)  # [B, n_slots, n_obj]
        return q
