"""
MOFD V8 + OACR (晚融合 critic) —— 最小测试版
============================================
只验一个假设: **omega 晚融合(先编码不含 omega 的 ctx, omega 在第 2 层才注入) 比
现在的早融合(omega 从第 1 层就和 state 混在一起) 在未见偏好上泛化更好?**

相对 V8 只换 critic 网络, 其余 (反馈扩散 actor / 干净 critic 损失 / COR / PopArt / SAC-α)
全继承 V8。critic 的 forward(s) 接口与 QValueNetV5 完全一致 (内部 split state),
所以 update / COR(_relabel_state_omega) / take_action 一行都不用改。

故意不做 (都是另一个独立问题, 留待这步验明再说):
  * aux 下一步 context 预测 —— 本环境信道 i.i.d.、valid_mask 恒定, 该任务半废;
  * actor 也用 OACR —— SAC actor 对输入维度敏感, 非最小;
  * 漂移/context-shift 实验 —— 已否决的方向。

容量配平 (堵 "只是参数变多" 的质疑):
  早融合 QValueNetV5: Linear(ctx+2, H) → ReLU → Linear(H,H) → ReLU → Linear(H, A*nobj)
  晚融合 QValueNetOACR: Linear(ctx, H) → ReLU → Linear(H+2, H) → ReLU → Linear(H, A*nobj)
  → 层数相同(3 Linear), omega 进哪一层不同, 总参数几乎相等。唯一变量 = 融合时机。

用法: 由 runner monkey-patch  mofd_main.MOFD_SAC_V8 = MOFD_SAC_V8_OACR  注入, 不污染主项目。
"""
import torch
import torch.nn as nn

from mofd_v5 import N_OBJ, OMEGA_STATE_SLICE
from mofd_v8 import MOFD_SAC_V8


class QValueNetOACR(nn.Module):
    """晚融合向量 Q: 第 1 层只吃 ctx(不含 omega), omega 在第 2 层注入。

    forward(s) 接口、输出形状 [B, action_dim, n_obj] 与 QValueNetV5 完全一致。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, n_obj=N_OBJ,
                 omega_slice=OMEGA_STATE_SLICE):
        super().__init__()
        self.action_dim = action_dim
        self.n_obj = n_obj
        self.omega_slice = omega_slice
        self.omega_dim = omega_slice.stop - omega_slice.start          # = 2
        ctx_dim = state_dim - self.omega_dim
        # 层1: 仅 ctx (无 omega)
        self.enc = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.ReLU(),
        )
        # 层2/3: omega 在此注入 (晚融合)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.omega_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_obj),
        )

    def forward(self, s):
        om = s[:, self.omega_slice]
        ctx = torch.cat([s[:, :self.omega_slice.start],
                         s[:, self.omega_slice.stop:]], dim=-1)
        z = self.enc(ctx)
        h = self.head(torch.cat([z, om], dim=-1))
        return h.view(-1, self.action_dim, self.n_obj)


class MOFD_SAC_V8_OACR(MOFD_SAC_V8):
    """V8 + 晚融合 critic。super 后重建 critic/target/优化器, 其余全继承 (NoDiffusion 同模式)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sd = self.critic1.net[0].in_features          # state_dim
        H = self.critic1.net[0].out_features          # hidden_dim
        n_obj = getattr(self.critic1, 'n_obj', N_OBJ)
        critic_lr = float(kwargs.get('critic_lr', 1e-3))

        def mk():
            return QValueNetOACR(sd, H, self.Emax, n_obj=n_obj).to(self.device)

        self.critic1, self.critic2 = mk(), mk()
        self.target1, self.target2 = mk(), mk()
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
