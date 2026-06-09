"""
MOFD V8 = V5 干净版 critic
==========================
只改 critic 的"学法", 其余 (反馈扩散 actor / 向量 Q / COR / PopArt / SAC-α) 全继承 V5。
相对 V5 的两处改动 (都通过覆写 V5 的 hook 实现, 不复制 update 主体):

  1) 目标不掺熵:  target_vec = r_vec + γ·V        (V5 是软 Q, 把 SAC 熵平摊进 Q_T/Q_E)
       → Q 通道回到"纯逐目标回报", 熵只留给 actor, 不再污染目标价值。
  2) critic 损失等权: 去掉逐目标的 ω·α 偏好权重, 仅留 PopArt 量级归一 (1/σ²)
       → 极端偏好下 (如完全不在乎能耗) 能耗通道不再被压成零梯度, 两目标都照学。
       这条同时作用于主损失 w_main 与 COR 的 w_sample (因为都走同一个 hook)。

注意: ω 偏好仍在 actor 的 _scalarize_q(ω) 里起作用, 一点没丢; 改的只是 critic 怎么学。
代价: Q 不再是"软" Q (价值里不含熵奖励), 但 actor 仍照常用熵探索, 实际影响很小。

用法 (在 mofd_main 里由 cfg 选择):
    mofd_main.main(cfg_override={'use_v8': True, 'file_prefix': 'v8', ...})
"""
import torch

from mofd_v5 import MOFD_SAC_V5, N_OBJ  # noqa: F401 (N_OBJ 供阅读对照)


class MOFD_SAC_V8(MOFD_SAC_V5):
    """V5 干净版: 纯逐目标 Q + 等权 critic loss。仅覆写两个 hook。"""

    def _critic_target(self, r_vec_t, V_vec, entropy_next):
        # 干净版: 只估纯逐目标回报, 不掺熵 (熵交给 actor)
        return r_vec_t + self.gamma * V_vec

    def _critic_weight(self, omega_2d, alpha_vec, sigma_sq):
        # 干净版: 两目标等权, 仅留 PopArt 量级归一 (去掉 ω·α 偏好权重)
        return torch.ones_like(omega_2d) / sigma_sq
