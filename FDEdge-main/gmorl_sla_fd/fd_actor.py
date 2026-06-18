"""Prior-feedback 扩散 Actor, 接在 GMORL set-conv 编码器上。

设计 (路 A: 让 prior-feedback 扩散在"按服务器"的空间里跑):
  1. SetConvEncoder 把 dict obs 编成条件向量 cond  —— 【只跑 1 次】
  2. WarmStartDiffusion 从 prior (上一步分配概率) 暖启动, 去噪 T 步
     —— 循环里只跑轻量 DenoiseNet, cond 复用, 不重编码 (修掉"编码×T"的爆炸)
  3. 输出 11 维服务器分配概率, 用 mask2 屏蔽无效服务器

关键修正 vs 原 feedback_diffusion.Diffusion:
  基类 p_sample_loop 从 torch.randn 起步; 这里 WarmStartDiffusion 从 prior 起步
  (start_mode='prior'), 并保留 'randn' 开关供 E/F/G 消融对照。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import SinusoidalPosEmb
from feedback_diffusion import Diffusion
from set_encoder import SetConvEncoder, N_SLOTS, SERVER_FEAT


# --------------------------------------------------------------
# 去噪网络 (ε-预测): 输入 (x_t, t, cond), 输出 11 维
# --------------------------------------------------------------
class DenoiseNet(nn.Module):
    def __init__(self, cond_dim=256, n_slots=N_SLOTS, t_dim=16, hidden=256, use_prior_cond=False):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(t_dim)
        self.use_prior_cond = use_prior_cond
        in_dim = n_slots + t_dim + cond_dim + (n_slots if use_prior_cond else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_slots),
        )

    def forward(self, x, time_step, state, prior_latent=None):
        # x: [B, n_slots] 当前去噪 latent; state: cond [B, cond_dim] (已编码, 循环里复用)
        # prior_latent: [B, n_slots] 上一步分配概率, P2 双通道 = 既当暖启动又当条件喂进来
        t_emb = self.time_emb(time_step)
        parts = [x, t_emb, state]
        if self.use_prior_cond:
            if prior_latent is None:
                prior_latent = torch.zeros_like(x)
            parts.append(prior_latent)
        h = torch.cat(parts, dim=1)
        return F.softmax(self.net(h), dim=1)  # 与 FDEdge 原实现一致: ε-网络输出 softmax 化


# --------------------------------------------------------------
# 暖启动扩散: 从 prior 起步 (而非 randn)
# --------------------------------------------------------------
class WarmStartDiffusion(Diffusion):
    def p_sample_loop(self, state, prior_probs, prior=None, start_mode='prior', det=False):
        # det=True: 确定性反向 (全程只走后验均值, 不注随机噪声)。
        #   推理/eval 用 —— 让动作成为 cond 的干净确定函数, argmax 稳、输出更尖 (low-temp diffusion)。
        #   训练 rollout 仍用 det=False (随机反向) 保留探索/熵。
        if start_mode == 'prior':
            x = prior_probs.clone()              # 从上一步分配概率暖启动
        else:
            x = torch.randn_like(prior_probs)    # 消融对照: 普通随机起步
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            if det:
                mean, _, _ = self.p_mean_variance(x=x, t=t, s=state, prior=prior)
                x = mean
            else:
                x = self.p_sample(x, t, state, prior=prior)
        return x


# --------------------------------------------------------------
# 完整 Actor: 编码器 + 扩散 + mask
# --------------------------------------------------------------
class FDActor(nn.Module):
    def __init__(self, conv_ch=256, cond_dim=256, n_slots=N_SLOTS,
                 denoising_steps=5, start_mode='prior', t_dim=16, hidden=256,
                 use_prior_cond=False):
        super().__init__()
        self.n_slots = n_slots
        self.start_mode = start_mode
        self.use_prior_cond = use_prior_cond
        self.encoder = SetConvEncoder(in_ch=SERVER_FEAT, conv_ch=conv_ch,
                                      cond_dim=cond_dim, n_slots=n_slots)
        self.denoise = DenoiseNet(cond_dim=cond_dim, n_slots=n_slots, t_dim=t_dim, hidden=hidden,
                                  use_prior_cond=use_prior_cond)
        self.diffusion = WarmStartDiffusion(state_dim=cond_dim, action_dim=n_slots,
                                            model=self.denoise, beta_schedule='vp',
                                            denoising_steps=denoising_steps)

    def forward(self, servers, preference, mask2, prior_probs, act_mask=None, det=False):
        """servers [B,67,11], preference [B,2], mask2 [B,11], prior_probs [B,11] -> probs [B,11].

        mask2:    padding mask (合法服务器), 喂编码器清零补零槽。
        act_mask: 可选输出掩码 (= mask2 ∩ 准入掩码); 给了就用它屏蔽动作, 否则退回 mask2。
                  准入掩码只会更严 (去掉预计超时的服务器), 故恒 ⊆ mask2。
        det:      True=确定性反向采样 (推理/eval); False=随机反向 (训练, 保留探索)。
        """
        cond = self.encoder(servers, preference, mask2)                      # 编码一次 (用 padding mask)
        cond_prior = prior_probs if self.use_prior_cond else None             # P2: prior 也当条件喂网络
        raw = self.diffusion.p_sample_loop(cond, prior_probs, prior=cond_prior,
                                           start_mode=self.start_mode, det=det)  # 去噪 T 步, 复用 cond
        probs = F.softmax(raw, dim=1)
        out_mask = mask2 if act_mask is None else act_mask
        probs = probs * out_mask                                             # 屏蔽无效/预计超时服务器
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)             # 重归一化
        return probs


# --------------------------------------------------------------
# 消融用: 普通 MLP actor (无扩散, 无 prior 暖启动) —— 与 FDActor 同接口
# --------------------------------------------------------------
class MLPActor(nn.Module):
    """公平对照: 同一个 set-conv 编码器 + MLP 头直接出 logits。
    prior_probs 参数接受但忽略 (MLP 不暖启动)。其余 (编码器/mask/容量) 与 FDActor 一致,
    唯一差别 = 扩散去噪 vs 一次前向, 故能干净隔离"扩散+prior反馈"的贡献。"""

    def __init__(self, conv_ch=256, cond_dim=256, n_slots=N_SLOTS, hidden=256, **kw):
        super().__init__()
        self.n_slots = n_slots
        self.encoder = SetConvEncoder(in_ch=SERVER_FEAT, conv_ch=conv_ch,
                                      cond_dim=cond_dim, n_slots=n_slots)
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_slots),
        )

    def forward(self, servers, preference, mask2, prior_probs, act_mask=None, det=False):
        cond = self.encoder(servers, preference, mask2)  # det 忽略 (MLP 无随机反向)
        probs = F.softmax(self.head(cond), dim=1)
        out_mask = mask2 if act_mask is None else act_mask
        probs = probs * out_mask
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        return probs


# --------------------------------------------------------------
# dict obs (numpy) -> torch tensors
# --------------------------------------------------------------
def obs_to_tensors(obs, device):
    """单个 obs dict 或 obs dict 列表 -> (servers, preference, mask2) tensors。"""
    if isinstance(obs, dict):
        servers = obs['servers'][None]
        pref = obs['preference'][None]
        mask2 = obs['mask2'][None]
    else:
        servers = np.stack([o['servers'] for o in obs])
        pref = np.stack([o['preference'] for o in obs])
        mask2 = np.stack([o['mask2'] for o in obs])
    return (torch.as_tensor(servers, dtype=torch.float32, device=device),
            torch.as_tensor(pref, dtype=torch.float32, device=device),
            torch.as_tensor(mask2, dtype=torch.float32, device=device))


def uniform_prior(mask2_np):
    """合法服务器上的均匀分布 prior (episode 起始 / last_probs 不可用时的回退)。"""
    m = np.asarray(mask2_np, dtype=np.float32)
    p = m / (m.sum() + 1e-8)
    return p


# --------------------------------------------------------------
# 形状冒烟测试
# --------------------------------------------------------------
if __name__ == '__main__':
    import os
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from env_gmorl_sla import MEC_Env

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', dev)
    actor = FDActor(denoising_steps=3).to(dev)
    n_params = sum(p.numel() for p in actor.parameters())
    print('FDActor params = %.2fM' % (n_params / 1e6))

    env = MEC_Env(conf_name='multi-part', w=0.3)
    obs = env.reset()
    print('obs servers shape =', obs['servers'].shape, ' mask2 =', obs['mask2'])

    # 单样本推理
    servers, pref, mask2 = obs_to_tensors(obs, dev)
    prior = torch.as_tensor(uniform_prior(obs['mask2'])[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        probs = actor(servers, pref, mask2, prior)
    p = probs.cpu().numpy()[0]
    print('probs =', np.round(p, 4))
    print('  sum = %.6f (应=1)' % p.sum())
    invalid_mass = float(p[np.asarray(obs['mask2']) == 0].sum())
    print('  无效槽概率和 = %.2e (应≈0)' % invalid_mass)

    # batch 推理 (训练时用)
    obs_list = [env.reset() for _ in range(8)]
    servers, pref, mask2 = obs_to_tensors(obs_list, dev)
    priors = torch.as_tensor(np.stack([uniform_prior(o['mask2']) for o in obs_list]),
                             dtype=torch.float32, device=dev)
    with torch.no_grad():
        probs = actor(servers, pref, mask2, priors)
    print('batch probs shape =', tuple(probs.shape), ' 每行 sum =', np.round(probs.sum(1).cpu().numpy(), 4))
    print('[ok] FDActor 形状冒烟通过')
