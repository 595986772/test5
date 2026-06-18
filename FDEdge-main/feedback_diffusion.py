













import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract_into_tensor,
    Losses
)


class Diffusion(nn.Module):
    """
    反馈扩散模型 (Feedback Diffusion Model)
    基于 DDPM 框架实现，用作 SAC 算法中的 Actor 网络。
    通过多步去噪过程从噪声中生成任务调度的动作概率分布。
    """

    def __init__(self, state_dim, action_dim, model, beta_schedule='linear', denoising_steps=5,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim    # 系统状态维度
        self.action_dim = action_dim  # 动作空间维度（即边缘服务器数量）
        self.model = model            # 噪声预测网络（MLP_PolicyNet）

        # 根据指定的调度策略生成 beta 序列（噪声调度参数）
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(denoising_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(denoising_steps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(denoising_steps)

        alphas = 1. - betas                                    # alpha_t = 1 - beta_t，每步的信号保留率
        alphas_cumprod = torch.cumprod(alphas, axis=0)         # alpha 的累积乘积 ᾱ_t，表示从起始到第 t 步的总信号保留率
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # 前一步的累积乘积 ᾱ_{t-1}

        self.n_timesteps = int(denoising_steps)  # 去噪总步数
        self.clip_denoised = clip_denoised       # 是否裁剪去噪结果到 [-1, 1]
        self.predict_epsilon = predict_epsilon   # 模型是否预测噪声（True）还是直接预测 x_0（False）

        # 将扩散过程所需的常量注册为 buffer（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # ===== 前向扩散过程 q(x_t | x_0) 所需的系数 =====
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))                # √ᾱ_t
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # √(1-ᾱ_t)
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))    # log(1-ᾱ_t)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))      # 1/√ᾱ_t
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # √(1/ᾱ_t - 1)

        # ===== 后验分布 q(x_{t-1} | x_t, x_0) 所需的系数 =====
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 后验方差
        self.register_buffer('posterior_variance', posterior_variance)

        # 对后验方差取对数并裁剪（防止 t=0 时方差为 0 导致 log 未定义）
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        # 后验均值的两个系数
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()  # 训练用的损失函数

    def predict_start_from_noise(self, x_t, t, noise):
        """
        从噪声预测中恢复 x_0（原始信号）
        公式：x_0 = (1/√ᾱ_t) * x_t - √(1/ᾱ_t - 1) * noise
        """
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_t_v(self, x_t, t, noise):
        """
        备用的 x_0 预测函数，支持两种模式：
        - predict_epsilon=True：模型输出噪声，由噪声反推 x_0
        - predict_epsilon=False：模型直接输出 x_0
        """
        if self.predict_epsilon:
            return (
                    extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        后验均值 = coef1 * x_0 + coef2 * x_t
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, prior=None):
        """
        计算反向去噪过程 p(x_{t-1} | x_t) 的均值和方差
        1. 用模型预测噪声
        2. 由噪声恢复 x_0
        3. 由 x_0 和 x_t 计算后验分布
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=self._call_model(x, t, s, prior))
        if self.clip_denoised:
            x_recon.clamp_(-1, 1)  # 可选：将恢复的 x_0 裁剪到 [-1, 1]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def _call_model(self, x, t, s, prior):
        """统一的 model 调用: 子类 PCPolicyNet 接收 prior_latent, 旧 model 不接收."""
        if prior is None:
            return self.model(x, t, s)
        try:
            return self.model(x, t, s, prior)
        except TypeError:
            return self.model(x, t, s)

    def p_sample(self, x, t, s, prior=None):
        """
        单步反向去噪采样：从 x_t 采样得到 x_{t-1}
        公式：x_{t-1} = 均值 + 非零掩码 * σ * 随机噪声
        当 t=0 时不再添加噪声（nonzero_mask=0）
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, prior=prior)
        probs = torch.randn_like(x)  # 采样随机噪声
        # 当 t=0 时掩码为 0，不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * probs

    def p_sample_loop(self, state, latent_action_probs, prior=None):
        """
        完整的反向去噪循环：从纯噪声开始，经过 n_timesteps 步去噪，生成最终的动作表示
        输入：
          - state: 系统状态
          - latent_action_probs: 历史潜在动作概率（用于反馈机制）
          - prior: 可选 [B, Emax] episode 级动作先验, 透传给 PCPolicyNet
        """
        device = self.betas.device
        batch_size = state.shape[0]
        # 从标准高斯噪声开始（与潜在动作概率同形状）
        x = torch.randn_like(latent_action_probs)
        # 从 t=T-1 逐步去噪到 t=0
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, prior=prior)
        return x

    def sample(self, state, latent_action_probs, prior=None, *args, **kwargs):
        """
        采样动作概率分布：执行完整去噪循环后，用 softmax 归一化为概率分布
        """
        action = self.p_sample_loop(state, latent_action_probs, prior=prior, *args, **kwargs)
        return F.softmax(action, dim=-1)

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：对原始数据 x_0 加噪到第 t 步
        公式：x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε，其中 ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, prior=None, weights=1.0):
        """
        计算扩散模型的训练损失
        1. 对 x_start 加噪到第 t 步得到 x_noisy
        2. 用模型从 x_noisy 预测噪声（或 x_0）
        3. 计算预测值与真实值之间的加权损失
        """
        noise = torch.randn_like(x_start)                      # 采样真实噪声
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 前向加噪
        x_recon = self._call_model(x_noisy, t, state, prior)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise, weights)   # 预测噪声模式：与真实噪声比较
        else:
            return self.loss_fn(x_recon, x_start, weights)  # 预测 x_0 模式：与原始信号比较

    def loss(self, x, state, prior=None, weights=1.0):
        """计算一个 batch 的训练损失，随机采样时间步 t"""
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, prior=prior, weights=weights)

    def forward(self, state, latent_action_probs, prior=None, *args, **kwargs):
        """前向传播：给定系统状态和历史潜在动作概率，生成动作概率分布"""
        return self.sample(state, latent_action_probs, prior=prior, *args, **kwargs)
