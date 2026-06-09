import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码模块，用于将扩散时间步编码为连续向量表示"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 编码维度

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # 计算频率因子：log(10000) / (half_dim - 1)
        emb = math.log(10000) / (half_dim - 1)
        # 生成不同频率的指数衰减序列
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入时间步与频率相乘，得到相位矩阵
        emb = x[:, None] * emb[None, :]
        # 拼接正弦和余弦编码，输出维度为 dim
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def extract_into_tensor(a, t, x_shape):
    """从一维张量 a 中按索引 t 提取值，并扩展维度以匹配 x_shape 的形状（用于广播）"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    余弦 beta 调度策略
    参考：https://openreview.net/forum?id=-NEXDKk8gZ
    在扩散初期噪声增加缓慢，后期加快，有助于保留更多信号信息
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    # 计算累积乘积 alpha_cumprod，基于余弦函数
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # 由累积乘积反推出每步的 beta 值
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # 将 beta 裁剪到 [0, 0.999] 范围内，防止数值不稳定
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    """线性 beta 调度策略：beta 从 beta_start 线性增长到 beta_end"""
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    """
    VP (Variance Preserving) beta 调度策略
    噪声调度遵循方差保持原则，beta 按二次函数从 b_min 增长到 b_max
    """
    t = np.arange(1, timesteps + 1)
    b_max = 10.   # 最大 beta 参数
    b_min = 0.1   # 最小 beta 参数
    # 计算每步的 alpha 值（信号保留率）
    alpha = np.exp(-b_min / timesteps - 0.5 * (b_max - b_min) * (2 * t - 1) / timesteps ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedLoss(nn.Module):
    """加权损失基类，支持对每个样本施加不同权重"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        loss = self._loss(pred, targ)
        # 将损失乘以权重后取平均
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):
    """加权 L1 损失（绝对值误差）"""

    def _loss(self, predict_values, target_values):
        return torch.abs(predict_values - target_values)


class WeightedL2(WeightedLoss):
    """加权 L2 损失（均方误差）"""

    def _loss(self, predict_values, target_values):
        return F.mse_loss(predict_values, target_values, reduction='none')


# 损失函数映射表，通过字符串选择对应的损失函数类
Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2
}

class EMA:
    """指数移动平均 (Exponential Moving Average)，用于平滑模型参数更新"""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta  # 平滑系数，越接近 1 则旧参数权重越大

    def update_model_average(self, ma_model, current_model):
        """用当前模型参数更新移动平均模型的参数"""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """计算指数移动平均值：old * beta + new * (1 - beta)"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# ==================== MOFD 多目标扩展工具 ====================
# 所有方法 final-eval 共享的 seed 偏移: eval_seed = train_seed * 100000 + SHARED_EVAL_SEED_OFFSET
# 保证不同方法在同一组任务到达 / 信道 / ES 频率上评估,跨方法 HV 才公平可比
SHARED_EVAL_SEED_OFFSET = 99999


def build_preference_set(n):
    """在 2 维偏好单纯形上均匀取点：[(ω_T, ω_E)]，ω_T+ω_E=1"""
    w_T = np.linspace(0.0, 1.0, n)
    w_E = 1.0 - w_T
    return np.stack([w_T, w_E], axis=1)


def pareto_front_2d(points):
    """
    2D 最小化问题的非支配集 (Pareto front)
    输入: points (N,2) —— 两维都越小越好
    输出: 按第一维升序 (→ 第二维降序) 排列的 Pareto 前沿点
    """
    if len(points) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(points, dtype=float)
    idx = np.argsort(pts[:, 0])
    sorted_pts = pts[idx]
    pf = []
    min_y = float('inf')
    for p in sorted_pts:
        if p[1] < min_y - 1e-12:
            pf.append(p)
            min_y = p[1]
    return np.array(pf)


def hypervolume_2d(points, ref):
    """
    2D 最小化问题的超体积 (hypervolume)
    points: (N,2)；ref: (r_x,r_y) 必须 >= 所有点
    """
    pf = pareto_front_2d(points)
    if len(pf) == 0:
        return 0.0
    hv = 0.0
    for i in range(len(pf)):
        x_i, y_i = pf[i]
        x_next = pf[i + 1][0] if i + 1 < len(pf) else ref[0]
        w = max(x_next - x_i, 0.0)
        h = max(ref[1] - y_i, 0.0)
        hv += w * h
    return float(max(hv, 0.0))


def masked_softmax(logits, mask, dim=-1, eps=1e-12):
    """
    对 logits 做带掩码的 softmax：mask=0 的位置概率直接置 0 后重新归一化。
    logits / mask 同形状，mask 可以是 bool 或 0/1 浮点。
    """
    mask_bool = mask > 0.5 if mask.dtype != torch.bool else mask
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask_bool, neg_inf)
    probs = F.softmax(masked_logits, dim=dim)
    probs = probs.masked_fill(~mask_bool, 0.0)
    s = probs.sum(dim=dim, keepdim=True) + eps
    return probs / s
