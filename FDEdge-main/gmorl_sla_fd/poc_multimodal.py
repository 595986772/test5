"""POC (路A 去风险第一步): 连续动作下 扩散 vs 高斯-MLP 的表征能力。

核心 claim 要测的必要条件:
  离散任务里扩散吃亏的"动作尖锐度地板"(迭代采样噪声), 在连续空间应反转成优势 ——
  当**最优动作分布是多峰**的(连续分数卸载中"两种同样好的分配方式"),
  扩散能表征整个多峰分布; 而高斯-MLP(= 标准连续 SAC actor 的分布族, 单峰高斯)
  只能覆盖一个峰、或为最小化 NLL 把质量摊到两峰中间的"谷底"(空区域)。

  若此处扩散都赢不了高斯-MLP, 说明扩散在我们的设置里连"必要条件"都不满足,
  路A 可直接否决, 省掉重写整个 MEC env 的成本。这是最小、最快的 go/no-go。

任务(受控探针, 非完整 env): 条件二维分布 a*|s 双峰。
  - s ∈ [-1,1]^2 当上下文; 两个峰 m_{1,2}(s) 关于原点对称, 方向随 s[0] 旋转(测条件化)。
  - a* = m_k(s) + σ0·ε, k∈{0,1} 各 50%。  → "两种同样好的分配"。
两个模型同等容量/同等训练步:
  - GaussMLP:  f(s)->(μ, logσ), 对 a* 做 NLL (高斯族, 标准 SAC 连续 actor)。
  - Diffusion: 条件 DDPM ε-网络, 去噪分数匹配 (扩散原生训练)。
判据: MMD(模型样本 vs 真值, 越小越好) + 双峰覆盖率 min(cov1,cov2)(越大越好) + 散点图。

用法: python poc_multimodal.py
产物: result2/poc/poc_multimodal.png + 控制台指标。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = 0.7      # 两峰半径
SIG0 = 0.07  # 每峰内部噪声


# ---------------- 真值条件分布 ----------------
def modes(s):
    """s: [B,2] -> 两个峰 m1,m2: [B,2]。方向随 s[0] 旋转, 关于原点对称。"""
    theta = np.pi * s[:, 0]                       # s[0]∈[-1,1] -> 角度
    d = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)   # [B,2]
    return d, -d

def sample_true(B):
    s = np.random.uniform(-1, 1, size=(B, 2)).astype(np.float32)
    m1, m2 = modes(s)
    pick = (np.random.rand(B) < 0.5)[:, None]
    m = np.where(pick, m1, m2)
    a = (m + SIG0 * np.random.randn(B, 2)).astype(np.float32)
    return s, a


# ---------------- 时间嵌入 ----------------
class TimeEmb(nn.Module):
    def __init__(self, dim=16):
        super().__init__(); self.dim = dim
    def forward(self, t):                          # t: [B] long
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


# ---------------- 高斯-MLP actor (标准连续 SAC 分布族) ----------------
class GaussMLP(nn.Module):
    def __init__(self, s_dim=2, a_dim=2, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(s_dim, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, a_dim)
        self.logstd = nn.Linear(h, a_dim)
    def forward(self, s):
        z = self.body(s)
        return self.mu(z), self.logstd(z).clamp(-5, 2)
    def nll(self, s, a):
        mu, logstd = self(s)
        var = (2 * logstd).exp()
        return (0.5 * ((a - mu) ** 2 / var + 2 * logstd + np.log(2 * np.pi))).sum(1).mean()
    @torch.no_grad()
    def sample(self, s):
        mu, logstd = self(s)
        return mu + logstd.exp() * torch.randn_like(mu)


# ---------------- 条件 DDPM 扩散 actor ----------------
class EpsNet(nn.Module):
    def __init__(self, s_dim=2, a_dim=2, t_dim=16, h=128):
        super().__init__()
        self.temb = TimeEmb(t_dim)
        self.net = nn.Sequential(nn.Linear(a_dim + t_dim + s_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, a_dim))
    def forward(self, a_t, t, s):
        return self.net(torch.cat([a_t, self.temb(t), s], dim=1))

class Diffusion(nn.Module):
    def __init__(self, T=50, s_dim=2, a_dim=2):
        super().__init__()
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        acp = torch.cumprod(alphas, 0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('acp', acp)
        self.register_buffer('acp_prev', torch.cat([torch.ones(1), acp[:-1]]))
        self.eps = EpsNet(s_dim, a_dim)
    def loss(self, s, a0):
        B = a0.size(0)
        t = torch.randint(0, self.T, (B,), device=a0.device)
        acp = self.acp[t][:, None]
        noise = torch.randn_like(a0)
        a_t = acp.sqrt() * a0 + (1 - acp).sqrt() * noise
        return F.mse_loss(self.eps(a_t, t, s), noise)
    @torch.no_grad()
    def sample(self, s):
        B = s.size(0)
        a = torch.randn(B, 2, device=s.device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=s.device, dtype=torch.long)
            eps = self.eps(a, t, s)
            acp = self.acp[i]; beta = self.betas[i]; alpha = self.alphas[i]
            mean = (a - beta / (1 - acp).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                var = beta * (1 - self.acp_prev[i]) / (1 - acp)
                a = mean + var.sqrt() * torch.randn_like(a)
            else:
                a = mean
        return a


# ---------------- 训练 ----------------
def train(model, steps, is_diff, bs=512, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(steps):
        s, a = sample_true(bs)
        s = torch.as_tensor(s, device=DEV); a = torch.as_tensor(a, device=DEV)
        loss = model.loss(s, a) if is_diff else model.nll(s, a)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % (steps // 5) == 0 or it == steps - 1:
            print('  step %4d  loss=%.4f' % (it, loss.item()), flush=True)
    return model


# ---------------- 指标 ----------------
def mmd(x, y, sigmas=(0.05, 0.1, 0.2, 0.5)):
    """无偏 MMD^2 (多带宽RBF), x,y: [N,2] numpy。"""
    def k(a, b):
        d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s * s)) for s in sigmas)
    xx, yy, xy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    np.fill_diagonal(xx, 0); np.fill_diagonal(yy, 0)
    return xx.sum() / (n * (n - 1)) + yy.sum() / (m * (m - 1)) - 2 * xy.mean()

def mode_coverage(model, is_diff, n_s=40, per=400, tau=0.25):
    """对若干固定 s, 各采 per 个动作, 看两峰是否都被覆盖。返回平均 min(cov1,cov2)。"""
    covs = []
    for _ in range(n_s):
        s0 = np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32)
        m1, m2 = modes(s0)
        s_rep = torch.as_tensor(np.repeat(s0, per, axis=0), device=DEV)
        a = model.sample(s_rep).cpu().numpy()
        d1 = np.linalg.norm(a - m1, axis=1); d2 = np.linalg.norm(a - m2, axis=1)
        cov1 = (d1 < tau).mean(); cov2 = (d2 < tau).mean()
        covs.append(min(cov1, cov2))
    return float(np.mean(covs))


def main():
    print('device =', DEV, '\n')
    STEPS = 4000
    print('[训练 GaussMLP]')
    g = train(GaussMLP().to(DEV), STEPS, is_diff=False)
    print('[训练 Diffusion]')
    d = train(Diffusion(T=50).to(DEV), STEPS, is_diff=True)

    # 整体分布 MMD (新采 s, 各模型采 a, 与真值比)
    s_np, a_true = sample_true(3000)
    s_t = torch.as_tensor(s_np, device=DEV)
    a_g = g.sample(s_t).cpu().numpy()
    a_d = d.sample(s_t).cpu().numpy()
    mmd_g = mmd(a_g, a_true); mmd_d = mmd(a_d, a_true)
    cov_g = mode_coverage(g, False); cov_d = mode_coverage(d, True)

    print('\n=== 指标 (越好的方向已标注) ===')
    print('%-12s %12s %16s' % ('model', 'MMD↓(vs真值)', '双峰覆盖↑ min(c1,c2)'))
    print('%-12s %12.4f %16.3f' % ('GaussMLP', mmd_g, cov_g))
    print('%-12s %12.4f %16.3f' % ('Diffusion', mmd_d, cov_d))
    win_mmd = 'Diffusion' if mmd_d < mmd_g else 'GaussMLP'
    win_cov = 'Diffusion' if cov_d > cov_g else 'GaussMLP'
    print('\n  MMD 更接近真值: %s' % win_mmd)
    print('  双峰覆盖更好  : %s' % win_cov)
    verdict = (mmd_d < mmd_g) and (cov_d > cov_g + 0.1)
    print('\n  >>> POC 判决: %s' %
          ('扩散在连续多峰上有明确表征优势 -> 路A 值得继续 (建最小连续卸载env)'
           if verdict else
           '扩散未显出多峰优势 -> 路A 必要条件不满足, 应否决, 别重写env'))

    # ---- 散点图: 3 个固定 s, true/Gauss/Diff ----
    test_s = np.array([[-0.6, 0.0], [0.0, 0.0], [0.6, 0.0]], dtype=np.float32)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    col_titles = ['True p(a|s)', 'GaussMLP', 'Diffusion']
    for r, s0 in enumerate(test_s):
        m1, m2 = modes(s0[None])
        s_rep = torch.as_tensor(np.repeat(s0[None], 800, axis=0), device=DEV)
        # true
        pick = (np.random.rand(800) < 0.5)[:, None]
        mt = np.where(pick, m1, m2)
        at = mt + SIG0 * np.random.randn(800, 2)
        ag = g.sample(s_rep).cpu().numpy()
        ad = d.sample(s_rep).cpu().numpy()
        for c, (data, name) in enumerate([(at, 'True'), (ag, 'Gauss'), (ad, 'Diff')]):
            ax = axes[r, c]
            ax.scatter(data[:, 0], data[:, 1], s=4, alpha=0.3, c='C0')
            ax.scatter(*m1[0], c='r', marker='x', s=80); ax.scatter(*m2[0], c='r', marker='x', s=80)
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal'); ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel('s[0]=%.1f' % s0[0])
    fig.suptitle('连续多峰动作表征: 红×=真峰. 高斯塌单峰/落谷底, 扩散应覆盖双峰', fontsize=11)
    fig.tight_layout()
    png = 'result2/poc/poc_multimodal.png'
    fig.savefig(png, dpi=130)
    print('\n[saved] %s' % png)


if __name__ == '__main__':
    main()
