"""连续分数卸载的 agent (路A 主体, Block 2)。

两个 actor 同架构对照, 测"连续多峰上扩散 vs 高斯":
  - DiffActor (Diffusion-QL 式): 条件 DDPM 在 logit 空间去噪, softmax 出分配 a∈Δ^N。
      训练 = −Q(s, a~π) (策略改进, 反传过采样链) + η·BC去噪正则 (拟合 buffer 动作分布)。
      **丢显式熵** (扩散密度不可解析, 见理论分析); 探索靠采样随机性 + BC 多样性。
  - GaussActor (标准连续 SAC): cond→(μ,logσ), x=μ+σε, a=softmax(x)。
      训练 = −Q(s,a) − α·H(高斯熵)。**单峰分布族 = 预期在多峰上塌/落谷底**。
两者共享: 同样的 MLP 编码器架构 + 同一个向量 critic (输出 [n_obj]=[Q_T,Q_E,Q_C])。
标量化 ω-无关 SLA: scalar = w·Q_T + (1−w)·Q_E + Q_C。

contextual bandit (γ=0, 每任务独立终止): critic target = 即时 r_vec, 直接回归。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_OBJ = 3


# ---------------- 工具 ----------------
def flat_state(servers, omega):
    """servers [B,N,F] (F=3: [f,q,warm]), omega [B] -> [B, N*F+1]。"""
    B = servers.shape[0]
    return torch.cat([servers.reshape(B, -1), omega.reshape(B, 1)], dim=1)

def alloc_to_latent(a, eps=1e-6):
    """分配 a∈Δ^N -> logit latent (中心化, 裁剪), softmax(x)≈a。"""
    x = torch.log(a.clamp_min(eps))
    return (x - x.mean(dim=1, keepdim=True)).clamp(-8, 8)


def sparsemax(z, dim=1):
    """sparsemax (Martins&Astudillo 2016): z∈R^N -> 单纯形上**带精确零**的稀疏点。
    = 到单纯形的欧氏投影; 故 z 若已在单纯形上则恒等 (-> 扩散隐空间=分配空间, BC 用 x0=a)。
    可微 (sort/gather/clamp 链), 支撑集 = active server 集 z_i=1。"""
    zs, _ = torch.sort(z, dim=dim, descending=True)
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shp = [1] * z.dim(); shp[dim] = -1; rng = rng.view(shp)
    cssv = zs.cumsum(dim) - 1
    cond = (1 + rng * zs) > cssv                       # 哪些进支撑集
    k = cond.to(z.dtype).sum(dim, keepdim=True)         # 支撑集大小
    tau = cssv.gather(dim, (k.long() - 1).clamp_min(0)) / k
    return torch.clamp(z - tau, min=0)


class OmegaFiLM(nn.Module):
    """ω-FiLM 条件: 把偏好 ω(标量)做成对 cond 的乘加调制 cond·(1+γ(ω))+β(ω)。
    目的: ω 原本只是 N*F+1 维输入里的 1 维, 被编码器淹没 -> actor ω-盲(K̄ 不随 ω 变)。
    乘性调制给 ω 一条强通道 -> actor 能学到"随 ω 切支撑"(w=0 稀疏绿 / w=1 全展开)。
    末层零初始化 -> 起步=恒等(不扰动 warmstart), 按需学习用 ω。"""
    def __init__(self, cond_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, 2 * cond_dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, cond, w):
        g, b = self.net(w).chunk(2, dim=1)
        return cond * (1 + g) + b


class TimeEmb(nn.Module):
    def __init__(self, dim=16):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        fr = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        ang = t.float()[:, None] * fr[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


# ---------------- 编码器 ----------------
class Enc(nn.Module):
    def __init__(self, state_dim, cond_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, cond_dim), nn.ReLU(),
                                 nn.Linear(cond_dim, cond_dim), nn.ReLU())
    def forward(self, s):
        return self.net(s)


# ---------------- 向量 critic: (cond, a) -> [B, n_obj] ----------------
class VCritic(nn.Module):
    def __init__(self, cond_dim, n_act, h=128, n_obj=N_OBJ):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + n_act, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_obj))
    def forward(self, cond, a):
        return self.net(torch.cat([cond, a], dim=1))


# ---------------- 扩散 actor (Diffusion-QL) ----------------
class DiffActor(nn.Module):
    def __init__(self, state_dim, n_act, cond_dim=128, T=5, t_dim=16, h=128, sparse=False, omega_film=False):
        super().__init__()
        self.n_act = n_act; self.T = T; self.sparse = sparse   # sparse=True -> sparsemax 输出(显式开关)
        self.enc = Enc(state_dim, cond_dim)
        self.film = OmegaFiLM(cond_dim) if omega_film else None  # ω-FiLM: 让支撑随 ω 切, 解 ω-盲
        self.temb = TimeEmb(t_dim)
        self.eps = nn.Sequential(nn.Linear(n_act + t_dim + cond_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_act))
        betas = torch.linspace(1e-4, 0.02, T)
        acp = torch.cumprod(1 - betas, 0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('acp', acp)
        self.register_buffer('acp_prev', torch.cat([torch.ones(1), acp[:-1]]))

    def _eps(self, x, t, cond):
        return self.eps(torch.cat([x, self.temb(t), cond], dim=1))

    def _p_sample(self, x, i, cond, det=False):
        t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
        eps = self._eps(x, t, cond)
        acp = self.acp[i]; beta = self.betas[i]; alpha = self.alphas[i]
        mean = (x - beta / (1 - acp).sqrt() * eps) / alpha.sqrt()
        if i > 0 and not det:
            var = beta * (1 - self.acp_prev[i]) / (1 - acp)
            return mean + var.sqrt() * torch.randn_like(x)
        return mean

    def sample(self, s, prior_latent=None, det=False):
        """s [B,state_dim] -> 分配 a∈Δ^N (可反传, 供策略改进)。"""
        cond = self.enc(s)
        if self.film is not None:
            cond = self.film(cond, s[:, -1:])     # ω 在 state 末维, 取出做 FiLM 调制
        x = prior_latent if prior_latent is not None else torch.randn(s.size(0), self.n_act, device=s.device)
        for i in reversed(range(self.T)):
            x = self._p_sample(x, i, cond, det=det)
        return sparsemax(x, dim=1) if self.sparse else F.softmax(x, dim=1)

    def bc_loss(self, s, a):
        """去噪分数匹配 (拟合 buffer 动作分布) = Diffusion-QL 的 BC 正则。"""
        cond = self.enc(s)
        if self.film is not None:
            cond = self.film(cond, s[:, -1:])
        x0 = a if self.sparse else alloc_to_latent(a)   # sparsemax(a)=a -> 隐空间即分配空间, 直接 x0=a
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        acp = self.acp[t][:, None]
        noise = torch.randn_like(x0)
        x_t = acp.sqrt() * x0 + (1 - acp).sqrt() * noise
        return F.mse_loss(self._eps(x_t, t, cond), noise)


# ---------------- 高斯 actor (标准连续 SAC) ----------------
class GaussActor(nn.Module):
    def __init__(self, state_dim, n_act, cond_dim=128, h=128, sparse=False, omega_film=False):
        super().__init__()
        self.sparse = sparse
        self.enc = Enc(state_dim, cond_dim)
        self.film = OmegaFiLM(cond_dim) if omega_film else None  # 同款 ω-FiLM(两 actor 同条件=公平对比)
        self.mu = nn.Linear(cond_dim, n_act)
        self.logstd = nn.Linear(cond_dim, n_act)
        nn.init.constant_(self.logstd.bias, -1.0)        # 起步合理方差, 防 logstd 顶爆
    def forward(self, s):
        z = self.enc(s)
        if self.film is not None:
            z = self.film(z, s[:, -1:])
        return self.mu(z), self.logstd(z).clamp(-5, 0.5)  # std≤1.65, 防高方差退化乱集中
    def sample(self, s, prior_latent=None, det=False):
        mu, logstd = self(s)
        x = mu if det else mu + logstd.exp() * torch.randn_like(mu)
        return sparsemax(x, dim=1) if self.sparse else F.softmax(x, dim=1)
    def entropy(self, s):
        _, logstd = self(s)
        return (0.5 + 0.5 * np.log(2 * np.pi) + logstd).sum(1)   # 高斯解析熵(pre-softmax)


# ---------------- replay buffer ----------------
class Buf:
    def __init__(self, cap, n_srv, n_obj=N_OBJ):
        self.cap = cap; self.i = 0; self.full = False
        self.srv = np.zeros((cap, n_srv, 3), np.float32)   # 每服务器 [f, q, warm]
        self.om = np.zeros(cap, np.float32)
        self.a = np.zeros((cap, n_srv), np.float32)
        self.r = np.zeros((cap, n_obj), np.float32)
    def add(self, srv, om, a, r):
        i = self.i
        self.srv[i] = srv; self.om[i] = om; self.a[i] = a; self.r[i] = r
        self.i = (i + 1) % self.cap
        if self.i == 0: self.full = True
    def size(self):
        return self.cap if self.full else self.i
    def sample(self, n):
        idx = np.random.randint(0, self.size(), n)
        return (self.srv[idx], self.om[idx], self.a[idx], self.r[idx])


# ---------------- agent ----------------
class FracAgent:
    def __init__(self, n_srv, actor_type='diffusion', cond_dim=128, T=5,
                 lr=3e-4, bc_eta=1.0, alpha=0.05, sla_lambda=1.0,
                 start_mode='prior', sparse=False, feat_dim=3, omega_film=False, device='cpu'):
        self.N = n_srv; self.actor_type = actor_type; self.dev = device
        self.bc_eta = bc_eta; self.alpha = alpha; self.sla_lambda = sla_lambda
        self.start_mode = start_mode; self.feat_dim = feat_dim
        sd = n_srv * feat_dim + 1                # 每服务器 feat_dim 特征 + ω (frac=3, hetero=5)
        if actor_type == 'diffusion':
            self.actor = DiffActor(sd, n_srv, cond_dim, T, sparse=sparse, omega_film=omega_film).to(device)
        else:
            self.actor = GaussActor(sd, n_srv, cond_dim, sparse=sparse, omega_film=omega_film).to(device)
        self.enc_c = Enc(sd, cond_dim).to(device)
        self.c1 = VCritic(cond_dim, n_srv).to(device)
        self.c2 = VCritic(cond_dim, n_srv).to(device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(list(self.enc_c.parameters()) +
                                      list(self.c1.parameters()) + list(self.c2.parameters()), lr=lr)

    def _prior_latent(self, prior_a):
        if self.start_mode != 'prior' or prior_a is None:
            return None
        return alloc_to_latent(torch.as_tensor(prior_a[None], dtype=torch.float32, device=self.dev))

    @torch.no_grad()
    def act(self, obs, prior_a=None, det=False):
        s = flat_state(torch.as_tensor(obs['servers'][None], device=self.dev),
                       torch.as_tensor(np.array([obs['omega']]), device=self.dev))
        pl = self._prior_latent(prior_a) if self.actor_type == 'diffusion' else None
        a = self.actor.sample(s, prior_latent=pl, det=det)[0].cpu().numpy()
        return a.astype(np.float32)

    def _scalarize(self, qv, w):
        return w * qv[:, 0] + (1 - w) * qv[:, 1] + self.sla_lambda * qv[:, 2]

    def update(self, batch):
        srv, om, a, r = [torch.as_tensor(x, device=self.dev) for x in batch]
        s = flat_state(srv, om)
        # --- critic: γ=0 bandit, target = 即时 r_vec ---
        cond_c = self.enc_c(s)
        q1, q2 = self.c1(cond_c, a), self.c2(cond_c, a)
        c_loss = F.mse_loss(q1, r) + F.mse_loss(q2, r)
        self.opt_c.zero_grad(); c_loss.backward(); self.opt_c.step()
        # --- actor: 最大化标量化 Q ---
        # ⚠️P1-4: 这里从**无 prior 随机起点**采样改进, 但 act() 部署用 prior -> 训练/部署不是同一条
        #   prior-条件策略 (扩散反向起点不同)。这是 train_frac.py 被弃用的原因之一; 主线
        #   train_frac_seq.py 已修 (SeqBuf 存 x/nx, 用 pl/npl 喂当前/下一 actor)。本方法仅供弃用脚本。
        pl = self._prior_latent(None)  # 训练期暖启动用 randn (无逐样本 prior); 简化
        a_pi = self.actor.sample(s)
        cond_cd = self.enc_c(s).detach()
        qv = torch.min(self.c1(cond_cd, a_pi), self.c2(cond_cd, a_pi))
        q_scalar = self._scalarize(qv, om)
        if self.actor_type == 'diffusion':
            bc = self.actor.bc_loss(s, a)
            a_loss = -q_scalar.mean() + self.bc_eta * bc
            extra = float(bc.item())
        else:
            H = self.actor.entropy(s)
            a_loss = -q_scalar.mean() - self.alpha * H.mean()
            extra = float(H.mean().item())
        self.opt_a.zero_grad(); a_loss.backward(); self.opt_a.step()
        return {'c_loss': float(c_loss.item()), 'q': float(q_scalar.mean().item()), 'extra': extra}

    def save(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pt'))
        torch.save(self.c1.state_dict(), os.path.join(path, 'c1.pt'))
        torch.save(self.enc_c.state_dict(), os.path.join(path, 'enc_c.pt'))

    def load(self, path):
        import os
        sd = torch.load(os.path.join(path, 'actor.pt'), map_location=self.dev)
        # 维度守卫: 旧代码 actor 输入是 N*2+1 (无 warm 通道), 当前是 N*3+1。直接 load_state_dict
        #   会抛 cryptic 的 size mismatch。这里先比对 encoder 首层输入维度, 给出可执行的报错。
        exp = self.N * self.feat_dim + 1
        for k, v in sd.items():
            if v.ndim == 2 and 'enc' in k:
                got = v.shape[1]
                if got != exp:
                    raise ValueError(
                        '[ckpt 维度不兼容] %s 的 actor 首层输入=%d, 当前代码需要 %d (=N*feat_dim+1)。\n'
                        '该 checkpoint 由旧代码 (无 warm 状态) 训练, 已不可复现/评估。请用当前代码从头重训:\n'
                        '  python train_frac_direct.py --actor %s --tag <new_tag>\n'
                        '  python train_frac_seq.py    --actor %s --tag <new_tag> --warmstart <new_tag>'
                        % (path, got, exp, self.actor_type, self.actor_type))
                break
        self.actor.load_state_dict(sd)


# ---------------- 冒烟 ----------------
if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    from env_frac_offload import FracOffloadEnv
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for at in ['diffusion', 'mlp']:
        env = FracOffloadEnv(n_servers=5, w=0.5)
        ag = FracAgent(5, actor_type=at, device=dev)
        buf = Buf(2000, 5)
        obs = env.reset(); prior = np.ones(5) / 5
        for t in range(200):
            a = ag.act(obs, prior)
            nobs, r, done, info = env.step(a)
            buf.add(obs['servers'], obs['omega'], a, info['r_vec'])
            obs = nobs if not done else env.reset(); prior = a
            if buf.size() >= 64:
                log = ag.update(buf.sample(64))
        print('[%s] act sum=%.3f  update ok: c_loss=%.4f q=%.3f extra=%.3f'
              % (at, a.sum(), log['c_loss'], log['q'], log['extra']))
    print('[ok] frac_agent 冒烟通过')
