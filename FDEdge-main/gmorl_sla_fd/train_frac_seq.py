"""序贯连续分数卸载训练 (路A, faithful 版): actor-critic + γ Bellman + 队列跨任务累积。

vs bandit (train_frac_direct): 序贯 env 队列跨任务累积 -> 决策有长期后果 -> 必须 critic 学长期账。
**鸡生蛋破法**: 用已训好的 bandit 扩散策略**热启动 actor 权重**(--warmstart) -> 一上来策略就会
"集中+铺开", buffer 立刻有多样动作、策略不差 -> critic 快速学到位, 不再死锁。

critic: 向量 Q[n_obj] on 原始 state, twin + target net, target = r_vec + γ(1-done)·min(tc1,tc2)(s',a')。
actor: 最大化标量化 min-Q; 扩散 +BC正则, 高斯 +熵。

用法:
  python train_frac_seq.py --actor diffusion --tag seq_diff --warmstart diff_l3 --episodes 300
  python train_frac_seq.py --actor mlp       --tag seq_gauss --warmstart gauss_l3 --episodes 300
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse, json, copy
import numpy as np
import torch
import torch.nn.functional as F
from env_frac_offload import FracOffloadEnv
from env_hetero_offload import HeteroOffloadEnv
from env_dyn_offload import DynOffloadEnv
from frac_agent import DiffActor, GaussActor, VCritic, flat_state, alloc_to_latent

RESULT = 'result2/frac'


class SeqBuf:
    def __init__(self, cap, N, feat_dim=3, n_obj=3):
        self.cap = cap; self.i = 0; self.full = False
        sd = N * feat_dim + 1                     # 每服务器 feat_dim 特征 + ω
        self.s = np.zeros((cap, sd), np.float32)
        self.x = np.zeros((cap, N), np.float32)   # feedback prior (上一动作/起始均分) = 扩散反向起点
        self.a = np.zeros((cap, N), np.float32)
        self.r = np.zeros((cap, n_obj), np.float32)
        self.ns = np.zeros((cap, sd), np.float32)
        self.nx = np.zeros((cap, N), np.float32)  # next-prior = 本步动作 a (= s' 时的 prior)
        self.d = np.zeros(cap, np.float32)
    def add(self, s, x, a, r, ns, nx, d):
        i = self.i
        self.s[i] = s; self.x[i] = x; self.a[i] = a; self.r[i] = r
        self.ns[i] = ns; self.nx[i] = nx; self.d[i] = d
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0
    def size(self): return self.cap if self.full else self.i
    def sample(self, n):
        idx = np.random.randint(0, self.size(), n)
        return (self.s[idx], self.x[idx], self.a[idx], self.r[idx], self.ns[idx], self.nx[idx], self.d[idx])


def obs_state(obs, dev):
    return flat_state(torch.as_tensor(obs['servers'][None], device=dev),
                      torch.as_tensor(np.array([obs['omega']], dtype=np.float32), device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor', choices=['diffusion', 'mlp'], default='diffusion')
    ap.add_argument('--tag', default='seq_diff')
    ap.add_argument('--warmstart', default='', help='bandit策略tag热启动actor权重(破鸡生蛋)')
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_srv', type=int, default=5)
    ap.add_argument('--deadline', type=float, default=7.0)
    ap.add_argument('--e_f_ratio', type=float, default=0.20)
    ap.add_argument('--agg_ratio', type=float, default=0.10)
    ap.add_argument('--idle_ratio', type=float, default=0.40, help='静态功率/满载计算功率 (idle能量=功率×arrival_dt, 控养几台热机)')
    ap.add_argument('--keep_alive', type=int, default=0, help='空闲后保持 warm 的槽数 (0=立即睡)')
    ap.add_argument('--arrival_dt', type=float, default=4.0)
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--gamma', type=float, default=0.9, help='降γ减horizon累积发散')
    ap.add_argument('--tau', type=float, default=0.01, help='target net 软更新')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--bc_eta', type=float, default=0.05)
    ap.add_argument('--alpha', type=float, default=0.005)
    ap.add_argument('--sla_lambda', type=float, default=3.0)
    ap.add_argument('--delay_scale', type=float, default=1.5, help='标量化延迟通道固定微推(常数, 不影响ω物理含义)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪(稳定critic)')
    ap.add_argument('--policy_delay', type=int, default=2, help='actor 每 N 次 critic 更新才更新一次(TD3式)')
    ap.add_argument('--warm_eps', type=int, default=20, help='热启动策略先灌 buffer 的 episode 数')
    ap.add_argument('--pa_beta', type=float, default=3e-3, help='PopArt 每通道统计 EMA 动量(0=关闭PopArt)')
    ap.add_argument('--hetero', action='store_true', help='异构服务器(诊断: 测"用哪k台"是否真多峰)')
    ap.add_argument('--randn_prior', action='store_true',
                    help='弃 feedback prior, 扩散从 randn 起步(解 prior 自锁; 配大 T)')
    ap.add_argument('--sparsemax', action='store_true',
                    help='输出层 softmax->sparsemax(显式开关 z=支撑集, 能耗展开 K=1..5; 多峰留扩散)')
    ap.add_argument('--env', choices=['frac', 'hetero', 'dyn'], default='frac',
                    help='hetero=异构画像+时变链路+背景队列; dyn=Shannon时变速率+随机任务类型+增强异构(主线现实化)')
    ap.add_argument('--task_size', type=float, default=20e6, help='hetero: 任务(批)大小 bit')
    ap.add_argument('--link_vol', type=float, default=0.35, help='hetero: 链路波动幅度')
    ap.add_argument('--bg_load', type=float, default=0.12, help='hetero: 背景负载强度')
    ap.add_argument('--dl_ratio', type=float, default=0.0, help='dyn: 结果回传比 ρ (输出/输入; 走下行)')
    ap.add_argument('--coord_ratio', type=float, default=0.0,
                    help='dyn: scatter/merge 协调开销/多一台 (×delay_ref, 随K增长 -> 切分非免费, 延迟角非平凡)')
    # --- 路②: 扩散-SAC 抗塌缩 (FDEdge 最大熵的连续多目标对应物) ---
    ap.add_argument('--m_div', type=int, default=1,
                    help='每状态采 M 候选算"跨样本支撑多样性"(>1 开启抗塌缩; 1=旧 Diffusion-QL)')
    ap.add_argument('--div_target', type=float, default=0.05,
                    help='目标支撑多样性(类比目标熵 H̃; 自动调温拉到此值; 唯一主旋钮, 太大会过度铺开伤能耗轴)')
    ap.add_argument('--div_kappa', type=float, default=20.0, help='软支撑指示器锐度 sigmoid(κ(a-τ0))')
    ap.add_argument('--div_tau0', type=float, default=0.05, help='软支撑阈值(a>τ0 视为活跃)')
    ap.add_argument('--div_lr', type=float, default=2e-2,
                    help='多样性温度 α 的学习率(需够快, 否则 α 来不及顶住 -Q 塌缩; 实测 3e-4 太慢)')
    ap.add_argument('--omega_film', action='store_true',
                    help='ω-FiLM 条件: 让 actor 随 ω 切支撑(解 ω-盲, 抢纯延迟角); 两 actor 同条件=公平')
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = os.path.join(RESULT, args.tag); os.makedirs(out, exist_ok=True)
    N = args.n_srv

    def make_env():
        if args.env == 'dyn':
            return DynOffloadEnv(n_servers=N, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                                 agg_ratio=args.agg_ratio, idle_ratio=args.idle_ratio,
                                 keep_alive=args.keep_alive, arrival_dt=args.arrival_dt,
                                 horizon=args.horizon, task_size=args.task_size,
                                 chan_vol=args.link_vol, bg_load=args.bg_load,
                                 dl_ratio=args.dl_ratio, coord_delay_ratio=args.coord_ratio)
        if args.env == 'hetero':
            return HeteroOffloadEnv(n_servers=N, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                                    agg_ratio=args.agg_ratio, idle_ratio=args.idle_ratio,
                                    keep_alive=args.keep_alive, arrival_dt=args.arrival_dt,
                                    horizon=args.horizon, task_size=args.task_size,
                                    link_vol=args.link_vol, bg_load=args.bg_load)
        return FracOffloadEnv(n_servers=N, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                              agg_ratio=args.agg_ratio, idle_ratio=args.idle_ratio,
                              keep_alive=args.keep_alive, arrival_dt=args.arrival_dt,
                              horizon=args.horizon, sequential=True, homogeneous=not args.hetero)
    env = make_env()
    feat_dim = env.reset()['servers'].shape[1]   # 从 obs 探每节点特征数 (frac=3, hetero=5)
    sd = N * feat_dim + 1
    Act = DiffActor if args.actor == 'diffusion' else GaussActor
    actor = (DiffActor(sd, N, T=args.T, sparse=args.sparsemax, omega_film=args.omega_film) if args.actor == 'diffusion'
             else GaussActor(sd, N, sparse=args.sparsemax, omega_film=args.omega_film)).to(dev)
    if args.warmstart:
        wpath = os.path.join(RESULT, args.warmstart, 'actor.pt')
        wsd = torch.load(wpath, map_location=dev)
        for k, v in wsd.items():           # 维度守卫: 旧码 11 维 ckpt 直接 load 会抛 cryptic mismatch
            if v.ndim == 2 and 'enc' in k:
                if v.shape[1] != sd:
                    raise ValueError('[warmstart 维度不兼容] %s 的 actor 首层输入=%d, 当前 sd=%d (=N*3+1)。'
                                     '该 warmstart ckpt 由旧码训练, 请先用当前码重训 direct 再 warmstart。'
                                     % (args.warmstart, v.shape[1], sd))
                break
        actor.load_state_dict(wsd)
        print('[warmstart] actor 权重载自 %s' % args.warmstart)
    c1, c2 = VCritic(sd, N).to(dev), VCritic(sd, N).to(dev)
    tc1, tc2 = copy.deepcopy(c1), copy.deepcopy(c2)
    # 正确 PopArt (每通道自适应归一化): critic 输出**归一化 Q**, loss 在归一化空间(energy 通道的
    #   小幅决策相关变化不再被大均值淹没 -> 可学); Bellman/actor 在**原始尺度**(denorm); 改 (μ,σ) 时
    #   **重参数化所有 critic(含 target)末层**使 Q_raw 不变 -> 物理单位稳、ω/sla_lambda 不漂。
    #   (修旧"假 PopArt"=只动 std 不重参数化 + actor 用归一化 Q 的两宗罪。)
    n_obj = 3
    pa_mu = torch.zeros(n_obj, device=dev)
    pa_nu = torch.ones(n_obj, device=dev)
    pa_sigma = torch.ones(n_obj, device=dev)
    pa_last = [c.net[-1] for c in (c1, c2, tc1, tc2)]   # 各 critic 的末层 Linear(h, n_obj)

    def denorm(qn):
        return pa_sigma * qn + pa_mu                     # 归一化 Q -> 原始尺度

    @torch.no_grad()
    def popart_step(target_raw):
        nonlocal pa_mu, pa_nu, pa_sigma
        if args.pa_beta <= 0:
            return
        mu_old, sig_old = pa_mu.clone(), pa_sigma.clone()
        pa_mu = (1 - args.pa_beta) * pa_mu + args.pa_beta * target_raw.mean(0)
        pa_nu = (1 - args.pa_beta) * pa_nu + args.pa_beta * (target_raw ** 2).mean(0)
        pa_sigma = torch.sqrt(torch.clamp(pa_nu - pa_mu ** 2, min=1e-4))
        ratio = sig_old / pa_sigma                       # W*=σ_old/σ_new; b=(σ_old·b+μ_old-μ_new)/σ_new
        shift = (mu_old - pa_mu) / pa_sigma
        for L in pa_last:
            L.weight.data.mul_(ratio[:, None]); L.bias.data.mul_(ratio).add_(shift)

    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=args.lr)
    # 多样性温度 α_div (自动调温, FDEdge 式17 对应): div<target -> α↑ 加大多样性压力
    div_on = (args.actor == 'diffusion' and args.m_div > 1)
    log_alpha_div = torch.tensor([-1.0], device=dev, requires_grad=True)   # softplus(-1)≈0.31 起步, 不一上来过度铺开
    opt_alpha = torch.optim.Adam([log_alpha_div], lr=args.div_lr)
    buf = SeqBuf(100000, N, feat_dim=feat_dim)

    def act(obs, prior, det=False):
        s = obs_state(obs, dev)
        pl = alloc_to_latent(torch.as_tensor(prior[None], dtype=torch.float32, device=dev)) \
            if (args.actor == 'diffusion' and not args.randn_prior and not args.sparsemax) else None
        with torch.no_grad():
            a = actor.sample(s, prior_latent=pl, det=det)[0].cpu().numpy()
        return a.astype(np.float32)

    def scalarize(qv, w):
        return w * args.delay_scale * qv[:, 0] + (1 - w) * qv[:, 1] + args.sla_lambda * qv[:, 2]

    def collect(ep_count, fill_only=False):
        for _ in range(ep_count):
            w = float(np.random.beta(0.5, 0.5))
            env.w = w; obs = env.reset(); prior = np.ones(N) / N; done = False
            while not done:
                a = act(obs, prior)
                nobs, r, done, info = env.step(a)
                # 增强 transition: x=prior(本步反向起点), nx=a(下一步的 prior) -> 改进/部署同一 prior-条件策略
                buf.add(obs['servers'].reshape(-1).tolist() + [obs['omega']], prior, a, info['r_vec'],
                        nobs['servers'].reshape(-1).tolist() + [nobs['omega']], a, float(done))
                obs = nobs; prior = a
        return env.episode_sla_summary()

    ucount = [0]
    def update(bs=128):
        ucount[0] += 1
        s, x, a, r, ns, nx, d = [torch.as_tensor(z, device=dev) for z in buf.sample(bs)]
        w = s[:, -1]
        # feedback prior latents (扩散反向起点; 高斯/randn_prior/sparsemax 忽略 prior -> None=从 randn 起)
        if args.actor == 'diffusion' and not args.randn_prior and not args.sparsemax:
            pl = alloc_to_latent(x); npl = alloc_to_latent(nx)
        else:
            pl = npl = None
        with torch.no_grad():
            na = actor.sample(ns, prior_latent=npl)     # 下一动作: 用 next-prior(=本步a) 起步, 对齐部署
            nqv_raw = denorm(torch.min(tc1(ns, na), tc2(ns, na)))   # target net 输出归一化 -> 还原原始尺度
            target_raw = r + args.gamma * (1 - d)[:, None] * nqv_raw
        popart_step(target_raw)                          # 更新每通道 (μ,σ) 并重参数化所有 critic 末层 (保 Q_raw)
        target_norm = (target_raw - pa_mu) / pa_sigma    # 归一化尺度回填 (energy 通道变化不再被淹没)
        c_loss = F.mse_loss(c1(s, a), target_norm) + F.mse_loss(c2(s, a), target_norm)
        opt_c.zero_grad(); c_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(c1.parameters()) + list(c2.parameters()), args.grad_clip)
        opt_c.step()
        q_log = float('nan'); div_log = float('nan'); alpha_log = float('nan')
        if ucount[0] % args.policy_delay == 0:          # TD3式: actor 延迟更新
            if div_on:
                # 路②: 每状态采 M 候选 -> -Q 改进(均值) + 跨样本支撑多样性(抗塌缩) + 自动调温
                M = args.m_div
                s_rep = s.repeat_interleave(M, dim=0)            # [bs*M, sd]
                w_rep = w.repeat_interleave(M)
                pl_rep = pl.repeat_interleave(M, dim=0) if pl is not None else None
                a_div = actor.sample(s_rep, prior_latent=pl_rep)  # [bs*M, N] 可反传
                qv = denorm(torch.min(c1(s_rep, a_div), c2(s_rep, a_div)))
                q_scalar = scalarize(qv, w_rep)
                # 软支撑指示器 u≈1 表示该台真活跃; 跨 M 样本方差高=支撑在样本间翻转=覆盖不同支撑
                #   (作用在"支撑是否翻转"而非"单次分配铺多开" -> 不动 sparsemax 稀疏/不伤能耗轴)
                u = torch.sigmoid(args.div_kappa * (a_div - args.div_tau0)).view(-1, M, N)
                div_mean = u.var(dim=1, unbiased=False).mean()    # 每服务器跨样本方差, 再对 (状态,服务器) 平均
                alpha_c = F.softplus(log_alpha_div).detach()      # 当前温度(对 actor 视为常数)
                bc = actor.bc_loss(s, a)
                a_loss = -q_scalar.mean() - alpha_c * div_mean + args.bc_eta * bc
                opt_a.zero_grad(); a_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                opt_a.step()
                # 自动调温: div<target -> 损失随 α 下降 -> α↑ (反之 α↓), 把 div 拉到 div_target
                alpha_loss = (F.softplus(log_alpha_div) * (div_mean.detach() - args.div_target)).mean()
                opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()
                div_log = float(div_mean.item()); alpha_log = float(F.softplus(log_alpha_div).item())
            else:
                a_pi = actor.sample(s, prior_latent=pl)  # 用 prior 起步 -> 改进=部署同一策略
                qv = denorm(torch.min(c1(s, a_pi), c2(s, a_pi)))   # 策略梯度用**原始尺度真 Q** (非归一化)
                q_scalar = scalarize(qv, w)
                if args.actor == 'diffusion':
                    extra = actor.bc_loss(s, a); a_loss = -q_scalar.mean() + args.bc_eta * extra
                else:
                    extra = -actor.entropy(s).mean(); a_loss = -q_scalar.mean() + args.alpha * extra
                opt_a.zero_grad(); a_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                opt_a.step()
            q_log = float(q_scalar.mean().item())
            for tp, p in [(tc1, c1), (tc2, c2)]:         # target 软更新随 actor
                for t, pp in zip(tp.parameters(), p.parameters()):
                    t.data.mul_(1 - args.tau).add_(args.tau * pp.data)
        return float(c_loss.item()), q_log, div_log, alpha_log

    print('dev=%s actor=%s seq deadline=%.1f dt=%.1f γ=%.2f warmstart=%s'
          % (dev, args.actor, args.deadline, args.arrival_dt, args.gamma, args.warmstart or '无'))
    print('[warm] 用当前(热启动)策略灌 buffer %d ep...' % args.warm_eps)
    collect(args.warm_eps, fill_only=True)
    print('  buffer=%d' % buf.size())
    rows = []
    for ep in range(args.episodes):
        sla = collect(1)
        logs = [update() for _ in range(args.horizon)]
        cl = np.mean([x[0] for x in logs]); q = np.nanmean([x[1] for x in logs])
        dv = np.nanmean([x[2] for x in logs]); al = np.nanmean([x[3] for x in logs])
        rows.append([ep, sla['mean_delay'], sla['mean_energy'], sla['violation_rate'], sla['p95_delay'], cl, q])
        if ep % 20 == 0 or ep == args.episodes - 1:
            ds = '' if not div_on else ' div=%.4f α=%.3f' % (dv, al)
            print('ep%03d | delay=%5.2f energy=%6.4f viol=%.3f p95=%5.2f | c_loss=%.3f q=%.3f%s'
                  % (ep, sla['mean_delay'], sla['mean_energy'], sla['violation_rate'], sla['p95_delay'], cl, q, ds),
                  flush=True)
    torch.save(actor.state_dict(), os.path.join(out, 'actor.pt'))
    # 存 critic + PopArt 统计 -> eval 可做 M-候选 critic 选择 (IDQL 式: actor 提议, critic 筛选)
    torch.save(c1.state_dict(), os.path.join(out, 'c1.pt'))
    torch.save(c2.state_dict(), os.path.join(out, 'c2.pt'))
    torch.save({'mu': pa_mu.cpu(), 'sigma': pa_sigma.cpu()}, os.path.join(out, 'popart.pt'))
    meta = vars(args).copy(); meta['sequential'] = True   # eval 据此还原训练 env (见 eval_frac 读 meta)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=2)
    print('[saved]', out)


if __name__ == '__main__':
    main()
