"""连续分数卸载 — 直接奖励最大化训练 (路A, 无 critic)。

1步 bandit + 已知可微奖励 => 不必学 critic(避开"策略不探索→critic无数据"的鸡生蛋)。
actor 直接最大化 torch_scalar_reward: 扩散 = −reward(过采样反传); 高斯 = −reward − α·H。
这是测"actor 能否表征/找到多峰最优"的最干净设置。eval 仍用硬 env。

**定位 (#8): 本脚本仅作序贯训练的 warm-start 原型, 非主实验算法**。可微奖励是硬 env 的近似
(τ 平滑 + delay_scale 为训练期梯度平衡常数, eval 不含), 扩散训练已对齐 eval 从均分 prior 起步。
主结果走 train_frac_seq (带 critic + γ Bellman + feedback prior) 在硬 env 上 eval。

用法:
  python train_frac_direct.py --actor diffusion --tag diff --iters 6000
  python train_frac_direct.py --actor mlp       --tag gauss --iters 6000
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse, json
import numpy as np
import torch
from env_frac_offload import (FracOffloadEnv, torch_scalar_reward, F_BASE)
from frac_agent import DiffActor, GaussActor, flat_state, alloc_to_latent

RESULT = 'result2/frac'


def sample_ctx(B, N, homo, span, q_max, dev):
    if homo:
        f = F_BASE * (1 + (torch.rand(B, N, device=dev) * 2 - 1) * span)
    else:
        f = F_BASE * (0.7 + torch.rand(B, N, device=dev) * 0.9)
    q = torch.rand(B, N, device=dev) * q_max
    # ushaped ω: 加权极值 (近似 Beta(0.5,0.5)) — 改善 delay/energy 两端欠训练
    u = torch.rand(B, device=dev)
    w = torch.where(torch.rand(B, device=dev) < 0.5, u ** 2, 1 - (1 - u) ** 2)
    return f, q, w


@torch.no_grad()
def eval_hard(actor, at, env, omegas, K, seed0, dev, randn_prior=False):
    """硬 env 上扫 ω 出 (delay,energy,viol)。"""
    pts = []
    for w in omegas:
        d, e, v = [], [], []
        for k in range(K):
            np.random.seed(seed0 + k); env.w = float(w)
            obs = env.reset(); prior = np.ones(env.N) / env.N; done = False
            while not done:
                s = flat_state(torch.as_tensor(obs['servers'][None], device=dev),
                               torch.as_tensor(np.array([obs['omega']]), dtype=torch.float32, device=dev))
                pl = None
                if at == 'diffusion' and not randn_prior:
                    from frac_agent import alloc_to_latent
                    pl = alloc_to_latent(torch.as_tensor(prior[None], dtype=torch.float32, device=dev))
                a = actor.sample(s, prior_latent=pl)[0].cpu().numpy()
                obs, r, done, info = env.step(a); prior = a
            sm = env.episode_sla_summary()
            d.append(sm['mean_delay']); e.append(sm['mean_energy']); v.append(sm['violation_rate'])
        pts.append((float(w), np.mean(d), np.mean(e), np.mean(v)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor', choices=['diffusion', 'mlp'], default='diffusion')
    ap.add_argument('--tag', default='diff')
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_srv', type=int, default=5)
    ap.add_argument('--deadline', type=float, default=7.0)
    ap.add_argument('--e_f_ratio', type=float, default=0.20)
    ap.add_argument('--agg_ratio', type=float, default=0.10)
    ap.add_argument('--q_max_ratio', type=float, default=0.2)
    ap.add_argument('--sla_lambda', type=float, default=3.0)
    ap.add_argument('--rew_beta', type=float, default=20.0, help='smooth-max 锐度(大=偏置小, 延迟侧敢铺开)')
    ap.add_argument('--delay_scale', type=float, default=1.8, help='延迟通道权重(平衡两通道梯度, 让延迟侧铺得开)')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.005, help='高斯熵正则(轻)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hetero', action='store_true')
    ap.add_argument('--randn_prior', action='store_true',
                    help='弃 feedback prior, 扩散从 randn 起步(解 prior 自锁; 配大 T)')
    ap.add_argument('--sparsemax', action='store_true',
                    help='输出层 softmax->sparsemax(显式开关 z=支撑集, 能耗展开 K=1..5)')
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 78)
    print('[WARM-START PROTOTYPE ONLY] 本脚本用可微 surrogate (torch_scalar_reward) 训练。')
    print('  它与硬 env 在小 fraction 区不可消除地不等: 硬 env 用硬阈值 a>1e-4 判 active,')
    print('  surrogate 用尺度 tau 的 soft 掩码 -> 如 a=[0.99,0.01,0] 给高队列服务器, 硬 delay≈20s')
    print('  而 surrogate≈10s。压 tau 到 1e-4 就没梯度 = 毁掉本脚本意义。')
    print('  => 产物**只能**作 train_frac_seq 的 --warmstart 种子 (seq 在硬 env+critic 上纠偏),')
    print('     不得作主结果或公平 baseline。主结果一律 train_frac_seq + eval_frac (都在硬 env)。')
    print('=' * 78)
    out = os.path.join(RESULT, args.tag); os.makedirs(out, exist_ok=True)
    N = args.n_srv; sd = N * 3 + 1   # 每服务器 [f,q,warm]+ω; bandit 下 warm 恒 0 (冷启动) 但保持架构统一
    env = FracOffloadEnv(n_servers=N, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                         agg_ratio=args.agg_ratio, q_max_ratio=args.q_max_ratio,
                         homogeneous=not args.hetero)
    span = env.hetero_span; q_max = env.q_max
    D, dl, ef, agg = env.D, env.delay_ref, env.e_f, env.agg
    eref = env.energy_ref
    rew_kw = dict(D=D, deadline=args.deadline, e_f=ef, agg=agg, delay_ref=dl,
                  energy_ref=eref, sla_lambda=args.sla_lambda, beta=args.rew_beta,
                  delay_scale=args.delay_scale)
    actor = (DiffActor(sd, N, T=args.T, sparse=args.sparsemax) if args.actor == 'diffusion'
             else GaussActor(sd, N, sparse=args.sparsemax)).to(dev)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    print('dev=%s actor=%s tag=%s iters=%d N=%d deadline=%.1f e_f=%.0f%% sla_λ=%.1f'
          % (dev, args.actor, args.tag, args.iters, N, args.deadline, 100 * args.e_f_ratio, args.sla_lambda))

    omegas = np.linspace(0, 1, 6)
    for it in range(args.iters):
        f, q, w = sample_ctx(args.batch, N, not args.hetero, span, q_max, dev)
        warm0 = torch.zeros_like(f)                           # bandit 冷启动 warm≡0
        feat = torch.stack([f / F_BASE, q / env.D, warm0], dim=2)   # [B,N,3], 与 env._obs 一致
        s = flat_state(feat, w)
        if args.actor == 'diffusion':
            # randn_prior/sparsemax: 不喂 prior(从 randn 起); 否则均分 prior(与 eval 起点一致)
            pl = None if (args.randn_prior or args.sparsemax) else alloc_to_latent(torch.ones(args.batch, N, device=dev) / N)
            a = actor.sample(s, prior_latent=pl)
        else:
            a = actor.sample(s)
        rew, delay, energy = torch_scalar_reward(a, f, q, w, **rew_kw)
        loss = -rew.mean()
        if args.actor == 'mlp':
            loss = loss - args.alpha * actor.entropy(s).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == args.iters - 1:
            pts = eval_hard(actor, args.actor, env, omegas, 8, 5000, dev,
                            randn_prior=args.randn_prior or args.sparsemax)
            lo = pts[0]; hi = pts[-1]   # w=0(能量) / w=1(延迟)
            print('it%5d loss=%.3f | w0: d=%.2f e=%.4f v=%.2f | w1: d=%.2f e=%.4f v=%.2f'
                  % (it, loss.item(), lo[1], lo[2], lo[3], hi[1], hi[2], hi[3]), flush=True)
    torch.save(actor.state_dict(), os.path.join(out, 'actor.pt'))
    meta = vars(args).copy(); meta['sequential'] = False   # bandit; eval 据此构建 env
    meta['role'] = 'warmstart_prototype'                    # 标记: 仅作 seq 的 warmstart, 非主结果
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=2)
    print('[saved]', out)


if __name__ == '__main__':
    main()
