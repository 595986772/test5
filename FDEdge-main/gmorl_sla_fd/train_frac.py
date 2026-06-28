"""[DEPRECATED] 连续分数卸载训练 (bandit + FracAgent.update)。

⚠️ 已弃用, 默认拒绝运行 (需 --force)。两个原因:
  1. 鸡生蛋死锁: 策略一直全切 -> buffer 无集中动作 -> critic 估不出集中价值 -> 无梯度。
  2. FracAgent.update 的 actor 改进从**无 prior 的随机起点**采样, 与 act() 部署时用 prior 不一致
     (P1-4): 训练/部署不是同一条 prior-条件策略。
主线 = train_frac_seq.py (序贯 critic + γ Bellman + feedback prior 增强 transition, 已修)。
本脚本仅留作历史参照, 勿用于主结果/对比。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse, json, csv
import numpy as np
import torch
from env_frac_offload import FracOffloadEnv
from frac_agent import FracAgent, Buf

RESULT = 'result2/frac'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actor', choices=['diffusion', 'mlp'], default='diffusion')
    ap.add_argument('--tag', default='diff')
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_srv', type=int, default=5)
    ap.add_argument('--deadline', type=float, default=7.0)
    ap.add_argument('--e_f_ratio', type=float, default=0.20, help='激活能量/单任务能量(红线①, 强到让能量区有牙)')
    ap.add_argument('--agg_ratio', type=float, default=0.10)
    ap.add_argument('--q_max_ratio', type=float, default=0.2)
    ap.add_argument('--hetero', action='store_true', help='异构服务器(敏感性, 压低多峰)')
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--T', type=int, default=5, help='扩散去噪步')
    ap.add_argument('--bc_eta', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=0.02, help='高斯SAC熵温度(轻探索, 防熵项压过Q)')
    ap.add_argument('--sla_lambda', type=float, default=1.0)
    ap.add_argument('--omega_sample', choices=['uniform', 'ushaped'], default='ushaped')
    ap.add_argument('--force', action='store_true', help='确认仍要跑这条已弃用路径')
    args = ap.parse_args()
    if not args.force:
        print('=' * 78)
        print('[DEPRECATED] train_frac.py 已弃用 (鸡生蛋死锁 + FracAgent.update 的 prior 不一致, P1-4)。')
        print('  主线请用: python train_frac_seq.py --actor <a> --tag <t> --warmstart <warmstart_tag>')
        print('  若确知自己在做什么, 加 --force 才会运行。')
        print('=' * 78)
        return
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = os.path.join(RESULT, args.tag); os.makedirs(out, exist_ok=True)
    print('dev=%s actor=%s tag=%s ep=%d N=%d deadline=%.1f e_f=%.0f%% agg=%.0f%% hetero=%s'
          % (dev, args.actor, args.tag, args.episodes, args.n_srv, args.deadline,
             100 * args.e_f_ratio, 100 * args.agg_ratio, args.hetero))

    env = FracOffloadEnv(n_servers=args.n_srv, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                         agg_ratio=args.agg_ratio, q_max_ratio=args.q_max_ratio,
                         homogeneous=not args.hetero, horizon=args.horizon)
    ag = FracAgent(args.n_srv, actor_type=args.actor, T=args.T, bc_eta=args.bc_eta,
                   alpha=args.alpha, sla_lambda=args.sla_lambda, device=dev)
    buf = Buf(100000, args.n_srv)
    WARMUP, BATCH = 500, 128
    rows = []
    for ep in range(args.episodes):
        w = float(np.random.beta(0.5, 0.5) if args.omega_sample == 'ushaped' else np.random.rand())
        env.w = w
        obs = env.reset(); prior = np.ones(args.n_srv) / args.n_srv
        logs = []
        done = False
        while not done:
            a = ag.act(obs, prior)
            nobs, r, done, info = env.step(a)
            buf.add(obs['servers'], obs['omega'], a, info['r_vec'])
            obs = nobs; prior = a
            if buf.size() >= WARMUP:
                logs.append(ag.update(buf.sample(BATCH)))
        s = env.episode_sla_summary()
        cl = np.mean([d['c_loss'] for d in logs]) if logs else float('nan')
        ex = np.mean([d['extra'] for d in logs]) if logs else float('nan')
        rows.append([ep, w, s['mean_delay'], s['mean_energy'], s['violation_rate'], s['p95_delay'], cl, ex])
        if ep % 20 == 0 or ep == args.episodes - 1:
            print('ep%03d w=%.2f | delay=%5.2f energy=%6.4f viol=%.3f p95=%5.2f | c_loss=%.4f %s=%.3f'
                  % (ep, w, s['mean_delay'], s['mean_energy'], s['violation_rate'], s['p95_delay'],
                     cl, ('bc' if args.actor == 'diffusion' else 'H'), ex), flush=True)
    ag.save(out)
    json.dump(vars(args), open(os.path.join(out, 'meta.json'), 'w'), indent=2)
    with open(os.path.join(out, 'log.csv'), 'w', newline='') as f:
        wc = csv.writer(f); wc.writerow(['ep', 'w', 'delay', 'energy', 'viol', 'p95', 'c_loss', 'extra'])
        wc.writerows(rows)
    print('[saved]', out)


if __name__ == '__main__':
    main()
