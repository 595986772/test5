"""正式训练 (带 checkpoint)。每 episode 采一个偏好 ω, prior-feedback 暖启动。

用法:
  python train.py                 # 默认 300 episode
  python train.py --episodes 600 --tag run1

产物: result2/<tag>/{actor.pt, critic1.pt, critic2.pt, meta.json, train_log.csv}
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import json
import numpy as np
import torch

from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent, ReplayBuffer
from fd_actor import uniform_prior

CONF = 'multi-part'
DEADLINE = 15.0
RESULT_DIR = 'result2'   # 本项目所有产物统一写这里 (与老项目 results/ 区分)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--tag', type=str, default='run1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--denoising_steps', type=int, default=3)
    ap.add_argument('--use_prior_cond', action='store_true', help='P2: prior 既当暖启动又当条件')
    ap.add_argument('--start_mode', type=str, default='prior', choices=['prior', 'randn'])
    ap.add_argument('--actor', type=str, default='diffusion', choices=['diffusion', 'mlp'],
                    help='消融: diffusion(扩散+prior反馈) 或 mlp(普通前馈, 隔离扩散贡献)')
    ap.add_argument('--no_sla', action='store_true', help='消融: 关掉 SLA 通道 (sla_lambda=0)')
    ap.add_argument('--use_popart', action='store_true', help='每目标 Q 归一化 (统一三通道尺度)')
    ap.add_argument('--omega_sample', type=str, default='uniform', choices=['uniform', 'ushaped'],
                    help='ushaped=Beta(0.5,0.5) 加权 ω 极值, 改善 delay/energy 极端欠训练')
    ap.add_argument('--fixed_alpha', type=float, default=None, help='固定温度(防熵崩); 不给=自动调温')
    ap.add_argument('--alpha_anneal', type=str, default=None,
                    help='温度线性退火 "hi,lo" (如 "0.3,0.05"): 前期高探索→后期低收尖, 优先级高于 fixed_alpha')
    ap.add_argument('--sla_lambda', type=float, default=1.0, help='SLA 标量化权重 (拧大=更重视合规)')
    ap.add_argument('--sla_penalty_scale', type=float, default=0.05, help='SLA 超时惩罚系数 (env 端)')
    ap.add_argument('--admission', action='store_true', help='训练期加截止期准入掩码 (硬机制)')
    ap.add_argument('--margin', type=float, default=1.0, help='准入安全裕度 (<1 更严)')
    ap.add_argument('--deadline', type=float, default=DEADLINE, help='截止期 (默认 15s)')
    ap.add_argument('--task_size_cap', type=float, default=None, help='任务大小封顶 bit (如 20e6)')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join(RESULT_DIR, args.tag)
    os.makedirs(out, exist_ok=True)
    print('device=%s  tag=%s  episodes=%d  start_mode=%s  sla=%s'
          % (dev, args.tag, args.episodes, args.start_mode, not args.no_sla))

    env = MEC_Env(conf_name=CONF, w=0.5, deadline=args.deadline,
                  sla_penalty_scale=args.sla_penalty_scale, task_size_cap=args.task_size_cap)
    print('  deadline=%.1f  admission=%s margin=%.2f  task_cap=%s'
          % (args.deadline, args.admission, args.margin, args.task_size_cap))
    anneal = None
    if args.alpha_anneal is not None:
        anneal = tuple(float(x) for x in args.alpha_anneal.split(','))   # (hi, lo)
    _auto = (args.fixed_alpha is None) and (anneal is None)
    _init_alpha = anneal[0] if anneal is not None else (0.05 if _auto else args.fixed_alpha)
    agent = FDSACAgent(denoising_steps=args.denoising_steps, start_mode=args.start_mode,
                       sla_lambda=(0.0 if args.no_sla else args.sla_lambda),
                       actor_type=args.actor, auto_alpha=_auto, use_prior_cond=args.use_prior_cond,
                       use_popart=args.use_popart,
                       reward_scale=((1.0, 1.0, 1.0) if args.use_popart else (0.1, 1.0, 1.0)),
                       alpha=_init_alpha, alpha_max=max(0.3, _init_alpha), device=dev)
    if anneal is not None:
        print('  alpha 退火: %.3f -> %.3f (线性, 按 episode)' % anneal)
    buf = ReplayBuffer(100000)

    WARMUP, BATCH = 500, 128
    log_rows = []
    for ep in range(args.episodes):
        if anneal is not None:                       # 温度退火: hi -> lo 线性
            frac = ep / max(1, args.episodes - 1)
            a_ep = anneal[0] + (anneal[1] - anneal[0]) * frac
            agent.log_alpha.data = torch.tensor(np.log(a_ep), dtype=torch.float, device=dev)
        w = float(np.random.beta(0.5, 0.5) if args.omega_sample == 'ushaped' else np.random.rand())
        env.w = w
        obs = env.reset()
        prior = uniform_prior(obs['mask2'])
        am = env.admission_mask(margin=args.margin) if args.admission else None
        done = False
        logs = []
        while not done:
            a, probs = agent.take_action(obs, prior, act_mask_np=am)
            next_obs, reward, done, info = env.step(a)
            n_am = env.admission_mask(margin=args.margin) if args.admission else None
            buf.add(obs['servers'], obs['preference'], obs['mask2'], prior, a,
                    info['r_vec'], next_obs['servers'], next_obs['preference'],
                    next_obs['mask2'], probs, float(done), act_mask=am, n_act_mask=n_am)
            obs = next_obs; prior = probs; am = n_am
            if buf.size() >= WARMUP:
                logs.append(agent.update(buf.sample(BATCH)))

        sla = env.episode_sla_summary()
        if logs:
            al = np.mean([d['alpha'] for d in logs]); H = np.mean([d['H'] for d in logs])
            cl = np.mean([d['c_loss'] for d in logs])
        else:
            al = H = cl = float('nan')
        log_rows.append([ep, w, sla['mean_delay'], sla['mean_energy'],
                         sla['violation_rate'], sla['p95_delay'], al, H, cl])
        if ep % 10 == 0 or ep == args.episodes - 1:
            print('ep%03d w=%.2f | delay=%6.2f energy=%6.3f viol=%.3f p95=%6.2f | alpha=%.4f H=%.3f c_loss=%.3f'
                  % (ep, w, sla['mean_delay'], sla['mean_energy'], sla['violation_rate'],
                     sla['p95_delay'], al, H, cl), flush=True)

    agent.save(out)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump({'conf': CONF, 'deadline': args.deadline, 'episodes': args.episodes,
                   'denoising_steps': args.denoising_steps, 'start_mode': args.start_mode,
                   'actor': args.actor, 'task_size_cap': args.task_size_cap,
                   'sla': (not args.no_sla), 'sla_lambda': args.sla_lambda, 'seed': args.seed}, f, indent=2)
    import csv
    with open(os.path.join(out, 'train_log.csv'), 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['ep', 'w', 'mean_delay', 'mean_energy', 'violation_rate', 'p95', 'alpha', 'H', 'c_loss'])
        wcsv.writerows(log_rows)
    print('\n[saved] %s  (actor/critic/meta/train_log)' % out)


if __name__ == '__main__':
    main()
