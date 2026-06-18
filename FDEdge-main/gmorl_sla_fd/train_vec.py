"""向量化训练 (同步, GMORL 风格): N 个 env 各固定一个 ω, 每步批量一次扩散前向。

= GMORL 的 DummyVectorEnv 做法 (串行步 env + 批量网络前向), 每个 batch 覆盖全 ω 谱。
两个挡位只差参数:
  务实挡:  python train_vec.py --n_envs 32 --total_steps 60000 --tag run_vec_ours
  严格对齐: python train_vec.py --n_envs 64 --total_steps 500000 --tag run_vec_full   (~过夜)

GMORL 原值参考: train_num=64, 总环境步≈3200万, batch=4096, buffer=1e6, updates≈25万。
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
import csv
import numpy as np
import torch

from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent, ReplayBuffer
from fd_actor import uniform_prior

CONF = 'multi-part'


class VecMECEnv:
    """N 个独立 MEC_Env, 各固定一个 ω = i/(N-1)。串行步进, 接口返回 list。"""
    def __init__(self, n_envs, deadline, task_size_cap, sla_penalty_scale):
        self.n = n_envs
        self.omegas = [(i / (n_envs - 1) if n_envs > 1 else 0.5) for i in range(n_envs)]
        self.envs = [MEC_Env(conf_name=CONF, w=self.omegas[i], deadline=deadline,
                             task_size_cap=task_size_cap, sla_penalty_scale=sla_penalty_scale)
                     for i in range(n_envs)]

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        obs, rew, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, d, inf = e.step(int(actions[i]))
            obs.append(o); rew.append(r); done.append(d); info.append(inf)
        return obs, rew, done, info

    def admission_masks(self, margin):
        return [e.admission_mask(margin) for e in self.envs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_envs', type=int, default=32)
    ap.add_argument('--total_steps', type=int, default=60000, help='向量化步数 (总环境步 = n_envs × 该值)')
    ap.add_argument('--tag', type=str, default='run_vec_ours')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--updates_per_step', type=int, default=1)
    ap.add_argument('--update_every', type=int, default=1, help='每 K 个向量步更新 1 次 (GMORL 风格低频)')
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--buffer', type=int, default=200000)
    ap.add_argument('--fixed_alpha', type=float, default=None, help='给值=固定温度(GMORL风格), 不给=自动调温')
    ap.add_argument('--actor_lr', type=float, default=1e-4)
    ap.add_argument('--critic_lr', type=float, default=3e-4)
    ap.add_argument('--denoising_steps', type=int, default=3)
    ap.add_argument('--start_mode', type=str, default='prior')
    ap.add_argument('--sla_lambda', type=float, default=1.0)
    ap.add_argument('--sla_penalty_scale', type=float, default=0.05)
    ap.add_argument('--no_sla', action='store_true')
    ap.add_argument('--admission', action='store_true')
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--deadline', type=float, default=20.0)
    ap.add_argument('--task_size_cap', type=float, default=20e6)
    ap.add_argument('--log_every', type=int, default=2000)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('result2', args.tag)
    os.makedirs(out, exist_ok=True)
    sla_lambda = 0.0 if args.no_sla else args.sla_lambda
    print('device=%s tag=%s | n_envs=%d total_steps=%d (env_steps=%d) batch=%d'
          % (dev, args.tag, args.n_envs, args.total_steps, args.n_envs * args.total_steps, args.batch))
    print('  deadline=%.1f cap=%s sla_lambda=%.2f admission=%s margin=%.2f'
          % (args.deadline, args.task_size_cap, sla_lambda, args.admission, args.margin))

    vec = VecMECEnv(args.n_envs, args.deadline, args.task_size_cap, args.sla_penalty_scale)
    auto = args.fixed_alpha is None
    agent = FDSACAgent(denoising_steps=args.denoising_steps, start_mode=args.start_mode,
                       sla_lambda=sla_lambda, auto_alpha=auto,
                       alpha=(0.05 if auto else args.fixed_alpha),
                       actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=dev)
    print('  auto_alpha=%s fixed_alpha=%s actor_lr=%g update_every=%d buffer=%d'
          % (auto, args.fixed_alpha, args.actor_lr, args.update_every, args.buffer))
    buf = ReplayBuffer(args.buffer)
    N = args.n_envs

    obs = vec.reset()
    prior = np.stack([uniform_prior(o['mask2']) for o in obs])           # [N, n_slots]
    am = vec.admission_masks(args.margin) if args.admission else [None] * N

    ep_done = 0
    recent = []   # 最近完成 episode 的 SLA 统计
    logs = []
    log_rows = []
    for t in range(args.total_steps):
        actions, probs = agent.take_action_batch(obs, prior, act_mask_np=(np.stack(am) if args.admission else None))
        next_obs, rew, done, info = vec.step(actions)
        n_am = vec.admission_masks(args.margin) if args.admission else [None] * N
        for i in range(N):
            buf.add(obs[i]['servers'], obs[i]['preference'], obs[i]['mask2'], prior[i], actions[i],
                    info[i]['r_vec'], next_obs[i]['servers'], next_obs[i]['preference'],
                    next_obs[i]['mask2'], probs[i], float(done[i]),
                    act_mask=am[i], n_act_mask=n_am[i])
        # 推进 + 处理 done -> 记账并重置
        prior = probs.copy()
        for i in range(N):
            if done[i]:
                recent.append(vec.envs[i].episode_sla_summary())
                ep_done += 1
                ro = vec.envs[i].reset()
                next_obs[i] = ro
                prior[i] = uniform_prior(ro['mask2'])
                if args.admission:
                    n_am[i] = vec.envs[i].admission_mask(args.margin)
        obs = next_obs
        am = n_am
        # 更新 (buffer 必须 >= max(warmup, batch); update_every 控制低频更新)
        if buf.size() >= max(args.warmup, args.batch) and (t % args.update_every == 0):
            for _ in range(args.updates_per_step):
                logs.append(agent.update(buf.sample(args.batch)))
        # 日志
        if (t + 1) % args.log_every == 0:
            if recent:
                mv = np.mean([s['violation_rate'] for s in recent])
                md = np.mean([s['mean_delay'] for s in recent])
                me = np.mean([s['mean_energy'] for s in recent])
            else:
                mv = md = me = float('nan')
            if logs:
                al = np.mean([d['alpha'] for d in logs[-500:]]); H = np.mean([d['H'] for d in logs[-500:]])
                cl = np.mean([d['c_loss'] for d in logs[-500:]])
            else:
                al = H = cl = float('nan')
            print('step%6d (env%8d ep%4d) | viol=%.3f delay=%6.2f energy=%6.3f | alpha=%.4f H=%.3f c_loss=%.3f'
                  % (t + 1, (t + 1) * N, ep_done, mv, md, me, al, H, cl), flush=True)
            log_rows.append([t + 1, (t + 1) * N, ep_done, mv, md, me, al, H, cl])
            recent = []

    agent.save(out)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump({'conf': CONF, 'deadline': args.deadline, 'task_size_cap': args.task_size_cap,
                   'n_envs': N, 'total_steps': args.total_steps, 'env_steps': N * args.total_steps,
                   'sla_lambda': sla_lambda, 'admission': args.admission, 'margin': args.margin,
                   'denoising_steps': args.denoising_steps, 'seed': args.seed}, f, indent=2)
    with open(os.path.join(out, 'train_log.csv'), 'w', newline='') as f:
        wc = csv.writer(f)
        wc.writerow(['step', 'env_step', 'ep_done', 'viol', 'mean_delay', 'mean_energy', 'alpha', 'H', 'c_loss'])
        wc.writerows(log_rows)
    print('\n[saved] %s  (env_steps=%d, episodes=%d)' % (out, N * args.total_steps, ep_done))


if __name__ == '__main__':
    main()
