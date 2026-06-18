"""动作生成稳定性 / 鲁棒性: 扩散 vs MLP, nominal vs perturbed。

对齐 introduction 贡献三 ("复杂动态环境下的动作生成稳定性"), 不含 ω 漂移(=ω泛化)。
扰动 = 系统侧 (信道每步重 roll + 负载尖峰), ω 固定。

两个核心指标:
  ① 动作波动 vol = mean_t ‖p_t − p_{t-1}‖_1  (策略输出的逐步抖动)
     —— prior-feedback 扩散从上一步分配暖启动, 机制上应更平滑; MLP 每步重前向应更抖。
  ② 尾部鲁棒性: 扰动下 p95/p99 尾延迟 + 违约率的退化 (perturbed − nominal)。

用法: python eval_stability.py --k 15
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import numpy as np
import torch
from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent
from fd_actor import uniform_prior

CONF = 'multi-part'


def rollout(agent, perturb, K, seed0, deadline, cap, omega=0.5):
    vols, viols, p95s, p99s, dls, ens = [], [], [], [], [], []
    env = MEC_Env(conf_name=CONF, w=omega, deadline=deadline, task_size_cap=cap, perturb=perturb)
    for k in range(K):
        np.random.seed(seed0 + k)
        env.w = omega
        obs = env.reset()
        prior = uniform_prior(obs['mask2'])
        prev, vol = None, []
        done = False
        while not done:
            a, probs = agent.take_action(obs, prior, stochastic=False)
            if prev is not None:
                vol.append(float(np.abs(probs - prev).sum()))
            obs, r, done, info = env.step(a)
            prior = probs; prev = probs
        s = env.episode_sla_summary()
        vols.append(np.mean(vol) if vol else 0.0)
        viols.append(s['violation_rate']); p95s.append(s['p95_delay']); p99s.append(s['p99_delay'])
        dls.append(s['mean_delay']); ens.append(s['mean_energy'])
    return dict(vol=np.mean(vols), viol=np.mean(viols), p95=np.mean(p95s),
               p99=np.mean(p99s), delay=np.mean(dls), energy=np.mean(ens))


def load(tag, actor, dev, steps=3):
    ag = FDSACAgent(denoising_steps=steps, actor_type=actor, device=dev)
    ag.load(os.path.join('result2', tag))
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--seed0', type=int, default=2000)
    ap.add_argument('--deadline', type=float, default=20.0)
    ap.add_argument('--task_size_cap', type=float, default=20e6)
    ap.add_argument('--diff_tag', type=str, default='run4_cap')
    ap.add_argument('--mlp_tag', type=str, default='run7_mlp_pop')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [('Diffusion', load(args.diff_tag, 'diffusion', dev)),
              ('MLP', load(args.mlp_tag, 'mlp', dev))]
    print('=== 动作稳定性/鲁棒性 (K=%d, ω=0.5) ===' % args.k)
    print('%-10s %-9s %7s %7s %7s %7s' % ('model', 'cond', 'vol', 'viol', 'p95', 'p99'))
    res = {}
    for name, ag in models:
        for cond, pb in [('nominal', False), ('perturb', True)]:
            r = rollout(ag, pb, args.k, args.seed0, args.deadline, args.task_size_cap)
            res[(name, cond)] = r
            print('%-10s %-9s %7.3f %7.3f %7.2f %7.2f' % (name, cond, r['vol'], r['viol'], r['p95'], r['p99']), flush=True)

    print('\n=== 对比 (扩散是否更稳) ===')
    for cond in ['nominal', 'perturb']:
        dv, mv = res[('Diffusion', cond)]['vol'], res[('MLP', cond)]['vol']
        print('  [%s] 动作波动 vol: 扩散=%.3f  MLP=%.3f  -> %s'
              % (cond, dv, mv, '扩散更平滑✓' if dv < mv else 'MLP更平滑'))
    # 尾部退化 (perturb - nominal)
    print('\n=== 扰动下尾部退化 (perturb − nominal, 越小越鲁棒) ===')
    for name, _ in models:
        dp95 = res[(name, 'perturb')]['p95'] - res[(name, 'nominal')]['p95']
        dviol = res[(name, 'perturb')]['viol'] - res[(name, 'nominal')]['viol']
        print('  %-10s Δp95=%+6.2f  Δviol=%+.3f' % (name, dp95, dviol))


if __name__ == '__main__':
    main()
