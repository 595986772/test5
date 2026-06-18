"""评估: 扫 ω 出主方法 Pareto 前沿, 与 baseline 同卷对比, 算 HV / feasible-HV, 画图。

主方法: 对每个 ω 跑 K 个固定场景 (greedy/argmax), 收 (mean_delay, mean_energy, violation_rate)。
        ω 从 0→1 扫 -> 一条前沿曲线。
baseline (ω-无关启发式, 各是一个点): random / round_robin / greedy_queue。
所有方法**同卷** (相同 per-episode seed -> 相同场景), 可直接比。

feasible-HV: 只保留 violation_rate <= 阈值 的点算 HV (SLA 可行域里的实用性)。

用法: python eval_pareto.py --tag run1 --k 10
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent
from fd_actor import uniform_prior
from helpers import pareto_front_2d, hypervolume_2d

CONF = 'multi-part'
DEADLINE = 15.0
VIOL_THRESH = 0.10   # feasible 判定: 违约率 <= 10%
RESULT_DIR = 'result2'   # 本项目所有产物统一写这里 (与老项目 results/ 区分)


# ---------- 各策略一步动作 ----------
def random_action(obs, st):
    valid = np.where(np.asarray(obs['mask2']) == 1)[0]
    return int(np.random.choice(valid))

def round_robin_action(obs, st):
    valid = np.where(np.asarray(obs['mask2']) == 1)[0]
    a = int(valid[st['rr'] % len(valid)]); st['rr'] += 1
    return a

def greedy_queue_action(obs, st):
    # servers[6, slot] = 该服务器执行队列长度 (越短越快空出) -> 选合法槽里队列最短
    valid = np.where(np.asarray(obs['mask2']) == 1)[0]
    qlen = np.asarray(obs['servers'])[6, :]
    a = int(valid[np.argmin(qlen[valid])])
    return a


def rollout_policy(policy_fn, env, w, K, seed0):
    """跑 K 个固定场景, 返回 (mean_delay, mean_energy, mean_viol, mean_p95)。"""
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k)          # 固定场景: 同 k -> 同 env (跨方法可比)
        env.w = w
        obs = env.reset()
        st = {'rr': 0}
        done = False
        while not done:
            a = policy_fn(obs, st)
            obs, r, done, info = env.step(a)
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy'])
        v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)


def rollout_agent(agent, env, w, K, seed0, admission=False, margin=1.0):
    """主方法: greedy(argmax) + prior 暖启动追踪。admission=True 时加截止期准入掩码。"""
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k)
        env.w = w
        obs = env.reset()
        prior = uniform_prior(obs['mask2'])
        done = False
        while not done:
            am = env.admission_mask(margin=margin) if admission else None
            a, probs = agent.take_action(obs, prior, stochastic=False, act_mask_np=am)
            obs, r, done, info = env.step(a)
            prior = probs
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy'])
        v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='run1')
    ap.add_argument('--k', type=int, default=10, help='每个 ω / baseline 的场景数')
    ap.add_argument('--n_omega', type=int, default=11)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--admission', action='store_true', help='给主方法加截止期准入掩码 (硬机制)')
    ap.add_argument('--margin', type=float, default=1.0, help='准入安全裕度 (<1 更严)')
    ap.add_argument('--deadline', type=float, default=DEADLINE, help='评估用截止期 (覆盖默认 15s)')
    ap.add_argument('--task_size_cap', type=float, default=None, help='任务大小封顶 bit (与训练一致, 如 20e6)')
    ap.add_argument('--actor', type=str, default='diffusion', choices=['diffusion', 'mlp'])
    ap.add_argument('--denoising_steps', type=int, default=3, help='须与训练一致')
    ap.add_argument('--use_prior_cond', action='store_true', help='须与训练一致')
    args = ap.parse_args()
    deadline = args.deadline
    suffix = '_d%g%s' % (args.deadline, ('_adm%g' % args.margin) if args.admission else '')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = os.path.join(RESULT_DIR, args.tag)
    env = MEC_Env(conf_name=CONF, w=0.5, deadline=deadline, task_size_cap=args.task_size_cap)
    agent = FDSACAgent(denoising_steps=args.denoising_steps, actor_type=args.actor,
                       use_prior_cond=args.use_prior_cond, device=dev)
    agent.load(ckpt)
    print('loaded %s  | device=%s  K=%d  n_omega=%d\n' % (ckpt, dev, args.k, args.n_omega))

    # ---- 主方法: 扫 ω ----
    omegas = np.linspace(0.0, 1.0, args.n_omega)
    main_pts = []   # (delay, energy, viol, p95, omega)
    print('=== 主方法 (扫 ω)%s ===' % ('  [+准入掩码]' if args.admission else ''))
    for w in omegas:
        d, e, v, p = rollout_agent(agent, env, float(w), args.k, args.seed0,
                                   admission=args.admission, margin=args.margin)
        main_pts.append((d, e, v, p, float(w)))
        print('  w=%.2f | delay=%6.2f energy=%6.3f viol=%.3f p95=%6.2f %s'
              % (w, d, e, v, p, '(feasible)' if v <= VIOL_THRESH else ''), flush=True)

    # ---- baselines (ω-无关, 各一个点) ----
    print('\n=== baselines (ω-无关启发式) ===')
    baselines = {'random': random_action, 'round_robin': round_robin_action,
                 'greedy_queue': greedy_queue_action}
    base_pts = {}
    for name, fn in baselines.items():
        d, e, v, p = rollout_policy(fn, env, 0.5, args.k, args.seed0)
        base_pts[name] = (d, e, v, p)
        print('  %-13s | delay=%6.2f energy=%6.3f viol=%.3f p95=%6.2f %s'
              % (name, d, e, v, p, '(feasible)' if v <= VIOL_THRESH else ''), flush=True)

    # ---- HV / feasible-HV ----
    all_de = [(d, e) for d, e, *_ in main_pts] + [(v[0], v[1]) for v in base_pts.values()]
    all_de = np.array(all_de)
    ref = (all_de[:, 0].max() * 1.1, all_de[:, 1].max() * 1.1)   # nadir × 1.1
    main_de = np.array([(d, e) for d, e, *_ in main_pts])
    main_feas = np.array([(d, e) for d, e, vio, *_ in main_pts if vio <= VIOL_THRESH])
    hv_main = hypervolume_2d(main_de, ref)
    hv_main_feas = hypervolume_2d(main_feas, ref) if len(main_feas) else 0.0
    print('\n=== HV (ref=%.2f, %.3f) ===' % ref)
    print('  主方法 HV (全 ω 前沿)      = %.3f' % hv_main)
    print('  主方法 feasible-HV (viol<=%.0f%%) = %.3f  [%d/%d 个 ω 点可行]'
          % (VIOL_THRESH * 100, hv_main_feas, len(main_feas), len(main_pts)))
    for name, (d, e, v, p) in base_pts.items():
        hv_b = hypervolume_2d(np.array([(d, e)]), ref)
        print('  %-13s HV = %.3f  %s' % (name, hv_b, '(feasible)' if v <= VIOL_THRESH else '(INfeasible)'))

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    pf = pareto_front_2d(main_de)
    if len(pf):
        ax.plot(pf[:, 0], pf[:, 1], '-', color='C0', lw=1.2, alpha=0.6, zorder=1, label='Ours (Pareto front)')
    # 主方法各 ω 点: feasible 实心, infeasible 空心
    for d, e, v, p, w in main_pts:
        feas = v <= VIOL_THRESH
        ax.scatter(d, e, c=[[0.12, 0.47, 0.71]] if feas else [[1, 1, 1]],
                   edgecolors='C0', s=60, marker='o', zorder=3,
                   linewidths=1.4)
    # baselines
    mk = {'random': ('s', 'C3'), 'round_robin': ('^', 'C2'), 'greedy_queue': ('D', 'C4')}
    for name, (d, e, v, p) in base_pts.items():
        m, c = mk[name]
        ax.scatter(d, e, c=c, marker=m, s=110, zorder=4, edgecolors='k', linewidths=0.8,
                   label='%s%s' % (name, '' if v <= VIOL_THRESH else ' (infeas)'))
    ax.set_xlabel('Mean completion delay (s)')
    ax.set_ylabel('Mean energy per task (J)')
    ax.set_title('Delay-Energy Pareto: Ours%s (ω-swept) vs baselines\n(filled=SLA-feasible viol≤%.0f%%, deadline=%.0fs)'
                 % (' +admission' if args.admission else '', VIOL_THRESH * 100, deadline))
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, args.tag, 'pareto%s.png' % suffix)
    fig.savefig(png, dpi=140)
    print('\n[saved] %s' % png)

    # ---- CSV ----
    import csv
    with open(os.path.join(RESULT_DIR, args.tag, 'pareto_points%s.csv' % suffix), 'w', newline='') as f:
        wc = csv.writer(f)
        wc.writerow(['method', 'omega', 'delay', 'energy', 'violation_rate', 'p95'])
        for d, e, v, p, w in main_pts:
            wc.writerow(['ours', '%.3f' % w, '%.4f' % d, '%.5f' % e, '%.4f' % v, '%.3f' % p])
        for name, (d, e, v, p) in base_pts.items():
            wc.writerow([name, '', '%.4f' % d, '%.5f' % e, '%.4f' % v, '%.3f' % p])
    print('[saved] %s/%s/pareto_points.csv' % (RESULT_DIR, args.tag))


if __name__ == '__main__':
    main()
