"""ω-感知启发式 baseline 前沿 (纯 CPU, 不碰 GPU, 可与扩散训练并行)。

greedy-ω: 每步对到达任务, 用 env 内部估计算出每个合法服务器的
  完成时间 ct 和能耗 en, 各自在候选集内 min-max 归一化, 按 ω 加权取最小:
    cost_a = ω·ct_norm_a + (1-ω)·en_norm_a   -> argmin
扫 ω -> 一条前沿 (区别于 random/round_robin/greedy_queue 那些 ω-无关单点)。
这是一个"知情"且"偏好可调"的强 baseline, 直接回应"baseline 不感知 ω 不公平"的批评。

同卷 (相同 per-episode seed) + 与 eval_pareto 同 ref 口径。
用法: python eval_heuristic_front.py --k 40 --n_omega 11
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import csv
import numpy as np
from env_gmorl_sla import MEC_Env
from helpers import pareto_front_2d, hypervolume_2d

CONF = 'multi-part'
VIOL_THRESH = 0.10


def greedy_omega_action(env, w):
    """ORACLE 上界: 用前向成本估计(completion/energy)按 ω 加权挑最优。
    在线调度里这相当于有完美模型, 不是公平 baseline, 仅作天花板参考。GMORL 未用 greedy。"""
    mask2 = np.asarray(env.get_obs()['mask2'])
    valid = np.where(mask2 == 1)[0]
    ct = env.predict_completion_times()[valid]
    en = env.predict_energies()[valid]
    def norm(z):
        z = np.asarray(z, dtype=float)
        rng = z.max() - z.min()
        return (z - z.min()) / rng if rng > 1e-12 else np.zeros_like(z)
    cost = w * norm(ct) + (1.0 - w) * norm(en)
    return int(valid[int(np.argmin(cost))])


def random_p_action(env, p):
    """公平在线 baseline (GMORL 用): 以概率 p 卸载到随机边缘, 否则上云。扫 p -> 前沿。"""
    mask2 = np.asarray(env.get_obs()['mask2'])
    valid = np.where(mask2 == 1)[0]
    edges = valid[valid > 0]
    if np.random.rand() < p and len(edges) > 0:
        return int(np.random.choice(edges))
    return 0  # cloud


def rollout(env, param, K, seed0, policy):
    d, e, v, p_ = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k)
        env.w = param if policy is greedy_omega_action else 0.5
        obs = env.reset()
        done = False
        while not done:
            a = policy(env, param)
            obs, r, done, info = env.step(a)
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy'])
        v.append(s['violation_rate']); p_.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--n_omega', type=int, default=11)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--deadline', type=float, default=20.0)
    ap.add_argument('--task_size_cap', type=float, default=20e6)
    args = ap.parse_args()

    env = MEC_Env(conf_name=CONF, w=0.5, deadline=args.deadline, task_size_cap=args.task_size_cap)
    grid = np.linspace(0.0, 1.0, args.n_omega)
    rows = []  # (method, param, d, e, v, p)

    # ---- 公平在线 baseline: Random-p (扫 p) ----
    print('=== Random-p 公平前沿 (CPU, K=%d) ===' % args.k)
    for p_off in grid:
        d, e, v, pp = rollout(env, float(p_off), args.k, args.seed0, random_p_action)
        rows.append(('random_p', float(p_off), d, e, v, pp))
        print('  p=%.2f | delay=%6.2f energy=%6.3f viol=%.3f' % (p_off, d, e, v), flush=True)

    # ---- ORACLE 上界: greedy-ω (扫 ω) —— 仅天花板参考, 非 baseline ----
    print('\n=== greedy-ω [ORACLE 上界, 非 baseline] ===')
    for w in grid:
        d, e, v, pp = rollout(env, float(w), args.k, args.seed0, greedy_omega_action)
        rows.append(('oracle_greedy', float(w), d, e, v, pp))
        print('  w=%.2f | delay=%6.2f energy=%6.3f viol=%.3f' % (w, d, e, v), flush=True)

    # ---- HV (公共 ref, 仅供组内参考) ----
    def front_hv(method):
        de = np.array([(r[2], r[3]) for r in rows if r[0] == method])
        ref = (de[:, 0].max() * 1.1, de[:, 1].max() * 1.1)
        feas = np.array([(r[2], r[3]) for r in rows if r[0] == method and r[4] <= VIOL_THRESH])
        return hypervolume_2d(de, ref), (hypervolume_2d(feas, ref) if len(feas) else 0.0), len(feas)
    for m in ['random_p', 'oracle_greedy']:
        hv, hvf, nf = front_hv(m)
        tag = ' [ORACLE上界]' if m == 'oracle_greedy' else ' [公平baseline]'
        print('  %-14s HV=%.3f feasible-HV=%.3f (%d feasible)%s' % (m, hv, hvf, nf, tag))

    os.makedirs('result2', exist_ok=True)
    with open(os.path.join('result2', 'heuristic_front.csv'), 'w', newline='') as f:
        wc = csv.writer(f); wc.writerow(['method', 'param', 'delay', 'energy', 'violation_rate', 'p95'])
        for m, pr, d, e, v, pp in rows:
            wc.writerow([m, '%.3f' % pr, '%.4f' % d, '%.5f' % e, '%.4f' % v, '%.3f' % pp])
    print('\n[saved] result2/heuristic_front.csv (random_p=公平; oracle_greedy=上界参考)')


if __name__ == '__main__':
    main()
