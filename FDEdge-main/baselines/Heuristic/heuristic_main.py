"""
Heuristic baselines (Random / RoundRobin / GreedyDelay / GreedyEnergy)
======================================================================
从 mofd_main.py 解耦出的 4 个无训练启发式基线, 使用相同 MOFD 环境与多 seed 协议.
输出文件与 compare_hv.py 对齐: results/{rand,rr,gd,ge}_pareto_aggregated.csv
各基线:
  * random        随机在有效 ES 中均匀采
  * rr            Round-Robin 循环
  * greedy_delay  估计延迟 (tran + comp + queue) 最小的 ES
  * greedy_energy 执行能耗最低的 ES (等价于最低 CPU 频率)
"""
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mofd_environment import MOFDEnvironment
from helpers import build_preference_set, hypervolume_2d, pareto_front_2d, SHARED_EVAL_SEED_OFFSET
from mofd_main import sample_tasks, set_task_generator
from task_generator import make_task_generator

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

METHODS = [
    dict(key='rand', label='Random',       color='gray',    marker='x'),
    dict(key='rr',   label='RoundRobin',   color='orange',  marker='s'),
    dict(key='gd',   label='GreedyDelay',  color='green',   marker='^'),
    dict(key='ge',   label='GreedyEnergy', color='purple',  marker='v'),
]


# ------------------------------------------------------------
# 启发式动作选择
# ------------------------------------------------------------
def heuristic_action(env, t, n, method, E, f_E, rng, rr_state):
    """rr_state 是 [rr_idx] 的 list (mutable), 便于跨调用累加."""
    if method == 'rand':
        return int(rng.integers(0, E))
    if method == 'rr':
        a = rr_state[0] % E
        rr_state[0] += 1
        return a
    if method == 'gd':
        d_n = float(env.tasks_bit[t][n])
        rho_d = float(env.comp_density[n] * d_n)
        best_e, best = 0, float('inf')
        for e_ in range(E):
            f_b = f_E[e_]
            v = env.effective_rate(t, n, e_)
            if f_b <= 0 or v <= 0:
                continue
            delay_e = (d_n / v + rho_d / f_b
                       + (env.proc_queue_len[t, e_] + env.proc_queue_bef[t, e_]) / f_b)
            if delay_e < best:
                best, best_e = delay_e, e_
        return best_e
    if method == 'ge':
        best_e, best = 0, float('inf')
        for e_ in range(E):
            if f_E[e_] > 0 and f_E[e_] < best:
                best, best_e = f_E[e_], e_
        return best_e
    return 0


def run_episode(env, tasks, E, f_E, tran_rate, omega, method, rng):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    rr_state = [0]
    sum_d, sum_e, n_tasks = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for n in range(T_len):
            a = heuristic_action(env, t, n, method, E, f_E, rng, rr_state)
            _, _, d, e, _ = env.step(t, n, a)
            sum_d += d; sum_e += e; n_tasks += 1
        env.update_proc_queues(t)
    return sum_d / max(n_tasks, 1), sum_e / max(n_tasks, 1)


def evaluate_pareto(env, method, n_pref, n_eval_epi, seed):
    # 拆分 env / action 两个 rng:
    #   env_rng 仅由 sample_context / sample_tasks 消耗 → 跨 heuristic 方法对齐环境
    #   act_rng 仅由 'rand' 方法的 rng.integers 消耗 → 不污染 env_rng 状态
    prefs = build_preference_set(n_pref)
    points = []
    for omega in prefs:
        # 每个 omega 评估前重置两个 rng, 让 21 个 omega 看到相同的 n_eval_epi 个环境
        # 否则环境随机性盖过 omega 信号 → Pareto 不单调
        env_rng = np.random.default_rng(seed)
        act_rng = np.random.default_rng(seed + 1)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(env_rng)
            tasks = sample_tasks(env, env_rng)
            d, e = run_episode(env, tasks, E, f_E, tran_rate, omega, method, act_rng)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
    return np.array(points)


def main():
    cfg = dict(
        # ---- 环境 (与 mofd_main.py 保持一致) ----
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        # ---- 评估 ----
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        # ---- 任务到达模式 ('random' / 'poisson' / 'trace') ----
        task_mode='random',
        task_kwargs=dict(),
    )
    print('[heuristic] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    pts_pool = {m['key']: [] for m in METHODS}
    t0 = time.time()
    for seed in cfg['seeds']:
        env = MOFDEnvironment(
            Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
            bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
            f_range=cfg['f_range'], seed=seed,
        )
        for idx, m in enumerate(METHODS):
            pts = evaluate_pareto(env, m['key'],
                                  n_pref=cfg['final_eval_n_pref'],
                                  n_eval_epi=cfg['final_eval_n_epi'],
                                  seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET)
            pts_pool[m['key']].append(pts)
            print(f"[heuristic] seed {seed} {m['label']:>12} "
                  f"pts={len(pts)} elapsed={time.time() - t0:.1f}s")

    # ---- 保存 & 计算 HV ----
    all_points = []
    for m in METHODS:
        pts_all = np.vstack(pts_pool[m['key']])
        pts_pool[m['key']] = pts_all
        all_points.append(pts_all)
        np.savetxt(os.path.join(RESULTS_DIR, f"{m['key']}_pareto_aggregated.csv"),
                   pts_all, fmt='%.5f')

    all_global = np.vstack(all_points)
    ref = (float(all_global[:, 0].max() * 1.1 + 1e-6),
           float(all_global[:, 1].max() * 1.1 + 1e-6))

    # ---- 写 summary ----
    summary_path = os.path.join(RESULTS_DIR, 'heuristic_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=== Heuristic Baselines Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\n')
        f.write(f'Config: {cfg}\n')
        f.write(f'Local HV ref (for reference only; cross-method use compare_hv.py): {ref}\n')
        for m in METHODS:
            pts = pts_pool[m['key']]
            # 按 seed 切片
            n_pref = cfg['final_eval_n_pref']
            n_seeds = len(cfg['seeds'])
            per_seed = pts.reshape(n_seeds, n_pref, 2)
            hvs = [hypervolume_2d(per_seed[s], ref) for s in range(n_seeds)]
            mu, sd = float(np.mean(hvs)), float(np.std(hvs))
            f.write(f'{m["label"]:>12}: HV={mu:.4f} ± {sd:.4f}\n')
            print(f"[heuristic] {m['label']:>12}: HV={mu:.4f} ± {sd:.4f}")

    # ---- Pareto 对比图 ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m in METHODS:
        pts = pts_pool[m['key']]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=m['color'], marker=m['marker'],
                   s=25, alpha=0.5, label=m['label'])
        pf = pareto_front_2d(pts)
        if len(pf) > 0:
            ax.plot(pf[:, 0], pf[:, 1], color=m['color'], alpha=0.6)
    ax.set_xlabel('Avg Delay (s)')
    ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'Heuristic Pareto Fronts (pooled {len(cfg["seeds"])} seeds)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'heuristic_pareto.png'), dpi=150)
    plt.close()

    print(f'\n[heuristic] Done. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
