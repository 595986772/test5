"""
SA baseline V2 (on MOFDEnvironmentV2 with cloud action)
========================================================
跟 GMORL 论文对齐:
  - n_iter = 10000 (论文原话: "searches 10000 episodes for each preference")
  - action space includes cloud (action 0)
  - 输出: results/sa_v2_pareto_aggregated.csv
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

from mofd_environment_v2 import MOFDEnvironmentV2
from helpers import (build_preference_set, hypervolume_2d, pareto_front_2d,
                     SHARED_EVAL_SEED_OFFSET)
from mofd_main import sample_tasks, set_task_generator
from task_generator import make_task_generator

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def flatten_indices(tasks, T):
    idx = []
    for t in range(T - 1):
        if t < len(tasks):
            for n in range(len(tasks[t])):
                idx.append((t, n))
    return idx


def simulate_with_chromosome(env, tasks, E, f_E, tran_rate, omega, chrom):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n = 0.0, 0.0, 0
    k = 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for j in range(T_len):
            a = int(chrom[k])
            # V2: action ∈ [0, env.action_dim - 1] = [0, Emax] (含 cloud)
            if a < 0 or a >= env.action_dim or env.valid_mask[a] < 0.5:
                a = 0  # 退回 cloud (永远 valid)
            _, _, d, e, _ = env.step(t, j, a)
            sum_d += d; sum_e += e; n += 1
            k += 1
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def sa_optimize(env, tasks, E, f_E, tran_rate, omega, rng,
                n_iter=10000, T0=1.0, T_min=0.001, decay=None):
    flat_idx = flatten_indices(tasks, env.time_slots)
    L = len(flat_idx)
    if L == 0:
        return 0.0, 0.0, np.zeros(0, dtype=np.int32)

    if decay is None:
        decay = (T_min / T0) ** (1.0 / max(n_iter - 1, 1))

    # 初始解: 在 [0, action_dim-1] 范围内随机 (V2)
    chrom = rng.integers(0, env.action_dim, size=L).astype(np.int32)
    d_cur, e_cur = simulate_with_chromosome(env, tasks, E, f_E, tran_rate,
                                            omega, chrom)
    cost_cur = float(omega[0]) * d_cur + float(omega[1]) * e_cur

    best_chrom = chrom.copy()
    best_d, best_e, best_cost = d_cur, e_cur, cost_cur

    T = T0
    for it in range(n_iter):
        K_max = max(2, L // 20)
        K = max(1, int(round(K_max * (T - T_min) / max(T0 - T_min, 1e-9))))
        cand = chrom.copy()
        pos = rng.choice(L, size=K, replace=False)
        cand[pos] = rng.integers(0, env.action_dim, size=K)

        d_c, e_c = simulate_with_chromosome(env, tasks, E, f_E, tran_rate,
                                            omega, cand)
        cost_c = float(omega[0]) * d_c + float(omega[1]) * e_c
        dE = cost_c - cost_cur
        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
            chrom = cand; cost_cur, d_cur, e_cur = cost_c, d_c, e_c
            if cost_c < best_cost:
                best_chrom = cand.copy()
                best_d, best_e, best_cost = d_c, e_c, cost_c
        T *= decay
    return best_d, best_e, best_chrom


def evaluate_pareto(env, n_pref, n_eval_epi, seed, n_iter):
    prefs = build_preference_set(n_pref)
    points = []
    for k_omega, omega in enumerate(prefs):
        env_rng = np.random.default_rng(seed)
        act_rng = np.random.default_rng(seed + 1)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(env_rng)
            tasks = sample_tasks(env, env_rng)
            d, e, _ = sa_optimize(env, tasks, E, f_E, tran_rate, omega,
                                  act_rng, n_iter=n_iter)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
        # 跨偏好进度提示
        print(f'  [SA-v2] omega {k_omega+1}/{n_pref}  '
              f'd={np.mean(delays):.3f}s  e={np.mean(energies):.3f}J', flush=True)
    return np.array(points)


def main():
    cfg = dict(
        # ---- env (V2 默认) ----
        Emax=6, num_tasks_max=10, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        cloud_f_range=(50, 70), cloud_tran_rate_range=(80, 120),
        kappa=1e-3, cloud_kappa=1e-4,
        # ---- 评估 ----
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        n_iter=10000,                  # 跟 GMORL 论文对齐
        task_mode='random', task_kwargs=dict(),
    )
    print('[SA-v2] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    pts_per_seed = []
    t0 = time.time()
    for seed in cfg['seeds']:
        env = MOFDEnvironmentV2(
            Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
            bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
            f_range=cfg['f_range'],
            cloud_f_range=cfg['cloud_f_range'],
            cloud_tran_rate_range=cfg['cloud_tran_rate_range'],
            kappa=cfg['kappa'], cloud_kappa=cfg['cloud_kappa'],
            seed=seed,
        )
        pts = evaluate_pareto(env, cfg['final_eval_n_pref'], cfg['final_eval_n_epi'],
                              seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                              n_iter=cfg['n_iter'])
        pts_per_seed.append(pts)
        print(f'[SA-v2] seed {seed} done  delay range=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]s  '
              f'energy range=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'sa_v2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'sa_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== SA-v2 Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\nHV: {mu:.4f} ± {sd:.4f}\n')
    print(f'[SA-v2] HV (local) = {mu:.4f}  total = {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
