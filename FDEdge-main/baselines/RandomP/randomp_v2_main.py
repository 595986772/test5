"""
Random-P baseline V2 (tunable cloud probability)
=================================================
跟 GMORL 论文对齐 (C.2 Baselines):
  "The random-based scheme has p probability to offload a task to the cloud
   server and 1−p probability to a random edge server. We tune the
   probability p and evaluate the scheme to obtain a Pareto front."

对每个偏好 ω, 反推它对应的 p_cloud (经验映射 ω → p), 然后跑该 p 下的
random episode. 这样 21 个 ω 自然沿 cloud-edge 谱铺出 Pareto.

输出: results/randomp_v2_pareto_aggregated.csv
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

from mofd_environment_v2 import MOFDEnvironmentV2
from helpers import (build_preference_set, hypervolume_2d, pareto_front_2d,
                     SHARED_EVAL_SEED_OFFSET)
from mofd_main import sample_tasks, set_task_generator
from task_generator import make_task_generator

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def run_episode(env, tasks, E, f_E, tran_rate, omega, p_cloud, rng):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        for j in range(len(env.tasks_bit[t])):
            if rng.random() < p_cloud:
                a = 0                                       # cloud
            else:
                a = int(rng.integers(1, E + 1))             # 随机 edge
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n += 1
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def evaluate_pareto(env, n_pref, n_eval_epi, seed):
    """对每个 ω 反推 p_cloud = ω_E (节能偏好 → 更多 cloud).

    映射: p_cloud = ω_E ∈ [0, 1]. 当 ω 偏延迟 (ω_E ≈ 0), p_cloud ≈ 0, 全 edge;
    当 ω 偏节能 (ω_E ≈ 1), p_cloud ≈ 1, 全 cloud. 21 个 ω 自然铺出 Pareto.
    """
    prefs = build_preference_set(n_pref)
    points = []
    for k_omega, omega in enumerate(prefs):
        p_cloud = float(omega[1])                  # ω_E
        env_rng = np.random.default_rng(seed)
        act_rng = np.random.default_rng(seed + 1)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(env_rng)
            tasks = sample_tasks(env, env_rng)
            d, e = run_episode(env, tasks, E, f_E, tran_rate, omega, p_cloud, act_rng)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
        print(f'  [RandomP-v2] omega {k_omega+1}/{n_pref}  p_cloud={p_cloud:.2f}  '
              f'd={np.mean(delays):.3f}  e={np.mean(energies):.3f}', flush=True)
    return np.array(points)


def main():
    cfg = dict(
        Emax=6, num_tasks_max=10, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        cloud_f_range=(50, 70), cloud_tran_rate_range=(80, 120),
        kappa=1e-3, cloud_kappa=1e-4,
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        task_mode='random', task_kwargs=dict(),
    )
    print('[RandomP-v2] Config:', cfg)
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
                              seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET)
        pts_per_seed.append(pts)
        print(f'[RandomP-v2] seed {seed} done  delay range=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]s  '
              f'energy range=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'randomp_v2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'randomp_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== RandomP-v2 Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\nHV: {mu:.4f} ± {sd:.4f}\n')
    print(f'[RandomP-v2] HV (local) = {mu:.4f}  total = {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
