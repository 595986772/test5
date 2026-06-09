"""
Opt baseline V2 (on MOFDEnvironmentV2 with cloud action)
=========================================================
逐任务步级穷举: 对每个到达任务, 枚举所有 (cloud + 有效 edge) 共 E+1 个动作,
按当前偏好 omega 做标量化, 选标量化成本最小的.

跟 V1 区别: 候选集多了 action 0 (cloud), 物理参数不同 (cloud_f / cloud_v / cloud_kappa).
输出: results/opt_v2_pareto_aggregated.csv
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


def opt_action(env, t, n, omega):
    """枚举所有 valid 动作 (0=cloud, 1..E=edge), 按 omega 标量化取最小."""
    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    best_a, best_cost = 0, float('inf')
    for a in range(env.action_dim):
        if env.valid_mask[a] < 0.5:
            continue
        f_b, base_v, qi = env._action_to_f_v(a)
        v = base_v * float(env.channel_gain[t, n, a])
        if f_b <= 0 or v <= 0:
            continue
        tran_delay = d_n / v
        comp_delay = rho_d / f_b
        wait_delay = (env.proc_queue_len[t, qi] + env.proc_queue_bef[t, qi]) / f_b
        delay = tran_delay + comp_delay + wait_delay
        k = env.cloud_kappa if a == 0 else env.kappa
        e_off = env.p_off * tran_delay
        e_exe = k * (f_b ** 2) * rho_d
        energy = e_off + e_exe
        cost = float(omega[0]) * delay + float(omega[1]) * energy
        if cost < best_cost:
            best_cost, best_a = cost, a
    return best_a


def run_opt_episode(env, tasks, E, f_E, tran_rate, omega):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n_tasks = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        for j in range(len(env.tasks_bit[t])):
            a = opt_action(env, t, j, omega)
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n_tasks += 1
        env.update_proc_queues(t)
    return sum_d / max(n_tasks, 1), sum_e / max(n_tasks, 1)


def evaluate_pareto(env, n_pref, n_eval_epi, seed):
    prefs = build_preference_set(n_pref)
    points = []
    for k_omega, omega in enumerate(prefs):
        rng = np.random.default_rng(seed)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            d, e = run_opt_episode(env, tasks, E, f_E, tran_rate, omega)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
        print(f'  [Opt-v2] omega {k_omega+1}/{n_pref}  '
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
    print('[Opt-v2] Config:', cfg)
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
        print(f'[Opt-v2] seed {seed} done  delay range=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]s  '
              f'energy range=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'opt_v2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'opt_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Opt-v2 Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\nHV: {mu:.4f} ± {sd:.4f}\n')
    print(f'[Opt-v2] HV (local) = {mu:.4f}  total = {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
