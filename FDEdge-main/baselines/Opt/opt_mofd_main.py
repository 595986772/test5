"""
Opt (Exhaustive per-step) baseline for MOFD.
==========================================
对每一次任务到达, 枚举所有有效 ES, 计算该动作下的 (延迟, 能耗) 并按
当前偏好 omega 做标量化, 选择标量化成本最小的 ES.

说明:
  - FDEdge 原文中的 "Opt" 是给定完整信息后的穷举最优. 真正的全时域组合最优
    在 time_slots=100 x 任务~25/slot 的规模下不可解, 故采用"逐任务步级穷举"
    的实用实现: 它等价于在当前队列状态下的最优单步决策, 是**在线决策**可达
    的紧上界, 并自然支持不同 omega 扫出 Pareto 前沿.
  - 该基线无训练, 直接评估即可, 运行很快.
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


def opt_action(env, t, n, omega, alpha_T, alpha_E):
    """枚举有效 ES 的 (delay, energy), 按 omega 标量化取最小."""
    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    best_a, best_cost = 0, float('inf')
    for e_ in range(env.E):
        f_b = env.f_E[e_]
        v = env.effective_rate(t, n, e_)
        if f_b <= 0 or v <= 0:
            continue
        tran_delay = d_n / v
        comp_delay = rho_d / f_b
        wait_delay = (env.proc_queue_len[t, e_] + env.proc_queue_bef[t, e_]) / f_b
        delay = tran_delay + comp_delay + wait_delay
        e_off = env.p_off * tran_delay
        e_exe = env.kappa * (f_b ** 2) * rho_d
        energy = e_off + e_exe
        cost = omega[0] * alpha_T * delay + omega[1] * alpha_E * energy
        if cost < best_cost:
            best_cost, best_a = cost, e_
    return best_a


def run_opt_episode(env, tasks, E, f_E, tran_rate, omega, alpha_T, alpha_E):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n_tasks = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for n in range(T_len):
            a = opt_action(env, t, n, omega, alpha_T, alpha_E)
            _, _, d, e, _ = env.step(t, n, a)
            sum_d += d; sum_e += e; n_tasks += 1
        env.update_proc_queues(t)
    return sum_d / max(n_tasks, 1), sum_e / max(n_tasks, 1)


def evaluate_opt_pareto(env, n_pref, n_eval_epi, seed, alpha_T, alpha_E):
    prefs = build_preference_set(n_pref)
    points = []
    for omega in prefs:
        # 每个 omega 评估前重置 rng, 让 21 个 omega 看到相同的 n_eval_epi 个环境
        # 否则环境随机性盖过 omega 信号 → Pareto 不单调
        rng = np.random.default_rng(seed)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            d, e = run_opt_episode(env, tasks, E, f_E, tran_rate, omega, alpha_T, alpha_E)
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
        alpha_T=1.0, alpha_E=0.25,
        # ---- 任务到达模式 ('random' / 'poisson' / 'trace') ----
        task_mode='random',
        task_kwargs=dict(),
    )
    print('[opt] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    all_pts = []
    t0 = time.time()
    for seed in cfg['seeds']:
        env = MOFDEnvironment(
            Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
            bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
            f_range=cfg['f_range'], seed=seed,
        )
        pts = evaluate_opt_pareto(env,
                                  n_pref=cfg['final_eval_n_pref'],
                                  n_eval_epi=cfg['final_eval_n_epi'],
                                  seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                                  alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'])
        all_pts.append(pts)
        print(f"[opt] seed {seed} done, pareto pts={len(pts)} elapsed={time.time() - t0:.1f}s")

    pts_pool = np.vstack(all_pts)
    np.savetxt(os.path.join(RESULTS_DIR, 'opt_pareto_aggregated.csv'), pts_pool, fmt='%.5f')

    ref = (float(pts_pool[:, 0].max() * 1.1 + 1e-6),
           float(pts_pool[:, 1].max() * 1.1 + 1e-6))
    hvs = [hypervolume_2d(pts, ref) for pts in all_pts]
    print(f'\n[opt] Final HV (ref={ref}): mean={np.mean(hvs):.4f} std={np.std(hvs):.4f}')

    with open(os.path.join(RESULTS_DIR, 'opt_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Opt (Per-step Exhaustive) Multi-seed Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\n')
        f.write(f'Config: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\n')
        f.write(f'HV mean ± std: {np.mean(hvs):.4f} ± {np.std(hvs):.4f}\n')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(pts_pool[:, 0], pts_pool[:, 1],
               s=25, alpha=0.6, color='red',
               label=f'Opt (HV={np.mean(hvs):.2f}±{np.std(hvs):.2f})')
    pf = pareto_front_2d(pts_pool)
    if len(pf) > 0:
        ax.plot(pf[:, 0], pf[:, 1], color='red', alpha=0.5)
    ax.set_xlabel('Avg Delay (s)'); ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'Opt Pareto Front (pooled {len(cfg["seeds"])} seeds)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'opt_pareto.png'), dpi=150)
    plt.close()

    print(f'[opt] Done. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
