"""
Simulated Annealing (SA) baseline
==================================
GMORL 论文 baseline 之一. 与现有 MOFD 环境 / 多 seed 协议对齐,
输出 results/sa_pareto_aggregated.csv (n_seeds * n_pref, 2), compare_hv.py 自动识别.

算法设计 (per-episode 全局 SA, 单目标标量化):
  - 决策变量: 长度 = total_tasks_in_episode 的整数向量, 每位 ∈ [0, E-1]
  - 目标函数: 模拟完整 episode 后的 (ω_T * mean_delay + ω_E * mean_energy)
              注意: 直接用 delay/energy 物理量, 不乘 delay_scale/energy_scale,
              和 Opt / Heuristic 对齐.
  - 邻域算子: 随机翻转 K 个位置 (K 随温度递减)
  - 接受准则: Metropolis (exp(-ΔE / T)), T 从 T0 指数衰减到 T_min
  - 初始化: 随机有效动作

为什么选 per-episode 全局 SA 而不是 per-slot greedy-SA:
  per-slot greedy 等价于 "Boltzmann GreedyDelay", 跟现有 Heuristic 的 'gd'
  几乎重叠, 失去 baseline 价值. 全局 SA 是真正的元启发式优化器,
  跟 NSGA-II / Opt 站在同一层面.

为防止过慢, 默认 SA 迭代数自适应: n_iter = max(200, 2 * total_tasks).
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
from helpers import (build_preference_set, hypervolume_2d, pareto_front_2d,
                     SHARED_EVAL_SEED_OFFSET)
from mofd_main import sample_tasks, set_task_generator
from task_generator import make_task_generator

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# ------------------------------------------------------------
# 把 episode 拆成 (slot, task) 平面索引, 用于 SA 染色体编码
# ------------------------------------------------------------
def flatten_indices(tasks, T):
    """返回 [(t0, n0), (t1, n1), ...], 顺序与 episode 内决策顺序一致.

    注意: 必须从 tasks (sample_tasks 输出) 直接算, 不能用 env.tasks_bit
    (后者只有在 reset_env 之后才有内容).
    """
    idx = []
    for t in range(T - 1):
        if t < len(tasks):
            for n in range(len(tasks[t])):
                idx.append((t, n))
    return idx


def simulate_with_chromosome(env, tasks, E, f_E, tran_rate, omega, chrom):
    """按给定动作染色体跑完整 episode, 返回 (mean_delay, mean_energy)."""
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n = 0.0, 0.0, 0
    k = 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for j in range(T_len):
            a = int(chrom[k])
            # 防越界 → 回退到 0 (env.step 内还有一次保护)
            if a < 0 or a >= E:
                a = 0
            _, _, d, e, _ = env.step(t, j, a)
            sum_d += d; sum_e += e; n += 1
            k += 1
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def sa_optimize(env, tasks, E, f_E, tran_rate, omega, rng,
                n_iter=None, T0=1.0, T_min=0.01, decay=None,
                init_chrom=None):
    """单 (omega, episode) 上的 SA 优化, 返回 (best_d, best_e, best_chrom)."""
    flat_idx = flatten_indices(tasks, env.time_slots)
    L = len(flat_idx)
    if L == 0:
        return 0.0, 0.0, np.zeros(0, dtype=np.int32)

    # 自适应迭代数 & 衰减率
    if n_iter is None:
        n_iter = max(200, 2 * L)
    if decay is None:
        # 让 T 从 T0 衰减到 T_min 恰好 n_iter 步
        decay = (T_min / T0) ** (1.0 / max(n_iter - 1, 1))

    # 初始解
    if init_chrom is None:
        chrom = rng.integers(0, E, size=L).astype(np.int32)
    else:
        chrom = np.asarray(init_chrom, dtype=np.int32).copy()

    d_cur, e_cur = simulate_with_chromosome(env, tasks, E, f_E, tran_rate, omega, chrom)
    cost_cur = float(omega[0]) * d_cur + float(omega[1]) * e_cur

    best_chrom = chrom.copy()
    best_d, best_e, best_cost = d_cur, e_cur, cost_cur

    T = T0
    for it in range(n_iter):
        # 邻域大小 K: 从 max(2, L//20) 衰到 1
        K_max = max(2, L // 20)
        K = max(1, int(round(K_max * (T - T_min) / max(T0 - T_min, 1e-9))))

        # 提议: 在 K 个随机位置上随机重赋值
        cand = chrom.copy()
        pos = rng.choice(L, size=K, replace=False)
        cand[pos] = rng.integers(0, E, size=K)

        d_c, e_c = simulate_with_chromosome(env, tasks, E, f_E, tran_rate, omega, cand)
        cost_c = float(omega[0]) * d_c + float(omega[1]) * e_c

        dE = cost_c - cost_cur
        if dE <= 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
            chrom = cand
            cost_cur, d_cur, e_cur = cost_c, d_c, e_c
            if cost_c < best_cost:
                best_chrom = cand.copy()
                best_d, best_e, best_cost = d_c, e_c, cost_c
        T *= decay

    return best_d, best_e, best_chrom


# ------------------------------------------------------------
# Pareto 评估: 与 Heuristic 一致协议
# ------------------------------------------------------------
def evaluate_pareto(env, n_pref, n_eval_epi, seed, n_iter):
    prefs = build_preference_set(n_pref)
    points = []
    for omega in prefs:
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
    return np.array(points)


def main():
    cfg = dict(
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        # SA 超参 (跟 NSGA-II 大致同量级以便公平比较)
        n_iter=200,
        task_mode='random', task_kwargs=dict(),
    )
    print('[SA] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    pts_per_seed = []
    t0 = time.time()
    for seed in cfg['seeds']:
        env = MOFDEnvironment(
            Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
            bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
            f_range=cfg['f_range'], seed=seed,
        )
        pts = evaluate_pareto(env, n_pref=cfg['final_eval_n_pref'],
                              n_eval_epi=cfg['final_eval_n_epi'],
                              seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                              n_iter=cfg['n_iter'])
        pts_per_seed.append(pts)
        print(f'[SA] seed {seed} done  pts={len(pts)}  '
              f'mean_delay={pts[:,0].mean():.3f}s  mean_energy={pts[:,1].mean():.3f}J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'sa_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    # 本地 HV (compare_hv.py 会用 random 重算统一参考点)
    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'sa_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Simulated Annealing (SA) Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\nHV mean ± std: {mu:.4f} ± {sd:.4f}\n')
    print(f'[SA] HV (local ref) = {mu:.4f} ± {sd:.4f}')

    # 单独 Pareto 图
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(all_pts[:, 0], all_pts[:, 1], color='tab:pink', marker='D',
               s=30, alpha=0.6, label=f'SA (HV={mu:.2f})')
    pf = pareto_front_2d(all_pts)
    if len(pf) > 0:
        ax.plot(pf[:, 0], pf[:, 1], color='tab:pink', alpha=0.7)
    ax.set_xlabel('Avg Delay (s)'); ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'SA Pareto Front (pooled {len(cfg["seeds"])} seeds)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sa_pareto.png'), dpi=150)
    plt.close()

    print(f'[SA] Done. Total time {time.time()-t0:.1f}s. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
