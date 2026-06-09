"""
NSGA-II baseline (Non-dominated Sorting Genetic Algorithm II)
=============================================================
GMORL 论文 baseline 之一 [36][37]. 多目标进化算法 (真正的 MO solver, 不需 ω).

实现:
  - 用 pymoo (>=0.6.0) 的 NSGA2 求解器
  - 决策变量: 整数向量, 长度 = total_tasks_in_episode, 每位 ∈ [0, E-1]
  - 两目标: (mean_delay, mean_energy), 都最小化
  - 每个 episode 跑一次 NSGA-II → 得到该 episode 的近似 Pareto front
  - 为对齐 compare_hv.py 的 21-ω 格式: 对每个 ω 从 NSGA-II 的 PF 上挑
    标量化 cost = ω_T*d + ω_E*e 最小的点

跨 ω 共享同一次进化结果 (每 episode 只跑一次 NSGA-II), 因此速度比
SA 快 ~21×. 注意结果是 "21 个偏好分别取自同一 PF 上的最优点", 自然在
2D 平面上排布成 Pareto 形状.
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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# ------------------------------------------------------------
# 解码 & 模拟
# ------------------------------------------------------------
def simulate_chrom(env, tasks, E, f_E, tran_rate, omega_for_reset, chrom):
    """跟 SA 的 simulate_with_chromosome 一样, 返回 (mean_delay, mean_energy)."""
    env.reset_env(tasks, E, f_E, tran_rate, omega_for_reset)
    sum_d, sum_e, n = 0.0, 0.0, 0
    k = 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for j in range(T_len):
            a = int(chrom[k])
            if a < 0 or a >= E:
                a = 0
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n += 1
            k += 1
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def flatten_len(tasks, T):
    """同 SA: 必须从 tasks 参数算, 不能用 env.tasks_bit."""
    L = 0
    for t in range(T - 1):
        if t < len(tasks):
            L += len(tasks[t])
    return L


class MOFDProblem(Problem):
    """pymoo 自定义问题: 离散整数变量, 双目标."""

    def __init__(self, env, tasks, E, f_E, tran_rate, omega_for_reset, L):
        super().__init__(n_var=L, n_obj=2, n_ieq_constr=0,
                         xl=0, xu=E - 1, vtype=int)
        self.env = env
        self.tasks = tasks
        self.E = E
        self.f_E = f_E
        self.tran_rate = tran_rate
        self.omega_for_reset = omega_for_reset

    def _evaluate(self, X, out, *args, **kwargs):
        # X: (pop_size, L) 整数矩阵
        F = np.zeros((X.shape[0], 2), dtype=np.float64)
        for i in range(X.shape[0]):
            d, e = simulate_chrom(self.env, self.tasks, self.E,
                                  self.f_E, self.tran_rate,
                                  self.omega_for_reset, X[i].astype(np.int32))
            F[i, 0] = d
            F[i, 1] = e
        out['F'] = F


def evaluate_pareto(env, n_pref, n_eval_epi, seed,
                    pop_size=40, n_gen=20):
    prefs = build_preference_set(n_pref)
    # 21 行 × 2 列 (per-omega 取 PF 上的标量化最优)
    accum = np.zeros((n_pref, 2), dtype=np.float64)
    accum_cnt = 0

    env_rng = np.random.default_rng(seed)

    for ep in range(n_eval_epi):
        E, f_E, tran_rate, _ = env.sample_context(env_rng)
        tasks = sample_tasks(env, env_rng)
        L = flatten_len(tasks, env.time_slots)
        if L == 0:
            continue

        # 用 ω=(0.5, 0.5) 重置 env (NSGA-II 本身不用 ω, 重置只是初始化)
        problem = MOFDProblem(env, tasks, E, f_E, tran_rate,
                              omega_for_reset=np.array([0.5, 0.5]),
                              L=L)

        # UniformCrossover 对整数染色体逐位选父代, 输出仍是整数, 无需 repair.
        # PM 是浮点突变, 必须 RoundingRepair 把结果四舍五入回 [0, E-1] 整数.
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=UniformCrossover(prob=0.7),
            mutation=PM(prob=1.0 / max(L, 1), eta=20, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
        res = minimize(problem, algorithm, ('n_gen', n_gen),
                       seed=int(seed + ep + 1), verbose=False)

        # 取最终种群的 F (非支配集 + 全部)
        pf_F = res.F  # shape (n_nondom, 2)
        if pf_F is None or len(pf_F) == 0:
            continue

        # 对每个 ω 在 pf_F 上挑标量化最优
        for i, omega in enumerate(prefs):
            costs = float(omega[0]) * pf_F[:, 0] + float(omega[1]) * pf_F[:, 1]
            best = int(np.argmin(costs))
            accum[i, 0] += float(pf_F[best, 0])
            accum[i, 1] += float(pf_F[best, 1])
        accum_cnt += 1

    if accum_cnt == 0:
        return np.zeros((n_pref, 2))
    return accum / accum_cnt


def main():
    cfg = dict(
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        # NSGA-II 超参
        pop_size=30,   # 每代种群大小
        n_gen=15,      # 进化代数 (pop_size * n_gen = 450 次模拟 / episode)
        task_mode='random', task_kwargs=dict(),
    )
    print('[NSGA-II] Config:', cfg)
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
                              pop_size=cfg['pop_size'], n_gen=cfg['n_gen'])
        pts_per_seed.append(pts)
        print(f'[NSGA-II] seed {seed} done  pts={len(pts)}  '
              f'mean_delay={pts[:,0].mean():.3f}s  mean_energy={pts[:,1].mean():.3f}J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'nsga2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'nsga2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== NSGA-II Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\nHV mean ± std: {mu:.4f} ± {sd:.4f}\n')
    print(f'[NSGA-II] HV (local ref) = {mu:.4f} ± {sd:.4f}')

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(all_pts[:, 0], all_pts[:, 1], color='tab:brown', marker='X',
               s=30, alpha=0.6, label=f'NSGA-II (HV={mu:.2f})')
    pf = pareto_front_2d(all_pts)
    if len(pf) > 0:
        ax.plot(pf[:, 0], pf[:, 1], color='tab:brown', alpha=0.7)
    ax.set_xlabel('Avg Delay (s)'); ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'NSGA-II Pareto Front (pooled {len(cfg["seeds"])} seeds)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'nsga2_pareto.png'), dpi=150)
    plt.close()

    print(f'[NSGA-II] Done. Total time {time.time()-t0:.1f}s. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
