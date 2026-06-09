"""
NSGA-II baseline V2 (on MOFDEnvironmentV2 with cloud action)
=============================================================
跟 GMORL 论文对齐: action space includes cloud (action 0..Emax).
输出: results/nsga2_v2_pareto_aggregated.csv
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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def simulate_chrom(env, tasks, E, f_E, tran_rate, omega_for_reset, chrom):
    env.reset_env(tasks, E, f_E, tran_rate, omega_for_reset)
    sum_d, sum_e, n = 0.0, 0.0, 0
    k = 0
    for t in range(env.time_slots - 1):
        for j in range(len(env.tasks_bit[t])):
            a = int(chrom[k])
            if a < 0 or a >= env.action_dim or env.valid_mask[a] < 0.5:
                a = 0
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n += 1
            k += 1
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def flatten_len(tasks, T):
    L = 0
    for t in range(T - 1):
        if t < len(tasks):
            L += len(tasks[t])
    return L


class MOFDProblemV2(Problem):
    def __init__(self, env, tasks, E, f_E, tran_rate, omega_for_reset, L):
        # V2: xu = action_dim - 1 = Emax (含 cloud)
        super().__init__(n_var=L, n_obj=2, n_ieq_constr=0,
                         xl=0, xu=env.action_dim - 1, vtype=int)
        self.env = env; self.tasks = tasks
        self.E = E; self.f_E = f_E; self.tran_rate = tran_rate
        self.omega_for_reset = omega_for_reset

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2), dtype=np.float64)
        for i in range(X.shape[0]):
            d, e = simulate_chrom(self.env, self.tasks, self.E, self.f_E,
                                  self.tran_rate, self.omega_for_reset,
                                  X[i].astype(np.int32))
            F[i, 0] = d; F[i, 1] = e
        out['F'] = F


def evaluate_pareto(env, n_pref, n_eval_epi, seed, pop_size=40, n_gen=30):
    prefs = build_preference_set(n_pref)
    accum = np.zeros((n_pref, 2), dtype=np.float64)
    accum_cnt = 0
    env_rng = np.random.default_rng(seed)
    for ep in range(n_eval_epi):
        E, f_E, tran_rate, _ = env.sample_context(env_rng)
        tasks = sample_tasks(env, env_rng)
        L = flatten_len(tasks, env.time_slots)
        if L == 0:
            continue
        problem = MOFDProblemV2(env, tasks, E, f_E, tran_rate,
                                omega_for_reset=np.array([0.5, 0.5]), L=L)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=UniformCrossover(prob=0.7),
            mutation=PM(prob=1.0 / max(L, 1), eta=20, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
        res = minimize(problem, algorithm, ('n_gen', n_gen),
                       seed=int(seed + ep + 1), verbose=False)
        pf_F = res.F
        if pf_F is None or len(pf_F) == 0:
            continue
        for i, omega in enumerate(prefs):
            costs = float(omega[0]) * pf_F[:, 0] + float(omega[1]) * pf_F[:, 1]
            best = int(np.argmin(costs))
            accum[i, 0] += float(pf_F[best, 0])
            accum[i, 1] += float(pf_F[best, 1])
        accum_cnt += 1
        print(f'  [NSGA-II-v2] episode {ep+1}/{n_eval_epi}  pf_size={len(pf_F)}',
              flush=True)
    if accum_cnt == 0:
        return np.zeros((n_pref, 2))
    return accum / accum_cnt


def main():
    cfg = dict(
        Emax=6, num_tasks_max=10, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        cloud_f_range=(50, 70), cloud_tran_rate_range=(80, 120),
        kappa=1e-3, cloud_kappa=1e-4,
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        pop_size=40, n_gen=30,
        task_mode='random', task_kwargs=dict(),
    )
    print('[NSGA-II-v2] Config:', cfg)
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
                              pop_size=cfg['pop_size'], n_gen=cfg['n_gen'])
        pts_per_seed.append(pts)
        print(f'[NSGA-II-v2] seed {seed} done  delay range=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]s  '
              f'energy range=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'nsga2_v2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'nsga2_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== NSGA-II-v2 Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\nHV: {mu:.4f} ± {sd:.4f}\n')
    print(f'[NSGA-II-v2] HV (local) = {mu:.4f}  total = {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
