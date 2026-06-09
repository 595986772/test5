"""
LinUCB baseline V2 (on MOFDEnvironmentV2 with cloud action)
============================================================
跟 GMORL 论文 (C.2 Baselines) 对齐:
  - **单一 agent**, ω 作为 context 一部分, 在 21 个 ω 上联合训练
    (论文原话: "We train this scheme in preference set Ω_101 and
    evaluate it for any preference in one.")
  - action space includes cloud (action 0), 共 Emax+1 个 arm
  - 输出: results/linucb_v2_pareto_aggregated.csv
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


def build_context(env, t, n, a, omega):
    """8 维 context: [bit_norm, comp_norm, ω_T, ω_E,
                       f_a_norm, q_a_norm, g_a_norm, valid_a]."""
    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    f_norm = max(env.cf_max, env.f_max, 1.0)
    q_norm = max(env.cf_max * env.duration * 10.0, 1.0)
    g_norm = max(env.fading_hi, 1e-6)
    bit_norm = d_n / max(env.max_bit, 1.0)
    comp_norm = rho_d / max(env.cd_max * env.max_bit, 1e-6)
    f_a, _, _ = env._action_to_f_v(a)
    return np.array([
        bit_norm, comp_norm,
        float(omega[0]), float(omega[1]),
        float(f_a) / f_norm if env.valid_mask[a] > 0.5 else 0.0,
        float(env.proc_queue_len[t, a]) / q_norm if env.valid_mask[a] > 0.5 else 0.0,
        float(env.channel_gain[t, n, a]) / g_norm,
        float(env.valid_mask[a]),
    ], dtype=np.float64)


class LinUCBAgent:
    def __init__(self, n_arms, d, alpha=1.0, lam=1.0):
        self.n_arms = n_arms; self.d = d
        self.alpha = alpha; self.lam = lam
        self.A_inv = [(1.0 / lam) * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.theta = [np.zeros(d) for _ in range(n_arms)]

    def select(self, contexts, mask, explore=True):
        ucb = np.full(self.n_arms, -np.inf)
        for a in range(self.n_arms):
            if mask[a] < 0.5:
                continue
            x = contexts[a]
            mean = float(self.theta[a] @ x)
            bonus = (self.alpha * float(np.sqrt(max(x @ self.A_inv[a] @ x, 0.0)))
                     if explore else 0.0)
            ucb[a] = mean + bonus
        return int(np.argmax(ucb))

    def update(self, arm, x, reward):
        Ainv = self.A_inv[arm]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[arm] = Ainv - np.outer(Ax, Ax) / max(denom, 1e-12)
        self.b[arm] += reward * x
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]


def run_episode(env, agent, tasks, E, f_E, tran_rate, omega, explore=True):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        for j in range(len(env.tasks_bit[t])):
            contexts = np.array([build_context(env, t, j, a, omega)
                                 for a in range(env.action_dim)])
            mask = env.get_valid_mask()
            a = agent.select(contexts, mask, explore=explore)
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n += 1
            if explore:
                r = -(float(omega[0]) * d + float(omega[1]) * e_val)
                agent.update(a, contexts[a], r)
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def evaluate_pareto(env, n_pref, n_eval_epi, seed,
                    n_warmup_per_omega=20, alpha=1.0, lam=1.0):
    """GMORL 风格: 单 agent 在所有 ω 上联合训练, 然后逐 ω evaluate."""
    prefs = build_preference_set(n_pref)
    d_context = 8
    agent = LinUCBAgent(env.action_dim, d_context, alpha=alpha, lam=lam)

    # ---- joint warmup: 在所有 ω 上轮流采样训练 ----
    train_rng = np.random.default_rng(seed)
    n_train_total = n_warmup_per_omega * n_pref
    for k in range(n_train_total):
        omega = prefs[k % n_pref]    # 轮换 ω
        E, f_E, tran_rate, _ = env.sample_context(train_rng)
        tasks = sample_tasks(env, train_rng)
        run_episode(env, agent, tasks, E, f_E, tran_rate, omega, explore=True)
        if (k + 1) % 50 == 0:
            print(f'  [LinUCB-v2 train] {k+1}/{n_train_total} episodes', flush=True)

    # ---- evaluate: exploit-only on each ω ----
    points = []
    for k_omega, omega in enumerate(prefs):
        env_rng_eval = np.random.default_rng(seed + 1)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(env_rng_eval)
            tasks = sample_tasks(env, env_rng_eval)
            d, e = run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                               explore=False)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
        print(f'  [LinUCB-v2 eval] omega {k_omega+1}/{n_pref}  '
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
        n_warmup_per_omega=20, alpha=1.0, lam=1.0,
        task_mode='random', task_kwargs=dict(),
    )
    print('[LinUCB-v2] Config:', cfg)
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
                              n_warmup_per_omega=cfg['n_warmup_per_omega'],
                              alpha=cfg['alpha'], lam=cfg['lam'])
        pts_per_seed.append(pts)
        print(f'[LinUCB-v2] seed {seed} done  delay range=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]s  '
              f'energy range=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'linucb_v2_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'linucb_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== LinUCB-v2 Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\nHV: {mu:.4f} ± {sd:.4f}\n')
    print(f'[LinUCB-v2] HV (local) = {mu:.4f}  total = {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
