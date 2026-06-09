"""
LinUCB (Linear Upper Confidence Bound) baseline
================================================
GMORL 论文 baseline 之一 [32]. Contextual multi-arm bandit.

算法 (Li et al., WWW 2010 经典 LinUCB):
  - 臂 (arm) = 卸载目的地 e ∈ [0, E-1]
  - 上下文 x ∈ R^d (每 (slot, task) 重新构造):
      [task_bit_norm, comp_load_norm, ω_T, ω_E,
       f_e_norm, q_e_norm, chan_gain_norm, valid_e]  -> 每个 arm 一份
    实际实现: 每个 arm 共享 d 维特征 (任务特征 + 该 arm 的服务器特征)
  - 每个 arm 维护:
      A_e = λI + Σ x x^T     (d × d)
      b_e = Σ r x            (d,)
      θ_e = A_e^{-1} b_e
  - 选臂: a = argmax_e [ θ_e^T x + α * sqrt(x^T A_e^{-1} x) ]    (mask 屏蔽无效)
  - reward = -(ω_T * delay + ω_E * energy)   (越大越好 → 与论文奖励语义对齐)

在线学习 = 训练 + 评估同步.
  - 训练: 前 n_warmup 个 episode, 每步根据 LinUCB 选臂 + 接收 reward 更新 A_e/b_e
  - 评估: 最后 n_eval_epi 个 episode, exploit-only (alpha=0), 报 mean(delay,energy)

每个 ω 单独跑一遍 (因为 reward 函数依赖 ω; 把 ω 喂进 context 后参数本质上是
不同 reward 模型的产物, 单独训练对 LinUCB 更友好且与论文一致).
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
# 上下文向量构造: 任务侧 + 单 arm 侧, 共 d 维
# ------------------------------------------------------------
def build_context(env, t, n, e, omega):
    """8 维上下文: [d_n_norm, comp_norm, ω_T, ω_E, f_e_norm, q_e_norm, g_e_norm, valid_e]."""
    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    f_norm = max(env.f_max, 1.0)
    q_norm = max(env.f_max * env.duration * 10.0, 1.0)
    g_norm = max(env.fading_hi, 1e-6)
    bit_norm = d_n / max(env.max_bit, 1.0)
    comp_norm = rho_d / max(env.cd_max * env.max_bit, 1e-6)
    return np.array([
        bit_norm, comp_norm,
        float(omega[0]), float(omega[1]),
        float(env.f_E[e]) / f_norm if env.valid_mask[e] > 0.5 else 0.0,
        float(env.proc_queue_len[t, e]) / q_norm if env.valid_mask[e] > 0.5 else 0.0,
        float(env.channel_gain[t, n, e]) / g_norm,
        float(env.valid_mask[e]),
    ], dtype=np.float64)


class LinUCBAgent:
    """每个 arm 独立维护 A_e, b_e (即论文 disjoint LinUCB)."""

    def __init__(self, n_arms, d, alpha=1.0, lam=1.0):
        self.n_arms = n_arms; self.d = d
        self.alpha = alpha; self.lam = lam
        self.A = [lam * np.eye(d) for _ in range(n_arms)]
        self.A_inv = [(1.0 / lam) * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.theta = [np.zeros(d) for _ in range(n_arms)]

    def select(self, contexts, mask, explore=True):
        """contexts: (n_arms, d).  mask: (n_arms,) 0/1.  return arm idx."""
        ucb = np.full(self.n_arms, -np.inf)
        for e in range(self.n_arms):
            if mask[e] < 0.5:
                continue
            x = contexts[e]
            mean = float(self.theta[e] @ x)
            if explore:
                bonus = self.alpha * float(np.sqrt(max(x @ self.A_inv[e] @ x, 0.0)))
            else:
                bonus = 0.0
            ucb[e] = mean + bonus
        return int(np.argmax(ucb))

    def update(self, arm, x, reward):
        """Sherman-Morrison 更新 A_inv (避免每步求逆 O(d^3))."""
        Ainv = self.A_inv[arm]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[arm] = Ainv - np.outer(Ax, Ax) / max(denom, 1e-12)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]


def run_episode(env, agent, tasks, E, f_E, tran_rate, omega, explore=True):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n = 0.0, 0.0, 0
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for j in range(T_len):
            contexts = np.array([build_context(env, t, j, e, omega)
                                 for e in range(env.Emax)])
            mask = env.get_valid_mask()
            a = agent.select(contexts, mask, explore=explore)
            _, _, d, e_val, _ = env.step(t, j, a)
            sum_d += d; sum_e += e_val; n += 1
            if explore:
                # 标量化 reward, 取负号 (LinUCB 默认 reward 越大越好)
                r = -(float(omega[0]) * d + float(omega[1]) * e_val)
                agent.update(a, contexts[a], r)
        env.update_proc_queues(t)
    return sum_d / max(n, 1), sum_e / max(n, 1)


def evaluate_pareto(env, n_pref, n_eval_epi, seed,
                    n_warmup=8, alpha=1.0, lam=1.0):
    prefs = build_preference_set(n_pref)
    points = []
    d_context = 8  # build_context 维度
    for omega in prefs:
        env_rng = np.random.default_rng(seed)
        # 每个 ω 独立 agent
        agent = LinUCBAgent(env.Emax, d_context, alpha=alpha, lam=lam)
        # ---- warmup (online 学习) ----
        for _ in range(n_warmup):
            E, f_E, tran_rate, _ = env.sample_context(env_rng)
            tasks = sample_tasks(env, env_rng)
            run_episode(env, agent, tasks, E, f_E, tran_rate, omega, explore=True)
        # ---- eval (exploit-only) ----
        # 重置 env_rng 让 21 个 ω 看相同评估 task
        env_rng_eval = np.random.default_rng(seed + 1)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(env_rng_eval)
            tasks = sample_tasks(env, env_rng_eval)
            d, e = run_episode(env, agent, tasks, E, f_E, tran_rate, omega, explore=False)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
    return np.array(points)


def main():
    cfg = dict(
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        seeds=[0],
        final_eval_n_pref=21, final_eval_n_epi=3,
        # LinUCB 超参
        n_warmup=20,   # 每 ω 在线学习 episode 数
        alpha=1.0,     # UCB 置信宽度
        lam=1.0,       # 岭回归正则
        task_mode='random', task_kwargs=dict(),
    )
    print('[LinUCB] Config:', cfg)
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
                              n_warmup=cfg['n_warmup'], alpha=cfg['alpha'],
                              lam=cfg['lam'])
        pts_per_seed.append(pts)
        print(f'[LinUCB] seed {seed} done  pts={len(pts)}  '
              f'mean_delay={pts[:,0].mean():.3f}s  mean_energy={pts[:,1].mean():.3f}J  '
              f'elapsed={time.time()-t0:.1f}s')

    all_pts = np.vstack(pts_per_seed)
    np.savetxt(os.path.join(RESULTS_DIR, 'linucb_pareto_aggregated.csv'),
               all_pts, fmt='%.5f')

    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))
    per_seed = np.array(pts_per_seed)
    hvs = [hypervolume_2d(per_seed[s], ref) for s in range(per_seed.shape[0])]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))

    with open(os.path.join(RESULTS_DIR, 'linucb_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== LinUCB Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\nConfig: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\nHV mean ± std: {mu:.4f} ± {sd:.4f}\n')
    print(f'[LinUCB] HV (local ref) = {mu:.4f} ± {sd:.4f}')

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(all_pts[:, 0], all_pts[:, 1], color='tab:olive', marker='P',
               s=30, alpha=0.6, label=f'LinUCB (HV={mu:.2f})')
    pf = pareto_front_2d(all_pts)
    if len(pf) > 0:
        ax.plot(pf[:, 0], pf[:, 1], color='tab:olive', alpha=0.7)
    ax.set_xlabel('Avg Delay (s)'); ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'LinUCB Pareto Front (pooled {len(cfg["seeds"])} seeds)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'linucb_pareto.png'), dpi=150)
    plt.close()

    print(f'[LinUCB] Done. Total time {time.time()-t0:.1f}s. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
