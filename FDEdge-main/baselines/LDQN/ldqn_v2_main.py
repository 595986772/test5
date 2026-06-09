"""
LDQN baseline V2 (on MOFDEnvironmentV2 with cloud action)
=========================================================
对齐 GMORL 论文 Pareto 协议:
  - action ∈ [0, Emax]  (action 0 = cloud, 1..Emax = edges)
  - num_tasks_max = 10  (V2 默认, 避免 cloud 单服务器队列爆)
  - 其余训练/评估流程与 V1 (ldqn_main.py) 完全一致
输出: results/ldqn_v2_pareto_aggregated.csv 等
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
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mofd_environment_v2 import MOFDEnvironmentV2
from helpers import (build_preference_set, hypervolume_2d, pareto_front_2d,
                     SHARED_EVAL_SEED_OFFSET)
from mofd_main import sample_tasks, smooth, set_task_generator
from task_generator import make_task_generator

from ldqn_model import LDQN, ReplayBuffer

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def compute_fixed_ref_v2(env, seed, scale=1.3, n_pref=11):
    """V2 版 random-action rollout 标定 HV ref. 动作均匀采 [0, action_dim)
    覆盖 cloud 极点, 避免 ref 系统性偏低."""
    rng = np.random.default_rng(seed)
    prefs = build_preference_set(n_pref)
    pts = []
    for omega in prefs:
        E, f_E, tran_rate, _ = env.sample_context(rng)
        tasks = sample_tasks(env, rng)
        env.reset_env(tasks, E, f_E, tran_rate, omega)
        sum_d, sum_e, n = 0.0, 0.0, 0
        for t in range(env.time_slots - 1):
            T_len = len(env.tasks_bit[t])
            for nn in range(T_len):
                action = int(rng.integers(0, env.action_dim))
                _, _, d, e, _ = env.step(t, nn, action)
                sum_d += d; sum_e += e; n += 1
            env.update_proc_queues(t)
        pts.append([sum_d / max(n, 1), sum_e / max(n, 1)])
    pts = np.array(pts)
    return (float(pts[:, 0].max() * scale + 1e-3),
            float(pts[:, 1].max() * scale + 1e-3))


def run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                stochastic=True, buf=None,
                batch_size=64, buffer_warmup=500,
                alpha_T=1.0, alpha_E=0.25):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    agent.reset_hidden()
    sum_d, sum_e, n_tasks, losses = 0.0, 0.0, 0, []
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for n in range(T_len):
            s = env.get_state(t, n)
            m = env.get_valid_mask()
            a = agent.take_action(s, m, stochastic=stochastic)
            r_T, r_E, d, e, real_a = env.step(t, n, a)
            r = float(omega[0] * alpha_T * r_T + omega[1] * alpha_E * r_E)

            if n == T_len - 1:
                nt, nn = t + 1, 0
            else:
                nt, nn = t, n + 1
            if nt < env.time_slots and nn < len(env.tasks_bit[nt]):
                s_next = env.get_state(nt, nn)
                m_next = env.get_valid_mask()
            else:
                s_next, m_next = s, m

            if buf is not None:
                buf.add(s, real_a, r, s_next, m_next)

            sum_d += d; sum_e += e; n_tasks += 1

            if buf is not None and buf.size() >= max(buffer_warmup, batch_size):
                losses.append(agent.update(buf.sample(batch_size)))
        env.update_proc_queues(t)
    return sum_d / max(n_tasks, 1), sum_e / max(n_tasks, 1), losses


def evaluate_pareto(env, agent, n_pref, n_eval_epi, seed, alpha_T, alpha_E):
    prefs = build_preference_set(n_pref)
    points = []
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    for omega in prefs:
        rng = np.random.default_rng(seed)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            d, e, _ = run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                                  stochastic=False, buf=None,
                                  alpha_T=alpha_T, alpha_E=alpha_E)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
    agent.epsilon = saved_eps
    return np.array(points)


def run_single_seed(cfg, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = MOFDEnvironmentV2(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'],
        cloud_f_range=cfg['cloud_f_range'],
        cloud_tran_rate_range=cfg['cloud_tran_rate_range'],
        kappa=cfg['kappa'], cloud_kappa=cfg['cloud_kappa'],
        seed=seed,
    )
    agent = LDQN(
        state_dim=env.state_dim, action_dim=env.action_dim,
        hidden_dim=cfg['hidden_dim'], emb_dim=cfg['emb_dim'],
        learning_rate=cfg['lr'], gamma=cfg['gamma'],
        epsilon_start=cfg['epsilon_start'],
        epsilon_min=cfg['epsilon_min'],
        epsilon_decay=cfg['epsilon_decay'],
        target_update=cfg['target_update'],
        device=device,
    )
    buf = ReplayBuffer(cfg['buf_size'])
    rng = np.random.default_rng(seed + 1)

    fixed_ref = compute_fixed_ref_v2(env, seed=seed + 7777)
    print(f"[ldqn-v2 seed {seed}] fixed HV ref = ({fixed_ref[0]:.3f}, {fixed_ref[1]:.3f})")

    log_hv, log_d, log_e = [], [], []
    t0 = time.time()
    for epoch in range(cfg['num_epochs']):
        ep_ds, ep_es, ep_losses = [], [], []
        for _ in range(cfg['n_prefs_per_epoch']):
            E, f_E, tran_rate, omega = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            d, e, losses = run_episode(
                env, agent, tasks, E, f_E, tran_rate, omega,
                stochastic=True, buf=buf,
                batch_size=cfg['batch_size'], buffer_warmup=cfg['buffer_warmup'],
                alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
            )
            ep_ds.append(d); ep_es.append(e); ep_losses.extend(losses)

        pts = evaluate_pareto(env, agent,
                              n_pref=cfg['train_eval_n_pref'],
                              n_eval_epi=cfg['train_eval_n_epi'],
                              seed=seed * 100000 + epoch,
                              alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'])
        hv = hypervolume_2d(pts, fixed_ref)
        log_hv.append(hv)
        log_d.append(float(np.mean(ep_ds)))
        log_e.append(float(np.mean(ep_es)))

        log_interval = max(1, cfg['num_epochs'] // 20)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            loss_mean = float(np.mean([l['loss'] for l in ep_losses])) if ep_losses else 0.0
            print(f"[ldqn-v2 seed {seed} epoch {epoch + 1:03d}/{cfg['num_epochs']}] "
                  f"HV={hv:.4f} d={log_d[-1]:.4f} e={log_e[-1]:.4f} "
                  f"loss={loss_mean:.4f} eps={agent.epsilon:.3f} "
                  f"elapsed={time.time() - t0:.1f}s")

    pts_final = evaluate_pareto(env, agent,
                                n_pref=cfg['final_eval_n_pref'],
                                n_eval_epi=cfg['final_eval_n_epi'],
                                seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                                alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'])
    return dict(
        seed=seed,
        log_hv=np.array(log_hv),
        log_d=np.array(log_d),
        log_e=np.array(log_e),
        fixed_ref=fixed_ref,
        pts=pts_final,
    )


def main():
    cfg = dict(
        # ---- 环境 (V2 默认, 跟其他 *_v2 baseline 对齐) ----
        Emax=6, num_tasks_max=10, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        cloud_f_range=(50, 70), cloud_tran_rate_range=(80, 120),
        kappa=1e-3, cloud_kappa=1e-4,
        # ---- 训练 ----
        num_epochs=100, n_prefs_per_epoch=8,
        seeds=[0], smooth_window=5,
        train_eval_n_pref=11, train_eval_n_epi=1,
        final_eval_n_pref=21, final_eval_n_epi=3,
        # ---- LDQN 超参 (与 V1 保持一致, 便于纯环境差异对比) ----
        alpha_T=1.0, alpha_E=0.25,
        lr=1e-3, gamma=0.95,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
        target_update=100,
        hidden_dim=128, emb_dim=64,
        buf_size=10000, batch_size=64, buffer_warmup=500,
        # ---- 任务到达模式 ----
        task_mode='random',
        task_kwargs=dict(),
    )
    print('[ldqn-v2] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[ldqn-v2] Device:', device)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    all_runs = []
    t_all = time.time()
    for seed in cfg['seeds']:
        print(f"\n========== [ldqn-v2] Seed {seed} ==========")
        all_runs.append(run_single_seed(cfg, seed, device))
    print(f"\n[ldqn-v2] all seeds done in {time.time() - t_all:.1f}s")

    hv_all = np.stack([r['log_hv'] for r in all_runs], axis=0)
    d_all = np.stack([r['log_d'] for r in all_runs], axis=0)
    e_all = np.stack([r['log_e'] for r in all_runs], axis=0)
    pts_pool = np.vstack([r['pts'] for r in all_runs])

    w = cfg['smooth_window']
    def agg(arr): return smooth(arr.mean(0), w), smooth(arr.std(0), w)
    hv_m, hv_s = agg(hv_all); d_m, d_s = agg(d_all); e_m, e_s = agg(e_all)

    np.savetxt(os.path.join(RESULTS_DIR, 'ldqn_v2_hv_all_seeds.csv'), hv_all, fmt='%.5f')
    np.savetxt(os.path.join(RESULTS_DIR, 'ldqn_v2_hv_mean_std.csv'),
               np.stack([hv_m, hv_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'ldqn_v2_avg_delay_mean_std.csv'),
               np.stack([d_m, d_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'ldqn_v2_avg_energy_mean_std.csv'),
               np.stack([e_m, e_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'ldqn_v2_pareto_aggregated.csv'), pts_pool, fmt='%.5f')

    epochs = np.arange(cfg['num_epochs'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, title, mu, sd, color in zip(
            axes, ['Hypervolume', 'Avg Delay (s)', 'Avg Energy (J)'],
            [hv_m, d_m, e_m], [hv_s, d_s, e_s],
            ['tab:blue', 'tab:orange', 'tab:green']):
        ax.plot(epochs, mu, color=color, linewidth=2)
        ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.25, color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_title(f'[LDQN-v2] {title} ({len(all_runs)} seeds, smooth={w})')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ldqn_v2_training_curves.png'), dpi=150)
    plt.close()

    ref = (float(pts_pool[:, 0].max() * 1.1 + 1e-6),
           float(pts_pool[:, 1].max() * 1.1 + 1e-6))
    hvs = [hypervolume_2d(r['pts'], ref) for r in all_runs]
    print(f'\n[ldqn-v2] Final HV (ref={ref}): mean={np.mean(hvs):.4f} std={np.std(hvs):.4f}')

    with open(os.path.join(RESULTS_DIR, 'ldqn_v2_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== LDQN-v2 Multi-seed Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\n')
        f.write(f'Config: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\n')
        f.write(f'HV mean ± std: {np.mean(hvs):.4f} ± {np.std(hvs):.4f}\n')

    print(f'[ldqn-v2] Done. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
