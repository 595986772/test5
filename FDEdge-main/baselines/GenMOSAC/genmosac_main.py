"""
GenMOSAC baseline main (同 MOFD 环境, 同多 seed 协议).
直接与 mofd_main.py 输出文件对齐: results/genmosac_pareto_aggregated.csv 等.
核心差异: 使用 dict-based generalizable state encoder (Set-pool + 队列直方图).
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

from mofd_environment import MOFDEnvironment
from helpers import build_preference_set, hypervolume_2d, pareto_front_2d, SHARED_EVAL_SEED_OFFSET
from mofd_main import sample_tasks, smooth, compute_fixed_ref, set_task_generator
from task_generator import make_task_generator

from genmosac_model import GenMOSAC, DictReplayBuffer, build_dict_state

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


class RewardNormalizer:
    """逐目标奖励归一化 (PopArt-lite, 与 baselines/DiscreteSAC 完全对齐).

    维护 (r_T, r_E) 各自的 running RMS = EMA(r²) 的平方根, 标量化前把两路奖励
    各除以自己的 σ, 抵消 delay(~30) 与 energy(~4) 的量级差. 仅训练期更新.
    """
    def __init__(self, beta=0.01, eps=1e-3):
        self.beta = float(beta)
        self.eps = float(eps)
        self._ms = np.ones(2, dtype=np.float64)   # EMA of r²
        self._init = False

    def update(self, r_T, r_E):
        sq = np.array([r_T, r_E], dtype=np.float64) ** 2
        if not self._init:
            self._ms = np.maximum(sq, self.eps ** 2)
            self._init = True
        else:
            self._ms = (1.0 - self.beta) * self._ms + self.beta * sq

    @property
    def sigma(self):
        return np.sqrt(self._ms)

    def normalize(self, r_T, r_E):
        s = np.maximum(np.sqrt(self._ms), self.eps)
        return r_T / s[0], r_E / s[1]


def run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                n_bins, stochastic=True, buf=None,
                batch_size=64, buffer_warmup=500,
                alpha_T=1.0, alpha_E=0.25, normalizer=None):
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    sum_d, sum_e, n_tasks, losses = 0.0, 0.0, 0, []
    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for n in range(T_len):
            s = build_dict_state(env, t, n, n_bins)
            m = env.get_valid_mask()
            a = agent.take_action(s, m, stochastic=stochastic)
            r_T, r_E, d, e, real_a = env.step(t, n, a)
            if normalizer is not None:
                normalizer.update(r_T, r_E)
                rn_T, rn_E = normalizer.normalize(r_T, r_E)
                r = float(omega[0] * alpha_T * rn_T + omega[1] * alpha_E * rn_E)
            else:
                r = float(omega[0] * alpha_T * r_T + omega[1] * alpha_E * r_E)

            if n == T_len - 1:
                nt, nn = t + 1, 0
            else:
                nt, nn = t, n + 1
            if nt < env.time_slots and nn < len(env.tasks_bit[nt]):
                s_next = build_dict_state(env, nt, nn, n_bins)
                m_next = env.get_valid_mask()
            else:
                s_next, m_next = s, m

            if buf is not None:
                buf.add(s, real_a, m, r, s_next, m_next)

            sum_d += d; sum_e += e; n_tasks += 1

            if buf is not None and buf.size() >= max(buffer_warmup, batch_size):
                losses.append(agent.update(buf.sample(batch_size)))
        env.update_proc_queues(t)
    return sum_d / max(n_tasks, 1), sum_e / max(n_tasks, 1), losses


def evaluate_pareto(env, agent, n_pref, n_eval_epi, seed, alpha_T, alpha_E, n_bins):
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
            d, e, _ = run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                                  n_bins, stochastic=False, buf=None,
                                  alpha_T=alpha_T, alpha_E=alpha_E)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
    return np.array(points)


def run_single_seed(cfg, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = MOFDEnvironment(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'], seed=seed,
    )
    agent = GenMOSAC(
        task_pref_dim=4, per_srv_dim=env.per_server_dim,
        n_bins=cfg['n_bins'], Emax=cfg['Emax'],
        hidden=cfg['hidden_dim'], emb=cfg['emb_dim'],
        actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'],
        alpha=cfg['alpha_init'], alpha_lr=cfg['alpha_lr'],
        target_entropy=cfg['target_entropy'],
        tau=cfg['tau'], gamma=cfg['gamma'],
        device=device,
    )
    buf = DictReplayBuffer(cfg['buf_size'])
    rng = np.random.default_rng(seed + 1)

    normalizer = RewardNormalizer(beta=cfg.get('reward_norm_beta', 0.01)) \
        if cfg.get('use_reward_norm', False) else None

    fixed_ref = compute_fixed_ref(env, seed=seed + 7777)
    print(f"[genmosac seed {seed}] fixed HV ref = ({fixed_ref[0]:.3f}, {fixed_ref[1]:.3f})")

    log_hv, log_d, log_e = [], [], []
    t0 = time.time()
    for epoch in range(cfg['num_epochs']):
        ep_ds, ep_es, ep_losses = [], [], []
        for _ in range(cfg['n_prefs_per_epoch']):
            E, f_E, tran_rate, omega = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            d, e, losses = run_episode(
                env, agent, tasks, E, f_E, tran_rate, omega,
                n_bins=cfg['n_bins'], stochastic=True, buf=buf,
                batch_size=cfg['batch_size'], buffer_warmup=cfg['buffer_warmup'],
                alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
                normalizer=normalizer,
            )
            ep_ds.append(d); ep_es.append(e); ep_losses.extend(losses)

        pts = evaluate_pareto(env, agent,
                              n_pref=cfg['train_eval_n_pref'],
                              n_eval_epi=cfg['train_eval_n_epi'],
                              seed=seed * 100000 + epoch,
                              alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
                              n_bins=cfg['n_bins'])
        hv = hypervolume_2d(pts, fixed_ref)
        log_hv.append(hv)
        log_d.append(float(np.mean(ep_ds)))
        log_e.append(float(np.mean(ep_es)))

        log_interval = max(1, cfg['num_epochs'] // 20)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            h_mean = float(np.mean([l['H'] for l in ep_losses])) if ep_losses else 0.0
            a_mean = float(np.mean([l['alpha'] for l in ep_losses])) if ep_losses else 0.0
            print(f"[genmosac seed {seed} epoch {epoch + 1:03d}/{cfg['num_epochs']}] "
                  f"HV={hv:.4f} d={log_d[-1]:.4f} e={log_e[-1]:.4f} "
                  f"alpha={a_mean:.4f} H(π)={h_mean:.3f} elapsed={time.time() - t0:.1f}s")

    pts_final = evaluate_pareto(env, agent,
                                n_pref=cfg['final_eval_n_pref'],
                                n_eval_epi=cfg['final_eval_n_epi'],
                                seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                                alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
                                n_bins=cfg['n_bins'])

    _name = cfg.get('file_tag') or 'genmosac'
    _nb = cfg['n_bins']

    # 存 checkpoint (策略网络含 DictEncoder + 重建元信息): 评估与训练解耦, 换卷子可离线重评
    try:
        from eval_baselines_on_testset import save_baseline_ckpt
        save_baseline_ckpt('genmosac', _name, agent, seed,
                           ctor_meta=dict(task_pref_dim=4,
                                          per_srv_dim=env.per_server_dim,
                                          n_bins=cfg['n_bins'], Emax=cfg['Emax'],
                                          hidden=cfg['hidden_dim'], emb=cfg['emb_dim']),
                           n_bins=cfg['n_bins'])
    except Exception as _e:
        print(f'[genmosac] ckpt save skipped: {_e}')

    # 路 B: 在固定卷子(校准秤)上评估, 落到统一可比表 results/testset_compare.*
    # genmosac 用 dict 状态, 故传 state_builder = build_dict_state(env,t,n,n_bins)
    try:
        from eval_baselines_on_testset import evaluate_trained_agent
        evaluate_trained_agent(_name, agent,
                               state_builder=lambda env, t, n: build_dict_state(env, t, n, _nb),
                               k_eval=cfg.get('testset_k', 20))
    except Exception as _e:
        print(f'[genmosac] testset eval skipped: {_e}')

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
        # ---- 环境 (与 mofd_main.py 保持一致) ----
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        # ---- 训练 ----
        num_epochs=100, n_prefs_per_epoch=8,
        seeds=[0], smooth_window=5,
        train_eval_n_pref=11, train_eval_n_epi=1,
        final_eval_n_pref=21, final_eval_n_epi=3,
        # ---- 奖励归一化 (与 baselines/DiscreteSAC 对齐, 抵消 r_T/r_E 量级差) ----
        use_reward_norm=True, reward_norm_beta=0.01,
        # ---- GenMOSAC 超参 ----
        alpha_T=1.0, alpha_E=0.25,
        actor_lr=1e-4, critic_lr=1e-3,
        alpha_init=0.05, alpha_lr=3e-4,
        tau=0.005, gamma=0.95,
        hidden_dim=128, emb_dim=64, target_entropy=0.5,   # 离散 |A|=6, H∈[0,log6≈1.79]; -1.0 不可达→α崩塌, 对齐 PC-FDN 取 0.5
        n_bins=10,   # 队列负载直方图 bin 数
        buf_size=10000, batch_size=64, buffer_warmup=500,
        # ---- 任务到达模式 ('random' / 'poisson' / 'trace') ----
        task_mode='random',
        task_kwargs=dict(),
    )
    print('[genmosac] Config:', cfg)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[genmosac] Device:', device)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    all_runs = []
    t_all = time.time()
    for seed in cfg['seeds']:
        print(f"\n========== [genmosac] Seed {seed} ==========")
        all_runs.append(run_single_seed(cfg, seed, device))
    print(f"\n[genmosac] all seeds done in {time.time() - t_all:.1f}s")

    hv_all = np.stack([r['log_hv'] for r in all_runs], axis=0)
    d_all = np.stack([r['log_d'] for r in all_runs], axis=0)
    e_all = np.stack([r['log_e'] for r in all_runs], axis=0)
    pts_pool = np.vstack([r['pts'] for r in all_runs])

    w = cfg['smooth_window']
    def agg(arr): return smooth(arr.mean(0), w), smooth(arr.std(0), w)
    hv_m, hv_s = agg(hv_all); d_m, d_s = agg(d_all); e_m, e_s = agg(e_all)

    np.savetxt(os.path.join(RESULTS_DIR, 'genmosac_hv_all_seeds.csv'), hv_all, fmt='%.5f')
    np.savetxt(os.path.join(RESULTS_DIR, 'genmosac_hv_mean_std.csv'),
               np.stack([hv_m, hv_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'genmosac_avg_delay_mean_std.csv'),
               np.stack([d_m, d_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'genmosac_avg_energy_mean_std.csv'),
               np.stack([e_m, e_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(RESULTS_DIR, 'genmosac_pareto_aggregated.csv'), pts_pool, fmt='%.5f')

    epochs = np.arange(cfg['num_epochs'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, title, mu, sd, color in zip(
            axes, ['Hypervolume', 'Avg Delay (s)', 'Avg Energy (J)'],
            [hv_m, d_m, e_m], [hv_s, d_s, e_s],
            ['tab:cyan', 'tab:orange', 'tab:green']):
        ax.plot(epochs, mu, color=color, linewidth=2)
        ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.25, color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_title(f'[GenMOSAC] {title} ({len(all_runs)} seeds, smooth={w})')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'genmosac_training_curves.png'), dpi=150)
    plt.close()

    ref = (float(pts_pool[:, 0].max() * 1.1 + 1e-6),
           float(pts_pool[:, 1].max() * 1.1 + 1e-6))
    hvs = [hypervolume_2d(r['pts'], ref) for r in all_runs]
    print(f'\n[genmosac] Final HV (ref={ref}): mean={np.mean(hvs):.4f} std={np.std(hvs):.4f}')

    with open(os.path.join(RESULTS_DIR, 'genmosac_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== GenMOSAC Multi-seed Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\n')
        f.write(f'Config: {cfg}\n')
        f.write(f'Local HV ref: {ref}\n')
        f.write(f'HV per seed: {hvs}\n')
        f.write(f'HV mean ± std: {np.mean(hvs):.4f} ± {np.std(hvs):.4f}\n')

    print(f'[genmosac] Done. Results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()
