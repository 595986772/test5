"""
ω-Drift (Time-Varying Preference) Experiment
=============================================
单 episode 内 ω 中途切换, 测 OmegaLatentBuffer 的 "drift-aware" 在线适应能力.

Drift schedules (每 episode T 个 slot):
  - sudden:   T/2 处突变 ω = (0,1) → (1,0)
  - gradual:  T 个 slot 线性 ω = (0,1) → (1,0)
  - cyclic:   5 段, 每段 T/5 slot, ω 取 prefs[0,5,10,15,20]

Methods:
  - C0          : 训练/评估都不用 buffer; 评估 fb 候选 = 均匀先验广播
  - C2-passive  : 训练用 buffer; 评估起点 fb = retrieve_prior(ω₀) 广播; 漂移后不重置
  - C2-aware    : 训练用 buffer; 漂移到新 ω 时 prior=retrieve_prior(ω_new), latent_slice[t:] 取 prior 广播
  注: #3 Tier-1 后 fb 候选不再来自 retrieve(shape=...)+retrieve_noise (与 rd=randn 同源),
      改成 retrieve_prior 广播, 让 H-MCSS Q-net 学到 “fb=pr (光滑) vs rd (噪声)” 二分.

Outputs → results_omega_drift/:
  delay_<method>_<schedule>.csv       — [eval_eps, T] per-slot 平均延迟
  energy_<method>_<schedule>.csv      — [eval_eps, T] per-slot 平均能耗
  drift_curves.png                    — 6 子图 (3 schedule × 2 metric)
  comparison.txt                      — 恢复时间表 + 速率比较

Note: 训练阶段沿用 mofd_main.run_single_seed 标准流程 (21 ω cycling),
      drift 仅在评估阶段出现, 测的是策略对 OOD ω 轨迹的实时适应能力.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
1
from mofd_main import (run_single_seed, set_task_generator, sample_tasks,
                       make_env_ctx, load_agent_from_ckpt)
from mofd_environment import MOFDEnvironment
from drift_detector import CUSUMDetector
from drift_detector_v7 import CUSUMRLDetector
from helpers import build_preference_set
from task_generator import make_task_generator


# ------------------------------------------------------------
# Drift 调度构造
# ------------------------------------------------------------
def build_drift_schedules(T, prefs):
    """
    Returns: dict[str, ndarray of shape [T, 2]]  — 每 slot 的 ω
    prefs: 21-ω 评估集 (build_preference_set(21))
    """
    schedules = {}

    # 1) sudden: T/2 突变 (0, 1) → (1, 0)
    sched = np.zeros((T, 2), dtype=np.float32)
    sched[:T // 2] = prefs[0]      # (0, 1)
    sched[T // 2:] = prefs[-1]     # (1, 0)
    schedules['sudden'] = sched

    # 2) gradual: 线性插值
    alpha = np.linspace(0.0, 1.0, T).astype(np.float32)
    sched = np.stack([alpha, 1.0 - alpha], axis=1)
    schedules['gradual'] = sched

    # 3) cyclic: 5 段
    sched = np.zeros((T, 2), dtype=np.float32)
    seg_len = T // 5
    pref_idx = [0, 5, 10, 15, 20]
    for i, idx in enumerate(pref_idx):
        sched[i * seg_len:(i + 1) * seg_len] = prefs[idx]
    sched[5 * seg_len:] = prefs[pref_idx[-1]]
    schedules['cyclic'] = sched

    return schedules


# ------------------------------------------------------------
# 单 episode drift 评估
# ------------------------------------------------------------
def _refresh_prior(omega_now, prior_source, agent, omega_buf, env_ctx, action_dim):
    """统一的 prior 刷新入口.

    prior_source = 'buffer'   → omega_buf.retrieve_prior  (V5/C2-aware 历史路径)
    prior_source = 'hypernet' → agent.compute_prior        (V6/V7 hypernet 路径)
    fallback                  → uniform (action_dim,)
    """
    if prior_source == 'hypernet' and hasattr(agent, 'compute_prior'):
        return agent.compute_prior(np.asarray(omega_now, dtype=np.float32),
                                   env_ctx_np=env_ctx)
    if omega_buf is not None:
        return omega_buf.retrieve_prior(
            omega_now, env_ctx=env_ctx, action_dim=action_dim)
    return np.full(action_dim, 1.0 / action_dim, dtype=np.float32)


def run_drift_episode(env, agent, tasks, E, f_E, tran_rate,
                      omega_schedule, latent_slice,
                      omega_buf=None, drift_aware=False, drift_tol=1e-3,
                      env_ctx=None, prior_latent=None,
                      cusum_detector=None,
                      prior_source='buffer',
                      detector_type='none'):
    """
    omega_schedule: [T, 2] — per-slot ω
    latent_slice:   [T, N, Emax] — initial latent (caller 提供)
    drift_aware:    若 True, 检测到 ω 变化时刷新 prior 与剩余 latent_slice
    cusum_detector: CUSUMDetector / CUSUMRLDetector 实例; 与 detector_type 配套
    detector_type:  'none' = oracle (cur_omega != prev) 触发
                    'cusum' = 监测 slot reward (V6)
                    'cusum_rl' = 监测 TD-error (V7, 要求 agent.compute_td_error)
    prior_source:   'buffer' = omega_buf.retrieve_prior
                    'hypernet' = agent.compute_prior  (V6/V7)
    env_ctx:        episode-level config 上下文
    prior_latent:   [Emax] episode-level 持久先验; None 时均匀
    Returns: delays_per_slot[T], energies_per_slot[T]
    """
    T = env.time_slots
    env.reset_env(tasks, E, f_E, tran_rate, omega_schedule[0])

    if prior_latent is None:
        prior_latent = np.full(env.action_dim, 1.0 / env.action_dim, dtype=np.float32)
    prior_latent = np.asarray(prior_latent, dtype=np.float32)

    delays = np.zeros(T, dtype=np.float32)
    energies = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)

    prev_omega = omega_schedule[0].copy()
    use_cusum = (detector_type in ('cusum', 'cusum_rl')) and (cusum_detector is not None)
    triggers = []  # per-episode 漂移触发时刻 (slot index)

    for t in range(T - 1):
        cur_omega = omega_schedule[t]
        env.omega = cur_omega.astype(np.float32)

        # 触发源:
        #   - detector_type='none' → oracle (cur_omega != prev_omega)
        #   - detector_type in (cusum, cusum_rl) → 由 inner loop 后判定
        oracle_trigger = (not use_cusum
                          and not np.allclose(cur_omega, prev_omega, atol=drift_tol))

        if drift_aware and oracle_trigger:
            prior_latent = _refresh_prior(
                cur_omega, prior_source, agent, omega_buf, env_ctx, env.action_dim)
            full_warm = np.broadcast_to(
                prior_latent, (T, env.n_tasks_max, env.action_dim)
            ).astype(np.float32).copy()
            latent_slice[t:] = full_warm[t:]
            triggers.append(int(t))

        T_len = len(env.tasks_bit[t])
        last_state = last_action = last_mask = None
        last_r_T = last_r_E = 0.0

        for n in range(T_len):
            state = env.get_state(t, n)
            mask = env.get_valid_mask()
            latent = latent_slice[t, n].copy()
            action, probs = agent.take_action(state, latent, prior_latent, mask, stochastic=False)
            latent_slice[t, n] = probs.astype(np.float32)
            r_T, r_E, delay, energy, _ = env.step(t, n, action)
            delays[t] += float(delay)
            energies[t] += float(energy)
            counts[t] += 1
            last_state, last_action, last_mask = state, action, mask
            last_r_T, last_r_E = float(r_T), float(r_E)

        env.update_proc_queues(t)

        # 漂移检测 (per-slot)
        triggered = False
        if drift_aware and use_cusum and counts[t] > 0:
            if detector_type == 'cusum':
                # V6 raw-reward: 用 slot 平均负延迟当 reward 信号
                slot_reward = -float(delays[t]) / float(counts[t])
                triggered = cusum_detector.update(slot_reward)
            elif detector_type == 'cusum_rl' and last_state is not None and t + 1 < T:
                # V7 TD-error: 用本 slot 最后一步的 (s, a, r_vec, s') 算 TD-err
                next_state = env.get_state(t + 1, 0)
                next_mask = env.get_valid_mask()
                next_latent = latent_slice[t + 1, 0].copy()
                r_vec = np.array([last_r_T, last_r_E], dtype=np.float32)
                try:
                    td_err = agent.compute_td_error(
                        last_state, last_action, r_vec, next_state,
                        next_latent, prior_latent, last_mask, next_mask)
                    triggered = cusum_detector.update(td_err)
                except AttributeError:
                    # agent 不支持 (V5/V6) → 静默降级为不触发
                    triggered = False

        if triggered:
            prior_latent = _refresh_prior(
                cur_omega, prior_source, agent, omega_buf, env_ctx, env.action_dim)
            full_warm = np.broadcast_to(
                prior_latent, (T, env.n_tasks_max, env.action_dim)
            ).astype(np.float32).copy()
            latent_slice[t + 1:] = full_warm[t + 1:]
            triggers.append(int(t))

        prev_omega = cur_omega.copy()

    counts = np.maximum(counts, 1)
    return delays / counts, energies / counts, triggers


def evaluate_drift(env, agent, schedule, n_episodes, seed,
                   omega_buf=None, drift_aware=False, init_from_buffer=False,
                   detector_type='none', cusum_kwargs=None,
                   prior_source='buffer'):
    """
    重复 n_episodes 次同一 schedule, 返回 [n_episodes, T] 的 delay 和 energy 矩阵.
    init_from_buffer: episode 起点 prior 来源
                      ('hypernet' → agent.compute_prior, 'buffer' → omega_buf, 否则 uniform)
    detector_type:    'none' (oracle) / 'cusum' (raw reward) / 'cusum_rl' (TD-error)
    prior_source:     drift 触发后 prior 刷新来源
    cusum_kwargs:     CUSUM[RL]Detector 构造参数
    """
    T = env.time_slots
    delays_runs, energies_runs, triggers_runs = [], [], []
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        E, f_E, tran_rate, _ = env.sample_context(rng)
        tasks = sample_tasks(env, rng)
        env_ctx = make_env_ctx(E, f_E, tran_rate, Emax=env.Emax)
        # 起点 prior
        if init_from_buffer and prior_source == 'hypernet' and hasattr(agent, 'compute_prior'):
            prior_latent = agent.compute_prior(
                np.asarray(schedule[0], dtype=np.float32), env_ctx_np=env_ctx)
        elif init_from_buffer and omega_buf is not None:
            prior_latent = omega_buf.retrieve_prior(
                schedule[0], env_ctx=env_ctx, action_dim=env.action_dim)
        else:
            prior_latent = np.full(env.action_dim, 1.0 / env.action_dim,
                                    dtype=np.float32)
        latent = np.broadcast_to(
            prior_latent, (T, env.n_tasks_max, env.action_dim)
        ).astype(np.float32).copy()
        # 每个 episode 新建一个 detector 实例 (跨 episode 不共享状态)
        cusum = None
        if detector_type in ('cusum', 'cusum_rl'):
            kw = dict(cusum_kwargs or {})
            w = int(kw.get('window', 20))
            th = float(kw.get('threshold', 3.0))
            dp = float(kw.get('drift_param', 0.5))
            if detector_type == 'cusum_rl':
                cusum = CUSUMRLDetector(window=w, threshold=th, drift_param=dp)
            else:
                cusum = CUSUMDetector(window=w, threshold=th, drift_param=dp)
        d, e, trig = run_drift_episode(env, agent, tasks, E, f_E, tran_rate,
                                       schedule, latent,
                                       omega_buf=omega_buf,
                                       drift_aware=drift_aware,
                                       env_ctx=env_ctx,
                                       prior_latent=prior_latent,
                                       cusum_detector=cusum,
                                       prior_source=prior_source,
                                       detector_type=detector_type)
        delays_runs.append(d)
        energies_runs.append(e)
        triggers_runs.append(trig)
    return np.array(delays_runs), np.array(energies_runs), triggers_runs


# ------------------------------------------------------------
# 恢复时间分析 (sudden schedule)
# ------------------------------------------------------------
def compute_recovery_time(delays, t_drift, tail_window=10, tol_factor=1.1):
    """
    漂移后 delay 多少 slot 才能稳定到 (尾部 tail_window 平均) × tol_factor 以内.
    返回 -1 表示未恢复.
    """
    target = float(delays[-tail_window:].mean())
    threshold = target * tol_factor
    for t in range(t_drift, len(delays)):
        if float(delays[t]) <= threshold:
            return t - t_drift
    return -1


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def main():
    base_cfg = dict(
        # 环境
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        # 训练 (短于主实验, 重点是 drift 评估)
        num_epochs=50, n_prefs_per_epoch=8,
        seeds=[0],
        smooth_window=5,
        train_eval_n_pref=11, train_eval_n_epi=1,
        final_eval_n_pref=21, final_eval_n_epi=2,
        # SAC
        # 与 mofd_main 改造保持一致: 真凸组合 (alpha_E=1.0), 通道归一化在 env 内
        alpha_T=1.0, alpha_E=1.0,
        actor_lr=1e-4, critic_lr=1e-3,
        alpha_init=0.05, alpha_lr=3e-4,
        tau=0.005, gamma=0.95,
        denoising_steps=3, hidden_dim=128,
        # 修复 α 崩塌: 与 mofd_main 对齐, 旧 -1.0 是不可达负数导致 α → 0
        target_entropy=0.5,
        buf_size=10000, batch_size=64, buffer_warmup=500,
        update_every=4,
        # ω-buffer
        obuf_decay=0.5, obuf_noise=0.05,
        # 纯 V5 (vector-Q + COR + PopArt-lite); V6/V7 模块 (Hypernet /
        # Diversity / NaP / CUSUM-RL) 已关闭 — 实测对漂移恢复无效甚至有害.
        use_v7=False, use_v6=False, use_v5=True,
        use_cor=True, cor_lambda=0.1, cor_c=0.0,
        use_popart=True, popart_beta=0.001,
        # V4 Envelope-MORL (默认关闭)
        use_envelope=False, n_relabel_omegas=4,
        # 任务模式
        task_mode='random', task_kwargs=dict(),
    )

    # ---- CLI: 支持从训练好的 ckpt 加载, 跳过训练 ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-c0', type=str, default=None,
                        help='C0 (no-buffer) ckpt 目录, 形如 results/mofd_*/ckpt_seed0')
    parser.add_argument('--ckpt-c2', type=str, default=None,
                        help='C2 (with-buffer) ckpt 目录')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖 num_epochs (用于快速验证)')
    args, _ = parser.parse_known_args()
    use_ckpt = args.ckpt_c0 is not None and args.ckpt_c2 is not None
    if args.epochs is not None:
        base_cfg['num_epochs'] = int(args.epochs)
        print(f'[drift] override num_epochs = {base_cfg["num_epochs"]}')

    # ---- 输出目录: 每次运行一个 timestamp 子目录 ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_root = 'results_omega_drift'
    results_root = os.path.join(base_root, f'drift_{ts}')
    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**base_cfg, 'ckpt_c0': args.ckpt_c0, 'ckpt_c2': args.ckpt_c2},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'[drift] run dir: {os.path.abspath(results_root)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[drift] device={device}')
    set_task_generator(make_task_generator(base_cfg['task_mode'],
                                           **base_cfg.get('task_kwargs', {})))

    # ===== 加载 or 训练 C0 / C2 =====
    trained = {}
    if use_ckpt:
        print('\n' + '=' * 60)
        print('=== Loading C0 / C2 from ckpts (skip training) ===')
        print('=' * 60)
        seed = base_cfg['seeds'][0]
        for tag, ckpt_dir in [('C0', args.ckpt_c0), ('C2', args.ckpt_c2)]:
            env = MOFDEnvironment(
                Emax=base_cfg['Emax'], num_tasks_max=base_cfg['num_tasks_max'],
                bit_range=base_cfg['bit_range'], time_slots=base_cfg['time_slots'],
                f_range=base_cfg['f_range'], seed=seed,
            )
            # V6/V7 模块已关闭, C0/C2 均为纯 V5 (差别仅 ω-buffer)
            agent, obuf = load_agent_from_ckpt(ckpt_dir, base_cfg, env, device)
            trained[tag] = dict(agent=agent, env=env, omega_buf=obuf)
            print(f'[{tag}] loaded from {ckpt_dir} (obuf={"yes" if obuf else "no"})')
    else:
        print('\n' + '=' * 60)
        print('=== Training C0 (no buffer) and C2 (with buffer) ===')
        print('=' * 60)
        for tag, use_buf in [('C0', False), ('C2', True)]:
            cfg = dict(base_cfg)
            cfg['use_omega_buffer'] = use_buf
            cfg['results_dir'] = results_root
            cfg['file_prefix'] = f'train_{tag}'
            # V6/V7 模块已关闭, C0/C2 均为纯 V5 (差别仅 ω-buffer)
            cfg['use_v5'] = True
            cfg['use_v6'] = False
            cfg['use_v7'] = False
            cfg['use_hypernet'] = False
            seed = cfg['seeds'][0]
            print(f'\n--- Training {tag} (seed={seed}, version=V5) ---')
            t0 = time.time()
            r = run_single_seed(cfg, seed, device)
            print(f'[{tag}] training done in {time.time() - t0:.1f}s')
            trained[tag] = r
            # ---- 关键: 立刻把 ckpt 备份到 ckpt_<tag>_seed{seed}/, 防 C2 覆盖 C0 ----
            src_ckpt = os.path.join(results_root, f'ckpt_seed{seed}')
            dst_ckpt = os.path.join(results_root, f'ckpt_{tag}_seed{seed}')
            if os.path.isdir(src_ckpt):
                if os.path.isdir(dst_ckpt):
                    shutil.rmtree(dst_ckpt, ignore_errors=True)
                shutil.copytree(src_ckpt, dst_ckpt)
                print(f'[{tag}] ckpt mirrored to {dst_ckpt}')

    # ===== 构造 drift schedules =====
    prefs = build_preference_set(21)
    schedules = build_drift_schedules(base_cfg['time_slots'], prefs)
    sched_names = list(schedules.keys())

    # ===== Drift 评估 =====
    print('\n' + '=' * 60)
    print('=== Drift evaluation ===')
    print('=' * 60)
    n_eval_eps = 5
    eval_seed = 99999

    # (label, train_tag, drift_aware, init_from_buffer, detector_type, prior_source)
    # C0           : V5 baseline, no buffer, no drift adaptation       (lower bound)
    # C2-passive   : V5 + buffer warm-start at t=0, 不感知漂移          (静态 buffer)
    # C2-cusum     : V5 + buffer prior + CUSUM(raw reward) 触发 retrieve (deployable)
    # C2-aware     : V5 + buffer prior + oracle 触发 retrieve           (上界)
    methods = [
        ('C0',          'C0', False, False, 'none',     'buffer'),
        ('C2-passive',  'C2', False, True,  'none',     'buffer'),
        ('C2-cusum',    'C2', True,  True,  'cusum',    'buffer'),
        ('C2-aware',    'C2', True,  True,  'none',     'buffer'),
    ]
    cusum_kwargs = dict(window=20, threshold=3.0, drift_param=0.5)

    drift_results = {}
    drift_triggers = {}  # (method, sched) -> list[list[int]] (episode -> trigger slots)
    pick_log = {}  # H-MCSS 候选胜出比例: (method, sched) -> [feedback, prior, random]
    for sched_name in sched_names:
        sched = schedules[sched_name]
        for method_label, train_tag, da, ib, det_type, prior_src in methods:
            agent = trained[train_tag]['agent']
            env = trained[train_tag]['env']
            obuf = trained[train_tag]['omega_buf']
            # 重置候选计数器, 单独统计这一组 (method × schedule)
            if hasattr(agent, '_mcss_pick_count'):
                agent._mcss_pick_count = [0, 0, 0]
            d_runs, e_runs, trig_runs = evaluate_drift(
                env, agent, sched, n_eval_eps, eval_seed,
                omega_buf=obuf, drift_aware=da, init_from_buffer=ib,
                detector_type=det_type, cusum_kwargs=cusum_kwargs,
                prior_source=prior_src,
            )
            drift_results[(method_label, sched_name)] = (d_runs, e_runs)
            drift_triggers[(method_label, sched_name)] = trig_runs
            pick_msg = ''
            if hasattr(agent, '_mcss_pick_count'):
                pc = list(agent._mcss_pick_count)
                tot = sum(pc) or 1
                pick_log[(method_label, sched_name)] = pc
                pick_msg = (f'  picks(fb/pr/rd)={pc[0]}/{pc[1]}/{pc[2]} '
                            f'({pc[0]/tot:.0%}/{pc[1]/tot:.0%}/{pc[2]/tot:.0%})')
            print(f'  {method_label:11s} {sched_name:8s}  '
                  f'mean_delay={d_runs.mean():7.3f}  '
                  f'mean_energy={e_runs.mean():.3f}{pick_msg}')

    # 把 pick_log 写到结果目录, 方便后续画图/分析
    if pick_log:
        with open(os.path.join(results_root, 'mcss_picks.csv'),
                  'w', encoding='utf-8') as f:
            f.write('method,schedule,feedback,prior,random,total\n')
            for (m, s), pc in pick_log.items():
                tot = sum(pc)
                f.write(f'{m},{s},{pc[0]},{pc[1]},{pc[2]},{tot}\n')

    # ===== 保存 raw csv =====
    for (method, sched), (d_runs, e_runs) in drift_results.items():
        np.savetxt(os.path.join(results_root, f'delay_{method}_{sched}.csv'),
                   d_runs, fmt='%.4f')
        np.savetxt(os.path.join(results_root, f'energy_{method}_{sched}.csv'),
                   e_runs, fmt='%.4f')

    # ===== 保存漂移触发记录 (per method × schedule × episode) =====
    with open(os.path.join(results_root, 'drift_triggers.csv'),
              'w', encoding='utf-8') as f:
        f.write('method,schedule,episode,n_triggers,trigger_slots\n')
        for (method, sched), trig_runs in drift_triggers.items():
            for ep_idx, trig in enumerate(trig_runs):
                slots_str = ';'.join(str(t) for t in trig)
                f.write(f'{method},{sched},{ep_idx},{len(trig)},{slots_str}\n')

    # ===== 保存 drift_results 完整 dict 为 npz (事后任意切片) =====
    npz_payload = {}
    for (method, sched), (d_runs, e_runs) in drift_results.items():
        npz_payload[f'delay__{method}__{sched}'] = d_runs
        npz_payload[f'energy__{method}__{sched}'] = e_runs
    for sched_name, sched_arr in schedules.items():
        npz_payload[f'schedule__{sched_name}'] = sched_arr
    np.savez(os.path.join(results_root, 'drift_results.npz'), **npz_payload)

    # ===== 恢复时间 (sudden 专属) =====
    t_drift = base_cfg['time_slots'] // 2
    rec_table = []
    for label, *_ in methods:
        d_runs, _ = drift_results[(label, 'sudden')]
        rts = []
        for d in d_runs:
            rt = compute_recovery_time(d, t_drift, tail_window=10)
            rts.append(rt if rt >= 0 else base_cfg['time_slots'])
        rec_table.append((label, float(np.mean(rts)), float(np.std(rts))))

    # ===== 画图: 3 schedule × 2 metric =====
    fig, axes = plt.subplots(len(sched_names), 2,
                             figsize=(14, 4.0 * len(sched_names)))
    if len(sched_names) == 1:
        axes = axes.reshape(1, 2)
    colors = {'C0': 'tab:gray', 'C2-passive': 'tab:blue',
              'C2-cusum': 'tab:green', 'C2-aware': 'tab:red'}
    for row, sched_name in enumerate(sched_names):
        for col, metric in enumerate(['delay', 'energy']):
            ax = axes[row, col]
            for label, *_ in methods:
                d_runs, e_runs = drift_results[(label, sched_name)]
                runs = d_runs if metric == 'delay' else e_runs
                mu, sd = runs.mean(0), runs.std(0)
                t = np.arange(len(mu))
                ax.plot(t, mu, color=colors[label], linewidth=2, label=label)
                ax.fill_between(t, mu - sd, mu + sd,
                                alpha=0.2, color=colors[label])
            # 漂移点标注
            sched = schedules[sched_name]
            change_idx = np.where(np.any(np.diff(sched, axis=0) != 0, axis=1))[0]
            for ci in change_idx:
                ax.axvline(ci + 1, color='black', linestyle='--',
                           alpha=0.25, linewidth=0.8)
            ax.set_xlabel('Slot')
            ax.set_ylabel('Avg Delay (s)' if metric == 'delay' else 'Avg Energy (J)')
            ax.set_title(f'{sched_name} — {metric}')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_root, 'drift_curves.png'), dpi=150)
    plt.close()

    # ===== Summary =====
    summary = ['=== ω-Drift (Time-Varying Preference) Experiment ===',
               f'Train epochs: {base_cfg["num_epochs"]}, '
               f'eval episodes per schedule: {n_eval_eps}, '
               f'time slots: {base_cfg["time_slots"]}',
               f'Drift schedules: {sched_names}',
               f'Methods: {[m[0] for m in methods]}',
               '',
               '--- Recovery time (sudden drift, slots to recover within 1.1× tail mean) ---',
               '| Method      | Recovery (slots)  |',
               '|-------------|-------------------|']
    for label, mu, sd in rec_table:
        summary.append(f'| {label:11s} | {mu:6.1f} ± {sd:5.1f}      |')

    rec_dict = {l: m for l, m, _ in rec_table}
    if rec_dict.get('C0', 0) > 0:
        base = rec_dict['C0']
        summary.append('')
        for tag in ('C2-passive', 'C2-cusum', 'C2-aware'):
            if tag in rec_dict:
                spd = (base - rec_dict[tag]) / base * 100
                summary.append(f'[{tag:11s} vs C0] Recovery speedup: {spd:+.1f}%')

    summary.append('')
    summary.append('--- Mean (delay, energy) per schedule ---')
    summary.append('| Method       | Schedule | Delay (s)        | Energy (J)       |')
    summary.append('|--------------|----------|------------------|------------------|')
    for sched_name in sched_names:
        for label, *_ in methods:
            d_runs, e_runs = drift_results[(label, sched_name)]
            summary.append(
                f'| {label:12s} | {sched_name:8s} | '
                f'{d_runs.mean():6.3f} ± {d_runs.std():5.3f}  | '
                f'{e_runs.mean():6.3f} ± {e_runs.std():5.3f}  |'
            )

    summary.append('')
    summary.append(f'Config: {base_cfg}')
    summary_text = '\n'.join(summary)
    print('\n' + summary_text)
    with open(os.path.join(results_root, 'comparison.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # ---- 同步 latest 镜像 ----
    latest_dir = os.path.join(base_root, 'latest')
    try:
        if os.path.lexists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                shutil.rmtree(latest_dir, ignore_errors=True)
        shutil.copytree(results_root, latest_dir)
        print(f'[drift] mirrored to {os.path.abspath(latest_dir)}')
    except Exception as e:
        print(f'[drift] latest mirror failed: {e}')

    print(f'\n[drift] Done. Results saved to {os.path.abspath(results_root)}')
    print('Key files:')
    print('  - drift_curves.png  (3 schedules × delay/energy with std bands)')
    print('  - comparison.txt    (recovery time + per-schedule means)')
    print('  - delay_*.csv / energy_*.csv  (raw per-slot data, [eval_eps, T])')
    print('Tip: 可用 --ckpt-c0/--ckpt-c2 复用 results/mofd_*/ckpt_seed0 跳过训练.')


if __name__ == '__main__':
    main()
