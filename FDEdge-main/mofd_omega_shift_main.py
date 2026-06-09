"""
ω-Shift Generalization Experiment
==================================
目的: 验证 OmegaLatentBuffer (Path B) 的"近邻泛化"能力 — 当训练 ω 不覆盖
      评估 ω 时, buffer 是否能让策略在 held-out ω 上仍然表现良好.

实验设计:
  - 评估 ω: 21 个 (build_preference_set(21), 在 [delay, energy] simplex 上等距)
  - 训练 ω: 隔点抽 11 个 (even-indexed: 0, 2, 4, ..., 20)
  - Held-out ω: 10 个 (odd-indexed: 1, 3, ..., 19), 每个落在两个训练 ω 中点
  - 跑 C0 (无 buffer) 和 C2 (有 buffer) 两组, 同 seed 同任务
  - 对比 HV 在 (seen / unseen / full) 三个子集上的差异

输出 → results_omega_shift/:
  C0_hv_all_seeds.csv / C0_pareto_aggregated.csv / ...
  C2_hv_all_seeds.csv / C2_pareto_aggregated.csv / C2_obuf_log_seed*.csv
  comparison.txt           — 数值对比 (HV-seen / HV-unseen / HV-full + Δ%)
  comparison.png           — 训练曲线 + Pareto + HV 分组柱状图
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import shutil
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mofd_main import run_single_seed, set_task_generator, load_run_from_ckpt
from helpers import build_preference_set, hypervolume_2d, pareto_front_2d
from task_generator import make_task_generator


def main():
    base_cfg = dict(
        # ---- 环境 (与 mofd_main.py 保持一致) ----
        Emax=6, num_tasks_max=50, bit_range=(10, 40),
        time_slots=100, f_range=(10, 40),
        num_epochs=100, n_prefs_per_epoch=8,
        seeds=[0], smooth_window=5,
        train_eval_n_pref=11, train_eval_n_epi=1,
        final_eval_n_pref=21, final_eval_n_epi=3,
        # ---- SAC ----
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
        # ---- ω-buffer ----
        obuf_decay=0.5, obuf_noise=0.05,
        # ---- V5 (默认): vector-Q + COR + PopArt-lite ----
        use_v5=True,
        use_cor=True, cor_lambda=0.1, cor_c=0.0,
        use_popart=True, popart_beta=0.001,
        # ---- V4 Envelope-MORL (use_v5=True 时不生效) ----
        use_envelope=False, n_relabel_omegas=4,
        # ---- 任务模式 ----
        task_mode='random', task_kwargs=dict(),
    )

    # ω-shift 切分
    eval_omegas = build_preference_set(21)        # [21, 2]
    seen_idx = list(range(0, 21, 2))               # 0, 2, ..., 20 (11 个)
    unseen_idx = list(range(1, 21, 2))             # 1, 3, ..., 19 (10 个)
    train_omegas_arr = eval_omegas[seen_idx]       # [11, 2]

    print('=== ω-Shift Generalization Experiment ===')
    print(f'Train ω indices ({len(seen_idx)}): {seen_idx}')
    print(f'Held-out ω indices ({len(unseen_idx)}): {unseen_idx}')
    print(f'Eval ω total: 21')
    print(f'Train ω values:\n{np.round(train_omegas_arr, 3)}')

    # ---- CLI ckpt 加载: 跳过训练 ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-c0', type=str, default=None,
                        help='C0 ckpt 目录(可多个,逗号分隔,对应多个 seed)')
    parser.add_argument('--ckpt-c2', type=str, default=None,
                        help='C2 ckpt 目录')
    args, _ = parser.parse_known_args()
    use_ckpt = args.ckpt_c0 is not None and args.ckpt_c2 is not None

    # 输出 timestamp 子目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_root = 'results_omega_shift'
    results_root = os.path.join(base_root, f'shift_{ts}')
    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**base_cfg,
                   'seen_idx': seen_idx, 'unseen_idx': unseen_idx,
                   'ckpt_c0': args.ckpt_c0, 'ckpt_c2': args.ckpt_c2},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'[ω-shift] run dir: {os.path.abspath(results_root)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_task_generator(make_task_generator(base_cfg['task_mode'],
                                           **base_cfg.get('task_kwargs', {})))

    # ========== 跑 C0 / C2 两组 ==========
    runs = {}
    if use_ckpt:
        # 注意: ckpt 目录数量必须等于 seeds 数量
        c0_dirs = [d.strip() for d in args.ckpt_c0.split(',') if d.strip()]
        c2_dirs = [d.strip() for d in args.ckpt_c2.split(',') if d.strip()]
        for tag, dirs in [('C0', c0_dirs), ('C2', c2_dirs)]:
            seeds_runs = []
            for seed, ckpt_dir in zip(base_cfg['seeds'], dirs):
                print(f'\n--- {tag} loading from {ckpt_dir} (seed={seed}) ---')
                cfg = dict(base_cfg)
                cfg['train_omegas'] = train_omegas_arr
                seeds_runs.append(load_run_from_ckpt(ckpt_dir, cfg, seed, device))
            runs[tag] = seeds_runs
    else:
        for tag, use_buf in [('C0', False), ('C2', True)]:
            cfg = dict(base_cfg)
            cfg['use_omega_buffer'] = use_buf
            cfg['train_omegas'] = train_omegas_arr
            cfg['results_dir'] = results_root
            cfg['file_prefix'] = tag

            seeds_runs = []
            t0 = time.time()
            for seed in cfg['seeds']:
                print(f"\n========== {tag} (use_buffer={use_buf}) seed={seed} ==========")
                r = run_single_seed(cfg, seed, device)
                seeds_runs.append(r)
            print(f'[{tag}] all seeds done in {time.time() - t0:.1f}s')
            runs[tag] = seeds_runs

    # ========== 保存原始训练曲线 ==========
    for tag in ['C0', 'C2']:
        hv_all = np.stack([r['log_hv'] for r in runs[tag]], axis=0)
        d_all = np.stack([r['log_avg_delay'] for r in runs[tag]], axis=0)
        e_all = np.stack([r['log_avg_energy'] for r in runs[tag]], axis=0)
        np.savetxt(os.path.join(results_root, f'{tag}_hv_all_seeds.csv'),
                   hv_all, fmt='%.5f')
        np.savetxt(os.path.join(results_root, f'{tag}_avg_delay_all_seeds.csv'),
                   d_all, fmt='%.5f')
        np.savetxt(os.path.join(results_root, f'{tag}_avg_energy_all_seeds.csv'),
                   e_all, fmt='%.5f')
        pts_pool = np.vstack([r['pts_gfd'] for r in runs[tag]])
        np.savetxt(os.path.join(results_root, f'{tag}_pareto_aggregated.csv'),
                   pts_pool, fmt='%.5f')

    # ========== 全局 HV 参考点 (C0 + C2 联合) ==========
    all_pts = np.vstack([r['pts_gfd'] for tag in ['C0', 'C2'] for r in runs[tag]])
    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))

    # ========== 计算 seen / unseen / full HV ==========
    metrics = {}
    for tag in ['C0', 'C2']:
        seen_hvs, unseen_hvs, full_hvs = [], [], []
        for r in runs[tag]:
            pts = r['pts_gfd']                              # [21, 2]
            seen_hvs.append(hypervolume_2d(pts[seen_idx], ref))
            unseen_hvs.append(hypervolume_2d(pts[unseen_idx], ref))
            full_hvs.append(hypervolume_2d(pts, ref))
        metrics[tag] = dict(
            seen=(float(np.mean(seen_hvs)), float(np.std(seen_hvs))),
            unseen=(float(np.mean(unseen_hvs)), float(np.std(unseen_hvs))),
            full=(float(np.mean(full_hvs)), float(np.std(full_hvs))),
        )

    def pct_delta(c2, c0):
        return (c2 - c0) / max(abs(c0), 1e-9) * 100

    delta_seen = pct_delta(metrics['C2']['seen'][0], metrics['C0']['seen'][0])
    delta_unseen = pct_delta(metrics['C2']['unseen'][0], metrics['C0']['unseen'][0])
    delta_full = pct_delta(metrics['C2']['full'][0], metrics['C0']['full'][0])

    # ========== 写 comparison.txt ==========
    summary_lines = [
        '=== ω-Shift Generalization Experiment ===',
        f'Train ω indices ({len(seen_idx)}): {seen_idx}',
        f'Held-out ω indices ({len(unseen_idx)}): {unseen_idx}',
        f'Seeds: {base_cfg["seeds"]}',
        f'Global HV ref: {ref}',
        '',
        '| Method | HV-seen (11 ω) | HV-unseen (10 ω) | HV-full (21 ω) |',
        '|--------|----------------|------------------|----------------|',
        f'| C0 (no buffer)     | {metrics["C0"]["seen"][0]:7.2f} ± {metrics["C0"]["seen"][1]:5.2f} '
        f'| {metrics["C0"]["unseen"][0]:7.2f} ± {metrics["C0"]["unseen"][1]:5.2f} '
        f'| {metrics["C0"]["full"][0]:7.2f} ± {metrics["C0"]["full"][1]:5.2f} |',
        f'| C2 (with buffer)   | {metrics["C2"]["seen"][0]:7.2f} ± {metrics["C2"]["seen"][1]:5.2f} '
        f'| {metrics["C2"]["unseen"][0]:7.2f} ± {metrics["C2"]["unseen"][1]:5.2f} '
        f'| {metrics["C2"]["full"][0]:7.2f} ± {metrics["C2"]["full"][1]:5.2f} |',
        '',
        f'[Buffer Benefit] Δ HV-seen   = {delta_seen:+.2f}%',
        f'[Buffer Benefit] Δ HV-unseen = {delta_unseen:+.2f}%   ★ key metric',
        f'[Buffer Benefit] Δ HV-full   = {delta_full:+.2f}%',
        '',
        f'Config: {base_cfg}',
    ]
    summary_text = '\n'.join(summary_lines)
    print('\n' + summary_text)
    with open(os.path.join(results_root, 'comparison.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # ========== 画图: 训练曲线 + Pareto + HV 柱状图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'C0': 'tab:blue', 'C2': 'tab:red'}

    # (1) Training HV (在训练 ω 子集上, run_single_seed 的 log_hv)
    for tag in ['C0', 'C2']:
        hv_all = np.stack([r['log_hv'] for r in runs[tag]], axis=0)
        hv_m = hv_all.mean(0)
        epochs = np.arange(len(hv_m))
        axes[0].plot(epochs, hv_m, color=colors[tag], linewidth=2, label=tag)
        if hv_all.shape[0] > 1:
            hv_s = hv_all.std(0)
            axes[0].fill_between(epochs, hv_m - hv_s, hv_m + hv_s,
                                 alpha=0.2, color=colors[tag])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train HV (11 ω, n_eval_epi=1)')
    axes[0].set_title('Training HV curve')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # (2) Final Pareto (21 ω, seen 标实心 / unseen 标空心)
    for tag in ['C0', 'C2']:
        pts_pool = np.vstack([r['pts_gfd'] for r in runs[tag]])    # [S*21, 2]
        # 拆 seen/unseen
        S = len(runs[tag])
        pts_per_run = [r['pts_gfd'] for r in runs[tag]]
        seen_pts = np.vstack([p[seen_idx] for p in pts_per_run])
        unseen_pts = np.vstack([p[unseen_idx] for p in pts_per_run])
        axes[1].scatter(seen_pts[:, 0], seen_pts[:, 1],
                        color=colors[tag], marker='o', alpha=0.7, s=35,
                        label=f'{tag} seen')
        axes[1].scatter(unseen_pts[:, 0], unseen_pts[:, 1],
                        color=colors[tag], marker='x', alpha=0.9, s=45,
                        label=f'{tag} unseen')
        pf = pareto_front_2d(pts_pool)
        if len(pf) > 0:
            axes[1].plot(pf[:, 0], pf[:, 1], color=colors[tag], alpha=0.5)
    axes[1].set_xlabel('Avg Delay (s)')
    axes[1].set_ylabel('Avg Energy (J)')
    axes[1].set_title('Final Pareto: seen ●  vs  unseen ✕')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # (3) HV bar: seen / unseen / full
    cats = ['HV-seen', 'HV-unseen ★', 'HV-full']
    x = np.arange(len(cats))
    w = 0.35
    for i, tag in enumerate(['C0', 'C2']):
        vals = [metrics[tag]['seen'][0], metrics[tag]['unseen'][0],
                metrics[tag]['full'][0]]
        errs = [metrics[tag]['seen'][1], metrics[tag]['unseen'][1],
                metrics[tag]['full'][1]]
        axes[2].bar(x + (i - 0.5) * w, vals, w, yerr=errs,
                    color=colors[tag], label=tag, alpha=0.85, capsize=5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cats)
    axes[2].set_ylabel('Hypervolume')
    axes[2].set_title(f'HV breakdown (Δ unseen = {delta_unseen:+.1f}%)')
    axes[2].legend(); axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_root, 'comparison.png'), dpi=150)
    plt.close()

    # latest 镜像
    latest_dir = os.path.join(base_root, 'latest')
    try:
        if os.path.lexists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                shutil.rmtree(latest_dir, ignore_errors=True)
        shutil.copytree(results_root, latest_dir)
        print(f'[ω-shift] mirrored to {os.path.abspath(latest_dir)}')
    except Exception as e:
        print(f'[ω-shift] latest mirror failed: {e}')

    print(f'\n[ω-shift] Done. Results saved to {os.path.abspath(results_root)}')
    print('Key files:')
    print('  - comparison.txt   (HV table + Δ%)')
    print('  - comparison.png   (3-panel: train curve / Pareto seen-vs-unseen / HV bars)')
    print('  - {C0,C2}_hv_all_seeds.csv, {C0,C2}_pareto_aggregated.csv')
    print('Tip: --ckpt-c0 dir1,dir2 --ckpt-c2 dir1,dir2  可跳过训练直接评估.')
    print('  - C2_obuf_log_seed*.csv (buffer 命中诊断, 应能看到 unseen ω 检索 nn_dist=1)')


if __name__ == '__main__':
    main()
