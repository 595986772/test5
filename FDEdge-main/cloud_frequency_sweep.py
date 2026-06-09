"""
cloud_frequency_sweep.py
========================
测试云端不同处理能力对延迟的影响。将云端频率分为 5 档，以当前默认值
(50,70) GHz 的中点 60 GHz 为中心，每档独立完成训练+评估，
最终输出延迟-频率曲线图。

5 档: 20 / 40 / 60 / 80 / 100 GHz (单点固定, 非范围)
输出: result_frequency/cloud_frequency_sweep.png + cloud_sweep_summary.csv

预计耗时: 5 档 × 20 epochs × ~3-5 min/epoch ≈ 5-8 小时
"""

import os
import sys
import glob as _glob

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mofd_environment_v2 import MOFDEnvironmentV2
import mofd_main as _mm

# ── 配置 ────────────────────────────────────────────────
CLOUD_F_LEVELS = [
    (20,  "20 GHz"),
    (40,  "40 GHz"),
    (60,  "60 GHz"),
    (80,  "80 GHz"),
    (100, "100 GHz"),
]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'result_frequency')
EPOCHS = 50          # 每档训练轮数
FINAL_EVAL_PREF = 21  # 最终评估偏好点数


def _make_adapter(cloud_f_val):
    """为固定 cloud_f 创建 V2 适配器, monkey-patch 到 mofd_main."""

    class _Adapter(MOFDEnvironmentV2):
        def __init__(self, Emax=7, num_tasks_max=10,
                     bit_range=(10, 40), time_slots=100,
                     f_range=(10, 40),
                     cloud_f_range=None,
                     cloud_tran_rate_range=(80, 120),
                     cloud_kappa=1e-4, kappa=1e-3,
                     delay_scale=0.05, energy_scale=0.25,
                     seed=0, **kwargs):
            n_edges = max(1, int(Emax) - 1)
            super().__init__(
                Emax=n_edges, num_tasks_max=num_tasks_max,
                bit_range=bit_range, time_slots=time_slots,
                f_range=f_range,
                cloud_f_range=(float(cloud_f_val), float(cloud_f_val)),
                cloud_tran_rate_range=cloud_tran_rate_range,
                kappa=kappa, cloud_kappa=cloud_kappa,
                delay_scale=delay_scale, energy_scale=energy_scale,
                seed=seed,
            )

    return _Adapter


def _base_cfg(prefix, cloud_f_val):
    """构建每次运行的 cfg_override."""
    return dict(
        Emax=7,
        num_tasks_max=10,
        bit_range=(10, 40),
        time_slots=100,
        f_range=(10, 40),

        num_epochs=EPOCHS,
        n_prefs_per_epoch=8,
        seeds=[0],
        smooth_window=3,
        train_eval_n_pref=11,
        train_eval_n_epi=1,
        final_eval_n_pref=FINAL_EVAL_PREF,
        final_eval_n_epi=3,

        delay_scale=0.05,
        energy_scale=0.25,
        alpha_T=1.0,
        alpha_E=1.0,

        actor_lr=1e-4,
        critic_lr=1e-3,
        alpha_init=0.05,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.95,
        denoising_steps=3,
        hidden_dim=128,
        target_entropy=0.5,
        buf_size=10000,
        batch_size=64,
        buffer_warmup=500,
        update_every=4,

        use_omega_buffer=True,
        obuf_decay=0.5,
        obuf_noise=0.05,
        use_v7=False,
        use_nap=True,
        nap_beta=0.01,
        use_v6=False,
        use_v5=True,
        use_envelope=False,
        use_hypernet=True,
        hyper_lr=1e-4,
        hyper_hidden=64,
        div_lambda=0.1,
        use_cor=True,
        cor_lambda=0.1,
        cor_c=0.0,
        use_popart=False,
        n_relabel_omegas=4,

        task_mode='random',
        task_kwargs=dict(),

        file_prefix=prefix,
        results_root=OUTPUT_DIR,
    )


def run_level(f_val, label):
    """训练+评估单档频率, 返回 (avg_delay, avg_energy, delays, energies)."""
    prefix = f'cloud_f{f_val}'
    print(f'\n{"=" * 60}')
    print(f'  Cloud Sweep: {label}  |  prefix = {prefix}')
    print(f'{"=" * 60}')

    # ── monkey-patch V2 adapter ──
    _mm.MOFDEnvironment = _make_adapter(f_val)

    # ── 训练 + 评估 ──
    cfg = _base_cfg(prefix, f_val)
    _mm.main(cfg_override=cfg)

    # ── 从输出目录读回最终 Pareto 数据 ──
    pattern = os.path.join(OUTPUT_DIR, f'{prefix}_*')
    dirs = sorted(_glob.glob(pattern))
    if not dirs:
        raise RuntimeError(f'未找到输出目录: {pattern}')
    results_dir = dirs[-1]

    pareto_csv = os.path.join(results_dir, f'{prefix}_pareto_aggregated.csv')
    if not os.path.isfile(pareto_csv):
        pareto_csv = os.path.join(OUTPUT_DIR, f'{prefix}_pareto_aggregated.csv')
    if not os.path.isfile(pareto_csv):
        raise FileNotFoundError(f'未找到 Pareto CSV: {pareto_csv}')

    pts = np.loadtxt(pareto_csv)          # shape (21, 2): delay, energy
    delays = pts[:, 0]
    energies = pts[:, 1]
    avg_d = float(delays.mean())
    avg_e = float(energies.mean())

    print(f'  [{label}] avg_delay={avg_d:.3f}s  avg_energy={avg_e:.3f}J  '
          f'min_delay={delays.min():.3f}s  max_energy={energies.max():.3f}J')
    return avg_d, avg_e, delays, energies


def plot(results, output_dir):
    """画延迟-频率 + 能耗-频率 双图"""
    os.makedirs(output_dir, exist_ok=True)

    f_vals = np.array([r['f'] for r in results])
    labels = [r['label'] for r in results]
    avg_d = np.array([r['avg_delay'] for r in results])
    avg_e = np.array([r['avg_energy'] for r in results])
    all_d = np.array([r['delays'] for r in results])   # (5, 21)
    all_e = np.array([r['energies'] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ── 延迟 ──
    ax = axes[0]
    # 平均线
    ax.plot(f_vals, avg_d, 'o-', color='#1976D2', linewidth=2.5, markersize=11, label='Mean delay')
    # 每个频率的偏好分布 (箱线: min-max 竖线)
    for i, fv in enumerate(f_vals):
        d_min, d_max = all_d[i].min(), all_d[i].max()
        ax.plot([fv, fv], [d_min, d_max], color='#90CAF9', linewidth=1.8, alpha=0.9)
        ax.plot(fv, d_min, 'v', color='#2196F3', markersize=5)
        ax.plot(fv, d_max, '^', color='#2196F3', markersize=5)

    ax.set_xlabel('Cloud CPU Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Average Delay (s)', fontsize=12)
    ax.set_title('Cloud Frequency → Task Delay', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # 标注数值
    for fv, d in zip(f_vals, avg_d):
        ax.annotate(f'{d:.2f}s', (fv, d), textcoords="offset points",
                     xytext=(0, 14), ha='center', fontsize=8.5, color='#1565C0')

    # ── 能耗 ──
    ax = axes[1]
    ax.plot(f_vals, avg_e, 's-', color='#E64A19', linewidth=2.5, markersize=11, label='Mean energy')
    for i, fv in enumerate(f_vals):
        e_min, e_max = all_e[i].min(), all_e[i].max()
        ax.plot([fv, fv], [e_min, e_max], color='#FFAB91', linewidth=1.8, alpha=0.9)
        ax.plot(fv, e_min, 'v', color='#FF5722', markersize=5)
        ax.plot(fv, e_max, '^', color='#FF5722', markersize=5)

    ax.set_xlabel('Cloud CPU Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Average Energy (J)', fontsize=12)
    ax.set_title('Cloud Frequency → Task Energy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    for fv, e in zip(f_vals, avg_e):
        ax.annotate(f'{e:.2f}J', (fv, e), textcoords="offset points",
                     xytext=(0, 14), ha='center', fontsize=8.5, color='#BF360C')

    fig.suptitle('Impact of Cloud Processing Capability on Task Offloading Performance',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'cloud_frequency_sweep.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[Plot] → {save_path}')


def main():
    print(f'Cloud Frequency Sweep: {len(CLOUD_F_LEVELS)} levels × {EPOCHS} epochs')
    print(f'Output dir: {OUTPUT_DIR}')
    print(f'Est. time: ~{len(CLOUD_F_LEVELS) * EPOCHS * 3}-{len(CLOUD_F_LEVELS) * EPOCHS * 5} min')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for f_val, label in CLOUD_F_LEVELS:
        avg_d, avg_e, delays, energies = run_level(f_val, label)
        results.append(dict(
            f=f_val, label=label,
            avg_delay=avg_d, avg_energy=avg_e,
            delays=delays, energies=energies,
        ))

    # ── 保存汇总 CSV ──
    csv_path = os.path.join(OUTPUT_DIR, 'cloud_sweep_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('cloud_f_ghz,label,avg_delay_s,avg_energy_j\n')
        for r in results:
            f.write(f'{r["f"]},{r["label"]},{r["avg_delay"]:.5f},{r["avg_energy"]:.5f}\n')
    print(f'\n[CSV] → {csv_path}')

    # ── 画图 ──
    plot(results, OUTPUT_DIR)
    print('\nDone.')


if __name__ == '__main__':
    main()
