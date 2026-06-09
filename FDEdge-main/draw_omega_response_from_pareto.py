"""
直接拿 mofd_*_pareto_aggregated.csv (两列: delay, energy) 画 ω 响应图.
CSV 行顺序 = build_preference_set(21) 的 ω 顺序: ω_T 从 0.0 → 1.0.

用法:
  python draw_omega_response_from_pareto.py
  python draw_omega_response_from_pareto.py \
      --csv results/mofd_20260515_175014/omega_resp_seed0.csv \
      --out results/mofd_20260515_175014/fig_omega_response
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='results/mofd_20260515_175014/omega_resp_seed0.csv')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-pref', type=int, default=21)
    args = ap.parse_args()

    assert os.path.exists(args.csv), f'CSV 不存在: {args.csv}'
    pts = np.loadtxt(args.csv)
    assert pts.shape == (args.n_pref, 2), \
        f'期望 ({args.n_pref}, 2) 形状, 实际 {pts.shape}'

    omega_T = np.linspace(0.0, 1.0, args.n_pref)
    delay   = pts[:, 0]
    energy  = pts[:, 1]

    rho_d, _ = spearmanr(omega_T, delay)
    rho_e, _ = spearmanr(omega_T, energy)

    # ---- 顶会风格 ----
    plt.rcParams.update({
        'font.size': 13, 'font.family': 'serif',
        'axes.linewidth': 1.2, 'lines.linewidth': 2.2,
        'figure.dpi': 150,
    })

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    c_delay, c_energy = '#1f77b4', '#d62728'

    ax1.plot(omega_T, delay, color=c_delay, marker='o',
             markersize=6, label='Delay')
    ax1.set_xlabel(r'Preference weight $\omega_{\mathrm{delay}}$')
    ax1.set_ylabel('Delay (s)', color=c_delay)
    ax1.tick_params(axis='y', labelcolor=c_delay)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(omega_T, energy, color=c_energy, marker='s',
             markersize=6, linestyle='--', label='Energy')
    ax2.set_ylabel('Energy (J)', color=c_energy)
    ax2.tick_params(axis='y', labelcolor=c_energy)

    ax1.annotate(
        r'$\omega_{\mathrm{delay}}\!\uparrow\;\Rightarrow\;$'
        r'delay $\downarrow$, energy $\uparrow$',
        xy=(0.5, 0.93), xycoords='axes fraction',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', fc='#fff7e6',
                  ec='gray', lw=0.8))

    ax1.text(0.02, 0.04,
             rf'Spearman $\rho$: delay={rho_d:+.2f}, energy={rho_e:+.2f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec='gray', lw=0.6, alpha=0.85))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', frameon=True, fontsize=11)

    plt.title(r'Preference-Response Curve of PC-FDN '
              r'(21 $\omega$ on simplex)',
              fontsize=13, pad=10)
    plt.tight_layout()

    out_prefix = args.out or os.path.join(
        os.path.dirname(args.csv), 'fig_omega_response')
    plt.savefig(out_prefix + '.pdf', bbox_inches='tight')
    plt.savefig(out_prefix + '.png', bbox_inches='tight', dpi=200)
    print(f'[saved] {out_prefix}.pdf / .png')
    print(f'[stats] delay range  = {delay.max()-delay.min():.3f} s')
    print(f'[stats] energy range = {energy.max()-energy.min():.3f} J')
    print(f'[stats] Spearman rho: delay={rho_d:+.3f}, energy={rho_e:+.3f}')


if __name__ == '__main__':
    main()
