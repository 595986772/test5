"""
ω-Response Plot (single seed)
=============================
读 omega_resp_seed*.csv (列: omega_T, delay, energy),
画 PC-FDN 偏好响应图.

用法:
  python draw_omega_response.py
  python draw_omega_response.py --csv results/latest/omega_response/omega_resp_seed0云端已输出.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          11,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/latest/omega_response/omega_resp_seed0.csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    assert os.path.exists(args.csv), f'CSV 不存在: {args.csv}'
    df      = pd.read_csv(args.csv)
    omega_T = df['omega_T'].values
    delay   = df['delay'].values
    energy  = df['energy'].values

    rho_d, _ = spearmanr(omega_T, delay)
    rho_e, _ = spearmanr(omega_T, energy)

    # smooth spline
    xf  = np.linspace(omega_T[0], omega_T[-1], 300)
    d_s = make_interp_spline(omega_T, delay,  k=3)(xf)
    e_s = make_interp_spline(omega_T, energy, k=3)(xf)

    C_D = '#1A6FBD'
    C_E = '#C0392B'

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))

    # ── delay (left axis) ────────────────────────────────────────────────────
    ax1.fill_between(xf, d_s, ax1.get_ylim()[0] if False else 0,
                     color=C_D, alpha=0.08, zorder=1)
    ax1.plot(xf, d_s, color=C_D, lw=2.0, zorder=3)
    ax1.scatter(omega_T, delay, color=C_D, s=28, zorder=4,
                edgecolors='white', linewidths=0.6)
    ax1.set_xlabel(r'Preference weight $\omega_{\mathrm{delay}}$', fontsize=11)
    ax1.set_ylabel('Avg. Delay (s)', color=C_D, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=C_D)
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(ls='--', lw=0.45, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(C_D)
    ax1.spines['left'].set_linewidth(0.8)
    ax1.spines['right'].set_color(C_E)
    ax1.spines['right'].set_linewidth(0.8)
    ax1.spines['bottom'].set_linewidth(0.8)

    # ── energy (right axis) ──────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.fill_between(xf, e_s, e_s.max() * 1.08,
                     color=C_E, alpha=0.08, zorder=1)
    ax2.plot(xf, e_s, color=C_E, lw=2.0, ls='--', zorder=3)
    ax2.scatter(omega_T, energy, color=C_E, s=28, zorder=4,
                marker='s', edgecolors='white', linewidths=0.6)
    ax2.set_ylabel('Avg. Energy (J)', color=C_E, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=C_E)
    ax2.spines['top'].set_visible(False)

    # ── annotation ───────────────────────────────────────────────────────────

    # combined legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_D, lw=2.0,
               marker='o', ms=5, markeredgecolor='white', label='Delay'),
        Line2D([0], [0], color=C_E, lw=2.0, ls='--',
               marker='s', ms=5, markeredgecolor='white', label='Energy'),
    ]
    ax1.legend(handles=handles, loc='center left',
               framealpha=0.88, edgecolor='#CCCCCC',
               handlelength=1.6, labelspacing=0.25,
               borderpad=0.4, fontsize=10)

    ax1.set_title('Edge+Cloud Offloading',
                  fontsize=10, pad=6, loc='left',
                  color='#214283', fontstyle='italic', fontweight='normal')
    plt.tight_layout()

    out_prefix = args.out or os.path.join(
        os.path.dirname(args.csv), 'fig_omega_response')
    plt.savefig(out_prefix + '.pdf', bbox_inches='tight')
    plt.savefig(out_prefix + '.png', bbox_inches='tight', dpi=300)
    print(f'[saved] {out_prefix}.pdf / .png')
    print(f'[stats] delay range={delay.max()-delay.min():.3f}s  '
          f'energy range={energy.max()-energy.min():.3f}J  '
          f'ρ_d={rho_d:+.3f}  ρ_e={rho_e:+.3f}')


if __name__ == '__main__':
    main()
