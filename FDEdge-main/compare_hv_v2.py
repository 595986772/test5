"""
V2 环境下的统一 HV 对比 + Pareto 叠加图
======================================
读各方法的 *_v2_pareto_aggregated.csv, 用 pooled-max × 1.05 作全局参考点,
在同一 ref 下重算 HV (跨方法可比), 并出 Pareto 散点叠加图.

为何用 pooled-max 而非 random-max:
  Opt-v2 物理下界给出极端低能耗点 (delay 32s, energy 1.02J), 远超 random
  max delay 范围. 若用 random-max × 1.1 作 ref, Opt 这些极端点全被剪掉, HV 不公平.

输出:
  results/v2_hv_comparison.csv / .txt
  results/v2_pareto_comparison.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from helpers import hypervolume_2d, pareto_front_2d

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

METHODS = [
    dict(key='pc-fdn_v2',     label='PC-FDN(ours)', color='tab:blue', marker='o'),
    dict(key='sa_v2',      label='SA',  color='tab:pink',   marker='D'),
    dict(key='linucb_v2',  label='LinUCB',             color='tab:olive',  marker='P'),
    dict(key='nsga2_v2',   label='NSGA-II',            color='tab:brown',  marker='X'),
    dict(key='randomp_v2', label='Random', color='gray',       marker='x'),
    dict(key='dsac_v2',       label='Discrete SAC',       color='tab:orange', marker='s', file='dsac_v2_pareto_aggregated.csv'),
    dict(key='genmosac',   label='GenMOSAC',           color='tab:cyan',   marker='^', file='genmosac_pareto_aggregated1.csv'),
    dict(key='ldqn_v2',       label='LDQN',               color='tab:green',  marker='v', file='ldqn_v2_pareto_aggregated.csv'),
]

N_SEEDS = 1
N_PREF = 21


def load_pts(key, filename=None):
    fname = filename if filename else f'{key}_pareto_aggregated.csv'
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        first = f.readline().strip()
    has_header = not first.replace('.', '').replace('-', '').replace(' ', '').replace(',', '').replace('\t', '').replace('e', '').replace('E', '').isdigit()
    skip = 1 if has_header else 0
    delimiter = ',' if ',' in first else None
    raw = np.loadtxt(path, skiprows=skip, delimiter=delimiter, ndmin=2)
    pts = raw[:, -2:] if raw.shape[1] >= 3 else raw
    if pts.size == 0:
        return None
    total = pts.shape[0]
    if total == N_SEEDS * N_PREF:
        return pts.reshape(N_SEEDS, N_PREF, 2)
    return pts.reshape(1, total, 2)


def main():
    method_data = {}
    pooled = []
    for m in METHODS:
        pts = load_pts(m['key'], m.get('file'))
        if pts is None:
            print(f'[skip] {m["key"]}: csv not found')
            continue
        method_data[m['key']] = pts
        pooled.append(pts.reshape(-1, 2))

    if len(method_data) < 2:
        print('[error] 至少需要 2 个方法')
        return

    all_pts = np.vstack(pooled)
    # 全局 pooled-max × 1.05
    ref = (float(all_pts[:, 0].max() * 1.05 + 1e-6),
           float(all_pts[:, 1].max() * 1.05 + 1e-6))
    print(f'[ref] global HV reference = ({ref[0]:.4f}, {ref[1]:.4f})')

    summary_rows = []
    for m in METHODS:
        if m['key'] not in method_data:
            continue
        pts = method_data[m['key']]
        hvs = [hypervolume_2d(pts[s], ref) for s in range(pts.shape[0])]
        hvs = np.array(hvs)
        summary_rows.append((m['key'], m['label'], hvs))
        print(f'[{m["key"]:>12}] HV = {hvs.mean():.4f} ± {hvs.std():.4f}')

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'v2_hv_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('method,hv_mean,hv_std,per_seed\n')
        for key, label, hvs in summary_rows:
            cells = [key, f'{hvs.mean():.5f}', f'{hvs.std():.5f}']
            cells += [f'{v:.5f}' for v in hvs]
            f.write(','.join(cells) + '\n')
    print(f'[save] {csv_path}')

    # TXT
    txt_path = os.path.join(RESULTS_DIR, 'v2_hv_comparison.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('=== V2 Unified-Reference HV Comparison ===\n')
        f.write(f'Reference point (pooled_max x 1.05): {ref}\n')
        f.write(f'# seeds = {N_SEEDS}, # preferences = {N_PREF}\n')
        f.write('Env: MOFDEnvironmentV2 (action 0 = cloud, 1..Emax = edges)\n\n')
        for key, label, hvs in summary_rows:
            f.write(f'{label:<26} HV = {hvs.mean():.4f} ± {hvs.std():.4f}\n')
    print(f'[save] {txt_path}')

    # Pareto plot - 2 subplots: full range + zoomed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    hv_by_key = {key: hvs for key, _, hvs in summary_rows}

    for m in METHODS:
        if m['key'] not in method_data:
            continue
        pts = method_data[m['key']].reshape(-1, 2)
        hv_arr = hv_by_key[m['key']]
        for ax in (ax1, ax2):
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=28, alpha=0.55, color=m['color'], marker=m['marker'],
                       label=m['label'])
            pf = pareto_front_2d(pts)
            if len(pf) > 0:
                ax.plot(pf[:, 0], pf[:, 1], color=m['color'], alpha=0.7, linewidth=1.6)

    ax1.set_xlabel('Avg Delay (s)'); ax1.set_ylabel('Avg Energy (J)')
    ax1.set_title('Pareto Fronts under Shared Reference(edge+cloud)')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    # zoomed view (delay <= 2.0)
    for m in METHODS:
        if m['key'] not in method_data:
            continue
    ax2.set_xlim(0, 2.0)
    ax2.set_xlabel('Avg Delay (s)'); ax2.set_ylabel('Avg Energy (J)')
    ax2.set_title('Pareto Fronts under Shared Reference(edge+cloud)')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'v2_pareto_comparison.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[save] {png_path}')


if __name__ == '__main__':
    main()
