"""
统一基准下的多方法 HV 对比
=========================
读取各方法训练已保存的 `*_pareto_aggregated.csv`（最终 Pareto 点），
用所有方法 pooled 的 max 构造**全局统一参考点**，在同一 ref 下重算
每方法每 seed 的 HV，得到可横向比较的 mean ± std 数值与对比图。

输入 (results/):
    omega_resp_seed0.csv   (n_seeds * n_pref, 2)
    dsac_pareto_aggregated.csv
    ldqn_pareto_aggregated.csv
    opt_pareto_aggregated.csv
  （缺失的方法会自动跳过）

输出 (results/):
    hv_comparison.csv            每方法 HV mean/std/每 seed 值
    hv_comparison.txt            纯文本摘要
    pareto_comparison.png        4 方法 Pareto 前沿叠加图
    hv_curves_comparison.png     训练 HV 曲线叠加（若 *_hv_all_seeds.csv 存在）
"""
import os
# Windows 上 torch/numpy(MKL)/matplotlib 可能各自链接一份 libiomp5md.dll, 引发 OMP Error #15
# 必须在 import numpy/matplotlib 之前设置
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
    dict(key='pc-fdn',      label='PC-FDN(ours)',    color='tab:red',    marker='o'),
    dict(key='pc-fdn-nobuf', label='PC-FDN w/o buf', color='tab:orange', marker='o'),
    dict(key='genmosac', label='GenMOSAC',     color='tab:cyan',   marker='*'),
    dict(key='dsac',     label='Discrete SAC', color='tab:blue',   marker='s'),
    dict(key='ldqn',     label='LDQN',         color='tab:green',  marker='^'),
    dict(key='linucb',   label='LinUCB',       color='tab:olive',  marker='P'),
    dict(key='sa',       label='SA',           color='tab:pink',   marker='D'),
    dict(key='nsga2',    label='NSGA-II',      color='tab:brown',  marker='X'),
    dict(key='rand',     label='Random',       color='gray',       marker='x'),
    dict(key='rr',       label='Round-Robin',  color='tab:purple', marker='v'),
]

N_SEEDS = 1          # 和各 *_main.py 中 seeds 的长度保持一致
N_PREF = 21          # 和 final_eval_n_pref 保持一致


def load_pts(key, filename=None):
    """加载某方法的 pareto_aggregated.csv，reshape 成 (n_seeds, n_pref, 2)."""
    fname = filename if filename else f'{key}_pareto_aggregated.csv'
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.isfile(path):
        return None
    # Handle both headerless 2-col and header+3-col (omega_T,delay,energy) formats
    with open(path) as f:
        first = f.readline().strip()
    has_header = not first.replace('.', '').replace('-', '').replace(' ', '').replace(',', '').replace('\t', '').replace('e', '').replace('E', '').isdigit()
    skip = 1 if has_header else 0
    delimiter = ',' if ',' in first else None
    raw = np.loadtxt(path, skiprows=skip, delimiter=delimiter, ndmin=2)
    # If 3+ columns, take last two (delay, energy); skip omega_T prefix
    pts = raw[:, -2:] if raw.shape[1] >= 3 else raw
    if pts.size == 0:
        return None
    total = pts.shape[0]
    if total != N_SEEDS * N_PREF:
        # 自适应：若行数可被 N_SEEDS 整除则推导 n_pref
        if total % N_SEEDS == 0:
            n_pref = total // N_SEEDS
            print(f'[{key}] auto-detected n_pref={n_pref} (rows={total})')
            return pts.reshape(N_SEEDS, n_pref, 2)
        print(f'[warn] {key}: rows={total} 不匹配 {N_SEEDS}*{N_PREF}, 整体当作 1 个 seed 处理')
        return pts.reshape(1, total, 2)
    return pts.reshape(N_SEEDS, N_PREF, 2)


def load_hv_curves(key):
    """加载某方法的 *_hv_all_seeds.csv 作为训练曲线（shape: n_seeds, n_epochs）."""
    path = os.path.join(RESULTS_DIR, f'{key}_hv_all_seeds.csv')
    if not os.path.isfile(path):
        return None
    return np.loadtxt(path, ndmin=2)


def main():
    # 1) 加载每方法的 Pareto 点
    method_data = {}
    pooled = []
    for m in METHODS:
        pts = load_pts(m['key'], m.get('file'))
        if pts is None:
            print(f'[skip] {m["key"]}: pareto_aggregated.csv 不存在')
            continue
        method_data[m['key']] = pts
        pooled.append(pts.reshape(-1, 2))

    if len(method_data) < 2:
        print('[error] 至少需要 2 个方法的结果才能做对比')
        return

    all_pts = np.vstack(pooled)

    # 2) 构造全局统一参考点 (pooled max × 1.05，与 v2 一致)
    ref = (float(all_pts[:, 0].max() * 1.05 + 1e-6),
           float(all_pts[:, 1].max() * 1.05 + 1e-6))
    print(f'[ref] global HV reference = ({ref[0]:.4f}, {ref[1]:.4f})')

    # 3) 每方法每 seed 在同一 ref 下重算 HV
    summary_rows = []
    for m in METHODS:
        if m['key'] not in method_data:
            continue
        pts = method_data[m['key']]
        hvs = [hypervolume_2d(pts[s], ref) for s in range(pts.shape[0])]
        hvs = np.array(hvs)
        summary_rows.append((m['key'], m['label'], hvs))
        print(f'[{m["key"]:>4}] HV = {hvs.mean():.4f} ± {hvs.std():.4f}   '
              f'per-seed: {np.round(hvs, 3).tolist()}')

    # 4) 保存 CSV（method, mean, std, 每 seed 的 HV）
    csv_path = os.path.join(RESULTS_DIR, 'hv_comparison.csv')
    max_seeds = max(len(h) for _, _, h in summary_rows)
    header = ['method', 'hv_mean', 'hv_std'] + [f'seed_{i}' for i in range(max_seeds)]
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for key, label, hvs in summary_rows:
            cells = [key, f'{hvs.mean():.5f}', f'{hvs.std():.5f}']
            cells += [f'{v:.5f}' for v in hvs] + [''] * (max_seeds - len(hvs))
            f.write(','.join(cells) + '\n')
    print(f'[save] {csv_path}')

    # 5) 文本摘要
    txt_path = os.path.join(RESULTS_DIR, 'hv_comparison.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('=== Unified-Reference HV Comparison ===\n')
        f.write(f'Reference point (shared across methods): {ref}\n')
        f.write(f'# seeds = {N_SEEDS}, # preferences = {N_PREF}\n\n')
        for key, label, hvs in summary_rows:
            f.write(f'{label:<22} HV = {hvs.mean():.4f} ± {hvs.std():.4f}\n')
    print(f'[save] {txt_path}')

    # 6) Pareto 前沿叠加图
    hv_by_key = {key: hvs for key, _, hvs in summary_rows}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m in METHODS:
        if m['key'] not in method_data or m['key'] not in hv_by_key:
            continue
        pts = method_data[m['key']].reshape(-1, 2)
        hv_arr = hv_by_key[m['key']]
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=22, alpha=0.45, color=m['color'], marker=m['marker'],
                   label=m['label'])
        pf = pareto_front_2d(pts)
        if len(pf) > 0:
            ax.plot(pf[:, 0], pf[:, 1], color=m['color'], alpha=0.7, linewidth=1.6)
    ax.set_xlabel('Avg Delay (s)')
    ax.set_ylabel('Avg Energy (J)')
    ax.set_title(f'Pareto Fronts under Shared Reference (only edge)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pf_png = os.path.join(RESULTS_DIR, 'pareto_comparison.png')
    plt.savefig(pf_png, dpi=150)
    plt.close()
    print(f'[save] {pf_png}')

    # 7) 训练 HV 曲线叠加（仅对有 *_hv_all_seeds.csv 的方法；Opt 无训练过程）
    curve_data = {}
    for m in METHODS:
        arr = load_hv_curves(m['key'])
        if arr is not None:
            curve_data[m['key']] = arr

    if curve_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in METHODS:
            if m['key'] not in curve_data:
                continue
            arr = curve_data[m['key']]
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            x = np.arange(len(mu))
            ax.plot(x, mu, color=m['color'], linewidth=2, label=m['label'])
            ax.fill_between(x, mu - sd, mu + sd, color=m['color'], alpha=0.18)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Hypervolume (per-method ref)')
        ax.set_title('Training HV Curves (note: each method uses its own ref — shape only)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        cv_png = os.path.join(RESULTS_DIR, 'hv_curves_comparison.png')
        plt.savefig(cv_png, dpi=150)
        plt.close()
        print(f'[save] {cv_png}')
    else:
        print('[info] 没找到任何 *_hv_all_seeds.csv, 跳过训练曲线图')

    print('\n[done] 统一 HV 对比完成')


if __name__ == '__main__':
    main()
