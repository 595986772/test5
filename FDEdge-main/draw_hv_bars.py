"""
HV 对比柱状图 (两环境)
=====================
解析 results/hv_comparison.txt (V1, 仅边缘) 与 results/v2_hv_comparison.txt
(V2, 边缘+云), 画双面板柱状图. PC-FDN(ours) 高亮, 其余基线统一配色,
柱顶标注 HV 数值, 按 HV 降序排列.

输出: pic/hv_comparison_bar.png
"""
import os
import re

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')
PIC = os.path.join(ROOT, 'pic')

LINE_RE = re.compile(r'^(.*?)\s+HV\s*=\s*([-\d.]+)\s*±\s*([-\d.]+)')


def parse_hv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m:
                label = m.group(1).strip()
                mean = float(m.group(2))
                std = float(m.group(3))
                rows.append((label, mean, std))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def draw_panel(ax, rows, title):
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ['#c0392b' if 'ours' in lab.lower() else '#5b8db8'
              for lab in labels]
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.6, capsize=3, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Hypervolume (HV)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_ylim(0, max(means) * 1.15)
    for rect, val in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f'{val:.1f}' if val >= 100 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)
    return bars


def main():
    v1 = parse_hv(os.path.join(RESULTS, 'hv_comparison.txt'))
    v2 = parse_hv(os.path.join(RESULTS, 'v2_hv_comparison.txt'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_panel(axes[0], v1, '(a) Edge-only Environment')
    draw_panel(axes[1], v2, '(b) Edge-Cloud Environment')

    # 图例: ours vs baseline
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color='#c0392b', label='PC-FDN (ours)'),
               mpatches.Patch(color='#5b8db8', label='Baselines')]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('HV Comparison under Two Environments', fontsize=13,
                 fontweight='bold', y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(PIC, exist_ok=True)
    out = os.path.join(PIC, 'hv_comparison_bar.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'[save] {out}')

    for tag, rows in [('V1 edge-only', v1), ('V2 edge-cloud', v2)]:
        print(f'\n[{tag}]')
        for lab, mean, std in rows:
            print(f'  {lab:<16} {mean:.4f}')


if __name__ == '__main__':
    main()
