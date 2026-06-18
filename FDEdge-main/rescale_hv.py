"""把所有方法在固定卷子上的前沿, 在多个候选 ref/盒子下重算 HV (秤校准 + 敏感性)
================================================================================
动因: 旧 HV 不裁 ref 框外的点 -> 高delay/低energy 的极端点凭空灌水 (已在
helpers.hypervolume_2d 修掉)。但「ref 框怎么定」本身是个判断: 框太小 (随机 nadir×1.1)
会把方法 1/3 的能耗侧前沿裁成 0; 裸取全局 nadir 又被 greedy_energy(delay~444) 顶到
量纲失衡。这里**不替你拍板**, 把同一批前沿在几个候选标尺下各算一遍, 把排名敏感性摊开:

  current        : eval_testset.pkl 里的 ref (随机策略 nadir×1.1) —— 旧标尺, 偏小
  union_deploy   : 可部署/训练方法并集的 per-轴 nadir ×1.1 (排除 oracle 与病态 greedy_energy)
  union_all      : 含 oracle greedy_omega 的并集 nadir ×1.1
  normalized     : 用 union_deploy 盒子把两轴各归一到 [0,1], ref=(1.1,1.1) —— 量纲平衡, 推荐

读 results/testset_*_pareto.csv (谁在就算谁), 写 results/testset_compare_fixed.txt。
**不碰 results/testset_compare.csv** (那是正在跑的 baseline 的 inline 产物)。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import glob
import csv
import numpy as np
from helpers import hypervolume_2d

OUT = 'results/testset_compare_fixed.txt'

# oracle (全信息上界, 单列不混排) 与 病态退化策略 (always-slowest, 不参与定盒子)
ORACLE = {'greedy_omega'}
DEGENERATE = {'greedy_energy'}


def load_fronts():
    fronts = {}
    for p in sorted(glob.glob('results/testset_*_pareto.csv')):
        name = os.path.basename(p)[len('testset_'):-len('_pareto.csv')]
        pts = []
        with open(p, encoding='utf-8') as f:
            r = csv.reader(f); next(r)
            for row in r:
                if len(row) >= 3:
                    pts.append([float(row[1]), float(row[2])])
        if pts:
            fronts[name] = np.array(pts, dtype=float)
    return fronts


def box_from(fronts, names):
    pts = np.vstack([fronts[n] for n in names if n in fronts])
    return pts.min(0), pts.max(0)


def norm_hv(pts, bmin, bmax, ref=(1.1, 1.1)):
    span = np.maximum(bmax - bmin, 1e-9)
    return hypervolume_2d((np.asarray(pts) - bmin) / span, ref)


def main():
    fronts = load_fronts()
    if not fronts:
        print('没有找到 results/testset_*_pareto.csv'); return
    names = list(fronts)
    deploy = [n for n in names if n not in ORACLE and n not in DEGENERATE]

    # ---- 候选标尺 ----
    try:
        from fixed_testset import load_testset
        cur_ref = tuple(load_testset()['ref'])
    except Exception:
        cur_ref = (39.788, 4.681)
    dmin, dmax = box_from(fronts, deploy)
    amin, amax = box_from(fronts, names)            # 含 oracle/全部
    ref_deploy = tuple(dmax * 1.1)
    ref_all = tuple(amax * 1.1)

    scales = [
        ('current',      lambda pts: hypervolume_2d(pts, cur_ref)),
        ('union_deploy', lambda pts: hypervolume_2d(pts, ref_deploy)),
        ('union_all',    lambda pts: hypervolume_2d(pts, ref_all)),
        ('normalized',   lambda pts: norm_hv(pts, dmin, dmax)),
    ]

    lines = []
    lines.append('=== 固定卷子 HV 多标尺重算 (helpers.hypervolume_2d 已修: 裁 ref 框外点) ===')
    lines.append(f'方法数={len(names)}  oracle={sorted(ORACLE)}  病态(不定盒子)={sorted(DEGENERATE)}')
    lines.append(f'current_ref   = ({cur_ref[0]:.2f}, {cur_ref[1]:.3f})  随机nadir×1.1, 偏小')
    lines.append(f'union_deploy  = ({ref_deploy[0]:.2f}, {ref_deploy[1]:.3f})  '
                 f'盒子 delay[{dmin[0]:.1f},{dmax[0]:.1f}] energy[{dmin[1]:.2f},{dmax[1]:.2f}]')
    lines.append(f'union_all     = ({ref_all[0]:.2f}, {ref_all[1]:.3f})  含 oracle')
    lines.append(f'normalized    = 两轴归一到[0,1] (用 union_deploy 盒子), ref=(1.1,1.1)')
    lines.append('')

    cols = [s[0] for s in scales]
    hdr = f'{"method":<16}{"type":>9}' + ''.join(f'{c:>14}' for c in cols)
    lines.append(hdr)
    lines.append('-' * len(hdr))

    def typ(n):
        return 'ORACLE' if n in ORACLE else ('degenerate' if n in DEGENERATE else '')

    # 按 normalized 排序展示 (推荐标尺), oracle/病态照常列出但标注
    order = sorted(names, key=lambda n: -norm_hv(fronts[n], dmin, dmax))
    for n in order:
        row = f'{n:<16}{typ(n):>9}'
        for _, fn in scales:
            row += f'{fn(fronts[n]):>14.3f}'
        lines.append(row)

    txt = '\n'.join(lines)
    print(txt)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f'\n[saved] {OUT}')


if __name__ == '__main__':
    main()
