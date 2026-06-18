"""多 ref 敏感性: 主方法(V8)能不能在某个合理 ref 下翻上来?
注意口径: V8 前沿来自训练协议 aggregated(自抽场景=主方法的有利场), baseline 前沿来自固定卷子。
不同卷, 近似比较 -> 这是"对主方法最有利"的探针: 它若连这都翻不上来, 就是稳的负结论。"""
import os, glob, csv
import numpy as np
from helpers import hypervolume_2d

ORACLE = {'greedy_omega'}; DEGEN = {'greedy_energy'}

fronts = {}
for src in ['prior', 'feedback', 'random', 'full3']:
    p = f'results/abl_mcss_v8_src_{src}_pareto_aggregated.csv'
    if os.path.exists(p):
        fronts[f'v8_{src}'] = np.loadtxt(p)            # 训练协议, 空格分隔
for p in glob.glob('results/testset_*_pareto.csv'):     # 固定卷子, 逗号+表头
    name = os.path.basename(p)[len('testset_'):-len('_pareto.csv')]
    pts = []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 3:
                pts.append([float(row[1]), float(row[2])])
    if pts:
        fronts[name] = np.array(pts)

allp = np.vstack(list(fronts.values()))
ref_E = float(allp[:, 1].max() * 1.02)          # 能耗 ref 罩住所有点 -> 只扫 delay 头寸
sweep = [40, 50, 70, 100, 160, 260, 440]
MAIN = 'v8_full3'                                # 完整方法

def tag(n):
    return 'ORC' if n in ORACLE else ('deg' if n in DEGEN else ('★' if n == MAIN else ''))

print(f'ref_E={ref_E:.2f} (罩住所有点, 不裁能耗轴)  ★=主方法({MAIN})  ORC=oracle上界  deg=退化')
print(f'{"method":<13}{"typ":>4}' + ''.join(f'{"d<"+str(d):>9}' for d in sweep) + f'{"norm":>9}')
print('-' * (17 + 9 * (len(sweep) + 1)))

# normalized: 用 deployable(非 oracle/退化) 的并集盒子
dep = [n for n in fronts if n not in ORACLE and n not in DEGEN]
bpts = np.vstack([fronts[n] for n in dep])
bmin, bmax = bpts.min(0), bpts.max(0)
def nhv(pts):
    span = np.maximum(bmax - bmin, 1e-9)
    return hypervolume_2d((pts - bmin) / span, (1.1, 1.1))

rows = {}
for n, pts in fronts.items():
    rows[n] = [hypervolume_2d(pts, (d, ref_E)) for d in sweep] + [nhv(pts)]

# 按 normalized 排序
for n in sorted(fronts, key=lambda k: -rows[k][-1]):
    print(f'{n:<13}{tag(n):>4}' + ''.join(f'{v:>9.2f}' for v in rows[n]))

# --- 翻盘判定 (分三组: 真 baseline / 同门兄弟源 / oracle) ---
TRUE_BASE = [n for n in fronts if n not in ORACLE and n not in DEGEN
             and not n.startswith('v8_')]
SIB = [n for n in fronts if n.startswith('v8_') and n != MAIN]
print('\n=== 主方法(v8_full3) 能否翻上来 ===')
print(f'真baseline={TRUE_BASE}\n同门源={SIB}  oracle={sorted(ORACLE)}')
print(f'{"ref":<7}{"main":>8}{"best_TRUEbase":>20}{"vs真base":>10}{"best_sib":>16}{"vs oracle":>12}')
for j, d in enumerate(sweep + ['norm']):
    col = 'norm' if d == 'norm' else f'd<{d}'
    mv = rows[MAIN][j]
    bb = max((rows[n][j], n) for n in TRUE_BASE)
    bs = max((rows[n][j], n) for n in SIB)
    orc = max((rows[n][j] for n in ORACLE), default=float('nan'))
    print(f'{col:<7}{mv:>8.2f}{bb[0]:>11.2f}({bb[1]:<7}){("WIN" if mv>bb[0] else "lose"):>10}'
          f'{bs[0]:>10.2f}({bs[1][3:]:<4}){("WIN" if mv>orc else "lose"):>8}({orc:.0f})')
