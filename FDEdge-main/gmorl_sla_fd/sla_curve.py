"""SLA 操作特性曲线: 违约率 ↔ Pareto-HV。

核心论证 (对比 GMORL):
  * GMORL-等价 (纯 delay-energy MORL, 无 SLA 机制) = 一个点 (高 HV, 高违约), 没有降违约的旋钮。
  * Ours = 一条曲线 (扫准入 margin): 从高HV/高违约 推向 低HV/低违约。
  * 若 Ours 曲线压在 GMORL 点的下方/左侧, 且延伸到 GMORL 到不了的低违约区
    -> 证明 "同 HV 下 Ours 违约更低 / 任给目标违约率 Ours 能达到, GMORL 不能"。

所有配置同卷 (相同 seed -> 相同场景) + 公共 ref (HV 才可比)。
用法: python sla_curve.py --k 10
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent
from eval_pareto import rollout_agent
from helpers import hypervolume_2d

CONF = 'multi-part'
VIOL_THRESH = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n_omega', type=int, default=11)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--ours_tag', type=str, default='run_vec_ours')
    ap.add_argument('--gmorl_tag', type=str, default='run_vec_gmorl')
    ap.add_argument('--deadline', type=float, default=20.0)
    ap.add_argument('--task_size_cap', type=float, default=20e6)
    ap.add_argument('--out', type=str, default='sla_curve')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    omegas = np.linspace(0.0, 1.0, args.n_omega)
    env = MEC_Env(conf_name=CONF, w=0.5, deadline=args.deadline, task_size_cap=args.task_size_cap)
    DEADLINE, CAP = args.deadline, args.task_size_cap
    CONFIGS = [
        ('GMORL-equiv (no SLA)', args.gmorl_tag, False, 1.0),
        ('Ours (SLA-ch, no adm)', args.ours_tag, False, 1.0),
        ('Ours +adm m=1.2',       args.ours_tag, True,  1.2),
        ('Ours +adm m=1.0',       args.ours_tag, True,  1.0),
        ('Ours +adm m=0.8',       args.ours_tag, True,  0.8),
        ('Ours +adm m=0.6',       args.ours_tag, True,  0.6),
    ]

    agents = {}
    def get_agent(tag):
        if tag not in agents:
            ag = FDSACAgent(denoising_steps=3, device=dev)
            ag.load(os.path.join('result2', tag))
            agents[tag] = ag
        return agents[tag]

    # ---- pass 1: 每个配置扫 ω, 收前沿 + 平均违约 ----
    results = []  # (label, de[N,2], mean_viol, is_gmorl)
    print('=== 采集各配置前沿 (K=%d, %d 档 ω) ===' % (args.k, args.n_omega))
    for label, tag, adm, margin in CONFIGS:
        ag = get_agent(tag)
        de, viols = [], []
        for w in omegas:
            d, e, v, p = rollout_agent(ag, env, float(w), args.k, args.seed0,
                                       admission=adm, margin=margin)
            de.append((d, e)); viols.append(v)
        is_gmorl = ('GMORL' in label)
        results.append((label, np.array(de), float(np.mean(viols)), is_gmorl))
        print('  %-24s mean_viol=%.3f' % (label, np.mean(viols)), flush=True)

    # ---- 公共 ref (所有点的 nadir × 1.1), HV 才可比 ----
    all_de = np.vstack([r[1] for r in results])
    ref = (all_de[:, 0].max() * 1.1, all_de[:, 1].max() * 1.1)
    pts = []  # (label, mean_viol, hv, is_gmorl)
    print('\n=== HV (公共 ref=%.2f, %.3f) ===' % ref)
    for label, de, mv, isg in results:
        hv = hypervolume_2d(de, ref)
        pts.append((label, mv, hv, isg))
        print('  %-24s viol=%.3f  HV=%.3f' % (label, mv, hv))

    # ---- 画 (违约率 x, HV y) 操作曲线 ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ours = [(p[1], p[2], p[0]) for p in pts if not p[3]]
    gm = [(p[1], p[2], p[0]) for p in pts if p[3]]
    ours.sort(key=lambda z: z[0])
    if ours:
        ox = [z[0] for z in ours]; oy = [z[1] for z in ours]
        ax.plot(ox, oy, '-o', color='C0', lw=1.6, ms=7, label='Ours (sweep admission)', zorder=3)
        for vx, vy, lab in ours:
            ax.annotate(lab.replace('Ours ', ''), (vx, vy), fontsize=7,
                        textcoords='offset points', xytext=(5, 5))
    for vx, vy, lab in gm:
        ax.scatter(vx, vy, color='C3', marker='*', s=320, edgecolors='k', zorder=4,
                   label='GMORL-equiv (no SLA knob)')
        ax.annotate(lab, (vx, vy), fontsize=8, textcoords='offset points', xytext=(6, -12))
    ax.axvline(VIOL_THRESH, color='gray', ls='--', lw=1, alpha=0.7)
    ax.text(VIOL_THRESH, ax.get_ylim()[1], ' SLA feasible', fontsize=8, color='gray', va='top')
    ax.set_xlabel('Mean violation rate (lower = better SLA)')
    ax.set_ylabel('Pareto-HV (higher = better trade-off)')
    ax.set_title('SLA operating curve: Ours (tunable) vs GMORL-equiv (single point)\n'
                 '(deadline=%.0fs, task cap=%.0fMb)' % (DEADLINE, CAP / 1e6))
    ax.legend(fontsize=9, loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join('result2', '%s.png' % args.out)
    fig.savefig(png, dpi=140)
    print('\n[saved] %s' % png)

    import csv
    with open(os.path.join('result2', '%s.csv' % args.out), 'w', newline='') as f:
        wc = csv.writer(f); wc.writerow(['config', 'mean_violation', 'HV', 'is_gmorl'])
        for label, mv, hv, isg in pts:
            wc.writerow([label, '%.4f' % mv, '%.4f' % hv, int(isg)])
    print('[saved] result2/sla_curve.csv')


if __name__ == '__main__':
    main()
