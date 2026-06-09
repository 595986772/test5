"""
最小 Pareto 版 A/B: ParetoMemory prior vs 普通加权平均 prior
==========================================================
两臂**唯一变量** = buffer 取 prior 的方式 (其余 V8 / eval_use_prior / 超参全同):
  baseline : V8 + OmegaLatentBuffer (附近历史动作加权平均, prior 偏糊)
  pareto   : V8 + ParetoMemoryBuffer (同桶非支配筛选 + 选单个好 prior)
两臂都开 eval_use_prior (训练/评估同款起点, 命中失败 gate→feedback 零), 所以对比公平:
HV 差只来自"Pareto 筛选 vs 加权平均"。

⚠️ 单 seed=0、单 run, 仅 go/no-go 探针。pareto > baseline 且 gate 率不高 → Pareto 筛选有料,
   值得上多 seed; pareto ≈/< baseline → 筛选没用, 这条线打住。

用法 (FDEdge-main/ 目录):
  python run_pareto_memory_ab.py --smoke
  python run_pareto_memory_ab.py --epochs 100         # 正式 (单 seed=0)
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

import mofd_main
from helpers import hypervolume_2d, pareto_front_2d

RESULTS = 'results'


def build_cfg(args):
    cfg = dict(seeds=[0], num_epochs=args.epochs, use_v8=True, eval_use_prior=True)
    if args.smoke:
        cfg.update(num_epochs=3, n_prefs_per_epoch=4, num_tasks_max=6, time_slots=12,
                   train_eval_n_pref=3, train_eval_n_epi=1,
                   final_eval_n_pref=5, final_eval_n_epi=1,
                   buffer_warmup=16, batch_size=8, update_every=4,
                   pmem_warmup_epochs=0, obuf_warmup_epochs=0)
    return cfg


def run_arm(prefix, cfg_extra, base):
    cfg = dict(base); cfg.update(cfg_extra); cfg['file_prefix'] = prefix
    print(f"\n############ ARM: {prefix}  {cfg_extra} ############")
    mofd_main.main(cfg_override=cfg)
    pts = np.loadtxt(os.path.join(RESULTS, f'{prefix}_pareto_aggregated.csv')).reshape(-1, 2)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    base = build_cfg(args)

    pts_base = run_arm('pmem_ab_baseline',
                       dict(use_pareto_memory=False, use_omega_buffer=True), base)
    pts_par = run_arm('pmem_ab_pareto',
                      dict(use_pareto_memory=True), base)

    # 共享 ref (两臂点并集) → HV 可比
    allp = np.vstack([pts_base, pts_par])
    ref = (float(allp[:, 0].max() * 1.1 + 1e-6), float(allp[:, 1].max() * 1.1 + 1e-6))
    hv_b = hypervolume_2d(pts_base, ref); hv_p = hypervolume_2d(pts_par, ref)

    def corner(p):
        return p[:, 0].min(), p[:, 1].min()
    db, eb = corner(pts_base); dp, ep = corner(pts_par)

    lines = [
        '=== 最小 Pareto 版 A/B (V8, eval_use_prior 两臂都开; 唯一变量=取prior方式) ===',
        f'共享 ref={ref}   单 seed=0',
        '',
        f'{"臂":<26}{"HV":>10}{"延迟下界":>10}{"能耗下界":>10}{"#pts":>6}',
        f'{"baseline (加权平均prior)":<26}{hv_b:>10.4f}{db:>10.2f}{eb:>10.3f}{len(pts_base):>6}',
        f'{"pareto   (非支配筛选prior)":<26}{hv_p:>10.4f}{dp:>10.2f}{ep:>10.3f}{len(pts_par):>6}',
        '',
        f'ΔHV (pareto-baseline) = {hv_p - hv_b:+.4f}  ({100*(hv_p-hv_b)/(hv_b+1e-9):+.1f}%)',
        '',
        '判读: ΔHV>0 且能耗/延迟下界不退 → Pareto 筛选有料, 上多 seed;',
        '      ΔHV≈/<0                  → 筛选没用, 瓶颈不在 prior 质量, 打住。',
        '注: 单 seed 单 run, 别下定论; gate 率见各 run 的 obuf 日志。',
    ]
    txt = '\n'.join(lines)
    out = os.path.join(RESULTS, 'pmem_ab_compare.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
