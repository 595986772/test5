"""
用固定卷子重测存疑结论 (纯评估, 不重训)
========================================
在校准好的"秤"(固定卷子, 同一 ref)上, 把两个存疑结论重测:
  1. oracle greedy 到底是不是真碾压训练模型?
  2. V8 源消融 prior > feedback 是真排名还是单批运气?

全部已训好的 ckpt, 纯评估。每个源模型忠实复现其训练评估协议
(MCSS_MODE + use_true_feedback + eval_use_prior + 各自 omega_buf)。
K_EVAL 个场景/档 (默认 20, 比 40 快一半, 仍远比旧的 2 稳)。

用法: python run_recheck_testset.py            # 默认 K_EVAL=20
      python run_recheck_testset.py --k 40     # 全卷
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
import glob
import numpy as np
import torch

from fixed_testset import load_testset, eval_agent_on_testset, eval_greedy_on_testset
from mofd_environment import MOFDEnvironment
from ablation_agents import MOFD_SAC_V5_HMCSS
from mofd_main import OmegaLatentBuffer

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 空时用 GPU, 扩散 eval 快很多


def load_src(mode, ts):
    ck = sorted(glob.glob(f'results/abl_mcss_v8_src_{mode}_*/ckpt_seed0'))[-1]
    env = MOFDEnvironment(**ts['env_params'])
    MOFD_SAC_V5_HMCSS.MCSS_MODE = mode
    ag = MOFD_SAC_V5_HMCSS(state_dim=env.state_dim, Emax=6, hidden_dim=128,
                           denoising_steps=3, alpha_T=1.0, alpha_E=1.0, device=DEV)
    ag.actor.load_state_dict(torch.load(f'{ck}/actor.pt', map_location=DEV))
    ag.critic1.load_state_dict(torch.load(f'{ck}/critic1.pt', map_location=DEV))
    ag.critic2.load_state_dict(torch.load(f'{ck}/critic2.pt', map_location=DEV))
    ag.actor.eval()
    ob = OmegaLatentBuffer()
    ob.load_pickle(f'{ck}/omega_buf.pkl')
    return ag, ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=20)
    args = ap.parse_args()
    ts = load_testset()
    ref = ts['ref']
    K = args.k
    print(f'固定卷子: 21档 × {K}场景/档  ref={ref}\n')

    # 把每个模型在固定卷子上的 21 点前沿存盘 (供 rescale_hv.py 在多个候选 ref 下重算)
    def _dump_pts(tag, pts):
        with open(f'results/testset_{tag}_pareto.csv', 'w', encoding='utf-8') as f:
            f.write('omega_T,delay,energy\n')
            for om, (d, e) in zip(ts['prefs'], pts):
                oT = float(np.asarray(om, dtype=float).ravel()[0])
                f.write(f'{oT:.4f},{float(d):.6f},{float(e):.6f}\n')

    rows = []
    # 1) oracle greedy (精确物理 + 全信息) —— 上界参照
    pts, hv = eval_greedy_on_testset(ts, hard=True, k_eval=K)
    _dump_pts('greedy_omega', pts)
    rows.append(('oracle-greedy(hard)', hv, 'heuristic'))
    print(f'  oracle-greedy(hard): HV={hv:.3f}', flush=True)

    # 2) 4 个 V8 源模型 (忠实协议: 各自 MCSS_MODE + true_feedback + eval_use_prior)
    for mode in ['prior', 'feedback', 'random', 'full3']:
        ag, ob = load_src(mode, ts)
        MOFD_SAC_V5_HMCSS.MCSS_MODE = mode
        pts, hv = eval_agent_on_testset(ag, ts, k_eval=K, eval_use_prior=True,
                                        omega_buf=ob, use_true_feedback=True)
        _dump_pts(f'v8_src_{mode}', pts)
        rows.append((f'V8-src_{mode}', hv, 'trained'))
        print(f'  V8-src_{mode}: HV={hv:.3f}', flush=True)

    rows.sort(key=lambda r: -r[1])
    lines = ['=== 固定卷子重测 (校准的秤, 同一 ref) ===',
             f'21档 × {K}场景/档   ref={ref}', '',
             f'{"方法":<22}{"HV":>10}{"类型":>10}']
    for n, hv, typ in rows:
        lines.append(f'{n:<22}{hv:>10.3f}{typ:>10}')
    # 关键对比
    d = dict((n, hv) for n, hv, _ in rows)
    g = d.get('oracle-greedy(hard)', float('nan'))
    pr = d.get('V8-src_prior', float('nan')); fb = d.get('V8-src_feedback', float('nan'))
    best_model = max(hv for n, hv, t in rows if t == 'trained')
    lines += ['',
              f'① greedy vs 最强训练模型: greedy={g:.1f}  best_trained={best_model:.1f}  '
              f'→ {"greedy 仍碾压" if g > best_model*1.05 else ("打平/接近" if abs(g-best_model)<=best_model*0.05 else "训练模型反超")}',
              f'② prior vs feedback: prior={pr:.1f}  feedback={fb:.1f}  '
              f'→ {"prior 仍赢" if pr > fb*1.03 else ("打平" if abs(pr-fb)<=fb*0.03 else "feedback 反超")}',
              '',
              '注: 这是校准秤上的单模型重测; 跨训练 seed 的稳健性仍需多 seed (但秤已不再是噪声源)。']
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open('results/recheck_testset.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n[saved] results/recheck_testset.txt')


if __name__ == '__main__':
    main()
