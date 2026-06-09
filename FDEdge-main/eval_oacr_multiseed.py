"""
OACR 稳健性核查: 把已训好的 early/late 两个存档, 在多批不同评估环境上各测一遍。
==================================================================
回答用户的问题: +20% 是 late 模型真的更好, 还是刚好原来那一批固定评估环境对 late 有利?
  每批: early、late 用**同一批环境**(fresh env + 同 eval seed)→ 受控对比;
  跨批: 换 eval seed = 换环境批次。
判读: late 每批都稳定赢 → +20% 真; 赢幅乱跳/翻盘 → 是评估批次的运气。
纯评估, 不重训。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import glob
import numpy as np
import torch

import mofd_main
from mofd_environment import MOFDEnvironment
from mofd_main import evaluate_pareto
from mofd_v8 import MOFD_SAC_V8
from mofd_v8_oacr import MOFD_SAC_V8_OACR
from helpers import hypervolume_2d, SHARED_EVAL_SEED_OFFSET

DEV = torch.device('cpu')


def build_env():
    # 每次新建 (seed=0) → comp_density / channel_rng 种子一致 → 同 eval seed 下 early/late 见到完全相同环境
    return MOFDEnvironment(Emax=6, num_tasks_max=50, bit_range=(10, 40), time_slots=100,
                           f_range=(10, 40), delay_scale=0.05, energy_scale=0.25, seed=0)


def load_agent(cls, ckpt):
    env = build_env()
    ag = cls(state_dim=env.state_dim, Emax=6, hidden_dim=128, denoising_steps=3,
             alpha_T=1.0, alpha_E=1.0, device=DEV)
    ag.actor.load_state_dict(torch.load(f'{ckpt}/actor.pt', map_location=DEV))
    ag.critic1.load_state_dict(torch.load(f'{ckpt}/critic1.pt', map_location=DEV))
    ag.critic2.load_state_dict(torch.load(f'{ckpt}/critic2.pt', map_location=DEV))
    ag.actor.eval()
    return ag


def main():
    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())
    ek = sorted(glob.glob('results/oacr_early_2*/ckpt_seed0'))[-1]
    lk = sorted(glob.glob('results/oacr_late_2*/ckpt_seed0'))[-1]
    print(f'early ckpt: {ek}\nlate  ckpt: {lk}')
    early = load_agent(MOFD_SAC_V8, ek)
    late = load_agent(MOFD_SAC_V8_OACR, lk)

    # 第一个种子用原始 final-eval 种子 → 应复现 ~+20% (验证重载正确); 其余是新环境批次
    seeds = [SHARED_EVAL_SEED_OFFSET, 101, 202, 303, 404]
    rows = []
    for s in seeds:
        pe = evaluate_pareto(build_env(), early, n_pref=21, n_eval_epi=2,
                             alpha_T=1.0, alpha_E=1.0, seed=int(s))
        pl = evaluate_pareto(build_env(), late, n_pref=21, n_eval_epi=2,
                             alpha_T=1.0, alpha_E=1.0, seed=int(s))
        allp = np.vstack([pe, pl])
        ref = (float(allp[:, 0].max() * 1.1 + 1e-6), float(allp[:, 1].max() * 1.1 + 1e-6))
        he = hypervolume_2d(pe, ref); hl = hypervolume_2d(pl, ref)
        rows.append((int(s), he, hl))
        print(f'  eval-seed {s:>5}:  early={he:8.2f}  late={hl:8.2f}  '
              f'diff={hl-he:+8.2f}  ({100*(hl-he)/(he+1e-9):+6.1f}%)', flush=True)

    diffs = np.array([100*(hl-he)/(he+1e-9) for _, he, hl in rows])
    win = int((diffs > 0).sum())
    lines = ['=== OACR early vs late 多批评估 (重载存档, 纯评估) ===',
             f'{"eval-seed":>10}{"early HV":>11}{"late HV":>11}{"diff%":>9}']
    for s, he, hl in rows:
        lines.append(f'{s:>10}{he:>11.2f}{hl:>11.2f}{100*(hl-he)/(he+1e-9):>8.1f}%')
    lines += ['',
              f'late 赢的批次: {win}/{len(rows)}   diff% 范围 [{diffs.min():+.1f}, {diffs.max():+.1f}]   均值 {diffs.mean():+.1f}%',
              '',
              '判读: 5/5 都明显赢且幅度稳 → +20% 是真 (late 模型确实更好);',
              '      赢幅乱跳/翻盘 → +20% 是评估批次运气, 非模型差异。']
    txt = '\n'.join(lines)
    print('\n' + txt)
    with open('results/oacr_multiseed_eval.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n[saved] results/oacr_multiseed_eval.txt')


if __name__ == '__main__':
    main()
