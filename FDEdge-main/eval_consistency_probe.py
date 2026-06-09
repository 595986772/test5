"""
训练/评估起点一致性探针
========================
问题: 训练时去噪起点 = retrieve_prior(buffer 池化 prior); 评估时却用零/均匀、且不查 buffer。
      → 评估的其实是和训练略有出入的策略。本探针在**同一张训好的网**上做 A/B:
        off: 现状起点 (零/均匀)
        on : 训练同款 retrieve_prior 起点 (命中失败 gate → 退回 feedback 零)
      只改"评估起点"这一个变量, take_action 结构、网络权重、评估上下文(同 seed)全不变,
      所以 HV 差 = 起点不一致的代价。不训练, 几分钟跑完。

⚠️ 注: 这里把 ablation ckpt 载入**基座 3 源 take_action** (load_agent_from_ckpt 的默认),
   所以 HV 绝对值不等于原 feedback 单源消融的 252; 看的是 **on vs off 的差**(take_action
   两边一致, 差只来自起点)。单 seed, 仅作探针。

用法 (FDEdge-main/ 目录):
  python eval_consistency_probe.py                 # 默认 feedback ckpt
  python eval_consistency_probe.py --n-pref 21 --n-epi 3
  python eval_consistency_probe.py --ckpt results/abl_mcss_src_random_xxx/ckpt_seed0
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

import mofd_main
from mofd_environment import MOFDEnvironment
from mofd_main import load_agent_from_ckpt, evaluate_pareto
from helpers import hypervolume_2d
from mechanism_diag import load_cfg_for_ckpt

DEVICE = torch.device('cpu')
EVAL_SEED = 4242   # off / on 用同一 seed → 评估上下文完全一致, 公平 A/B


def pick_ckpt():
    for src in ('feedback', 'random'):
        c = sorted(glob.glob(f'results/abl_mcss_src_{src}_*/ckpt_seed0'))
        if c:
            return c[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--n-pref', type=int, default=21)
    ap.add_argument('--n-epi', type=int, default=3)
    args = ap.parse_args()
    ckpt = args.ckpt or pick_ckpt()
    if not ckpt:
        print('[err] 未找到 ckpt'); return

    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())
    cfg = load_cfg_for_ckpt(ckpt)
    aT = float(cfg.get('alpha_T', 1.0)); aE = float(cfg.get('alpha_E', 1.0))
    env = MOFDEnvironment(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'],
        delay_scale=cfg.get('delay_scale', 0.05),
        energy_scale=cfg.get('energy_scale', 0.25), seed=0)
    agent, omega_buf = load_agent_from_ckpt(ckpt, cfg, env, DEVICE)
    agent.actor.eval()
    if omega_buf is None:
        print('[err] 该 ckpt 无 omega_buf.pkl, 无法测 on 模式'); return
    print(f'[probe] ckpt={ckpt}  buffer entries={len(omega_buf.entries)}')

    # off: 现状零/均匀起点
    pts_off = evaluate_pareto(env, agent, n_pref=args.n_pref, n_eval_epi=args.n_epi,
                              alpha_T=aT, alpha_E=aE, seed=EVAL_SEED,
                              eval_use_prior=False)
    # on: 训练同款 retrieve_prior 起点, gate→feedback(零)
    gate_log = []
    pts_on = evaluate_pareto(env, agent, n_pref=args.n_pref, n_eval_epi=args.n_epi,
                             alpha_T=aT, alpha_E=aE, seed=EVAL_SEED,
                             omega_buf=omega_buf, eval_use_prior=True, gate_log=gate_log)

    # 共享 ref (两组点并集), HV 才可比
    allp = np.vstack([pts_off, pts_on])
    ref = (float(allp[:, 0].max() * 1.1 + 1e-6), float(allp[:, 1].max() * 1.1 + 1e-6))
    hv_off = hypervolume_2d(pts_off, ref)
    hv_on = hypervolume_2d(pts_on, ref)
    gate_rate = float(np.mean(gate_log)) if gate_log else float('nan')

    def corner(pts):
        return pts[:, 0].min(), pts[:, 1].min()   # (最低延迟, 最低能耗)
    d_off, e_off = corner(pts_off); d_on, e_on = corner(pts_on)

    lines = [
        f'=== 训练/评估起点一致性 A/B  @ {os.path.basename(os.path.dirname(ckpt))} ===',
        f'共享 ref={ref}   eval_seed={EVAL_SEED}   n_pref={args.n_pref} n_epi={args.n_epi}',
        f'(基座 3 源 take_action; on vs off 只差"评估起点", 看差不看绝对值)',
        '',
        f'{"模式":<28}{"HV":>10}{"延迟下界":>10}{"能耗下界":>10}',
        f'{"off (现状 零/均匀起点)":<28}{hv_off:>10.4f}{d_off:>10.2f}{e_off:>10.3f}',
        f'{"on  (训练同款prior, gate→fb)":<28}{hv_on:>10.4f}{d_on:>10.2f}{e_on:>10.3f}',
        '',
        f'ΔHV (on-off)      = {hv_on - hv_off:+.4f}  ({100*(hv_on-hv_off)/(hv_off+1e-9):+.1f}%)',
        f'gate 率 (on 模式) = {gate_rate:.1%}   (命中失败、退回 feedback 零起点的比例)',
        '',
        '判读:',
        '  ΔHV > 0 且 gate 率不高  → 消除起点不一致确实有收益, 一致性修复值得保留, 再上 Pareto 筛选;',
        '  ΔHV ≈ 0               → 起点不一致代价很小 (因零/均匀/池化prior 都低方差, 网络不敏感);',
        '  ΔHV < 0 或 gate 率很高 → buffer 在评估上下文几乎没命中(退回零=回到 off), 一致性没料,',
        '                           说明瓶颈在 buffer 覆盖, 不在"用不用 prior"。',
    ]
    txt = '\n'.join(lines)
    out = os.path.join(os.path.dirname(ckpt), 'eval_consistency_probe.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
