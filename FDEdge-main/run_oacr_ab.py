"""
OACR 最小测试 A/B: omega 早融合 vs 晚融合 (未见偏好插值)
======================================================
唯一变量 = critic 里 omega 的融合时机 (其余 V8、超参、训练/评估全同):
  early : V8                (omega 从第 1 层混入, 现状)
  late  : V8 + QValueNetOACR (omega 第 2 层注入, 晚融合)

决定性设置 = **未见偏好插值** (这才是 OACR 的真赌注, 标准 21/21 方案自己也预测 HV≈):
  训练只给粗网格 ω_T ∈ {0, .25, .5, .75, 1} (5 点);
  评估细网格 21 点 → 其中 16 点训练没见过。
  晚融合若泛化更好, 它的 21 点 HV 应更高、偏好响应 ρ 更单调。

⚠️ 单 seed、单 run、40 epoch —— go/no-go 探针。late 明显赢 → 值得多 seed; ≈ → 整套 OACR 别建。
⚠️ 别和别的训练同时跑 (抢 CPU)。

用法 (FDEdge-main/ 目录):
  python run_oacr_ab.py --smoke
  python run_oacr_ab.py                 # 正式 (默认 40 epoch, 单 seed=0)
  python run_oacr_ab.py --epochs 100    # 想要全尺度
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
from mofd_v8_oacr import MOFD_SAC_V8_OACR
from helpers import hypervolume_2d, build_preference_set

RESULTS = 'results'
# 粗训练网格 (5 点); 是 21 点评估网格的子集 → seen/unseen 划分干净
COARSE_OMEGA = [[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]]


def spearman(x, y):
    """无依赖 Spearman 秩相关 (omega/energy 取值互异, 不处理并列)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12
    return float((rx * ry).sum() / d)


def build_cfg(args):
    cfg = dict(seeds=[0], num_epochs=args.epochs, use_v8=True,
               train_omegas=COARSE_OMEGA, final_eval_n_pref=21)
    if args.smoke:
        cfg.update(num_epochs=3, n_prefs_per_epoch=5, num_tasks_max=6, time_slots=12,
                   train_eval_n_pref=3, train_eval_n_epi=1,
                   final_eval_n_pref=21, final_eval_n_epi=1,
                   buffer_warmup=16, batch_size=8, update_every=4)
    return cfg


def run_arm(prefix, patch_oacr, base):
    cfg = dict(base); cfg['file_prefix'] = prefix
    print(f"\n############ OACR ARM: {prefix}  (late={patch_oacr}) ############")
    orig = mofd_main.MOFD_SAC_V8
    if patch_oacr:
        mofd_main.MOFD_SAC_V8 = MOFD_SAC_V8_OACR     # use_v8=True → AgentClass 解析到它
    try:
        mofd_main.main(cfg_override=cfg)
    finally:
        mofd_main.MOFD_SAC_V8 = orig
    # 单 seed: pareto_aggregated.csv = 按 ω 顺序的 21 个 (delay, energy)
    return np.loadtxt(os.path.join(RESULTS, f'{prefix}_pareto_aggregated.csv')).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    base = build_cfg(args)

    pts_early = run_arm('oacr_early', False, base)
    pts_late = run_arm('oacr_late', True, base)

    prefs = build_preference_set(21)
    wE = prefs[:, 1]
    seen_wT = {round(o[0], 4) for o in COARSE_OMEGA}
    unseen = np.array([round(float(w), 4) not in seen_wT for w in prefs[:, 0]])

    allp = np.vstack([pts_early, pts_late])
    ref = (float(allp[:, 0].max() * 1.1 + 1e-6), float(allp[:, 1].max() * 1.1 + 1e-6))

    def row(name, pts):
        # pts 与 prefs 同序 (单 seed); 若行数!=21 (异常) 则跳过 ρ
        hv = hypervolume_2d(pts, ref)
        hv_unseen = (hypervolume_2d(pts[unseen], ref)
                     if pts.shape[0] == 21 else float('nan'))
        if pts.shape[0] == 21:
            r_e = spearman(wE, pts[:, 1])     # ρ(ω_E, energy)  想更负
            r_d = spearman(wE, pts[:, 0])     # ρ(ω_E, delay)   想更正
        else:
            r_e = r_d = float('nan')
        return name, hv, hv_unseen, r_e, r_d, pts[:, 0].min(), pts[:, 1].min()

    re_ = row('early(现状,早融合)', pts_early)
    rl_ = row('late (OACR,晚融合)', pts_late)

    lines = [
        '=== OACR 最小测试 A/B (未见偏好插值: 训练 5 点 / 评估 21 点) ===',
        f'共享 ref={ref}   单 seed=0   epochs={args.epochs}',
        '唯一变量 = critic 里 omega 早融合 vs 晚融合',
        '',
        f'{"臂":<20}{"HV(21)":>10}{"HV(unseen16)":>14}{"ρ(ωE,E)":>10}{"ρ(ωE,d)":>10}{"d下界":>8}{"E下界":>8}',
    ]
    for n, hv, hvu, re2, rd, dmin, emin in (re_, rl_):
        lines.append(f'{n:<20}{hv:>10.3f}{hvu:>14.3f}{re2:>10.3f}{rd:>10.3f}{dmin:>8.2f}{emin:>8.3f}')
    dhv = rl_[1] - re_[1]; dhvu = rl_[2] - re_[2]
    lines += [
        '',
        f'ΔHV(21)      = {dhv:+.3f}  ({100*dhv/(re_[1]+1e-9):+.1f}%)',
        f'ΔHV(unseen)  = {dhvu:+.3f}  ({100*dhvu/(re_[2]+1e-9):+.1f}%)   ← 这条最关键',
        '',
        '判读: late 的 HV(unseen) 明显高 且 ρ(ωE,E) 更负/ρ(ωE,d) 更正',
        '        → omega 晚融合确实更泛化, 值得上多 seed, 再考虑 encoder/aux 完整版;',
        '      late ≈ early  → 融合时机没用, 整套 OACR(含 encoder+aux) 别建, 省几百行。',
        '注: 单 seed go/no-go, 别下定论; ρ 想要 ρ(ωE,energy)→-1、ρ(ωE,delay)→+1。',
    ]
    txt = '\n'.join(lines)
    out = os.path.join(RESULTS, 'oacr_ab_compare.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
