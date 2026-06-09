"""
路由源实验: 按偏好分区路由扩散起点 (单候选, 无 Critic 选优)
==========================================================
  能耗优先 ω_E ≥ τ → feedback latent (占低能耗角)
  延迟优先 ω_E < τ → random 起点     (占低延迟角)
默认 τ=0.5. 这是把 H-MCSS 失败的"每步 Critic 选源"换成"按干净 ω 信号确定性选源",
**把路由训练进策略**后检验能否真吃到多源红利, 而不是 post-hoc 把两条前沿拼起来。

⚠️ 单 seed=0 / 单 run —— 只是 go/no-go 探针。阳性也只说明"值得上多 seed", 不是定论;
   阴性 (≤ feedback) 则说明训练进策略后路由失效, 与 post-hoc 并集上界是两回事。

跑完直接打印对照表 (与 ablation_hmcss_summary 完全相同的固定 ref, HV 可逐行对比),
并把表写到 results/abl_routed_*_compare.txt。

用法 (FDEdge-main/ 目录):
  python ablation_routed_main.py --smoke
  python ablation_routed_main.py --epochs 100            # 正式, 同 H-MCSS 配置 (单 seed=0)
  python ablation_routed_main.py --epochs 100 --tau 0.5
"""
import argparse
import os
import numpy as np

from ablation_agents import MOFD_SAC_V5_RoutedSource, run_variant
from helpers import hypervolume_2d, pareto_front_2d

# 与 ablation_hmcss_summary.txt 完全一致的固定 ref —— HV 才能逐行直接对比
REF = (151.9987, 4.5547)
RESULTS = 'results'
SRC_CSV = {
    'feedback': 'abl_mcss_src_feedback_pareto_aggregated.csv',
    'prior':    'abl_mcss_src_prior_pareto_aggregated.csv',
    'random':   'abl_mcss_src_random_pareto_aggregated.csv',
    'full3':    'abl_mcss_src_full3_pareto_aggregated.csv',
}


def _load(name):
    return np.loadtxt(os.path.join(RESULTS, SRC_CSV[name])).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--tau', type=float, default=0.5,
                    help='能耗优先 ω_E≥τ 用 feedback, 否则 random (默认 0.5)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-buffer', dest='no_buffer', action='store_true')
    args = ap.parse_args()

    cfg = dict(seeds=list(args.seeds), num_epochs=args.epochs)
    if args.no_buffer:
        cfg['use_omega_buffer'] = False
    if args.smoke:
        cfg.update(num_epochs=1, n_prefs_per_epoch=2, num_tasks_max=6,
                   time_slots=12, train_eval_n_pref=3, train_eval_n_epi=1,
                   final_eval_n_pref=3, final_eval_n_epi=1,
                   buffer_warmup=16, batch_size=8, update_every=4, seeds=[0])

    MOFD_SAC_V5_RoutedSource.ROUTE_TAU = float(args.tau)
    tag = f"abl_routed_tau{str(args.tau).replace('.', '')}"
    res = run_variant(tag, MOFD_SAC_V5_RoutedSource, cfg)
    routed = np.asarray(res['pts']).reshape(-1, 2)

    # ---- 对照表 (固定 ref) ----
    rows = []
    for name in ['feedback', 'prior', 'random', 'full3']:
        try:
            pts = _load(name)
            rows.append((name, hypervolume_2d(pts, REF), len(pts)))
        except OSError:
            pass
    rows.append((f'routed(tau={args.tau})', hypervolume_2d(routed, REF), len(routed)))

    # post-hoc 并集上界 (oracle: 把已训好的两/三条前沿拼起取非支配)
    try:
        fb, rd = _load('feedback'), _load('random')
        u2 = pareto_front_2d(np.vstack([fb, rd]))
        rows.append(('union(fb|rd) oracle', hypervolume_2d(u2, REF), len(u2)))
        u3 = pareto_front_2d(np.vstack([fb, _load('prior'), rd]))
        rows.append(('union(3src) oracle', hypervolume_2d(u3, REF), len(u3)))
    except OSError:
        pass

    base = dict((n, hv) for n, hv, _ in rows).get('feedback') or 1.0
    lines = [f'=== 路由源 vs 单源 vs 并集上界  (固定 ref={REF}) ===',
             f'routed: ω_E≥{args.tau}→feedback, 否则→random   |   单 seed=0, 单 run',
             '', f'{"variant":<22}{"HV":>11}{"vs_fb%":>9}{"#pts":>6}']
    for n, hv, k in rows:
        lines.append(f'{n:<22}{hv:>11.4f}{100 * (hv - base) / base:>8.1f}%{k:>6}')
    lines += ['',
              '判据:',
              f'  routed > feedback({base:.2f}) 且逼近 union(fb|rd) → 训练进路由真能吃红利, 值得上多 seed;',
              '  routed ≈/≤ feedback                          → 训练进策略后路由失效, 这条线就此打住。',
              '注: union 是 post-hoc oracle 上界 (两条已训好的前沿事后拼), 不是可部署方法。']
    txt = '\n'.join(lines)
    print('\n' + txt)

    out = os.path.join(res['run_dir'] or RESULTS, f'{tag}_compare.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f'\n[compare] saved -> {out}')


if __name__ == '__main__':
    main()
