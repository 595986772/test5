"""
消融实验 2: H-MCSS (候选源 + 候选数)
====================================
A. 候选源消融 (C4 核心论点 "三源候选 + Critic 选优"):
     src_full3   : feedback + prior + random + Critic 选优  (= 完整方法, anchor)
     src_feedback: 仅反馈 latent 单候选
     src_prior   : 仅 prior 单候选
     src_random  : 仅随机起点单候选
B. 候选数敏感性 (n_candidates ∈ {1,3,6,10}):
     反馈 latent + (n-1) 个高斯扰动候选, Critic 选优. n=1 即无选优.

跑完在 results/ 下产出各 abl_mcss_*_pareto_aggregated.csv +
  ablation_hmcss_summary.{csv,txt}   ← 共享 ref 下 HV 对比 (主结论看这个)

⚠️ buffer 交互: H-MCSS 三源里 feedback 与 prior 的差异依赖 ω-buffer(cfg 默认 ON).
   若已决定删 buffer(use_omega_buffer=False), feedback 源会塌成 prior, 此时
   src_feedback ≈ src_prior, 只剩 "有/无随机源" 与候选数扫有意义 —— 结论照常成立,
   但解读时要说明. 想关 buffer 跑: 加 --no-buffer.

用法 (在 FDEdge-main/ 目录下运行):
  python ablation_hmcss_main.py --smoke
  python ablation_hmcss_main.py --epochs 100              # 正式 (默认单 seed=0)
  python ablation_hmcss_main.py --epochs 100 --no-buffer  # 删 buffer 跑
"""
import argparse
from ablation_agents import MOFD_SAC_V5_HMCSS, run_variant, summarize


def build_cfg(args):
    # use_true_feedback: feedback 源走真·within-episode 反馈链 (起点=上一步输出),
    #   训练上才真正区别于 prior 源 (静态 buffer prior); 二者不再训练时同物.
    # eval_use_prior: 评估也用真 buffer prior 作 prior 源起点 (否则评估退化成 uniform,
    #   prior 源照样没被测到), 顺带训练/评估起点一致.
    cfg = dict(seeds=list(args.seeds), num_epochs=args.epochs,
               use_true_feedback=True, eval_use_prior=True)
    if args.no_buffer:
        cfg['use_omega_buffer'] = False
    if args.smoke:
        cfg.update(num_epochs=1, n_prefs_per_epoch=2, num_tasks_max=6,
                   time_slots=12, train_eval_n_pref=3, train_eval_n_epi=1,
                   final_eval_n_pref=3, final_eval_n_epi=1,
                   buffer_warmup=16, batch_size=8, update_every=4, seeds=[0])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-buffer', dest='no_buffer', action='store_true',
                    help='use_omega_buffer=False (删 buffer 时三源里 feedback≈prior)')
    ap.add_argument('--skip-sources', action='store_true', help='只跑候选数扫')
    ap.add_argument('--skip-ncand', action='store_true', help='只跑候选源消融')
    args = ap.parse_args()
    base = build_cfg(args)

    variants = {}
    # A. 候选源  (模型已是 V8 干净版 critic; v8 前缀, 不覆盖旧 V5 消融数据)
    if not args.skip_sources:
        for mode in ['full3', 'feedback', 'prior', 'random']:
            variants[f'src_{mode}'] = run_variant(
                f'abl_mcss_v8_src_{mode}', MOFD_SAC_V5_HMCSS, dict(base), mcss_mode=mode)
    # B. 候选数 (暂不跑, 已注释; 想跑回时取消注释即可)
    # if not args.skip_ncand:
    #     for n in [1, 3, 6, 10]:
    #         variants[f'ncand_{n}'] = run_variant(
    #             f'abl_mcss_n{n}', MOFD_SAC_V5_HMCSS, dict(base),
    #             mcss_mode='perturb', mcss_n=n)

    summarize(variants, 'ablation_hmcss_v8',
              anchor='src_full3' if not args.skip_sources else None)


if __name__ == '__main__':
    main()
