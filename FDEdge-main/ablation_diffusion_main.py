"""
消融实验 1: w/o 反馈扩散
========================
对照两组, 其余一切相同 (vector-Q / COR / PopArt / ω-buffer / 超参):
  - full         : 原方法 PC-FDN (反馈扩散 actor)
  - no_diffusion : 扩散 actor 换成普通 MLP 策略

模型: V8 (干净版 critic: 目标去熵 + 等权), 两组都一样, 只隔离"反馈扩散 actor"这一个变量。
跑完在 results/ 下产出 (v8 前缀, 不覆盖旧 V5 消融数据):
  abl_diff_v8_full_pareto_aggregated.csv
  abl_diff_v8_nodiff_pareto_aggregated.csv
  ablation_diffusion_v8_summary.{csv,txt}   ← 共享 ref 下的 HV 对比 (主结论看这个)

用法 (在 FDEdge-main/ 目录下运行):
  python ablation_diffusion_main.py --smoke           # 先验证管线 (~1 分钟)
  python ablation_diffusion_main.py --epochs 100      # 正式 (默认单 seed=0)
  python ablation_diffusion_main.py --epochs 100 --seeds 0 1 2   # 想多 seed 再显式传
"""
import argparse
from ablation_agents import MOFD_SAC_V5_NoDiffusion, run_variant, summarize
from mofd_v8 import MOFD_SAC_V8


def build_cfg(args):
    cfg = dict(seeds=list(args.seeds), num_epochs=args.epochs)
    if args.smoke:
        # 极小规模, 仅验证不崩 + 文件链路通
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
    args = ap.parse_args()
    base = build_cfg(args)

    variants = {}
    variants['full'] = run_variant('abl_diff_v8_full', MOFD_SAC_V8, dict(base))
    variants['no_diffusion'] = run_variant('abl_diff_v8_nodiff', MOFD_SAC_V5_NoDiffusion, dict(base))
    summarize(variants, 'ablation_diffusion_v8', anchor='full')


if __name__ == '__main__':
    main()
