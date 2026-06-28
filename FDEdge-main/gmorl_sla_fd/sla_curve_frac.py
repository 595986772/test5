"""SLA 操作曲线 (连续分数卸载): 违约率 ↔ HV, 扩散 vs 高斯。

扫 SLA 权重 λ → 每个 λ 一个策略 → 一个 (mean违约率, HV) 操作点。
λ 小: 不重视SLA → 违约高、前沿HV高; λ 大: 重视SLA → 违约低、HV 让步。连成操作曲线。
**左上更优 = 任给目标违约率达到更高 HV** (谁更优由数据说话, 不预设)。

P1-3 修: 与 eval_frac 共用 `resolve_eval_cfg` 还原训练 env + 强制各 tag env 一致 + 固定
  torch/numpy 双随机 (det 采样可复现)。tag 前缀可配, 默认 diff_l*/gauss_l* (旧 11 维已作废,
  需用当前码按 λ 重训新 tag 后再传 --diff_pre/--gauss_pre)。

用法: python sla_curve_frac.py --lams 0,1,3,6 --k 12 --diff_pre diff_l --gauss_pre gauss_l
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
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
from env_frac_offload import FracOffloadEnv
from frac_agent import FracAgent
from helpers import hypervolume_2d
from eval_frac import resolve_eval_cfg

RESULT = 'result2/frac'


def rollout(ag, env, w, K, seed0):
    d, e, v = [], [], []
    for k in range(K):
        np.random.seed(seed0 + k); torch.manual_seed(seed0 + k); env.w = w   # 固定 np+torch (扩散采样可复现)
        obs = env.reset(); prior = np.ones(env.N) / env.N; done = False
        while not done:
            a = ag.act(obs, prior); obs, r, done, info = env.step(a); prior = a
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy']); v.append(s['violation_rate'])
    return np.mean(d), np.mean(e), np.mean(v)


def sweep(tag, actor, env, omegas, K, seed0, dev, T=5):
    ag = FracAgent(env.N, actor_type=actor, T=T, device=dev)
    ag.load(os.path.join(RESULT, tag))
    pts = [rollout(ag, env, float(w), K, seed0) for w in omegas]   # (d,e,v)
    return np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lams', default='0,1,3,6')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--n_omega', type=int, default=9)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--n_srv', type=int, default=5)
    ap.add_argument('--deadline', type=float, default=7.0)
    ap.add_argument('--e_f_ratio', type=float, default=0.20)
    ap.add_argument('--agg_ratio', type=float, default=0.10)
    ap.add_argument('--idle_ratio', type=float, default=0.40)
    ap.add_argument('--keep_alive', type=int, default=0)
    ap.add_argument('--q_max_ratio', type=float, default=0.2)
    ap.add_argument('--sequential', action='store_true')
    ap.add_argument('--arrival_dt', type=float, default=5.0)
    ap.add_argument('--diff_pre', default='diff_l', help='扩散按λ训练的 tag 前缀, 完整 tag=<pre><λ>')
    ap.add_argument('--gauss_pre', default='gauss_l', help='高斯按λ训练的 tag 前缀')
    ap.add_argument('--allow_incomplete', action='store_true',
                    help='允许某方法缺 λ 点仍出图(标记为不完整); 默认缺点即中止(论文图需完整同网格)')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lams = [int(x) for x in args.lams.split(',')]

    # 收集已训练 tag, 复用 eval 的 meta 还原 + 一致性守卫 (拒绝跨 env 混比)。
    plan = []   # (actor, lam, tag)
    for actor, pre in [('diffusion', args.diff_pre), ('mlp', args.gauss_pre)]:
        for lam in lams:
            tag = '%s%d' % (pre, lam)
            if os.path.exists(os.path.join(RESULT, tag, 'actor.pt')):
                plan.append((actor, lam, tag))
            else:
                print('  [skip] %s 未训练' % tag)
    if not plan:
        print('[err] 无任何已训练 tag, 退出 (请先按 λ 用当前码重训)'); return
    # P2-1: 论文图要求每个方法拥有完整且相同的 λ 网格; 缺点默认中止 (除非 --allow_incomplete)。
    expected = set(lams)
    cover = {}
    for actor, lam, _ in plan:
        cover.setdefault(actor, set()).add(lam)
    incomplete = {a: sorted(expected - s) for a, s in cover.items() if s != expected}
    is_incomplete = bool(incomplete)
    if is_incomplete and not args.allow_incomplete:
        print('[err] 以下方法缺 λ 点, 论文图需完整同网格 -> 中止 (加 --allow_incomplete 强画为不完整结果):')
        for a, miss in incomplete.items():
            print('   %-10s 缺 λ=%s' % (a, miss))
        return
    if is_incomplete:
        print('[warn] λ 网格不完整, 出图将标 [不完整] 且用虚线: %s' % incomplete)
    defaults = dict(n_servers=args.n_srv, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                    agg_ratio=args.agg_ratio, idle_ratio=args.idle_ratio, keep_alive=args.keep_alive,
                    q_max_ratio=args.q_max_ratio, sequential=args.sequential, arrival_dt=args.arrival_dt,
                    horizon=30, homogeneous=True)
    ecfg, per_tag = resolve_eval_cfg([t for _, _, t in plan], defaults)
    env = FracOffloadEnv(**ecfg)
    omegas = np.linspace(0, 1, args.n_omega)

    data = {'diffusion': {}, 'mlp': {}}
    allde = []
    for actor, lam, tag in plan:
        pts = sweep(tag, actor, env, omegas, args.k, args.seed0, dev, T=per_tag[tag]['T'])
        data[actor][lam] = pts
        allde.append(pts[:, :2])
    allde = np.vstack(allde)
    ref = (allde[:, 0].max() * 1.05, allde[:, 1].max() * 1.05)

    print('\n=== SLA 操作点 (公共 ref delay=%.2f energy=%.4f) ===' % ref)
    print('%-10s %4s %12s %8s' % ('actor', 'λ', 'mean_viol', 'HV'))
    curve = {'diffusion': [], 'mlp': []}
    for actor in ['diffusion', 'mlp']:
        for lam in lams:
            if lam not in data[actor]:
                continue
            pts = data[actor][lam]
            mv = float(pts[:, 2].mean())
            hv = hypervolume_2d(pts[:, :2], ref)
            curve[actor].append((mv, hv, lam))
            print('%-10s %4d %12.3f %8.3f' % (actor, lam, mv, hv))

    # ---- 操作曲线: x=违约率, y=HV ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sty = {'diffusion': ('C0', 'o', '扩散 (Diffusion-QL)'), 'mlp': ('C3', 's', '高斯 (MLP-SAC)')}
    for actor in ['diffusion', 'mlp']:
        c = sorted(curve[actor])   # 按违约率排序
        if not c:
            continue
        col, mk, lab = sty[actor]
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        ls = '--' if is_incomplete else '-'
        ax.plot(xs, ys, ls, color=col, marker=mk, ms=8, label=lab, lw=1.6)
        for mv, hv, lam in c:
            ax.annotate('λ=%d' % lam, (mv, hv), fontsize=7, xytext=(3, 4), textcoords='offset points')
    # 不反转 x 轴: 违约率小在左 (x 越右越大), HV 大在上 -> 左上 = 低违约+高HV = 最优角, 与标题一致。
    ax.set_xlabel('Mean SLA violation rate'); ax.set_ylabel('Hypervolume (delay-energy)')
    title = 'SLA 操作曲线: 违约率 ↔ HV (deadline=%.0fs)\n左上更优 (低违约 + 高HV)' % ecfg['deadline']
    if is_incomplete:
        title += '  [不完整 λ 网格]'
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(RESULT, 'cmp'), exist_ok=True)
    png = os.path.join(RESULT, 'cmp', 'sla_curve_frac.png'); fig.savefig(png, dpi=130)
    print('\n[saved]', png)


if __name__ == '__main__':
    main()
