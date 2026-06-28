"""连续分数卸载评估 (路A 主体, Block 3b): 扩散 vs 高斯, 同 ref HV/feasHV + 图。

扫 ω 出各方法前沿; 公共 ref 算 HV/feasHV; 含 uniform/random 参照点。
重点看**偏能量区制**(低 w)扩散是否兑现 POC 预测的多峰优势。

用法:
  python eval_frac.py --tags diff,gauss --actors diffusion,mlp --k 12
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
from env_hetero_offload import HeteroOffloadEnv
from env_dyn_offload import DynOffloadEnv
from frac_agent import FracAgent, VCritic, flat_state
from helpers import pareto_front_2d, hypervolume_2d

RESULT = 'result2/frac'
VIOL_THRESH = 0.10

# ---- 训练 env 配置还原 + 多 tag 一致性守卫 (P1-2: 防多模型静默混配/错配 env) ----
META_KEYMAP = {'n_srv': 'n_servers', 'n_servers': 'n_servers', 'deadline': 'deadline',
               'e_f_ratio': 'e_f_ratio', 'agg_ratio': 'agg_ratio', 'idle_ratio': 'idle_ratio',
               'keep_alive': 'keep_alive', 'sequential': 'sequential', 'arrival_dt': 'arrival_dt',
               'horizon': 'horizon', 'q_max_ratio': 'q_max_ratio'}
# 这些 env 关键参数必须在所有被比较的 tag 间完全一致, 否则"同环境对比"无效。
# (q_max_ratio 影响 bandit/direct 的队列分布; homogeneous 影响算力分布 — 由 meta 的 hetero 反推)
ENV_CONSISTENCY_KEYS = ['n_servers', 'sequential', 'arrival_dt', 'keep_alive', 'deadline',
                        'e_f_ratio', 'agg_ratio', 'idle_ratio', 'horizon', 'q_max_ratio', 'homogeneous']


def _read_meta(tag):
    import json
    p = os.path.join(RESULT, tag, 'meta.json')
    if not os.path.exists(p):
        raise FileNotFoundError('[meta] %s 缺 meta.json -> 无法还原训练 env, 拒绝评估 (请用当前码重训生成)' % tag)
    return json.load(open(p))


def resolve_eval_cfg(tags, defaults):
    """读每个 tag 的 meta.json 还原其训练 env, 并强制所有 tag 的 env 关键参数完全一致;
    不一致即抛错拒绝比较。每个 actor 也按各自 ckpt 的 (n_srv, T) 构建, 不再统一吃命令行 args。
    返回 (ecfg, per_tag): ecfg=共享 env kwargs; per_tag[tag]={'n_srv','T'}。"""
    metas = {t: _read_meta(t) for t in tags}
    env_types = {metas[t].get('env', 'frac') for t in tags}
    if len(env_types) > 1:
        raise ValueError('[eval 拒绝比较] tag 间 env 类型不一致: %s' % env_types)
    env_type = env_types.pop()
    if env_type == 'dyn':
        keymap = {'n_srv': 'n_servers', 'deadline': 'deadline', 'e_f_ratio': 'e_f_ratio',
                  'agg_ratio': 'agg_ratio', 'idle_ratio': 'idle_ratio', 'keep_alive': 'keep_alive',
                  'arrival_dt': 'arrival_dt', 'horizon': 'horizon', 'task_size': 'task_size',
                  'link_vol': 'chan_vol', 'bg_load': 'bg_load',
                  'dl_ratio': 'dl_ratio', 'coord_ratio': 'coord_delay_ratio'}
        ckeys = sorted(set(keymap.values()))
        base = dict(n_servers=defaults['n_servers'], deadline=11.0, e_f_ratio=0.2, agg_ratio=0.1,
                    idle_ratio=0.4, keep_alive=0, arrival_dt=6.0, horizon=30,
                    task_size=20e6, chan_vol=0.40, bg_load=0.12, dl_ratio=0.0, coord_delay_ratio=0.0)
    elif env_type == 'hetero':
        keymap = {'n_srv': 'n_servers', 'deadline': 'deadline', 'e_f_ratio': 'e_f_ratio',
                  'agg_ratio': 'agg_ratio', 'idle_ratio': 'idle_ratio', 'keep_alive': 'keep_alive',
                  'arrival_dt': 'arrival_dt', 'horizon': 'horizon', 'task_size': 'task_size',
                  'link_vol': 'link_vol', 'bg_load': 'bg_load'}
        ckeys = sorted(set(keymap.values()))
        base = dict(n_servers=defaults['n_servers'], deadline=11.0, e_f_ratio=0.2, agg_ratio=0.1,
                    idle_ratio=0.4, keep_alive=0, arrival_dt=6.0, horizon=30,
                    task_size=20e6, link_vol=0.35, bg_load=0.12)
    else:
        keymap = META_KEYMAP; ckeys = ENV_CONSISTENCY_KEYS; base = dict(defaults)
    per_env = {}
    for t, m in metas.items():
        cfg = dict(base)
        for mk, ek in keymap.items():
            if mk in m and m[mk] is not None:
                cfg[ek] = m[mk]
        if env_type == 'frac' and 'hetero' in m and m['hetero'] is not None:
            cfg['homogeneous'] = not bool(m['hetero'])
        per_env[t] = cfg
    disagree = {}
    for k in ckeys:
        vals = {t: per_env[t].get(k) for t in tags}
        if len({repr(v) for v in vals.values()}) > 1:
            disagree[k] = vals
    if disagree:
        lines = '\n'.join('   %-12s %s' % (k, v) for k, v in disagree.items())
        raise ValueError('[eval 拒绝比较] 以下 env 关键参数在各 tag 间不一致, 同环境对比无效:\n'
                         + lines + '\n只能比较同 env 训练的方法 (或分别评估)。')
    ecfg = per_env[tags[0]]
    per_tag = {t: {'n_srv': int(per_env[t]['n_servers']),
                   'T': int(metas[t].get('T', 5)),
                   'start_mode': 'randn' if (metas[t].get('randn_prior') or metas[t].get('sparsemax')) else 'prior',
                   'sparse': bool(metas[t].get('sparsemax', False)),
                   'omega_film': bool(metas[t].get('omega_film', False)),
                   'delay_scale': float(metas[t].get('delay_scale', 1.5)),
                   'sla_lambda': float(metas[t].get('sla_lambda', 3.0))} for t in tags}
    print('[meta] %d tag env=%s 一致: dt=%s e_f=%s idle=%s deadline=%s horizon=%s'
          % (len(tags), env_type, ecfg.get('arrival_dt'), ecfg.get('e_f_ratio'),
             ecfg.get('idle_ratio'), ecfg.get('deadline'), ecfg.get('horizon')))
    return env_type, ecfg, per_tag


def rollout_agent(ag, env, w, K, seed0):
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k); torch.manual_seed(seed0 + k); env.w = w   # 固定 np+torch (扩散采样可复现)
        obs = env.reset(); prior = np.ones(env.N) / env.N; done = False
        while not done:
            a = ag.act(obs, prior, det=False)
            obs, r, done, info = env.step(a); prior = a
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy'])
        v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)


def load_critics(tag, sd, N, dev):
    """加载 train_frac_seq 存的 twin critic + PopArt 统计 (没存则 None)。"""
    p1 = os.path.join(RESULT, tag, 'c1.pt')
    if not os.path.exists(p1):
        return None
    c1 = VCritic(sd, N).to(dev); c1.load_state_dict(torch.load(p1, map_location=dev))
    c2 = VCritic(sd, N).to(dev); c2.load_state_dict(torch.load(os.path.join(RESULT, tag, 'c2.pt'), map_location=dev))
    pa = torch.load(os.path.join(RESULT, tag, 'popart.pt'), map_location=dev)
    return c1.eval(), c2.eval(), pa['mu'].to(dev), pa['sigma'].to(dev)


def rollout_agent_select(ag, crit, env, w, K, seed0, M, delay_scale, sla_lambda, viol_filter=True):
    """M-候选 critic 选择 (IDQL 式): 每步 actor 采 M 个 sparse 候选, twin-min 原始尺度 Q 打分,
    先按 Q_C(SLA 通道)过滤预测最差的一半, 再选标量化最优执行。两法同协议=公平。"""
    c1, c2, pa_mu, pa_sig = crit; dev = ag.dev
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k); torch.manual_seed(seed0 + k); env.w = w
        obs = env.reset(); prior = np.ones(env.N) / env.N; done = False
        while not done:
            s1 = flat_state(torch.as_tensor(obs['servers'][None], device=dev),
                            torch.as_tensor(np.array([obs['omega']], dtype=np.float32), device=dev))
            sM = s1.repeat(M, 1)
            pl = ag._prior_latent(prior) if (ag.actor_type == 'diffusion' and ag.start_mode == 'prior') else None
            if pl is not None:
                pl = pl.repeat(M, 1)
            with torch.no_grad():
                acts = ag.actor.sample(sM, prior_latent=pl, det=False)        # [M,N] 候选
                qraw = pa_sig * torch.min(c1(sM, acts), c2(sM, acts)) + pa_mu   # [M,n_obj] 原始尺度
                scal = w * delay_scale * qraw[:, 0] + (1 - w) * qraw[:, 1] + sla_lambda * qraw[:, 2]
                if viol_filter and M > 1:                                       # 先滤预测违约最重的一半 (Q_C 越大越好)
                    qc = qraw[:, 2]; keep = qc >= qc.median()
                    scal = torch.where(keep, scal, torch.full_like(scal, -1e30))
                a = acts[int(torch.argmax(scal))].cpu().numpy()
            obs, r, done, info = env.step(a); prior = a
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy']); v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)


def rollout_fixed(fn, env, K, seed0):
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k); torch.manual_seed(seed0 + k); env.w = 0.5
        obs = env.reset(); done = False
        while not done:
            obs, r, done, info = env.step(fn(env.N));
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy'])
        v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='diff,gauss')
    ap.add_argument('--actors', default='diffusion,mlp')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--n_omega', type=int, default=11)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--n_srv', type=int, default=5)
    ap.add_argument('--deadline', type=float, default=7.0)
    ap.add_argument('--e_f_ratio', type=float, default=0.20)
    ap.add_argument('--agg_ratio', type=float, default=0.10)
    ap.add_argument('--idle_ratio', type=float, default=0.40)
    ap.add_argument('--keep_alive', type=int, default=0)
    ap.add_argument('--q_max_ratio', type=float, default=0.2)
    ap.add_argument('--sequential', action='store_true', help='序贯模式(队列跨任务累积)')
    ap.add_argument('--arrival_dt', type=float, default=5.0)
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--n_cand', type=int, default=1,
                    help='M-候选 critic 选择 (>1 启用 IDQL 式提议+筛选; 需训练已存 critic)')
    ap.add_argument('--deadline_eval', type=float, default=0.0,
                    help='>0: 覆盖 meta 的 deadline 做评测 (单动作 n_cand=1 下只改可行性标注/viol, 不改策略) -> deadline sweep')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tags = args.tags.split(','); actors = args.actors.split(',')
    # P1-2: 读每个 tag 的 meta 还原训练 env + 强制一致性, 并按各自 ckpt 建 actor (不再统一吃 args)。
    defaults = dict(n_servers=args.n_srv, deadline=args.deadline, e_f_ratio=args.e_f_ratio,
                    agg_ratio=args.agg_ratio, idle_ratio=args.idle_ratio, keep_alive=args.keep_alive,
                    q_max_ratio=args.q_max_ratio, sequential=args.sequential, arrival_dt=args.arrival_dt,
                    horizon=30, homogeneous=True)
    env_type, ecfg, per_tag = resolve_eval_cfg(tags, defaults)
    if args.deadline_eval > 0:
        ecfg['deadline'] = args.deadline_eval   # deadline sweep: 同策略, 改SLA阈值
        print('[deadline_eval] 覆盖 deadline -> %.1f (单动作只改可行性标注)' % args.deadline_eval)
    if env_type == 'dyn':
        env = DynOffloadEnv(**ecfg)
    elif env_type == 'hetero':
        env = HeteroOffloadEnv(**ecfg)
    else:
        env = FracOffloadEnv(**ecfg)
    feat_dim = env.reset()['servers'].shape[1]   # 每节点特征数 (frac=3, hetero=5)
    omegas = np.linspace(0, 1, args.n_omega)

    methods = {}   # name -> list of (d,e,v,p,w)
    for tag, at in zip(tags, actors):
        ptg = per_tag[tag]; N_t = ptg['n_srv']; sd_t = N_t * feat_dim + 1
        ag = FracAgent(N_t, actor_type=at, T=ptg['T'], start_mode=ptg['start_mode'],
                       sparse=ptg['sparse'], feat_dim=feat_dim, omega_film=ptg['omega_film'], device=dev)
        ag.load(os.path.join(RESULT, tag))
        crit = load_critics(tag, sd_t, N_t, dev) if args.n_cand > 1 else None
        use_sel = crit is not None
        if args.n_cand > 1 and crit is None:
            print('  [warn] %s 无存 critic -> 退回单采样 (需用存 critic 的新版 train_frac_seq 重训)' % tag)
        pts = []
        print('\n=== %s (%s)%s 扫 ω ===' % (tag, at, ' [M=%d候选选择]' % args.n_cand if use_sel else ''))
        for w in omegas:
            if use_sel:
                d, e, v, p = rollout_agent_select(ag, crit, env, float(w), args.k, args.seed0,
                                                  args.n_cand, ptg['delay_scale'], ptg['sla_lambda'])
            else:
                d, e, v, p = rollout_agent(ag, env, float(w), args.k, args.seed0)
            pts.append((d, e, v, p, float(w)))
            print('  w=%.2f | delay=%5.2f energy=%6.4f viol=%.3f p95=%5.2f %s'
                  % (w, d, e, v, p, '(feas)' if v <= VIOL_THRESH else ''), flush=True)
        methods[tag] = pts

    # baselines
    base = {}
    base['uniform'] = rollout_fixed(lambda N: np.ones(N) / N, env, args.k, args.seed0)
    base['random'] = rollout_fixed(lambda N: np.random.dirichlet(np.ones(N)), env, args.k, args.seed0)
    print('\n=== baselines ===')
    for n, (d, e, v, p) in base.items():
        print('  %-8s delay=%5.2f energy=%6.4f viol=%.3f p95=%5.2f' % (n, d, e, v, p))

    # 公共 ref
    allde = np.array([(d, e) for pts in methods.values() for d, e, *_ in pts] +
                     [(v[0], v[1]) for v in base.values()])
    ref = (allde[:, 0].max() * 1.05, allde[:, 1].max() * 1.05)
    print('\n=== HV (公共 ref delay=%.2f energy=%.4f) ===' % ref)
    print('%-10s %8s %8s %9s %7s' % ('method', 'HV', 'feasHV', 'min_delay', '#feas'))
    rowsout = []
    for tag, pts in methods.items():
        de = np.array([(d, e) for d, e, *_ in pts])
        feas = np.array([(d, e) for d, e, v, *_ in pts if v <= VIOL_THRESH])
        hv = hypervolume_2d(de, ref)
        fhv = hypervolume_2d(feas, ref) if len(feas) else 0.0
        print('%-10s %8.3f %8.3f %9.2f %7d' % (tag, hv, fhv, de[:, 0].min(), len(feas)))
        rowsout.append((tag, hv, fhv, float(de[:, 0].min()), len(feas)))

    # ---- 保存原始评估数据 (P1-2: 论文可复现; 之前 rowsout/methods 只在内存) ----
    import csv, json as _json
    cmp_dir = os.path.join(RESULT, 'cmp'); os.makedirs(cmp_dir, exist_ok=True)
    with open(os.path.join(cmp_dir, 'frac_eval_points.csv'), 'w', newline='', encoding='utf-8') as f:
        wc = csv.writer(f); wc.writerow(['method', 'actor', 'omega', 'delay', 'energy', 'viol', 'p95', 'feasible'])
        for tag, at in zip(tags, actors):
            for d, e, v, p, w in methods[tag]:
                wc.writerow([tag, at, '%.4f' % w, '%.6f' % d, '%.6f' % e, '%.6f' % v, '%.6f' % p,
                             int(v <= VIOL_THRESH)])
        for n, (d, e, v, p) in base.items():
            wc.writerow([n, 'baseline', 'NA', '%.6f' % d, '%.6f' % e, '%.6f' % v, '%.6f' % p,
                         int(v <= VIOL_THRESH)])
    with open(os.path.join(cmp_dir, 'frac_eval_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        wc = csv.writer(f); wc.writerow(['method', 'HV', 'feasHV', 'min_delay', 'n_feas'])
        for tag, hv, fhv, mind, nf in rowsout:
            wc.writerow([tag, '%.6f' % hv, '%.6f' % fhv, '%.4f' % mind, nf])
    _json.dump({'ref_delay': float(ref[0]), 'ref_energy': float(ref[1]), 'seed0': args.seed0,
                'k': args.k, 'n_omega': args.n_omega, 'viol_thresh': VIOL_THRESH,
                'tags': tags, 'actors': actors, 'ecfg': ecfg, 'per_tag': per_tag},
               open(os.path.join(cmp_dir, 'frac_eval_meta.json'), 'w'), indent=2)
    print('[saved] frac_eval_points.csv / frac_eval_summary.csv / frac_eval_meta.json (in %s)' % cmp_dir)

    # ---- 图: delay-energy 前沿 ----
    os.makedirs(os.path.join(RESULT, 'cmp'), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {'diffusion': 'C0', 'mlp': 'C3'}
    for tag, at in zip(tags, actors):
        pts = methods[tag]; de = np.array([(d, e) for d, e, *_ in pts])
        c = colors.get(at, 'C1')
        pf = pareto_front_2d(de)
        if len(pf):
            ax.plot(pf[:, 0], pf[:, 1], '-', color=c, alpha=0.5, lw=1.2)
        for d, e, v, p, w in pts:
            feas = v <= VIOL_THRESH
            ax.scatter(d, e, c=[c] if feas else [[1, 1, 1]], edgecolors=c, s=55,
                       marker='o', linewidths=1.4, zorder=3)
        ax.scatter([], [], c=c, label='%s (%s)' % (tag, at))
    for n, (d, e, v, p) in base.items():
        ax.scatter(d, e, marker='*' if n == 'uniform' else 'x', c='k', s=130, zorder=4, label=n)
    ax.set_xlabel('Mean makespan delay (s)'); ax.set_ylabel('Mean energy per task (J)')
    ax.set_title('连续分数卸载: 扩散 vs 高斯 (实心=SLA可行 viol≤10%%, deadline=%.0fs)\n'
                 '重点看低能耗角(右下)谁支配' % args.deadline)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    png = os.path.join(RESULT, 'cmp', 'frac_pareto.png')
    try:                                  # 图保存失败(如 PNG 被查看器占用)不得毁掉已算好的结果
        fig.savefig(png, dpi=130)
        print('\n[saved]', png)
    except OSError as e:
        print('\n[warn] 图保存失败 (%s) — 多半是该 PNG 正被图片查看器占用。'
              '原始数据 CSV/JSON 已保存, 不影响结果; 关掉查看器重跑或手动重绘即可。' % e)


if __name__ == '__main__':
    main()
