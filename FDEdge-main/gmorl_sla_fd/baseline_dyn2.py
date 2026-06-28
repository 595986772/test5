# -*- coding: utf-8 -*-
"""dyn2 启发式 baseline 对比 (A 随机稀疏 / B 比例 / C 贪婪延迟 / D 贪婪SLA-能耗) vs diffM1/gauss。
同 env / 同 ref / 同 HV-feasHV-#feas 协议; 单动作确定性评测。
用法: python baseline_dyn2.py --deadline 14 [--k 10]
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys, argparse
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch
from eval_frac import hypervolume_2d, rollout_agent, VIOL_THRESH
from frac_agent import FracAgent
from env_dyn_offload import DynOffloadEnv

ROOT = 'result2/frac'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 启发式策略 fn(env)->a ----------
def _energy_of(env, a):
    d, be, K, active = env._eval_alloc(a)
    warm_b = env.warm > 1e-9; on = active | (env.q > 1e-9); newly = on & (~warm_b)
    energy = be + float((env.e_f * newly).sum()) + float((env.p_idle * on).sum()) * env.arrival_dt
    return d, energy

def _cand_menu(env):
    N = env.N; eye = np.eye(N); cands = [eye[i] for i in range(N)]   # 单台
    of = np.argsort(-env.f); orr = np.argsort(-env.rate)
    for k in range(2, N + 1):                                         # top-k 按算力比例
        s = of[:k]; a = np.zeros(N); a[s] = env.f[s]; cands.append(a / a.sum())
    for k in [2, 3]:                                                  # top-k 按信道比例
        s = orr[:k]; a = np.zeros(N); a[s] = env.rate[s]; cands.append(a / a.sum())
    cands.append(np.ones(N) / N)                                      # 均分
    return cands

def bl_random_sparse(env):       # A: 随机选 1-2 台
    k = np.random.randint(1, 3); idx = np.random.choice(env.N, k, replace=False)
    a = np.zeros(env.N); a[idx] = np.random.dirichlet(np.ones(k)); return a

def bl_proportional(env):        # B: 按算力比例 (无学习负载均衡)
    a = env.f.astype(float).copy(); return a / a.sum()

def bl_greedy_delay(env):        # C: 选预测完成时间最短
    cs = _cand_menu(env); ds = [env._eval_alloc(a)[0] for a in cs]
    return cs[int(np.argmin(ds))]

def bl_greedy_sla_energy(env):   # D: 可行(<=deadline)中能耗最低; 否则违约最小
    cs = _cand_menu(env); rows = [(_energy_of(env, a), a) for a in cs]
    feas = [(e, a) for (d, e), a in rows if d <= env.deadline]
    if feas: return min(feas, key=lambda x: x[0])[1]
    return min(rows, key=lambda x: x[0][0])[1]

BASELINES = {'A_random_sparse': bl_random_sparse, 'B_proportional': bl_proportional,
             'C_greedy_delay': bl_greedy_delay, 'D_greedy_sla_energy': bl_greedy_sla_energy}

def rollout_baseline(fn, env, K, seed0, w=0.5):
    d, e, v, p = [], [], [], []
    for k in range(K):
        np.random.seed(seed0 + k); torch.manual_seed(seed0 + k); env.w = w
        env.reset(); done = False
        while not done:
            _, _, done, _ = env.step(fn(env))
        s = env.episode_sla_summary()
        d.append(s['mean_delay']); e.append(s['mean_energy']); v.append(s['violation_rate']); p.append(s['p95_delay'])
    return np.mean(d), np.mean(e), np.mean(v), np.mean(p)

def make_agent(tag, actor_type, T):
    ag = FracAgent(5, actor_type=actor_type, T=T, start_mode='randn', sparse=True,
                   feat_dim=8, omega_film=False, device=dev)
    ag.load(os.path.join(ROOT, tag)); return ag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline', type=float, default=14.0)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n_omega', type=int, default=11)
    ap.add_argument('--seed0', type=int, default=1000)
    args = ap.parse_args()

    def mkenv():
        return DynOffloadEnv(n_servers=5, deadline=args.deadline, arrival_dt=6.0,
                             coord_delay_ratio=0.10, dl_ratio=0.15)
    env = mkenv()
    omegas = np.linspace(0, 1, args.n_omega)

    methods = {}   # name -> list of (d,e,v,p,w)
    # 学习方法: ω-swept 前沿
    for tag, at, T in [('dyn2_diffM1', 'diffusion', 20), ('dyn2_gauss14', 'mlp', 5)]:
        ag = make_agent(tag, at, T); pts = []
        for w in omegas:
            d, e, v, p = rollout_agent(ag, env, float(w), args.k, args.seed0)
            pts.append((d, e, v, p, float(w)))
        methods[tag] = pts
        print('[%s] min_delay=%.2f' % (tag, min(x[0] for x in pts)))
    # 启发式: 单点
    print('\n=== 启发式 baselines (deadline=%.1f) ===' % args.deadline)
    for nm, fn in BASELINES.items():
        d, e, v, p = rollout_baseline(fn, env, args.k, args.seed0)
        methods[nm] = [(d, e, v, p, 0.5)]
        print('  %-20s delay=%6.2f energy=%6.4f viol=%.3f p95=%6.2f  %s'
              % (nm, d, e, v, p, '(feas)' if v <= VIOL_THRESH else ''))

    # 公共 ref + 统一 HV 表
    allde = np.array([(d, e) for pts in methods.values() for d, e, *_ in pts])
    ref = (allde[:, 0].max() * 1.05, allde[:, 1].max() * 1.05)
    print('\n=== HV (公共 ref delay=%.2f energy=%.4f, deadline=%.1f) ===' % (ref[0], ref[1], args.deadline))
    print('%-22s %8s %8s %9s %7s' % ('method', 'HV', 'feasHV', 'min_delay', '#feas'))
    for tag, pts in methods.items():
        de = np.array([(d, e) for d, e, *_ in pts])
        feas = np.array([(d, e) for d, e, v, *_ in pts if v <= VIOL_THRESH])
        hv = hypervolume_2d(de, ref); fhv = hypervolume_2d(feas, ref) if len(feas) else 0.0
        print('%-22s %8.3f %8.3f %9.2f %7d' % (tag, hv, fhv, de[:, 0].min(), len(feas)))

    # 存点 (画图用)
    import csv
    os.makedirs(os.path.join(ROOT, 'cmp'), exist_ok=True)
    with open(os.path.join(ROOT, 'cmp', 'baseline_dyn2_d%d_points.csv' % int(args.deadline)), 'w', newline='', encoding='utf-8') as f:
        wc = csv.writer(f); wc.writerow(['method', 'omega', 'delay', 'energy', 'viol', 'p95', 'feasible'])
        for nm, pts in methods.items():
            for d, e, v, p, w in pts:
                wc.writerow([nm, '%.3f' % w, '%.5f' % d, '%.5f' % e, '%.5f' % v, '%.5f' % p, int(v <= VIOL_THRESH)])
    print('\n[saved] cmp/baseline_dyn2_d%d_points.csv' % int(args.deadline))

if __name__ == '__main__':
    main()
