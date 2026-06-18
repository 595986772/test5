"""
路 B: baseline 统一评估适配 (在固定卷子上把所有方法放同一把秤)
=================================================================
背景: baselines/* 训练完**不存模型**, 各自用 compute_fixed_ref(seed+7777) + 随机 3 场景
      评估 (= fixed_testset 刚修掉的"坏秤"), HV 和 V8 的固定卷子 HV **不可比**。

本模块提供"路 B": 不改 baseline 的训练, 只把它的"取动作"接到 fixed_testset 的统一评估
(共享 ref + 钉死种子 + 固定卷子), 直接吐 21 点 Pareto + HV, 累积进一张可比表。

两类用法:
  1) 启发式 (免训练, 立即可跑): rand / rr / greedy-delay / greedy-energy / ω-greedy
        python eval_baselines_on_testset.py            # K=20
        python eval_baselines_on_testset.py --k 40
  2) RL baseline (需先训练, 在其 main 末尾调一行, agent 在内存里):
        from eval_baselines_on_testset import evaluate_trained_agent
        evaluate_trained_agent('dsac', agent)                       # dsac/ldqn: 默认 get_state
        evaluate_trained_agent('genmosac', agent,
                               state_builder=lambda env,t,n: build_dict_state(env,t,n))

产物:
  results/testset_<name>_pareto.csv   每方法 21 点 (omega_T, delay, energy)
  results/testset_compare.csv/.txt    所有方法的可比汇总 (同 ref), 多次运行自动 upsert
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
import json
import numpy as np

from fixed_testset import load_testset, eval_policy_on_testset, eval_greedy_on_testset

COMPARE_CSV = 'results/testset_compare.csv'
COMPARE_TXT = 'results/testset_compare.txt'


# ---------------------------------------------------------------------------
# 启发式 action_fn (与 baselines/Heuristic/heuristic_main.py 同口径)
# ---------------------------------------------------------------------------
def rand_action_fn(env, t, n, omega, rng, E, f_E):
    return int(rng.integers(0, max(E, 1)))


def make_rr_action_fn():
    """Round-Robin: 返回 (action_fn, reset_fn); reset_fn 在每个场景开始清零计数器。"""
    state = {'i': 0}

    def fn(env, t, n, omega, rng, E, f_E):
        a = state['i'] % max(E, 1)
        state['i'] += 1
        return a

    def reset():
        state['i'] = 0

    return fn, reset


def gd_action_fn(env, t, n, omega, rng, E, f_E):
    """Greedy-Delay: 选即时 delay 最小的有效服务器 (含队列等待)。"""
    d_n = float(env.tasks_bit[t][n])
    rho_d = float(env.comp_density[n] * d_n)
    best_e, best = 0, float('inf')
    for e_ in range(E):
        f_b = f_E[e_]
        v = env.effective_rate(t, n, e_)
        if f_b <= 0 or v <= 0:
            continue
        delay_e = (d_n / v + rho_d / f_b
                   + (env.proc_queue_len[t, e_] + env.proc_queue_bef[t, e_]) / f_b)
        if delay_e < best:
            best, best_e = delay_e, e_
    return best_e


def ge_action_fn(env, t, n, omega, rng, E, f_E):
    """Greedy-Energy: 选频率最低 (最省能) 的有效服务器。"""
    best_e, best = 0, float('inf')
    for e_ in range(E):
        if f_E[e_] > 0 and f_E[e_] < best:
            best, best_e = f_E[e_], e_
    return best_e


# ---------------------------------------------------------------------------
# NN baseline 适配 (dsac / ldqn / genmosac: 统一 take_action(state, mask, stochastic))
# ---------------------------------------------------------------------------
def make_nn_action_fn(agent, state_builder=None, stochastic=False):
    """把一个训练好的 NN agent 包成 action_fn。

    agent.take_action(state, mask, stochastic=False) -> int   (dsac/ldqn/genmosac 同签名)
    state_builder(env, t, n): 可选; genmosac 用 dict 状态时传它, 否则默认 env.get_state(t,n)。
    """
    def fn(env, t, n, omega, rng, E, f_E):
        s = state_builder(env, t, n) if state_builder is not None else env.get_state(t, n)
        return int(agent.take_action(s, env.get_valid_mask(), stochastic=stochastic))
    return fn


def evaluate_trained_agent(name, agent, ts=None, state_builder=None, k_eval=20,
                           stochastic=False):
    """RL baseline 在其训练 main 末尾调用: 在固定卷子上评估并落盘到可比表。"""
    if ts is None:
        ts = load_testset()
    fn = make_nn_action_fn(agent, state_builder=state_builder, stochastic=stochastic)
    # LDQN 是循环网络: 每个场景(episode)开头重置隐状态, 与其训练评估口径一致;
    # dsac/genmosac 无 reset_hidden → None, eval_policy_on_testset 跳过 (它们非循环)。
    reset_fn = getattr(agent, 'reset_hidden', None)
    pts, hv = eval_policy_on_testset(ts, fn, k_eval=k_eval, reset_fn=reset_fn)
    dump_result(name, pts, hv, ts, 'trained', k_eval=k_eval)
    return pts, hv


# ---------------------------------------------------------------------------
# checkpoint 存/取: 把"评估"与"训练"解耦 (照主方法 results/<...>_ckpt_seed<seed>/ 格式)
# 只存策略网络 + 重建元信息 → 换卷子可离线重评, 不必重训。
# ---------------------------------------------------------------------------
def _policy_net(baseline, agent):
    if baseline in ('dsac', 'genmosac'):
        return agent.actor            # 含状态编码器 (genmosac 的 DictEncoder)
    if baseline == 'ldqn':
        return agent.q_net
    raise ValueError(f'unknown baseline {baseline}')


def save_baseline_ckpt(baseline, tag, agent, seed, ctor_meta, n_bins=None,
                       out_dir='results'):
    """存 baseline 策略网络 + 重建元信息到 results/<tag>_ckpt_seed<seed>/。

    之后可: python eval_baselines_on_testset.py --eval_ckpt <tag> <seed>  离线重评。
    """
    import torch
    d = os.path.join(out_dir, f'{tag}_ckpt_seed{seed}')
    os.makedirs(d, exist_ok=True)
    torch.save(_policy_net(baseline, agent).state_dict(), os.path.join(d, 'policy.pt'))
    with open(os.path.join(d, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(baseline=baseline, ctor=ctor_meta, n_bins=n_bins), f,
                  ensure_ascii=False, indent=2)
    print(f'  [ckpt] saved {d}/policy.pt', flush=True)
    return d


def load_baseline_agent(tag, seed, out_dir='results', device=None):
    """从 results/<tag>_ckpt_seed<seed>/ 重建 baseline agent (仅策略, 供卷子评估)。

    返回 (agent, state_builder); state_builder 仅 genmosac 非 None。
    """
    import torch
    if device is None:
        device = torch.device('cpu')
    d = os.path.join(out_dir, f'{tag}_ckpt_seed{seed}')
    with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    baseline, ctor = meta['baseline'], meta['ctor']
    here = os.path.dirname(os.path.abspath(__file__))
    sb = None
    if baseline == 'dsac':
        sys.path.insert(0, os.path.join(here, 'baselines', 'DiscreteSAC'))
        from dsac_model import DiscreteSAC
        ag = DiscreteSAC(device=device, **ctor)
    elif baseline == 'ldqn':
        sys.path.insert(0, os.path.join(here, 'baselines', 'LDQN'))
        from ldqn_model import LDQN
        ag = LDQN(device=device, **ctor)
        ag.epsilon = 0.0
    elif baseline == 'genmosac':
        sys.path.insert(0, os.path.join(here, 'baselines', 'GenMOSAC'))
        from genmosac_model import GenMOSAC, build_dict_state
        ag = GenMOSAC(device=device, **ctor)
        nb = meta.get('n_bins') or ctor.get('n_bins')
        sb = lambda env, t, n: build_dict_state(env, t, n, nb)
    else:
        raise ValueError(f'unknown baseline {baseline}')
    net = _policy_net(baseline, ag)
    net.load_state_dict(torch.load(os.path.join(d, 'policy.pt'), map_location=device,
                                   weights_only=True))
    net.eval()
    return ag, sb


def eval_linucb_on_testset(ts, k_eval=20, n_warmup=20, alpha=1.0, lam=1.0):
    """LinUCB 专用评估 (per-ω 在线 bandit, 无单一持久模型, 所以不走 action_fn)。

    每个 ω 新建 agent, 在该 ω 的固定场景上 explore 在线学 n_warmup 次, 再在 k_eval 个
    固定场景上 exploit-only 评估。同卷同 ref, 复用 LinUCB 自己的 run_episode/build_context。
    注: warmup 与 eval 用同一批固定场景 (LinUCB 看过的上下文), 是其有利情形, 已知偏乐观。
    """
    import sys as _sys
    _lu = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'LinUCB')
    if _lu not in _sys.path:
        _sys.path.insert(0, _lu)
    from linucb_main import LinUCBAgent, run_episode as lu_run   # noqa: E402
    from mofd_environment import MOFDEnvironment
    from helpers import hypervolume_2d

    env = MOFDEnvironment(**ts['env_params'])
    d_context = 8                                  # = build_context 维度 (与 linucb_main 一致)
    pts = []
    for oi, omega in enumerate(ts['prefs']):
        scen = ts['scenarios'][oi] if k_eval is None else ts['scenarios'][oi][:k_eval]
        om = np.asarray(omega, np.float32)
        agent = LinUCBAgent(env.Emax, d_context, alpha=alpha, lam=lam)
        for w in range(n_warmup):                  # 在线 warmup (explore)
            E, f_E, tran_rate, tasks = scen[w % len(scen)]
            lu_run(env, agent, tasks, E, f_E, tran_rate, om, explore=True)
        d_all, e_all = [], []
        for (E, f_E, tran_rate, tasks) in scen:    # exploit-only 评估
            d, e = lu_run(env, agent, tasks, E, f_E, tran_rate, om, explore=False)
            d_all.append(d); e_all.append(e)
        pts.append([float(np.mean(d_all)), float(np.mean(e_all))])
    pts = np.array(pts)
    return pts, float(hypervolume_2d(pts, ts['ref']))


# ---------------------------------------------------------------------------
# 落盘: 每方法 21 点 + 可比汇总表 (upsert, 多次运行累积)
# ---------------------------------------------------------------------------
def _metrics(pts):
    d, e = pts[:, 0], pts[:, 1]
    e_sprd = (e.max() - e.min()) / e.mean() if e.mean() > 0 else 0.0
    return d.min(), d.max(), e.min(), e.max(), e_sprd


def dump_result(name, pts, hv, ts, typ='heuristic', k_eval=None, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    prefs = ts['prefs']
    p = os.path.join(out_dir, f'testset_{name}_pareto.csv')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('omega_T,delay,energy\n')
        for om, (d, e) in zip(prefs, pts):
            oT = float(np.asarray(om, dtype=float).ravel()[0])
            f.write(f'{oT:.4f},{d:.6f},{e:.6f}\n')
    _upsert_compare(name, typ, pts, hv, ts, k_eval)
    print(f'  [saved] {p}  HV={hv:.3f}', flush=True)


def _upsert_compare(name, typ, pts, hv, ts, k_eval=None):
    """读现有可比表 → 替换/新增本方法这一行 → 按 HV 降序重写 csv + txt。"""
    ref = ts['ref']
    K = int(k_eval) if k_eval is not None else int(ts.get('K', 0))
    dmin, dmax, emin, emax, espr = _metrics(pts)
    row = dict(name=name, type=typ, HV=hv, delay_min=dmin, delay_max=dmax,
               energy_min=emin, energy_max=emax, E_sprd_mean=espr,
               ref_T=ref[0], ref_E=ref[1], K=K)
    rows = {}
    if os.path.exists(COMPARE_CSV):
        with open(COMPARE_CSV, encoding='utf-8') as f:
            hdr = f.readline().strip().split(',')
            for ln in f:
                vals = ln.strip().split(',')
                if len(vals) != len(hdr):
                    continue
                r = dict(zip(hdr, vals))
                rows[r['name']] = r
    rows[name] = {k: (f'{v:.5f}' if isinstance(v, float) else str(v)) for k, v in row.items()}
    ordered = sorted(rows.values(), key=lambda r: -float(r['HV']))
    cols = ['name', 'type', 'HV', 'delay_min', 'delay_max',
            'energy_min', 'energy_max', 'E_sprd_mean', 'ref_T', 'ref_E', 'K']
    os.makedirs(os.path.dirname(COMPARE_CSV), exist_ok=True)
    with open(COMPARE_CSV, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in ordered:
            f.write(','.join(r.get(c, '') for c in cols) + '\n')
    with open(COMPARE_TXT, 'w', encoding='utf-8') as f:
        f.write('=== 固定卷子统一评估 (同卷同 ref, 可直接比) ===\n')
        f.write(f'ref=({float(ref[0]):.3f}, {float(ref[1]):.3f})  K={K}\n\n')
        f.write(f'{"method":<18}{"type":>10}{"HV":>11}'
                f'{"d[min,max]":>20}{"e[min,max]":>18}{"Espr/m":>9}\n')
        for r in ordered:
            f.write(f'{r["name"]:<18}{r["type"]:>10}{float(r["HV"]):>11.3f}'
                    f'{"["+r["delay_min"][:6]+","+r["delay_max"][:7]+"]":>20}'
                    f'{"["+r["energy_min"][:5]+","+r["energy_max"][:5]+"]":>18}'
                    f'{float(r["E_sprd_mean"]):>9.3f}\n')


# ---------------------------------------------------------------------------
# main: 立即跑全部启发式 (免训练) 落到可比表
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--linucb', action='store_true',
                    help='额外评估 LinUCB (per-ω 在线 bandit, 较慢)')
    ap.add_argument('--eval_ckpt', nargs=2, metavar=('TAG', 'SEED'), default=None,
                    help='离线模式: 从 results/<TAG>_ckpt_seed<SEED>/ 载入 baseline 策略, '
                         '在固定卷子上评估并落可比表 (不跑启发式)')
    args = ap.parse_args()

    if args.eval_ckpt:
        tag, seed = args.eval_ckpt[0], int(args.eval_ckpt[1])
        ts = load_testset()
        ag, sb = load_baseline_agent(tag, seed)
        print(f'[eval_ckpt] {tag} seed{seed} 在固定卷子上评估 (K={args.k}) ...', flush=True)
        evaluate_trained_agent(tag, ag, ts=ts, state_builder=sb, k_eval=args.k)
        print('\n' + open(COMPARE_TXT, encoding='utf-8').read())
        print(f'[done] 可比表: {COMPARE_CSV} / {COMPARE_TXT}')
        return

    ts = load_testset()
    K = args.k
    print(f'固定卷子统一评估 (启发式, 免训练): 21档 × {K}场景  ref={ts["ref"]}\n', flush=True)

    pts, hv = eval_policy_on_testset(ts, rand_action_fn, k_eval=K)
    dump_result('rand', pts, hv, ts, 'heuristic', k_eval=K)

    rr_fn, rr_reset = make_rr_action_fn()
    pts, hv = eval_policy_on_testset(ts, rr_fn, k_eval=K, reset_fn=rr_reset)
    dump_result('round_robin', pts, hv, ts, 'heuristic', k_eval=K)

    pts, hv = eval_policy_on_testset(ts, gd_action_fn, k_eval=K)
    dump_result('greedy_delay', pts, hv, ts, 'heuristic', k_eval=K)

    pts, hv = eval_policy_on_testset(ts, ge_action_fn, k_eval=K)
    dump_result('greedy_energy', pts, hv, ts, 'heuristic', k_eval=K)

    # ω-条件贪心 (= PGW arm A): 复用现成的精确物理 oracle greedy
    pts, hv = eval_greedy_on_testset(ts, hard=True, k_eval=K)
    dump_result('greedy_omega', pts, hv, ts, 'heuristic', k_eval=K)

    if args.linucb:
        pts, hv = eval_linucb_on_testset(ts, k_eval=K)
        dump_result('linucb', pts, hv, ts, 'bandit', k_eval=K)

    print('\n' + open(COMPARE_TXT, encoding='utf-8').read())
    print(f'[done] 可比表: {COMPARE_CSV} / {COMPARE_TXT}')


if __name__ == '__main__':
    main()
