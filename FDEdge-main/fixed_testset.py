"""
固定大测试集 (eval "固定卷子")
=============================
问题: evaluate_pareto 每个 ω 只抽 2 个随机场景就平均, 同一个固定模型 HV 能跳 4.5 倍
      (实测 44→201)。根因 = 评估方差太大, 不是训练。

修法: 预先生成一套**固定**场景 (每个 ω 共 K 个), 存盘, 所有模型考**同一套**;
      评估时**钉死随机种子** (torch+numpy), 连去噪/候选的随机也固定。
      → 同一个模型每次考分数完全一致 (不再跳); 不同模型同卷 → 直接可比。

一张卷子里固定了什么:
  * 每个 ω 的 K 个场景: (E, f_E, tran_rate, tasks) —— 存盘;
  * 信道: 不显式存, 但靠 "每次新建 env(seed=0) + 固定顺序遍历" 使信道序列可复现;
  * 去噪/候选随机: eval 时 torch.manual_seed + np.random.seed 钉死;
  * ref: 用随机策略在本卷上的 nadir×1.1 (方法无关, 固定), 存进卷子 → HV 跨方法可比。

用法见文件末 __main__ 与 README 注释。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import pickle
import numpy as np
import torch

from mofd_environment import MOFDEnvironment
from helpers import build_preference_set, hypervolume_2d
import mofd_main
from mofd_main import sample_tasks, run_episode

# 全尺度, 对齐之前消融 config.json
ENV_PARAMS = dict(Emax=6, num_tasks_max=50, bit_range=(10, 40), time_slots=100,
                  f_range=(10, 40), delay_scale=0.05, energy_scale=0.25, seed=0)
TESTSET_PATH = 'results/eval_testset.pkl'
EVAL_SEED = 12345          # 钉死去噪/候选随机, 保证同模型同分


def build_testset(n_pref=21, K=40, gen_seed=20260608, path=TESTSET_PATH):
    """生成并存盘固定卷子: 每个 ω 的 K 个 (E, f_E, tran_rate, tasks) + 固定 ref。"""
    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())
    rng = np.random.default_rng(gen_seed)
    env = MOFDEnvironment(**ENV_PARAMS)
    prefs = build_preference_set(n_pref)
    scenarios = []
    for _ in prefs:
        lst = []
        for _ in range(K):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            lst.append((int(E), np.asarray(f_E, np.float32), np.asarray(tran_rate, np.float32),
                        [np.asarray(t, np.float32) for t in tasks]))
        scenarios.append(lst)
    ts = dict(prefs=prefs, K=int(K), scenarios=scenarios, env_params=ENV_PARAMS)
    ts['ref'] = _fixed_ref(ts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(ts, f)
    print(f'[build] 固定卷子已存: {path}  (n_pref={n_pref}, K={K}, ref={ts["ref"]})')
    return ts


def load_testset(path=TESTSET_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _fixed_ref(ts):
    """随机策略 nadir×1.1 作固定 ref (方法无关)。"""
    env = MOFDEnvironment(**ts['env_params'])
    rng = np.random.default_rng(7)
    pts = []
    for oi, omega in enumerate(ts['prefs']):
        d_all, e_all = [], []
        for (E, f_E, tran_rate, tasks) in ts['scenarios'][oi]:
            env.reset_env(tasks, E, f_E, tran_rate, omega)
            sd = se = n = 0.0
            for t in range(env.time_slots - 1):
                for nn in range(len(env.tasks_bit[t])):
                    a = int(rng.integers(0, max(int(E), 1)))
                    _, _, d, e, _ = env.step(t, nn, a)
                    sd += d; se += e; n += 1
                env.update_proc_queues(t)
            d_all.append(sd / max(n, 1)); e_all.append(se / max(n, 1))
        pts.append([np.mean(d_all), np.mean(e_all)])
    pts = np.array(pts)
    return (float(pts[:, 0].max() * 1.1 + 1e-6), float(pts[:, 1].max() * 1.1 + 1e-6))


def eval_agent_on_testset(agent, ts, eval_seed=EVAL_SEED, k_eval=None,
                          eval_use_prior=False, omega_buf=None, use_true_feedback=False):
    """在固定卷子上评估一个 NN agent → (21点[delay,energy], HV)。确定性: 钉死随机种子。

    k_eval: 只用前 k_eval 个场景 (None=全 40); eval_use_prior/omega_buf/use_true_feedback:
    复现各模型的训练评估协议 (与 evaluate_pareto 一致), 保证忠实对比。"""
    from mofd_main import make_env_ctx
    torch.manual_seed(eval_seed); np.random.seed(eval_seed)
    env = MOFDEnvironment(**ts['env_params'])
    A = env.action_dim
    uni = np.full(A, 1.0 / A, dtype=np.float32)
    pts = []
    for oi, omega in enumerate(ts['prefs']):
        scen = ts['scenarios'][oi] if k_eval is None else ts['scenarios'][oi][:k_eval]
        d_all, e_all = [], []
        for (E, f_E, tran_rate, tasks) in scen:
            if eval_use_prior and omega_buf is not None:
                env_ctx = make_env_ctx(E, f_E, tran_rate, Emax=env.Emax)
                start = omega_buf.retrieve_prior(omega, env_ctx=env_ctx, action_dim=A,
                                                 current_epoch=None)
                if np.allclose(start, uni, atol=1e-6):
                    start = np.zeros(A, dtype=np.float32)
                latent = np.broadcast_to(
                    start, [env.time_slots, env.n_tasks_max, A]).astype(np.float32).copy()
                prior_arg = start.astype(np.float32)
            else:
                latent = np.zeros([env.time_slots, env.n_tasks_max, A], dtype=np.float32)
                prior_arg = None
            d, e, _ = run_episode(env, agent, tasks, E, f_E, tran_rate,
                                  np.asarray(omega, np.float32), latent, stochastic=False,
                                  prior_latent=prior_arg, use_true_feedback=use_true_feedback)
            d_all.append(d); e_all.append(e)
        pts.append([float(np.mean(d_all)), float(np.mean(e_all))])
    pts = np.array(pts)
    return pts, float(hypervolume_2d(pts, ts['ref']))


def eval_greedy_on_testset(ts, temperature=1.0, hard=False, k_eval=None,
                           aT=1.0, aE=1.0, ds=0.05, es=0.25, eval_seed=EVAL_SEED):
    """在固定卷子上评估纯贪心 (免训练, oracle 精确物理) → (21点, HV)。"""
    from greedy_warmstart import greedy_omega_prior
    rng = np.random.default_rng(eval_seed)
    env = MOFDEnvironment(**ts['env_params'])
    pts = []
    for oi, omega in enumerate(ts['prefs']):
        scen = ts['scenarios'][oi] if k_eval is None else ts['scenarios'][oi][:k_eval]
        d_all, e_all = [], []
        for (E, f_E, tran_rate, tasks) in scen:
            env.reset_env(tasks, E, f_E, tran_rate, omega)
            sd = se = n = 0.0
            for t in range(env.time_slots - 1):
                for nn in range(len(env.tasks_bit[t])):
                    p = greedy_omega_prior(env, t, nn, omega, aT, aE, ds, es, temperature, hard)
                    a = int(np.argmax(p)) if hard else int(rng.choice(env.Emax, p=p))
                    _, _, d, e, _ = env.step(t, nn, a)
                    sd += d; se += e; n += 1
                env.update_proc_queues(t)
            d_all.append(sd / max(n, 1)); e_all.append(se / max(n, 1))
        pts.append([float(np.mean(d_all)), float(np.mean(e_all))])
    pts = np.array(pts)
    return pts, float(hypervolume_2d(pts, ts['ref']))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=40)
    ap.add_argument('--proveK', type=int, default=8, help='确定性自证用的小 K')
    args = ap.parse_args()

    ts = build_testset(K=args.K)

    # ---- 确定性自证: 拿一个已训好的模型, 在小 K 切片上考两遍, 看 HV 是否完全一致 ----
    import glob
    from mofd_v8 import MOFD_SAC_V8
    ek = sorted(glob.glob('results/oacr_early_2*/ckpt_seed0'))
    if ek:
        ck = ek[-1]
        env = MOFDEnvironment(**ENV_PARAMS)
        ag = MOFD_SAC_V8(state_dim=env.state_dim, Emax=6, hidden_dim=128,
                         denoising_steps=3, alpha_T=1.0, alpha_E=1.0, device=torch.device('cpu'))
        ag.actor.load_state_dict(torch.load(f'{ck}/actor.pt', map_location='cpu'))
        ag.critic1.load_state_dict(torch.load(f'{ck}/critic1.pt', map_location='cpu'))
        ag.critic2.load_state_dict(torch.load(f'{ck}/critic2.pt', map_location='cpu'))
        ag.actor.eval()
        ts_small = dict(ts); ts_small['scenarios'] = [s[:args.proveK] for s in ts['scenarios']]
        print(f'\n确定性自证: 同一模型在固定卷子(K={args.proveK})上考两遍 ...')
        _, hv1 = eval_agent_on_testset(ag, ts_small)
        _, hv2 = eval_agent_on_testset(ag, ts_small)
        print(f'  第一遍 HV = {hv1:.6f}')
        print(f'  第二遍 HV = {hv2:.6f}')
        print(f'  差 = {abs(hv1-hv2):.2e}  →  {"完全一致, 秤校准成功 (对比随机评估那 4.5 倍跳)" if abs(hv1-hv2)<1e-6 else "仍有差, 需检查随机源"}')
    else:
        print('[proof] 未找到 oacr_early ckpt, 跳过确定性自证')
