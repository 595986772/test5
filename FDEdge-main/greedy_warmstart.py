"""
PGW 地基: 偏好条件化贪心物理先验 + 纯贪心 Pareto 评估 (免训练)
==============================================================
greedy_omega_prior(env,t,n,ω,...): 逐决策按当前 ω 现算每台有效服务器的即时代价,
  返回温度 softmax 先验 [Emax] (hard=True 则 one-hot)。代价完全复刻 env reward 物理。

eval_greedy_pareto(...): 不用任何训练好的网络, 纯按贪心先验采动作, 扫 21 个 ω 出 Pareto 点。
  用来回答最便宜的第一问: "贪心一个人能打多少分?" —— 决定 PGW 还有多少发挥空间。

⚠️ 诚实: 贪心用了 tran_rate、proc_queue_bef 等策略状态向量里没有的量 (编排器侧可知,
   部署不算作弊; 但科学归因要靠 state-augmented 对照, 见 PGW 方案 §4.4)。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

from mofd_environment import MOFDEnvironment
from helpers import build_preference_set, hypervolume_2d, SHARED_EVAL_SEED_OFFSET
import mofd_main
from mofd_main import sample_tasks


def server_costs(env, t, n, omega, aT, aE, ds, es):
    """每台有效服务器的即时 (delay, energy, ω-标量代价); 复刻 env.step 物理, 不改状态。"""
    A = env.action_dim
    d_n = float(env.tasks_bit[t][n]); rho = float(env.comp_density[n] * d_n)
    cost = np.full(A, np.inf, dtype=np.float64)
    delay = np.full(A, np.inf); energy = np.full(A, np.inf)
    for a in range(A):
        if env.valid_mask[a] <= 0.5:
            continue
        f_b = float(env.f_E[a])
        if f_b <= 0:
            continue
        v = env.effective_rate(t, n, a)
        td = d_n / max(v, 1e-6); cd = rho / f_b
        wd = (env.proc_queue_len[t, a] + env.proc_queue_bef[t, a]) / f_b
        delay[a] = td + cd + wd
        energy[a] = env.p_off * td + env.kappa * (f_b ** 2) * rho
        cost[a] = float(omega[0]) * aT * (delay[a] * ds) + float(omega[1]) * aE * (energy[a] * es)
    return delay, energy, cost


def greedy_omega_prior(env, t, n, omega, aT, aE, ds, es, temperature=1.0, hard=False):
    """逐决策贪心物理先验 [Emax]: softmax(-cost/τ) over valid; hard=True 则 one-hot(argmin cost)。"""
    A = env.action_dim
    _, _, cost = server_costs(env, t, n, omega, aT, aE, ds, es)
    valid = np.isfinite(cost)
    prior = np.zeros(A, dtype=np.float32)
    if not valid.any():
        prior[:] = 1.0 / A
        return prior
    if hard:
        prior[int(np.argmin(np.where(valid, cost, np.inf)))] = 1.0
        return prior
    z = np.where(valid, -cost / max(temperature, 1e-6), -np.inf)
    z -= z[valid].max()                         # 数值稳定
    e = np.where(valid, np.exp(z), 0.0)
    s = e.sum()
    return (e / s).astype(np.float32) if s > 1e-12 else np.where(valid, 1.0/valid.sum(), 0.0).astype(np.float32)


def eval_greedy_pareto(env, n_pref=21, n_epi=3, temperature=1.0, hard=False,
                       aT=1.0, aE=1.0, ds=0.05, es=0.25, seed=None):
    """纯贪心 rollout 扫 ω 出 Pareto 点 (协议对齐 evaluate_pareto: 每个 ω 重置 rng)。"""
    if seed is None:
        seed = SHARED_EVAL_SEED_OFFSET                 # 与训练模型 final eval 同环境, HV 可比
    prefs = build_preference_set(n_pref)
    pts = []
    for omega in prefs:
        rng = np.random.default_rng(seed)
        ds_list, es_list = [], []
        for _ in range(n_epi):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            env.reset_env(tasks, E, f_E, tran_rate, omega)
            sd = se = nt = 0.0
            for t in range(env.time_slots - 1):
                for nn in range(len(env.tasks_bit[t])):
                    p = greedy_omega_prior(env, t, nn, omega, aT, aE, ds, es, temperature, hard)
                    action = int(np.argmax(p)) if hard else int(rng.choice(env.Emax, p=p))
                    _, _, d, e, _ = env.step(t, nn, action)
                    sd += d; se += e; nt += 1
                env.update_proc_queues(t)
            ds_list.append(sd / max(nt, 1)); es_list.append(se / max(nt, 1))
        pts.append([float(np.mean(ds_list)), float(np.mean(es_list))])
    return np.array(pts)


if __name__ == '__main__':
    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())
    # 全尺度 env (对齐之前消融 config.json)
    env = MOFDEnvironment(Emax=6, num_tasks_max=50, bit_range=(10, 40),
                          time_slots=100, f_range=(10, 40),
                          delay_scale=0.05, energy_scale=0.25, seed=0)
    print('评估纯贪心 (免训练) 在 21 个 ω 上的 Pareto / HV ...')
    rows = []
    for name, hard, temp in [('hard greedy', True, None), ('soft greedy τ=0.5', False, 0.5),
                             ('soft greedy τ=1.0', False, 1.0), ('soft greedy τ=2.0', False, 2.0)]:
        pts = eval_greedy_pareto(env, n_pref=21, n_epi=3, temperature=(temp or 1.0), hard=hard)
        rows.append((name, pts))
    allp = np.vstack([p for _, p in rows])
    ref = (float(allp[:, 0].max() * 1.1 + 1e-6), float(allp[:, 1].max() * 1.1 + 1e-6))
    print(f'\n共享 ref = {ref}')
    print(f'{"贪心变体":<20}{"HV":>10}{"延迟下界":>10}{"能耗下界":>10}{"延迟上界":>10}')
    out = []
    for name, pts in rows:
        hv = hypervolume_2d(pts, ref)
        line = f'{name:<20}{hv:>10.3f}{pts[:,0].min():>10.2f}{pts[:,1].min():>10.3f}{pts[:,0].max():>10.2f}'
        print(line); out.append(line)
    txt = f'共享 ref = {ref}\n' + '\n'.join(out)
    with open('results/pgw_greedy_probe.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    np.savetxt('results/pgw_greedy_hard_pts.csv', rows[0][1], fmt='%.5f')
    print('\n[saved] results/pgw_greedy_probe.txt, pgw_greedy_hard_pts.csv')
    print('\n参考: 之前 V8/V5 训练模型在同尺度 final eval 的 HV 约 240~252。')
    print('若贪心已接近/超过该区间 → diffusion 发挥空间小, PGW 要赢得靠"修正拥塞";')
    print('若贪心明显低 → 说明 diffusion 本来就在贪心之上加了值, PGW 起点更好可能放大优势。')
