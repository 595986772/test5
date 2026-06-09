"""
内容敏感性探针: prior 的"内容"能不能传过 3 步去噪, 影响最终动作?
================================================================
背景: Pareto-memory 的卖点是"挑出一个高质量动作当 prior 起点"。但机制诊断显示起点
      主要只传"方差→尖锐度→延迟", 内容(具体哪个动作)可能在去噪里被冲掉。若内容传不
      下去, 则任何 Pareto 筛选都白做。本探针只验这一个前提, 不建 archive、不训练。

做法 (纯推理, 跑现成 ckpt):
  每个决策先算出"当前 state 下每台服务器的即时 ω-代价", 得到 best/worst 服务器, 构造:
    good prior = one-hot(best)    bad prior = one-hot(worst)    uniform = 均匀
  good 与 bad **尖锐度一样、内容相反** → 隔离"内容"与"尖锐度"。
  分两条注入通道各测一遍:
    start 通道: 把 prior 当**去噪起点** (conditioning 固定 uniform)  ← 用户设想的设计
    cond  通道: 把 prior 当**conditioning** (起点固定 zeros)         ← 另一条可能更强的通道
  比较: 喂 good 时动作是否更常落在 best 上 (follow 率), 以及实得即时代价 good vs bad。

判据:
  follow_good 明显高于 uniform 基线(uni_best) 且 cost(good) < cost(bad)
    → 内容能传下去, 该通道值得做 Pareto 筛选;
  follow_good ≈ uni_best 且 cost 不分高低
    → 内容传不下去, 筛选无意义, archive 别建。

用法 (FDEdge-main/ 目录):
  python prior_content_probe.py                 # 自动找 feedback+random ckpt
  python prior_content_probe.py --n-pref 11 --n-epi 1
  python prior_content_probe.py --ckpt results/abl_mcss_src_feedback_xxx/ckpt_seed0
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')   # Windows GBK 控制台无法编码 −/Δ 等, 强制 utf-8
except Exception:
    pass
import argparse
import numpy as np
import torch

import mofd_main
from mofd_environment import MOFDEnvironment
from mofd_main import load_agent_from_ckpt
from mechanism_diag import discover_ckpts, load_cfg_for_ckpt, build_scenarios

DEVICE = torch.device('cpu')


def server_costs(env, t, n, omega, aT, aE, ds, es):
    """当前 (t,n) 下每台有效服务器的即时 (delay, energy, ω-标量代价). 复刻 env.step 数学, 不改状态."""
    A = env.action_dim
    d_n = float(env.tasks_bit[t][n]); rho = float(env.comp_density[n] * d_n)
    delay = np.full(A, np.inf, dtype=np.float64)
    energy = np.full(A, np.inf, dtype=np.float64)
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
    finite = np.isfinite(delay) & np.isfinite(energy)
    cost = np.full(A, np.inf, dtype=np.float64)
    with np.errstate(invalid='ignore', over='ignore'):
        c = float(omega[0]) * aT * (delay * ds) + float(omega[1]) * aE * (energy * es)
    cost[finite] = c[finite]
    return delay, energy, cost


@torch.no_grad()
def act(agent, state, start_vec, cond_vec, mask):
    """单候选确定性决策: start_vec=去噪起点, cond_vec=conditioning. 返回 argmax 动作."""
    s_t = torch.tensor(state[None], dtype=torch.float, device=DEVICE)
    x_t = torch.tensor(start_vec[None], dtype=torch.float, device=DEVICE)
    p_t = torch.tensor(cond_vec[None], dtype=torch.float, device=DEVICE)
    probs = agent.actor(s_t, x_t, prior=p_t)[0]
    probs = probs * torch.tensor(mask, dtype=torch.float, device=DEVICE)
    probs = probs / (probs.sum() + 1e-8)
    p = probs.cpu().numpy()
    s = p.sum()
    if s < 1e-8:
        valid = np.where(mask > 0.5)[0]
        return int(valid[0]) if len(valid) else 0
    return int(np.argmax(p / s))


def probe_channel(agent, env, scen, channel, aT, aE, ds, es):
    """channel='start': prior 当起点(cond=uniform); 'cond': prior 当conditioning(起点=zeros)."""
    A = env.action_dim
    uni = np.full(A, 1.0 / A, dtype=np.float32)
    zeros = np.zeros(A, dtype=np.float32)
    rec = dict(fg=[], fb=[], ub=[], uw=[], agree=[], cg=[], cb=[], cu=[])
    for omega, epis in scen:
        for E, f_E, tran_rate, tasks in epis:
            env.reset_env(tasks, E, f_E, tran_rate, omega)
            for t in range(env.time_slots - 1):
                for n in range(len(env.tasks_bit[t])):
                    state = env.get_state(t, n)
                    mask = env.get_valid_mask()
                    delay, energy, cost = server_costs(env, t, n, omega, aT, aE, ds, es)
                    valid = np.where(np.isfinite(cost))[0]
                    if len(valid) == 0:
                        continue
                    best = int(valid[np.argmin(cost[valid])])
                    worst = int(valid[np.argmax(cost[valid])])
                    good = np.zeros(A, dtype=np.float32); good[best] = 1.0
                    bad = np.zeros(A, dtype=np.float32); bad[worst] = 1.0

                    def run(pvec):
                        if channel == 'start':
                            return act(agent, state, pvec, uni, mask)
                        return act(agent, state, zeros, pvec, mask)

                    a_g, a_b, a_u = run(good), run(bad), run(uni)
                    rec['fg'].append(a_g == best)          # good prior → 落在 best?
                    rec['fb'].append(a_b == worst)         # bad prior  → 落在 worst?
                    rec['ub'].append(a_u == best)          # 基线: uniform 自己落 best 的概率
                    rec['uw'].append(a_u == worst)
                    rec['agree'].append(a_g == a_b)        # good 与 bad 选同一台?
                    rec['cg'].append(cost[a_g]); rec['cb'].append(cost[a_b]); rec['cu'].append(cost[a_u])
                    # 推进轨迹: 用中性的 uniform 动作 (good/bad 只是探针, 不改轨迹方向)
                    env.step(t, n, a_u)
                env.update_proc_queues(t)
    return {k: float(np.mean(v)) for k, v in rec.items()}


def run_ckpt(ckpt_dir, n_pref, n_epi):
    print(f'\n{"="*72}\nckpt: {ckpt_dir}\n{"="*72}')
    cfg = load_cfg_for_ckpt(ckpt_dir)
    aT = float(cfg.get('alpha_T', 1.0)); aE = float(cfg.get('alpha_E', 1.0))
    ds = float(cfg.get('delay_scale', 0.05)); es = float(cfg.get('energy_scale', 0.25))
    env = MOFDEnvironment(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'], delay_scale=ds, energy_scale=es, seed=0)
    agent, _ = load_agent_from_ckpt(ckpt_dir, cfg, env, DEVICE)
    agent.actor.eval()
    scen = build_scenarios(env, n_pref, n_epi)

    lines = [f'内容敏感性 @ {os.path.basename(os.path.dirname(ckpt_dir))}  '
             f'(Emax={cfg["Emax"]}, denoise_steps={cfg["denoising_steps"]})',
             '通道含义: start=prior当去噪起点(用户设想) | cond=prior当conditioning',
             '',
             f'{"通道":<8}{"follow_good":>12}{"uni_best(基线)":>15}{"delta":>8}'
             f'{"follow_bad":>11}{"good=bad率":>11}{"cost_good":>11}{"cost_uni":>10}{"cost_bad":>10}']
    verdicts = []
    for channel in ('start', 'cond'):
        r = probe_channel(agent, env, scen, channel, aT, aE, ds, es)
        delta = r['fg'] - r['ub']
        lines.append(f'{channel:<8}{r["fg"]:>12.3f}{r["ub"]:>15.3f}{delta:>+8.3f}'
                     f'{r["fb"]:>11.3f}{r["agree"]:>11.3f}{r["cg"]:>11.4f}{r["cu"]:>10.4f}{r["cb"]:>10.4f}')
        # 判据: good prior 显著抬高"落 best"率 且 cost(good)<cost(bad)
        passed = (delta > 0.10) and (r['cg'] < r['cb'] - 1e-9)
        verdicts.append((channel, passed, delta, r))

    lines += ['', '判据 (内容能否传过去噪):']
    for channel, passed, delta, r in verdicts:
        tag = 'PASS 内容能传' if passed else 'FAIL 内容传不下去'
        lines.append(f'  [{channel:<5}] follow_good-uni_best={delta:+.3f} (>0.10?), '
                     f'cost_good({r["cg"]:.3f}) < cost_bad({r["cb"]:.3f})?  -> {tag}')
    any_pass = any(p for _, p, _, _ in verdicts)
    lines += ['',
              ('读法: 若某通道 PASS → 该通道能携带 prior 内容, 那个 Pareto 筛选值得做'
               ' (start PASS=你的设计可行; 只有 cond PASS=应把 prior 当 conditioning 而非起点);'),
              '      若两通道都 FAIL → prior 内容传不过去噪, Pareto 筛选无意义, archive 别建。',
              f'本 ckpt 结论: {"至少一条通道能传内容, 值得继续" if any_pass else "两通道都传不动内容, 这条路堵死"}']

    txt = '\n'.join(lines)
    out = os.path.join(os.path.dirname(ckpt_dir), 'prior_content_probe.txt')
    with open(out, 'w', encoding='utf-8') as f:      # 先落盘, 再 print, 防控制台编码崩了丢结果
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\n[saved] {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--n-pref', type=int, default=11)
    ap.add_argument('--n-epi', type=int, default=1)
    args = ap.parse_args()
    ckpts = [args.ckpt] if args.ckpt else discover_ckpts()
    if not ckpts:
        print('[err] 未找到 ckpt. 先跑过源消融, 或用 --ckpt 指定.')
        return
    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())
    for ck in ckpts:
        run_ckpt(ck, args.n_pref, args.n_epi)


if __name__ == '__main__':
    main()
