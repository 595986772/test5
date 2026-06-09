"""
机制诊断: "扩散起点的方差/几何 → 选中服务器频率 f_b → 延迟/能耗 corner"
=====================================================================
为什么写这个 (诚实交代, 修正了之前的口头错误):
  评估 (evaluate_pareto) 把 latent 传成 **全零** (mofd_main.py:519), FeedbackDiffusion
  从该起点去噪 (mofd_v5.py:59 `x = latent.clone()`). 所以三个"源"在评估时是三种
  **去噪起点几何**, 不是时序反馈:
      feedback = 零向量(σ≈0)   prior = 均匀向量   random = 高斯噪声(σ=1)
  而去噪器是按 "加噪的真实动作" 训练的 (q_sample=√ᾱx₀+√(1-ᾱ)ε), 高斯起点在分布内,
  零/均匀起点离流形 → 这是源会分化的**代码层真实**原因。本脚本只在已训好的 ckpt 上
  做**推理**, 不重训, 来验证/证伪这个机制。

测什么 (全部纯推理, 几分钟跑完):
  T3 (承重墙): 每个起点 regime 实际选中服务器的均值 f_b。
       预测: random(高方差) 的 f_b 高于 feedback(零)/prior(均匀)。
  剂量响应:   把起点方差 σ ∈ {0, 0.5, 1, 2} 连续扫, 看 f_b 是否随 σ 单调上升,
       且 delay 随之下降、energy 随之上升 (κf²ρ 的物理). 这是比"二元安慰剂"更强的证据。
  物理链:     pool 所有决策, corr(f_b, energy)>0 且 corr(f_b, delay)<0?
  安慰剂:     feedback(零) 与 prior(均匀) 是否聚在一起、而 random 是离群点
       → 起作用的是"随机性/方差", 不是某个具体结构。

能/不能证明 (写论文照此口径):
  * 能: 在固定已训网络上, 起点方差是否**因果**驱动 f_b/corner (推理级证据)。
  * 不能: (a) 这是**单 seed 的网络**, 仍需多 seed; (b) 最干净的"去噪步数"因果对照
          需要在不同 denoising_steps 下**重训** (本脚本 --step-probe 只能向下扫 1~3,
          升到 6/10 会越界, 必须重训), 推理级步数探针有 off-distribution 偏差。

用法 (FDEdge-main/ 目录):
  python mechanism_diag.py                      # 自动找 feedback+random 两个源 ckpt
  python mechanism_diag.py --n-pref 21 --n-epi 2
  python mechanism_diag.py --step-probe         # 额外跑去噪步数(1~3)探针
  python mechanism_diag.py --ckpt results/abl_mcss_src_feedback_xxx/ckpt_seed0
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import glob
import json
import numpy as np
import torch

import mofd_main
from mofd_environment import MOFDEnvironment
from mofd_main import load_agent_from_ckpt, sample_tasks
from helpers import build_preference_set

DEVICE = torch.device('cpu')

# 起点 regime: (名字, kind, sigma)  kind ∈ {'zeros','uniform','gauss'}
REGIMES = [
    ('feedback(σ0)', 'zeros',   0.0),
    ('prior(unif)',  'uniform', 0.0),
    ('gauss σ0.5',   'gauss',   0.5),
    ('random(σ1)',   'gauss',   1.0),
    ('gauss σ2.0',   'gauss',   2.0),
]


def make_init(kind, sigma, action_dim, rng):
    """构造一次决策的去噪起点向量 [action_dim]."""
    if kind == 'zeros':
        return np.zeros(action_dim, dtype=np.float32)
    if kind == 'uniform':
        return np.full(action_dim, 1.0 / action_dim, dtype=np.float32)
    return (sigma * rng.standard_normal(action_dim)).astype(np.float32)


def discover_ckpts():
    out = []
    for src in ('feedback', 'random'):
        cands = sorted(glob.glob(f'results/abl_mcss_src_{src}_*/ckpt_seed0'))
        if cands:
            out.append(cands[-1])
    return out


def load_cfg_for_ckpt(ckpt_dir):
    """ckpt_dir 的父目录里有 config.json (run_single_seed 落盘的全量 cfg)."""
    cfg_path = os.path.join(os.path.dirname(ckpt_dir), 'config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['bit_range'] = tuple(cfg['bit_range'])
    cfg['f_range'] = tuple(cfg['f_range'])
    return cfg


def build_scenarios(env, n_pref, n_epi, seed=20260605):
    """对每个 ω 预生成 n_epi 个固定 (E,f_E,tran,tasks), 所有 regime 共享 → 环境完全一致."""
    prefs = build_preference_set(n_pref)
    rng = np.random.default_rng(seed)
    scen = []
    for omega in prefs:
        epis = []
        for _ in range(n_epi):
            E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            epis.append((E, f_E, tran_rate, tasks))
        scen.append((np.asarray(omega, dtype=np.float32), epis))
    return scen


@torch.no_grad()
def rollout(agent, env, omega, scenario, kind, sigma, init_rng):
    """确定性 (argmax) 跑一个 episode, 直接调 actor (绕过 critic 选优), 逐决策记录."""
    E, f_E, tran_rate, tasks = scenario
    env.reset_env(tasks, E, f_E, tran_rate, omega)
    A = env.action_dim
    prior = np.full(A, 1.0 / A, dtype=np.float32)             # eval prior = 均匀 (同 evaluate_pareto)
    prior_t = torch.tensor(prior[None], dtype=torch.float, device=DEVICE)
    fb_list, d_list, e_list, ent_list, maxp_list = [], [], [], [], []
    for t in range(env.time_slots - 1):
        for n in range(len(env.tasks_bit[t])):
            state = env.get_state(t, n)
            mask = env.get_valid_mask()
            init = make_init(kind, sigma, A, init_rng)
            s_t = torch.tensor(state[None], dtype=torch.float, device=DEVICE)
            x_t = torch.tensor(init[None], dtype=torch.float, device=DEVICE)
            probs = agent.actor(s_t, x_t, prior=prior_t)[0]
            probs = probs * torch.tensor(mask, dtype=torch.float, device=DEVICE)
            probs = probs / (probs.sum() + 1e-8)
            p = probs.cpu().numpy()
            s = p.sum()
            if s < 1e-8:
                valid = np.where(mask > 0.5)[0]
                action = int(valid[0]) if len(valid) else 0
            else:
                p = p / s
                action = int(np.argmax(p))
            _, _, delay, energy, real_action = env.step(t, n, action)
            fb_list.append(float(env.f_E[real_action]))
            d_list.append(float(delay)); e_list.append(float(energy))
            pc = np.clip(p, 1e-8, 1.0)
            ent_list.append(float(-np.sum(pc * np.log(pc))))
            maxp_list.append(float(p.max()))
        env.update_proc_queues(t)
    return (np.array(fb_list), np.array(d_list), np.array(e_list),
            np.array(ent_list), np.array(maxp_list))


def run_ckpt(ckpt_dir, n_pref, n_epi, step_probe):
    print(f'\n{"="*70}\nckpt: {ckpt_dir}\n{"="*70}')
    cfg = load_cfg_for_ckpt(ckpt_dir)
    env = MOFDEnvironment(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'],
        delay_scale=cfg.get('delay_scale', 0.05),
        energy_scale=cfg.get('energy_scale', 0.25), seed=0)
    agent, _ = load_agent_from_ckpt(ckpt_dir, cfg, env, DEVICE)
    agent.actor.eval()

    scen = build_scenarios(env, n_pref, n_epi)
    init_rng = np.random.default_rng(777)

    # 每 regime: 汇总全决策 + 按 ω 区间(延迟优先 ω_E<0.5 / 能耗优先 ω_E≥0.5)分桶
    agg = {}
    pool_fb, pool_d, pool_e, pool_ent, pool_mp = [], [], [], [], []
    for name, kind, sigma in REGIMES:
        fb_all, d_all, e_all, ent_all, mp_all = [], [], [], [], []
        fb_lo, fb_hi = [], []                                  # lo=延迟优先, hi=能耗优先
        for omega, epis in scen:
            for sc in epis:
                fb, d, e, ent, mp = rollout(agent, env, omega, sc, kind, sigma, init_rng)
                fb_all.append(fb); d_all.append(d); e_all.append(e)
                ent_all.append(ent); mp_all.append(mp)
                (fb_hi if float(omega[1]) >= 0.5 else fb_lo).append(fb)
        fb_all = np.concatenate(fb_all); d_all = np.concatenate(d_all)
        e_all = np.concatenate(e_all)
        agg[name] = dict(
            fb=fb_all.mean(), delay=d_all.mean(), energy=e_all.mean(),
            ent=np.concatenate(ent_all).mean(), maxp=np.concatenate(mp_all).mean(),
            fb_lo=np.concatenate(fb_lo).mean() if fb_lo else float('nan'),
            fb_hi=np.concatenate(fb_hi).mean() if fb_hi else float('nan'),
        )
        pool_fb.append(fb_all); pool_d.append(d_all); pool_e.append(e_all)
        pool_ent.append(np.concatenate(ent_all)); pool_mp.append(np.concatenate(mp_all))

    # ---- 打印主表 ----
    lines = [f'机制诊断 @ {os.path.basename(os.path.dirname(ckpt_dir))}  '
             f'(f_max={cfg["f_range"][1]}, denoise_steps={cfg["denoising_steps"]})',
             f'{"regime":<14}{"mean f_b":>10}{"f_b|延迟优先":>14}{"f_b|能耗优先":>14}'
             f'{"delay":>9}{"energy":>9}{"entropy":>9}{"maxp":>7}']
    for name, _, _ in REGIMES:
        a = agg[name]
        lines.append(f'{name:<14}{a["fb"]:>10.3f}{a["fb_lo"]:>14.3f}{a["fb_hi"]:>14.3f}'
                     f'{a["delay"]:>9.3f}{a["energy"]:>9.3f}{a["ent"]:>9.3f}{a["maxp"]:>7.3f}')

    # ---- 判据 ----
    fb_z = agg['feedback(σ0)']['fb']; fb_u = agg['prior(unif)']['fb']
    fb_r = agg['random(σ1)']['fb']
    dose = [agg[n]['fb'] for n in ['feedback(σ0)', 'gauss σ0.5', 'random(σ1)', 'gauss σ2.0']]
    pool_fb = np.concatenate(pool_fb); pool_d = np.concatenate(pool_d); pool_e = np.concatenate(pool_e)
    pool_ent = np.concatenate(pool_ent); pool_mp = np.concatenate(pool_mp)
    cor_e = float(np.corrcoef(pool_fb, pool_e)[0, 1])
    cor_d = float(np.corrcoef(pool_fb, pool_d)[0, 1])
    cor_ent_d = float(np.corrcoef(pool_ent, pool_d)[0, 1])   # 策略越散→延迟越高?
    cor_mp_d = float(np.corrcoef(pool_mp, pool_d)[0, 1])     # 策略越尖→延迟越低?
    # 真实涌现机制: 起点方差→策略尖锐度(entropy)→延迟
    ent_dose = [agg[n]['ent'] for n in ['feedback(σ0)', 'gauss σ0.5', 'random(σ1)', 'gauss σ2.0']]
    d_dose = [agg[n]['delay'] for n in ['feedback(σ0)', 'gauss σ0.5', 'random(σ1)', 'gauss σ2.0']]
    sharp_mono = all(ent_dose[i + 1] <= ent_dose[i] + 1e-3 for i in range(len(ent_dose) - 1))
    delay_mono = all(d_dose[i + 1] <= d_dose[i] + 1e-3 for i in range(len(d_dose) - 1))
    mono = all(dose[i + 1] >= dose[i] - 1e-3 for i in range(len(dose) - 1))
    t3 = (fb_r - fb_z) / (abs(fb_z) + 1e-6)
    placebo_ok = abs(fb_z - fb_u) < 0.5 * abs(fb_z - fb_r) + 1e-9

    def yn(b):
        return 'PASS' if b else 'FAIL'

    lines += ['',
              '判据 (启发式阈值, 仅作快速读数):',
              f'  T3 选频: random f_b 比 feedback 高 {t3*100:+.1f}%   '
              f'[{yn(t3 > 0.03)}: 期望 >+3%]',
              f'  剂量响应: f_b 随起点方差 σ 单调上升  [{yn(mono)}]  '
              f'σ序列 f_b={[round(x,3) for x in dose]}',
              f'  物理链: corr(f_b,energy)={cor_e:+.3f} (>0?)  '
              f'corr(f_b,delay)={cor_d:+.3f} (<0?)  '
              f'[{yn(cor_e > 0.1 and cor_d < -0.1)}]',
              f'  安慰剂: feedback(零)与prior(均匀)聚类、random离群  '
              f'[{yn(placebo_ok)}]  (|Δzu|={abs(fb_z-fb_u):.3f} vs |Δzr|={abs(fb_z-fb_r):.3f})',
              '',
              '读法: 若 T3+剂量+物理链均 PASS → "起点方差驱动选频→corner" 机制在此网络成立,',
              '      值得上多 seed + 重训去噪步数对照; 若 T3 或物理链 FAIL → 该机制证伪, 埋掉。',
              '',
              '--- 涌现机制 (smoke 中浮现, 这里复核): 起点方差 → 策略尖锐度 → 延迟 ---',
              f'  尖锐度剂量: entropy 随 σ 单调下降  [{yn(sharp_mono)}]  '
              f'entropy={[round(x,3) for x in ent_dose]}',
              f'  延迟剂量:   delay 随 σ 单调下降    [{yn(delay_mono)}]  '
              f'delay={[round(x,2) for x in d_dose]}',
              f'  关联: corr(entropy,delay)={cor_ent_d:+.3f} (>0?)  '
              f'corr(maxp,delay)={cor_mp_d:+.3f} (<0?)  '
              f'[{yn(cor_ent_d > 0.1 and cor_mp_d < -0.1)}]',
              '  → 若这组 PASS 而上面 f_b 组 FAIL: 真正的杠杆是"起点方差控制策略尖锐度,',
              '    尖锐策略压低延迟", 与服务器频率 f_b 无关; energy 基本不被起点撬动',
              '    (energy 更像训练效应而非推理起点效应)。']

    if step_probe:
        lines += ['', '--- 去噪步数探针 (仅 1~3, 升到 6/10 需重训; off-distribution 注意) ---']
        omega_mid = np.array([0.5, 0.5], dtype=np.float32)
        sc0 = scen[len(scen) // 2][1][0]
        orig = agent.actor.n_timesteps
        for k in (1, 2, 3):
            agent.actor.n_timesteps = k
            fz, *_ = rollout(agent, env, omega_mid, sc0, 'zeros', 0.0, np.random.default_rng(1))
            fr, *_ = rollout(agent, env, omega_mid, sc0, 'gauss', 2.0, np.random.default_rng(1))
            gap = fr.mean() - fz.mean()
            lines.append(f'  steps={k}: f_b(σ0)={fz.mean():.3f}  f_b(σ2)={fr.mean():.3f}  '
                         f'gap={gap:+.3f}  (期望: gap 随 steps↑ 收窄=起点被洗掉)')
        agent.actor.n_timesteps = orig

    txt = '\n'.join(lines)
    print('\n' + txt)
    out = os.path.join(os.path.dirname(ckpt_dir), 'mechanism_diag.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print(f'\n[saved] {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=None,
                    help='单个 ckpt_seedX 目录; 缺省自动找 feedback+random 两个源')
    ap.add_argument('--n-pref', type=int, default=21)
    ap.add_argument('--n-epi', type=int, default=1)
    ap.add_argument('--step-probe', action='store_true')
    args = ap.parse_args()

    ckpts = [args.ckpt] if args.ckpt else discover_ckpts()
    if not ckpts:
        print('[err] 未找到 ckpt. 先跑过 H-MCSS 源消融, 或用 --ckpt 指定.')
        return
    mofd_main.set_task_generator(mofd_main.RandomTaskGenerator())  # 与消融一致
    for ck in ckpts:
        run_ckpt(ck, args.n_pref, args.n_epi, args.step_probe)


if __name__ == '__main__':
    main()
