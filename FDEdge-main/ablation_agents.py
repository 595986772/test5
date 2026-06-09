"""
消融实验用 Agent 变体 + 统一 runner
===================================
两个消融, 都通过 **monkey-patch** 注入, 不改动 mofd_v5.py / mofd_main.py:

  1) w/o 反馈扩散  (MOFD_SAC_V5_NoDiffusion)
       扩散 actor → 普通 MLP 策略 (忽略 latent/prior, 无去噪).
       其余组件 (vector-Q / ω-weighted critic loss / COR / PopArt / SAC-α)
       完全保留, 只隔离 "反馈扩散主干" 这一个变量.

  2) H-MCSS 候选源 / 候选数  (MOFD_SAC_V5_HMCSS)
       覆写 take_action, 由类属性 MCSS_MODE / MCSS_N 控制候选构造.
       注意: 原 mofd_v5.take_action 硬编码三源 cat([feedback, prior, random]),
       并不读 n_candidates(=死参数), 所以候选数扫必须靠覆写, 改 cfg 无效.

注入方式 (见 run_variant):
    mofd_main.MOFD_SAC_V5 = <变体类>      # run_single_seed 在调用时按全局名解析
    mofd_main.main(cfg_override={...})    # 跑完写 results/<prefix>_pareto_aggregated.csv
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mofd_main
from mofd_v5 import MOFD_SAC_V5, OMEGA_STATE_SLICE
from mofd_v8 import MOFD_SAC_V8
from helpers import hypervolume_2d


# ==============================================================
# 变体 1: 普通 MLP actor (无反馈扩散)
# ==============================================================
class PlainMLPActor(nn.Module):
    """与 FeedbackDiffusion 同接口的普通策略: state -> softmax(probs over Emax).

    刻意忽略 latent_action_probs / prior —— 即去掉 "反馈起点" 与 "多步去噪",
    只留一个直接 state→动作分布的 MLP. 这正是 "w/o 反馈扩散" 要隔离的变量.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, latent_action_probs=None, prior=None, *args, **kwargs):
        return F.softmax(self.net(state), dim=-1)


class MOFD_SAC_V5_NoDiffusion(MOFD_SAC_V8):
    """构造后把扩散 actor 替换成 PlainMLPActor, 重建 actor 优化器.

    ⚠️ 基类已改为 V8 (干净版 critic: 目标去熵 + 等权), 名字保留 V5_ 仅为兼容 import.
    take_action 继承父类的三源逻辑即可: 普通 MLP 忽略 cand, 三个候选输出相同,
    选优自然退化为 idx0, 不会报错, 语义即 "无候选差异". critic/COR/PopArt 不变.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.actor.state_dim                 # FeedbackDiffusion 存了 state_dim
        hidden_dim = int(kwargs.get('hidden_dim', 128))
        actor_lr = float(kwargs.get('actor_lr', 1e-4))
        self.actor = PlainMLPActor(state_dim, self.Emax, hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)


# ==============================================================
# 变体 2: H-MCSS 候选源 / 候选数消融
# ==============================================================
class MOFD_SAC_V5_HMCSS(MOFD_SAC_V8):
    """覆写 take_action, 由类属性控制候选集 (runner 在每个变体前设置).

    ⚠️ 基类已改为 V8 (干净版 critic: 目标去熵 + 等权), 名字保留 V5_ 仅为兼容 import.

    MCSS_MODE:
      'full3'    原方法: cat([feedback, prior, random]) 三源 + Critic 选优
      'feedback' 仅反馈 latent (单候选, 无选优)
      'prior'    仅 prior     (单候选)
      'random'   仅随机起点    (单候选)
      'perturb'  反馈 latent + (MCSS_N-1) 个高斯扰动候选, Critic 选优 (候选数敏感性)
    """

    MCSS_MODE = 'full3'
    MCSS_N = 3

    def take_action(self, state_np, latent_np, prior_np, mask_np, stochastic=True):
        mode = getattr(type(self), 'MCSS_MODE', 'full3')
        n_cand = int(getattr(type(self), 'MCSS_N', 3))

        state = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
        mask = torch.tensor(mask_np[None], dtype=torch.float, device=self.device)
        prior = torch.tensor(prior_np[None], dtype=torch.float, device=self.device)
        latent_arr = np.asarray(latent_np, dtype=np.float32)

        with torch.no_grad():
            if latent_arr.ndim == 1:
                latent = torch.tensor(latent_arr[None], dtype=torch.float, device=self.device)
                rand_init = torch.randn_like(latent)
                if mode == 'feedback':
                    cand = latent
                elif mode == 'prior':
                    cand = prior
                elif mode == 'random':
                    cand = rand_init
                elif mode == 'perturb':
                    if n_cand <= 1:
                        cand = latent
                    else:
                        base = latent.expand(n_cand, -1).clone()
                        noise = torch.randn_like(base) * self.perturb_std
                        noise[0].zero_()                 # 候选 0 = 原始反馈 latent
                        cand = base + noise
                else:  # 'full3'
                    cand = torch.cat([latent, prior, rand_init], dim=0)
                track_pick = (mode == 'full3')
            else:
                cand = torch.tensor(latent_arr, dtype=torch.float, device=self.device)
                track_pick = False

            N = cand.size(0)
            state_rep = state.expand(N, -1)
            mask_rep = mask.expand(N, -1)
            prior_rep = prior.expand(N, -1)

            probs_batch = self.actor(state_rep, cand, prior=prior_rep)
            probs_batch = probs_batch * mask_rep
            probs_batch = probs_batch / (probs_batch.sum(dim=1, keepdim=True) + 1e-8)

            q1_vec = self.critic1(state_rep)
            q2_vec = self.critic2(state_rep)
            min_q_vec = torch.min(q1_vec, q2_vec)
            omega_rep = state_rep[:, OMEGA_STATE_SLICE]
            q_eff = self._scalarize_q(min_q_vec, omega_rep)
            expected_q = torch.sum(probs_batch * q_eff, dim=1)
            best_idx = int(torch.argmax(expected_q).item())
            best_probs = probs_batch[best_idx]
            if track_pick and 0 <= best_idx < 3:
                self._mcss_pick_count[best_idx] += 1

        probs_np = best_probs.detach().cpu().numpy()
        probs_np = np.clip(probs_np, 0.0, 1.0)
        s = probs_np.sum()
        if s < 1e-8:
            valid_idx = np.where(mask_np > 0.5)[0]
            action = int(np.random.choice(valid_idx)) if len(valid_idx) else 0
            probs_np = np.zeros_like(probs_np)
            probs_np[action] = 1.0
            return action, probs_np
        probs_np = probs_np / s
        if stochastic:
            action = int(np.random.choice(self.Emax, p=probs_np))
        else:
            action = int(np.argmax(probs_np))
        return action, probs_np


# ==============================================================
# 变体 3: 偏好分区路由源 (单候选, 无 Critic 选优)
# ==============================================================
class MOFD_SAC_V5_RoutedSource(MOFD_SAC_V5):
    """按偏好分区**确定性**路由扩散起点 (每步只 1 个候选, 不做 Critic 选优):

        能耗优先  ω_E ≥ ROUTE_TAU  →  feedback latent   (占低能耗角)
        延迟优先  ω_E <  ROUTE_TAU  →  random 起点        (占低延迟角)

    动机: H-MCSS 的 full3 让 Critic 每步选源, 实测最差 (219.48 < 单源 feedback 252.08),
    即"每步用不准的 Q 瞎选"是负贡献. 这里把选源信号换成**干净的 ω**: ω 是外生偏好,
    不依赖 Q 估计, 也不在每步抖动. 路由在训练 rollout 与评估时都生效, 故测的是
    "把路由训练进策略后" 的真实表现, 而非 post-hoc 把两条前沿拼起来的 oracle 上界.
    """

    ROUTE_TAU = 0.5

    def take_action(self, state_np, latent_np, prior_np, mask_np, stochastic=True):
        tau = float(getattr(type(self), 'ROUTE_TAU', 0.5))
        omega_E = float(np.asarray(state_np)[OMEGA_STATE_SLICE][1])   # ω_E

        state = torch.tensor(state_np[None], dtype=torch.float, device=self.device)
        mask = torch.tensor(mask_np[None], dtype=torch.float, device=self.device)
        prior = torch.tensor(prior_np[None], dtype=torch.float, device=self.device)
        latent_arr = np.asarray(latent_np, dtype=np.float32)

        with torch.no_grad():
            if latent_arr.ndim == 1:
                latent = torch.tensor(latent_arr[None], dtype=torch.float, device=self.device)
                if omega_E >= tau:
                    cand = latent                       # 能耗优先 → feedback
                else:
                    cand = torch.randn_like(latent)     # 延迟优先 → random
            else:
                cand = torch.tensor(latent_arr, dtype=torch.float, device=self.device)

            probs = self.actor(state, cand, prior=prior)
            probs = probs * mask
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            best_probs = probs[0]

        probs_np = best_probs.detach().cpu().numpy()
        probs_np = np.clip(probs_np, 0.0, 1.0)
        s = probs_np.sum()
        if s < 1e-8:
            valid_idx = np.where(mask_np > 0.5)[0]
            action = int(np.random.choice(valid_idx)) if len(valid_idx) else 0
            probs_np = np.zeros_like(probs_np)
            probs_np[action] = 1.0
            return action, probs_np
        probs_np = probs_np / s
        if stochastic:
            action = int(np.random.choice(self.Emax, p=probs_np))
        else:
            action = int(np.argmax(probs_np))
        return action, probs_np


# ==============================================================
# 统一 runner
# ==============================================================
def run_variant(prefix, agent_class, cfg_override=None, mcss_mode=None, mcss_n=None):
    """跑一个消融变体. monkey-patch agent 类后调 mofd_main.main, 返回 pooled Pareto 点.

    prefix:        输出文件前缀 → results/<prefix>_pareto_aggregated.csv
    agent_class:   MOFD_SAC_V5 (原方法) / MOFD_SAC_V5_NoDiffusion / MOFD_SAC_V5_HMCSS
    cfg_override:  传给 mofd_main.main 的 cfg 覆盖 (seeds / num_epochs / smoke 等)
    mcss_mode/n:   仅对 MOFD_SAC_V5_HMCSS 生效
    """
    cfg_override = dict(cfg_override or {})
    cfg_override.setdefault('file_prefix', prefix)
    results_root = cfg_override.get('results_root', 'results')

    if mcss_mode is not None:
        agent_class.MCSS_MODE = mcss_mode
    if mcss_n is not None:
        agent_class.MCSS_N = int(mcss_n)

    print(f"\n############ ABLATION VARIANT: {prefix} "
          f"(class={agent_class.__name__}, mode={mcss_mode}, n={mcss_n}) ############")

    orig = mofd_main.MOFD_SAC_V5
    mofd_main.MOFD_SAC_V5 = agent_class
    try:
        mofd_main.main(cfg_override=cfg_override)
    finally:
        mofd_main.MOFD_SAC_V5 = orig

    path = os.path.join(results_root, f'{prefix}_pareto_aggregated.csv')
    pts = np.loadtxt(path).reshape(-1, 2)
    run_dir = _find_run_dir(prefix, results_root)
    return dict(prefix=prefix, pts=pts, run_dir=run_dir)


def _find_run_dir(prefix, results_root):
    """定位本变体完整 run 目录 results/<prefix>_<ts>/ (取最新一个)."""
    cands = [d for d in glob.glob(os.path.join(results_root, f'{prefix}_*'))
             if os.path.isdir(d)]
    return max(cands, key=os.path.getmtime) if cands else ''


def _variant_metrics(pts):
    """从 pooled Pareto 点 [N,2]=(delay,energy) 提取逐实验测评指标."""
    pts = np.asarray(pts).reshape(-1, 2)
    d, e = pts[:, 0], pts[:, 1]
    mean_e = float(e.mean())
    return dict(
        n_points=int(len(pts)),
        delay_min=float(d.min()), delay_max=float(d.max()), delay_mean=float(d.mean()),
        energy_min=float(e.min()), energy_max=float(e.max()), energy_mean=mean_e,
        # C1 卖点指标: 能耗 spread / 均值, 越大说明按 ω 分配能耗的幅度越大
        spread_e_over_mean=(float(e.max() - e.min()) / mean_e) if mean_e else 0.0,
    )


def summarize(variants, out_prefix, results_root='results', anchor=None):
    """variants: dict {name: {'prefix','pts','run_dir'}} (run_variant 的返回).

    逐实验记录: HV(共享 ref) / ΔHV% / Pareto 点数 / 延迟·能耗范围与均值 /
    能耗 spread÷均值(C1 指标) / 该实验完整数据目录.
    产出: results/<out_prefix>_summary.csv (机读) + .txt (人读) + _manifest.json (全量).
    """
    valid = {k: v for k, v in variants.items()
             if v is not None and len(np.asarray(v['pts']).reshape(-1, 2)) > 0}
    if not valid:
        print('[summarize] no valid variants'); return
    all_pts = np.vstack([np.asarray(v['pts']).reshape(-1, 2) for v in valid.values()])
    ref = (float(all_pts[:, 0].max() * 1.1 + 1e-6),
           float(all_pts[:, 1].max() * 1.1 + 1e-6))

    rows = []
    for name, v in valid.items():
        pts = np.asarray(v['pts']).reshape(-1, 2)
        rows.append(dict(name=name, prefix=v.get('prefix', name),
                         hv=float(hypervolume_2d(pts, ref)),
                         run_dir=v.get('run_dir', ''), **_variant_metrics(pts)))
    rows.sort(key=lambda r: -r['hv'])
    base_hv = next((r['hv'] for r in rows if r['name'] == anchor), rows[0]['hv']) or 1.0
    for r in rows:
        r['delta_pct'] = 100.0 * (r['hv'] - base_hv) / base_hv

    os.makedirs(results_root, exist_ok=True)
    csv_path = os.path.join(results_root, f'{out_prefix}_summary.csv')
    txt_path = os.path.join(results_root, f'{out_prefix}_summary.txt')
    json_path = os.path.join(results_root, f'{out_prefix}_manifest.json')

    cols = ['name', 'prefix', 'hv', 'delta_pct', 'n_points',
            'delay_min', 'delay_max', 'delay_mean',
            'energy_min', 'energy_max', 'energy_mean',
            'spread_e_over_mean', 'run_dir']
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(
                (f'{r[c]:.5f}' if isinstance(r[c], float) else str(r[c]))
                for c in cols) + '\n')

    header = (f'{"variant":<14}{"HV":>10}{"dHV%":>8}{"#pts":>6}  '
              f'{"delay[min,max]":>16}  {"energy[min,max]":>16}  {"E_sprd/mean":>12}')
    lines = [f'=== Ablation 测评结果 ({out_prefix}) ===',
             f'shared HV ref = ({ref[0]:.4f}, {ref[1]:.4f})   anchor = {anchor}',
             '', header]
    for r in rows:
        dr = f'[{r["delay_min"]:.2f},{r["delay_max"]:.2f}]'
        er = f'[{r["energy_min"]:.2f},{r["energy_max"]:.2f}]'
        lines.append(f'{r["name"]:<14}{r["hv"]:>10.4f}{r["delta_pct"]:>7.1f}%'
                     f'{r["n_points"]:>6}  {dr:>16}  {er:>16}{r["spread_e_over_mean"]:>13.3f}')
    lines += ['', '逐实验完整产出 (全 ω Pareto 点 / 训练曲线 / 监控 / ckpt) 见各 run 目录:']
    for r in rows:
        lines.append(f'  {r["name"]:<14} -> {r["run_dir"] or "(未找到)"}')
    txt = '\n'.join(lines)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dict(out_prefix=out_prefix, ref=ref, anchor=anchor, rows=rows),
                  f, indent=2, ensure_ascii=False)

    print('\n' + txt)
    print(f'\n[summarize] saved:\n  {csv_path}\n  {txt_path}\n  {json_path}')
