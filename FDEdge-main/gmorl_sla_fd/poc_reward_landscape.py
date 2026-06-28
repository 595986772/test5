"""POC-2 (路A 充分性探针): 连续分数卸载的奖励地形 —— 最优分配到底是不是多峰?

动机(承接 POC-1 表征探针 + FDEdge 理论分析):
  扩散在连续动作的优势, 只有当**最优动作分布真多峰**时才非空。
  但朴素可分负载分数卸载是**凸**的(delay=makespan=max(线性) 凸, energy=Σ线性):
    -> 标量化仍凸 -> 单纯形上唯一极小 -> 单峰 -> 扩散无优势。
  因此本探针诚实地测两件事:
    ① 当前 env 物理(无固定开销): 扫分配, 确认确实单峰(坐实风险)。
    ② 加入**现实非凸要素**(每服务器固定激活/建连开销 + 结果聚合惩罚)后:
       多峰最优会不会出现? 在哪个 ω 区制? 需要多大开销(占单任务成本几成)?
  这决定路A是"问题自带多峰"(强), 还是"得人为造峰"(弱, 审稿人会打)。

物理用 env 真实常数: exe_rate=f/C, 算力 f≈2GHz, C=1000 -> 2e6 bit/s; k=5e-31; off_power=0.01。
分数卸载: 任务 D bit 切分 a∈Δ^N, 服务器 i 得 a_i·D bit。
  completion_i = wait_i + [a_i>0]·setup_t + a_i·D/rate_off_i + a_i·D·C/f_i
  delay = makespan = max_{i:a_i>0} completion_i        (并行 -> 取 max)
  energy = Σ_i ( [a_i>0]·setup_e + a_i·D/rate_off_i·off_power + a_i·D·k·C·f_i² )
           + agg_e·(K_active-1)                         (聚合惩罚)
标量化成本(最小化): cost = w·delay_norm + (1-w)·energy_norm   (各自 min-max 归一, 与 PopArt 后标量化一致)
  w=1 延迟优先, w=0 能量优先 (沿用项目 reward = w·r_T+(1-w)·r_E 约定)。

用法: python poc_reward_landscape.py
产物: result2/poc/reward_landscape.png + 控制台多峰判决 + 阈值表。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env_gmorl_sla import MEC_Env

D = 20e6          # 任务大小 bit (= 项目 task_size_cap)
C = 1000.0        # cycles/bit
KCO = 5e-31       # 能量系数 k
OFFP = 0.01       # 传输功率 W


def real_snapshot():
    """从 env reset 抓一组真实服务器参数(算力/传输率/队列), 供探针用真物理。"""
    env = MEC_Env(conf_name='multi-part', w=0.5, task_size_cap=D)
    env.reset()
    f = list(env.edge_cpu_freq)                          # 各边缘算力 Hz
    rate = env.edge_off_datarate[:, 0].astype(float)     # 各边缘对 user0 的传输率 bit/s
    return np.array(f), np.array(rate)


def server_cost_coeffs(f, rate, wait):
    """每服务器: 单位分配(a_i=1)的 完成时间斜率 s_i 与 能量斜率 g_i, 以及队列offset o_i。"""
    s = D / rate + D * C / f          # off_time + exe_time (a_i=1 时)
    g = (D / rate) * OFFP + D * KCO * C * (f ** 2)   # off_energy + exe_energy (a_i=1 时)
    o = wait                          # 队列等待 offset (s)
    return s, g, o


def eval_alloc(a, s, g, o, setup_t, setup_e, agg_e):
    """给定分配 a (sum=1), 返回 (delay=makespan, energy)。"""
    active = a > 1e-6
    comp = o + active * setup_t + a * s          # completion_i
    delay = comp[active].max() if active.any() else 1e9
    K = int(active.sum())
    energy = (active * setup_e + a * g).sum() + agg_e * max(K - 1, 0)
    return delay, energy


def sweep_2server(s, g, o, w, setup_t, setup_e, agg_e, n=401):
    """N=2: 扫 a0∈[0,1] (a1=1-a0)。返回 a0网格 + 归一标量化成本。"""
    a0 = np.linspace(0, 1, n)
    dl = np.zeros(n); en = np.zeros(n)
    for i, x in enumerate(a0):
        dl[i], en[i] = eval_alloc(np.array([x, 1 - x]), s[:2], g[:2], o[:2], setup_t, setup_e, agg_e)
    dn = (dl - dl.min()) / (dl.ptp() + 1e-12)
    en_ = (en - en.min()) / (en.ptp() + 1e-12)
    cost = w * dn + (1 - w) * en_
    return a0, cost, dl, en


def count_wells(cost, barrier=0.05, compet=0.2):
    """数真实"势阱"(被势垒隔开的有竞争力极小)。返回 (标签, 极小索引list)。
    - 平地(ptp≈0) -> ('flat', []): 分配无关, 非多峰。
    - 否则: 找局部极小(含端点), 只留 ≤全局min+compet 的有竞争力者, 且相邻两极小间必须有
      高出两者 ≥barrier 的势垒才算两个独立阱(否则同阱取更低点)。"""
    n = len(cost)
    if cost.ptp() < 1e-9:
        return 'flat', []
    mins = []
    for i in range(n):
        left = cost[i - 1] if i > 0 else np.inf
        right = cost[i + 1] if i < n - 1 else np.inf
        if (cost[i] < left and cost[i] <= right) or (cost[i] <= left and cost[i] < right):
            mins.append(i)
    gmin = cost.min()
    mins = [i for i in mins if cost[i] <= gmin + compet]
    kept = []
    for i in mins:
        if not kept:
            kept.append(i); continue
        peak = cost[kept[-1]:i + 1].max()
        if peak >= cost[kept[-1]] + barrier and peak >= cost[i] + barrier:
            kept.append(i)                       # 真势垒隔开 -> 新阱
        elif cost[i] < cost[kept[-1]]:
            kept[-1] = i                         # 同阱, 取更低
    return len(kept), kept


def count_minima(cost):
    lab, idx = count_wells(cost)
    return idx if lab != 'flat' else ['flat']


def main():
    print('=== 抓 env 真实快照 ===')
    f, rate = real_snapshot()
    print('边缘算力 f (GHz):', np.round(f / 1e9, 3))
    print('传输率 rate (Mbit/s):', np.round(rate / 1e6, 2))
    fmean = float(np.mean(f[:2])) if len(f) >= 2 else float(f[0])
    # 构造两组服务器: 对称(易出多峰) 与 非对称(真实)
    f_sym = np.array([fmean, fmean])
    rate_sym = np.array([rate[:2].mean()] * 2) if len(rate) >= 2 else np.array([rate[0]] * 2)
    f_asym = f[:2] if len(f) >= 2 else np.array([f[0], f[0] * 0.85])
    rate_asym = rate[:2] if len(rate) >= 2 else np.array([rate[0], rate[0] * 0.9])
    wait = np.array([0.0, 0.0])   # 空队列基线 (offset 不改变凸性结论)

    # 单任务参考成本 (做 setup 的相对尺度)
    s_ref, g_ref, _ = server_cost_coeffs(f_sym, rate_sym, wait)
    t_one = float(s_ref[0]); e_one = float(g_ref[0])
    print('\n单任务(全给一台) 完成时间 ≈ %.2fs, 能量 ≈ %.4fJ' % (t_one, e_one))

    s_sym, g_sym, o_sym = server_cost_coeffs(f_sym, rate_sym, wait)
    s_as, g_as, o_as = server_cost_coeffs(f_asym, rate_asym, wait)

    omegas = [1.0, 0.5, 0.0]   # 延迟优先 / 均衡 / 能量优先
    def report(s, g, o, w, su_t, su_e, agg, tag):
        a0, cost, dl, en = sweep_2server(s, g, o, w, su_t, su_e, agg)
        lab, idx = count_wells(cost)
        if lab == 'flat':
            desc = '平地(分配无关, 非多峰)'
        elif lab == 1:
            desc = '单峰@a0=%.2f' % a0[idx[0]]
        else:
            desc = '%d峰@a0=%s ← 多峰!' % (lab, np.round(a0[idx], 2).tolist())
        print('  %s ω=%.1f: %s' % (tag, w, desc))
        return lab

    # ---- ① 当前物理(无开销): 确认凸/单峰/平地 ----
    print('\n=== ① 当前 env 物理 (setup=0, agg=0): 期望单峰/平地, 无真多峰 ===')
    for w in omegas:
        report(s_sym, g_sym, o_sym, w, 0, 0, 0, '对称')
    for w in omegas:
        report(s_as, g_as, o_as, w, 0, 0, 0, '非对称')

    # ---- ② 加每服务器激活能量开销: 扫强度找多峰阈值 ----
    print('\n=== ② 加每服务器激活能量开销 setup_e=γ·单任务能量: 找真双峰阈值 ===')
    gammas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    thr = {}
    for w in omegas:
        first_bi = None
        for gm in gammas:
            lab, idx = count_wells(sweep_2server(s_sym, g_sym, o_sym, w, 0, gm * e_one, 0)[1])
            if isinstance(lab, int) and lab >= 2 and first_bi is None:
                first_bi = gm
        thr[w] = first_bi
        print('  对称 ω=%.1f: 首次真双峰 γ(激活能量/单任务能量) = %s'
              % (w, ('%.2f' % first_bi) if first_bi is not None else '未出现(≤0.8)'))

    # ---- 画图: 对称服务器, 3个ω, 无开销 vs 有开销(γ=0.3) ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    def plot_one(ax, cost, a0, title):
        lab, idx = count_wells(cost)
        ax.plot(a0, cost, lw=1.5)
        if lab != 'flat':
            ax.scatter(a0[idx], cost[idx], c='r', zorder=3, s=45)
        tag = 'flat' if lab == 'flat' else ('%d峰' % lab)
        ax.set_title('%s  (%s)' % (title, tag)); ax.set_xlabel('a0'); ax.grid(alpha=0.3)
    for c, w in enumerate(omegas):
        a0, cost0, _, _ = sweep_2server(s_sym, g_sym, o_sym, w, 0, 0, 0)
        plot_one(axes[0, c], cost0, a0, 'ω=%.1f 当前物理(无开销)' % w)
        a0, cost1, _, _ = sweep_2server(s_sym, g_sym, o_sym, w, 0, 0.3 * e_one, 0)
        plot_one(axes[1, c], cost1, a0, 'ω=%.1f +激活开销γ=0.3' % w)
    axes[0, 0].set_ylabel('标量化成本'); axes[1, 0].set_ylabel('标量化成本')
    fig.suptitle('连续分数卸载奖励地形(对称双服务器): 上=当前物理(凸/单峰) 下=加现实激活开销\n'
                 '能量优先(ω=0)下出现"全给0 / 全给1"双峰; 延迟优先(ω=1)切分单峰', fontsize=11)
    fig.tight_layout()
    png = 'result2/poc/reward_landscape.png'
    fig.savefig(png, dpi=130)

    print('\n=== 判决 ===')
    cand = [thr[w] for w in omegas if thr[w] is not None]
    bi_min = min(cand) if cand else None
    if bi_min is not None and bi_min <= 0.2:
        print('  真双峰在 激活能量≈单任务能量的 %.0f%% 起出现(对称服务器)。' % (bi_min * 100))
        print('  机理: 延迟要并行(切分), 能量+激活开销要集中(少用服务器); 两股力在中/偏能量ω下')
        print('        形成"全给0 / 全给1"双势阱 = 真多峰 -> 高斯塌单峰/落谷底, 扩散能覆盖双峰。')
        print('  -> 路A充分条件: 成立但**有条件** = 需(a)每服务器固定激活/聚合开销(现实) (b)偏能量区制。')
        print('  -> 连续env 设计红线: 必须含每服务器激活/聚合开销, 否则退化成凸 -> 扩散白搭。')
        print('  -> 诚实边界: 纯延迟(ω=1)永远单峰(切分), 扩散优势不在延迟侧; 卖点要落在能量/SLA权衡侧。')
    else:
        print('  即便 γ=0.8 也难出真双峰 -> 多峰需人为强开销, 路A充分性弱, 需重审动作语义。')
    print('\n[saved] %s' % png)


if __name__ == '__main__':
    main()
