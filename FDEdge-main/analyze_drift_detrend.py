"""
Drift Detrend Analysis (R5-A + R5-B)
====================================
仅基于 results_omega_drift/ 现有 CSV 做分析, 不动训练:

  R5-A 去趋势:    per-slot delay 减去一阶多项式趋势, 让 drift spike 显形
  R5-B 相对指标:  画 (C2 - C0) 的 diff 曲线, 直观看 buffer 何时反超

输出 → results_omega_drift/analysis/
  detrended_curves.png      — 3 schedule × delay/energy 的 detrend 后曲线
  diff_curves.png           — 3 schedule × delay/energy 的 (C2 - C0) 差值曲线
  recovery_v2.txt           — 用 detrended 数据重算的 recovery time
  detrend_summary.csv       — 各方法 × schedule 的去趋势统计
"""
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = 'results_omega_drift'
OUT_DIR = os.path.join(RESULTS_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['C0', 'C2-passive', 'C2-cusum', 'C2-aware']
SCHEDULES = ['sudden', 'gradual', 'cyclic']
COLORS = {'C0': 'tab:gray', 'C2-passive': 'tab:blue',
          'C2-cusum': 'tab:green', 'C2-aware': 'tab:red'}


def load(method, sched, metric):
    path = os.path.join(RESULTS_DIR, f'{metric}_{method}_{sched}.csv')
    arr = np.loadtxt(path)
    # 去掉最后一个 terminal 0
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :-1]  # [eps, T-1]


def detrend(curve_1d):
    """一阶多项式去趋势."""
    T = len(curve_1d)
    x = np.arange(T)
    coef = np.polyfit(x, curve_1d, 1)
    trend = np.polyval(coef, x)
    return curve_1d - trend, trend


def get_drift_points(sched, T):
    """每个调度的 drift 触发 slot."""
    if sched == 'sudden':
        return [T // 2]
    if sched == 'cyclic':
        seg = T // 5
        return [seg, 2 * seg, 3 * seg, 4 * seg]
    if sched == 'gradual':
        return []  # 连续渐变, 没有离散 drift point
    return []


def recovery_v2(delay_1d, t_drift, look_ahead=20):
    """
    在 detrended 曲线上算恢复时间.
    思路:
      baseline = drift 前 10 slot 的 detrended mean
      spike    = drift 后 5 slot 的 detrended max
      threshold = baseline + 0.3 * (spike - baseline)
      返回 detrended 首次 ≤ threshold 的相对 slot
    """
    detrended, _ = detrend(delay_1d)
    pre = detrended[max(0, t_drift - 10):t_drift]
    post_window = detrended[t_drift:t_drift + look_ahead]
    if len(pre) == 0 or len(post_window) == 0:
        return -1
    baseline = pre.mean()
    spike = post_window.max()
    if spike <= baseline:
        return 0  # 没 spike, 算 0 也是合理的
    threshold = baseline + 0.3 * (spike - baseline)
    for k, v in enumerate(post_window):
        if v <= threshold:
            return k
    return -1  # look_ahead 内没恢复


# ========================================================================
# 1) 加载所有数据
# ========================================================================
data = {}  # data[(method, sched, metric)] = [eps, T-1] ndarray
for method in METHODS:
    for sched in SCHEDULES:
        for metric in ['delay', 'energy']:
            data[(method, sched, metric)] = load(method, sched, metric)

T = data[('C0', 'sudden', 'delay')].shape[1]
print(f'Loaded data, T={T}')


# ========================================================================
# 2) 画 detrended 曲线 (3 schedule × 2 metric)
# ========================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
for r, sched in enumerate(SCHEDULES):
    for c, metric in enumerate(['delay', 'energy']):
        ax = axes[r, c]
        for method in METHODS:
            curves = data[(method, sched, metric)]
            # 对每条 episode 单独 detrend, 再求均值/std
            detrended_runs = np.stack([detrend(curves[i])[0]
                                        for i in range(curves.shape[0])])
            mu = detrended_runs.mean(0)
            sd = detrended_runs.std(0)
            t = np.arange(T)
            ax.plot(t, mu, color=COLORS[method], lw=2, label=method)
            ax.fill_between(t, mu - sd, mu + sd,
                            alpha=0.2, color=COLORS[method])
        # drift 点垂直线
        for dp in get_drift_points(sched, T):
            ax.axvline(dp, color='k', ls='--', alpha=0.4, lw=0.8)
        ax.axhline(0, color='k', alpha=0.2, lw=0.5)
        unit = 's' if metric == 'delay' else 'J'
        ax.set_xlabel('Slot')
        ax.set_ylabel(f'Detrended {metric} ({unit})')
        ax.set_title(f'{sched} - {metric} (detrended)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'detrended_curves.png'), dpi=150)
plt.close()
print(f'Saved detrended_curves.png')


# ========================================================================
# 3) 画 diff 曲线 (C2 - C0), 6 subpanel
# ========================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
for r, sched in enumerate(SCHEDULES):
    for c, metric in enumerate(['delay', 'energy']):
        ax = axes[r, c]
        c0 = data[('C0', sched, metric)].mean(0)
        for method in ['C2-passive', 'C2-cusum', 'C2-aware']:
            cm = data[(method, sched, metric)].mean(0)
            diff = cm - c0
            t = np.arange(T)
            ax.plot(t, diff, color=COLORS[method], lw=2, label=f'{method} - C0')
            # 0 线
        ax.axhline(0, color='k', lw=1, alpha=0.5)
        ax.fill_between(np.arange(T), 0, 0, alpha=0)  # placeholder
        for dp in get_drift_points(sched, T):
            ax.axvline(dp, color='k', ls='--', alpha=0.4, lw=0.8)
        unit = 's' if metric == 'delay' else 'J'
        ax.set_xlabel('Slot')
        ax.set_ylabel(f'Δ{metric} (C2 - C0, {unit})')
        ax.set_title(f'{sched} - {metric} difference  '
                     f'(neg = C2 better)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'diff_curves.png'), dpi=150)
plt.close()
print(f'Saved diff_curves.png')


# ========================================================================
# 4) 重算 recovery time on detrended (sudden / cyclic 适用)
# ========================================================================
rec_lines = ['=== Recovery Time v2 (on detrended delay) ===',
             'Threshold: baseline + 0.3 * (spike - baseline)',
             'baseline = mean of [t_drift-10, t_drift) detrended delay',
             'spike    = max  of [t_drift, t_drift+5) detrended delay',
             '',
             '| Method      | Schedule | Drift slot | Recovery (slots, mean ± std) |',
             '|-------------|----------|------------|------------------------------|']

for sched in ['sudden', 'cyclic']:
    drift_pts = get_drift_points(sched, T)
    for dp in drift_pts:
        for method in METHODS:
            curves = data[(method, sched, 'delay')]
            rts = []
            for i in range(curves.shape[0]):
                rt = recovery_v2(curves[i], dp, look_ahead=20)
                if rt >= 0:
                    rts.append(rt)
            if rts:
                rec_lines.append(
                    f'| {method:11s} | {sched:8s} | {dp:10d} | '
                    f'{np.mean(rts):5.2f} ± {np.std(rts):5.2f} (n={len(rts)})   |'
                )
            else:
                rec_lines.append(
                    f'| {method:11s} | {sched:8s} | {dp:10d} | '
                    f'未恢复 in 20 slots                       |'
                )
    rec_lines.append('|-------------|----------|------------|------------------------------|')

# C2 vs C0 速率
rec_lines.append('')
rec_lines.append('=== Recovery speedup (drift-aware vs no-buffer) ===')
for sched in ['sudden', 'cyclic']:
    drift_pts = get_drift_points(sched, T)
    for dp in drift_pts:
        c0_curves = data[('C0', sched, 'delay')]
        c0_rts = [recovery_v2(c0_curves[i], dp, 20)
                  for i in range(c0_curves.shape[0])]
        c0_rts = [r for r in c0_rts if r >= 0]
        if not c0_rts:
            continue
        c0_mean = np.mean(c0_rts)
        for m in ['C2-passive', 'C2-cusum', 'C2-aware']:
            cm_curves = data[(m, sched, 'delay')]
            cm_rts = [recovery_v2(cm_curves[i], dp, 20)
                      for i in range(cm_curves.shape[0])]
            cm_rts = [r for r in cm_rts if r >= 0]
            if not cm_rts:
                continue
            cm_mean = np.mean(cm_rts)
            speedup = (c0_mean - cm_mean) / max(c0_mean, 1e-6) * 100
            rec_lines.append(
                f'  {sched:8s} t={dp:3d}  {m:11s} vs C0:  '
                f'{c0_mean:.2f} -> {cm_mean:.2f}  ({speedup:+.1f}%)'
            )

rec_text = '\n'.join(rec_lines)
print('\n' + rec_text)
with open(os.path.join(OUT_DIR, 'recovery_v2.txt'), 'w', encoding='utf-8') as f:
    f.write(rec_text)
print(f'Saved recovery_v2.txt')


# ========================================================================
# 5) Summary CSV: 关键统计量
# ========================================================================
import csv

rows = []
for method in METHODS:
    for sched in SCHEDULES:
        for metric in ['delay', 'energy']:
            curves = data[(method, sched, metric)]
            detrended_runs = np.stack([detrend(curves[i])[0]
                                        for i in range(curves.shape[0])])
            row = {
                'method': method,
                'schedule': sched,
                'metric': metric,
                'raw_mean': float(curves.mean()),
                'raw_std': float(curves.std()),
                'detrended_mean_abs': float(np.abs(detrended_runs).mean()),
                'detrended_max_abs': float(np.abs(detrended_runs).max()),
            }
            # 漂移点附近 detrended 极值
            dps = get_drift_points(sched, T)
            if dps:
                spikes = []
                for dp in dps:
                    for i in range(detrended_runs.shape[0]):
                        win = detrended_runs[i, dp:dp + 10]
                        if len(win):
                            spikes.append(win.max() - win.min())
                row['drift_spike_amplitude'] = float(np.mean(spikes)) if spikes else 0.0
            else:
                row['drift_spike_amplitude'] = 0.0
            rows.append(row)

with open(os.path.join(OUT_DIR, 'detrend_summary.csv'), 'w',
          encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved detrend_summary.csv')


# ========================================================================
# 6) 终端打印关键发现
# ========================================================================
print('\n' + '=' * 60)
print('=== Key findings ===')
print('=' * 60)

# (a) drift point 附近的 spike amplitude 对比
print('\n[A] Drift spike amplitude (max-min in 10 slot post-drift, on detrended)')
for sched in ['sudden', 'cyclic']:
    print(f'\n  {sched}:')
    for method in METHODS:
        sub = [r for r in rows
               if r['method'] == method and r['schedule'] == sched
               and r['metric'] == 'delay']
        if sub:
            print(f'    {method:11s}  spike={sub[0]["drift_spike_amplitude"]:.3f}')

# (b) diff 曲线在 drift 后 10 slot 的 mean
print('\n[B] Mean (C2 - C0) delay in [t_drift, t_drift+15] window')
for sched in ['sudden', 'cyclic']:
    drift_pts = get_drift_points(sched, T)
    for dp in drift_pts:
        c0 = data[('C0', sched, 'delay')].mean(0)
        for m in ['C2-passive', 'C2-cusum', 'C2-aware']:
            cm = data[(m, sched, 'delay')].mean(0)
            diff_window = (cm - c0)[dp:dp + 15]
            print(f'  {sched:8s} t={dp:3d}  {m:11s}  '
                  f'mean Δ={diff_window.mean():+.3f}s  '
                  f'(neg = C2 wins)')

print('\nAll outputs saved to:', os.path.abspath(OUT_DIR))
