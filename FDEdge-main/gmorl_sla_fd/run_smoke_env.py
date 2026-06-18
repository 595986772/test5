"""SLA-env 冒烟测试 + deadline 标定。

目的:
  1. 验证 env_gmorl_sla.py 能跑通、SLA 统计合理。
  2. 收集随机策略下的真实完成时延分布 -> 据此选一个有意义的 deadline。
  3. 验证 r_C (SLA 惩罚通道) 与 ω 无关:同一随机种子下, w=0 / 0.5 / 1.0 的 r_C 累计应一致。

用法: python run_smoke_env.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
from env_gmorl_sla import MEC_Env

CONF = 'multi-part'   # GMORL 训练用的配置: edge_num 1~8


def random_rollout(env, seed):
    """跑一个 episode, 随机选合法动作; 返回 (channel_sums, 所有完成时延)。"""
    np.random.seed(seed)
    obs = env.reset()
    sums = np.zeros(3, dtype=np.float64)   # [r_T, r_E, r_C]
    done = False
    while not done:
        mask2 = obs['mask2']
        valid = np.where(np.asarray(mask2) == 1)[0]
        action = int(np.random.choice(valid))
        obs, reward, done, info = env.step(action)
        sums += info['r_vec']
    return sums, np.array(env._episode_delays, dtype=np.float64)


def main():
    print('=== SLA-env 冒烟测试 (config=%s) ===\n' % CONF)

    # ---- 1) 时延分布 (用一个不太可能违约的大 deadline 收集真实分布) ----
    all_delays = []
    for s in range(8):
        env = MEC_Env(conf_name=CONF, w=0.5, deadline=1e9)  # deadline 设极大 -> 不影响分布收集
        _, delays = random_rollout(env, seed=100 + s)
        all_delays.append(delays)
    all_delays = np.concatenate(all_delays)
    print('随机策略完成时延分布 (8 episodes, n=%d 个任务):' % len(all_delays))
    for p in [50, 75, 90, 95, 99]:
        print('  p%-2d = %8.3f s' % (p, np.percentile(all_delays, p)))
    print('  mean = %8.3f s   max = %8.3f s' % (all_delays.mean(), all_delays.max()))
    # 选 deadline ~ p75: 让随机策略有 ~25% 违约率, SLA 才有区分度
    deadline = float(np.percentile(all_delays, 75))
    print('\n-> 选 deadline = p75 = %.3f s (随机策略下应 ~25%% 违约)\n' % deadline)

    # ---- 2) 在该 deadline 下看 SLA 统计 ----
    env = MEC_Env(conf_name=CONF, w=0.5, deadline=deadline)
    _, _ = random_rollout(env, seed=7)
    summ = env.episode_sla_summary()
    print('单 episode SLA 统计 (w=0.5, deadline=%.2f):' % deadline)
    for k, v in summ.items():
        print('  %-15s = %s' % (k, ('%.4f' % v if isinstance(v, float) else v)))

    # ---- 3) 验证 r_C 与 ω 无关: 固定种子, 只变 w, r_C 累计必须一致 ----
    print('\nω-无关性检查 (同种子 seed=42, deadline=%.2f, 只变 w):' % deadline)
    rc_by_w = {}
    for w in [0.0, 0.5, 1.0]:
        env = MEC_Env(conf_name=CONF, w=w, deadline=deadline)
        sums, _ = random_rollout(env, seed=42)
        rc_by_w[w] = sums[2]
        print('  w=%.1f :  r_T=%9.3f  r_E=%9.3f  r_C=%9.4f' % (w, sums[0], sums[1], sums[2]))
    spread = max(rc_by_w.values()) - min(rc_by_w.values())
    ok = abs(spread) < 1e-6
    print('  r_C 跨 w 的极差 = %.2e  -> %s' % (spread, 'OK: r_C 与 ω 无关' if ok else '!! r_C 受 w 影响, 有 bug'))

    print('\n[done] env 冒烟测试结束')


if __name__ == '__main__':
    main()
