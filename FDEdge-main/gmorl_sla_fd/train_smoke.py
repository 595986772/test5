"""冒烟训练: 验证整条链路 (env -> 扩散 actor -> 向量 critic -> SAC 更新) 能跑、不崩。

不追求收敛, 只验证:
  1. rollout + prior 追踪 + buffer 存取无误。
  2. update() 反向传播无 shape/device 错误。
  3. α 不奔 0、H 不崩到 0 (target_entropy=0.5 的修复在新架构上仍成立)。
  4. 3 通道奖励都在动 (delay/energy/SLA 都参与学习)。

用法: python train_smoke.py
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
import torch

from env_gmorl_sla import MEC_Env
from fd_agent import FDSACAgent, ReplayBuffer
from fd_actor import uniform_prior

CONF = 'multi-part'
DEADLINE = 15.0
N_EPISODES = 40
WARMUP = 500          # buffer 攒够才开始 update
BATCH = 128
UPDATES_PER_STEP = 1


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', dev)

    env = MEC_Env(conf_name=CONF, w=0.5, deadline=DEADLINE)
    agent = FDSACAgent(denoising_steps=3, device=dev)
    buf = ReplayBuffer(50000)

    total_steps = 0
    for ep in range(N_EPISODES):
        w = float(np.random.rand())           # 每 episode 采一个偏好 ω
        env.w = w
        obs = env.reset()
        prior = uniform_prior(obs['mask2'])    # episode 起始: 均匀 prior
        ch_sum = np.zeros(3)
        logs = []
        done = False
        while not done:
            a, probs = agent.take_action(obs, prior)
            next_obs, reward, done, info = env.step(a)
            next_prior = probs                 # 本步输出概率 -> 下一步暖启动
            buf.add(obs['servers'], obs['preference'], obs['mask2'], prior, a,
                    info['r_vec'], next_obs['servers'], next_obs['preference'],
                    next_obs['mask2'], next_prior, float(done))
            ch_sum += info['r_vec']
            obs = next_obs
            prior = next_prior
            total_steps += 1
            # 更新
            if buf.size() >= WARMUP:
                for _ in range(UPDATES_PER_STEP):
                    logs.append(agent.update(buf.sample(BATCH)))

        sla = env.episode_sla_summary()
        if logs:
            al = np.mean([d['alpha'] for d in logs]); H = np.mean([d['H'] for d in logs])
            cl = np.mean([d['c_loss'] for d in logs]); aloss = np.mean([d['a_loss'] for d in logs])
            train_str = 'alpha=%.4f H=%.3f c_loss=%.3f a_loss=%.3f' % (al, H, cl, aloss)
        else:
            train_str = '(warmup)'
        print('ep%02d w=%.2f | rT=%7.2f rE=%7.2f rC=%7.3f | viol=%.3f p95=%6.2f n=%3d | %s'
              % (ep, w, ch_sum[0], ch_sum[1], ch_sum[2],
                 sla['violation_rate'], sla['p95_delay'], sla['n_finished'], train_str),
              flush=True)

    print('\n[done] 冒烟训练结束, total_steps=%d, buffer=%d' % (total_steps, buf.size()))
    # 健康检查
    if logs:
        last = logs[-1]
        print('末步 alpha=%.4f (应 >0.01 不奔0)  H=%.3f (应 ~target 0.5 不崩0)'
              % (last['alpha'], last['H']))


if __name__ == '__main__':
    main()
