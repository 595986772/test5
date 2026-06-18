# GMORL + SLA + Prior-Feedback Diffusion 完整修改方案

> 面向 Claude 代码修改使用。目标是在 GMORL 项目基础上，保留其 MEC 环境与偏好条件化 Pareto 设定，移除 Tianshou 训练框架，接入当前项目的 PyTorch 扩散 Actor-Critic 框架，并加入 SLA 约束与统计。

## 1. 总目标

在以下 GMORL 项目基础上修改：

```text
D:/python_project/实验版5/FDEdge-main/Generalizable-Pareto-Optimal-Offloading-with-Reinforcement-Learning-in-Mobile-Edge-Computing-main
```

最终形成一个新的可运行分支：

```text
FDEdge-main/gmorl_sla_fd/
```

新系统结构为：

```text
GMORL MEC Env
  -> obs_to_state_adapter
  -> prior builder
  -> prior-feedback diffusion actor
  -> action
  -> env.step()
  -> custom replay buffer
  -> custom SAC / vector-critic update
```

核心修改目标：

1. 去掉 GMORL 原有 Tianshou 依赖。
2. 在 GMORL 环境里加入 SLA deadline、SLA violation、SLA penalty。
3. 使用当前项目的 PyTorch feedback diffusion actor。
4. 将扩散起点从随机噪声或复杂多源候选，改为先验反馈 prior。
5. 第一版不加入 H-MCSS、omega buffer、COR、Pareto memory，避免把旧项目复杂度搬进新底座。

## 2. 不要做的事

第一版请不要做：

```text
不要继续使用 tianshou
不要使用 DiscreteSACPolicy
不要使用 Collector / offpolicy_trainer
不要搬 mofd_main.py 整套训练流程
不要加入 H-MCSS 三源候选
不要加入 omega buffer
不要加入 COR
不要加入 Pareto memory
不要一开始就接真实数据集
```

理由：第一版目标是跑通 GMORL + SLA + prior-feedback diffusion 的最小闭环。旧模块过多会导致无法判断到底是哪一个模块带来收益或问题。

## 3. 新目录结构

建议新建：

```text
FDEdge-main/gmorl_sla_fd/
  config_sla.json
  env_gmorl_sla.py
  gmorl_adapter.py
  prior_builder.py
  fd_model.py
  fd_agent.py
  train_gmorl_sla_fd.py
  eval_gmorl_sla_fd.py
  metrics.py
```

各文件职责：

```text
config_sla.json
  保存 GMORL 原配置 + SLA 参数。

env_gmorl_sla.py
  从 GMORL Env.py 精简而来，去掉 tianshou，加入 SLA 完成时间与违约统计。

gmorl_adapter.py
  将 GMORL dict obs 转成当前扩散模型使用的 flat state。

prior_builder.py
  生成 uniform prior、last-action prior、last-probs prior。

fd_model.py
  放置 diffusion actor 内部网络，尽量复用当前项目 feedback_diffusion.py / helpers.py。

fd_agent.py
  自写 PyTorch SAC agent，复用 V8 的 vector critic 思想，动作由 prior-feedback diffusion actor 生成。

train_gmorl_sla_fd.py
  新训练入口，不使用 Tianshou。

eval_gmorl_sla_fd.py
  固定 omega 网格评估 Pareto、SLA violation、delay、energy。

metrics.py
  保存 HV、平均 delay、平均 energy、SLA violation rate、p95/p99 delay 等指标。
```

## 4. 环境修改：env_gmorl_sla.py

从 GMORL 原始文件复制：

```text
GMORL/Env.py -> gmorl_sla_fd/env_gmorl_sla.py
```

删除无用依赖：

```python
import gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from tqdm import tqdm
```

保留或新增：

```python
import os
import time
import json
import math
import numpy as np
from copy import deepcopy
```

### 4.1 新增 SLA 配置

在 `config_sla.json` 的 `multi-part` 配置里加入：

```json
"sla_enable": true,
"sla_deadline_base": 1.0,
"sla_deadline_per_mbit": 0.05,
"sla_penalty_beta": 1.0
```

含义：

```text
deadline = sla_deadline_base + sla_deadline_per_mbit * task_size_Mbit
```

也就是说任务越大，deadline 适当放宽，不使用所有任务统一 deadline。

### 4.2 __init__ 中读取 SLA 参数

在 `MEC_Env.__init__()` 中加入：

```python
self.sla_enable = bool(param.get("sla_enable", False))
self.sla_deadline_base = float(param.get("sla_deadline_base", 1.0))
self.sla_deadline_per_mbit = float(param.get("sla_deadline_per_mbit", 0.05))
self.sla_penalty_beta = float(param.get("sla_penalty_beta", 1.0))
```

### 4.3 任务创建时加入 deadline

在 `step()` 中创建 `the_task` 后加入：

```python
task_mbit = float(the_task["size"]) / 1e6
the_task["deadline"] = self.sla_deadline_base + self.sla_deadline_per_mbit * task_mbit
the_task["completion_delay"] = 0.0
the_task["sla_violation"] = 0.0
```

上下文类似：

```python
the_task = {}
the_task["start_step"] = self.step_cnt
the_task["user_id"] = self.task_user_id
the_task["size"] = self.task_size
the_task["remain"] = self.task_size
the_task["off_time"] = 0
the_task["wait_time"] = 0
the_task["exe_time"] = 0
the_task["off_energy"] = 0
the_task["exe_energy"] = 0

task_mbit = float(the_task["size"]) / 1e6
the_task["deadline"] = self.sla_deadline_base + self.sla_deadline_per_mbit * task_mbit
the_task["completion_delay"] = 0.0
the_task["sla_violation"] = 0.0
```

### 4.4 任务完成时 append 到 finished_task

GMORL 原始代码里有：

```python
finished_task = []
```

但任务完成后没有真正 append，这需要修。

cloud 执行完成处，原逻辑类似：

```python
if self.cloud_exe_list[i]["remain"] <= ZERO_RES:
    retain_flag_exe[i] = False
```

改成：

```python
if self.cloud_exe_list[i]["remain"] <= ZERO_RES:
    retain_flag_exe[i] = False
    done_task = deepcopy(self.cloud_exe_list[i])
    done_task["completion_delay"] = (
        float(done_task.get("off_time", 0.0))
        + float(done_task.get("wait_time", 0.0))
        + float(done_task.get("exe_time", 0.0))
    )
    done_task["sla_violation"] = float(
        self.sla_enable and done_task["completion_delay"] > done_task["deadline"]
    )
    finished_task.append(done_task)
```

edge 执行完成处同样改：

```python
if self.edge_exe_lists[n][i]["remain"] <= ZERO_RES:
    retain_flag_exe[i] = False
    done_task = deepcopy(self.edge_exe_lists[n][i])
    done_task["completion_delay"] = (
        float(done_task.get("off_time", 0.0))
        + float(done_task.get("wait_time", 0.0))
        + float(done_task.get("exe_time", 0.0))
    )
    done_task["sla_violation"] = float(
        self.sla_enable and done_task["completion_delay"] > done_task["deadline"]
    )
    finished_task.append(done_task)
```

### 4.5 注意 wait_time 问题

GMORL 原代码里 `wait_time` 初始化了，但没有充分累计。如果第一版不重构事件仿真，可以先使用：

```text
completion_delay = off_time + exe_time
```

但论文中更严谨的版本建议把排队等待也计入：

```text
completion_delay = finish_clock_time - arrival_clock_time
```

建议第一版先保证代码闭环，第二版再补更严格的 completion time。

### 4.6 get_reward 改成 SLA-aware

原始：

```python
def get_reward(self, finished_task):
    reward = self.w * self.rew_t + (1.0 - self.w) * self.rew_e
    return reward
```

改成：

```python
def get_reward(self, finished_task):
    sla_violations = sum(float(t.get("sla_violation", 0.0)) for t in finished_task)
    sla_penalty = -self.sla_penalty_beta * sla_violations
    reward = self.w * self.rew_t + (1.0 - self.w) * self.rew_e + sla_penalty
    return reward, sla_penalty, sla_violations
```

### 4.7 step 返回 info

`step()` 末尾改成：

```python
reward, sla_penalty, sla_violations = self.get_reward(finished_task)

info = {
    "r_T": float(self.rew_t),
    "r_E": float(self.rew_e),
    "sla_penalty": float(sla_penalty),
    "sla_violations": float(sla_violations),
    "finished_count": int(len(finished_task)),
    "finished_delay": [float(t["completion_delay"]) for t in finished_task],
}

return obs, reward, done, info
```

第一版训练时建议把 SLA penalty 合并到 delay 通道：

```python
r_vec = np.array([
    info["r_T"] + info["sla_penalty"],
    info["r_E"],
], dtype=np.float32)
```

这样不需要立刻把 critic 从 2 目标扩成 3 目标。

## 5. 状态适配：gmorl_adapter.py

GMORL 原 obs 是 dict：

```python
obs["preference"]  # [omega_T, omega_E]
obs["servers"]     # [67, MAX_EDGE_NUM + 1]
obs["mask2"]       # [MAX_EDGE_NUM + 1]
```

当前扩散模型更适合 flat state，并且 V5/V8 默认：

```text
state[2:4] = omega
```

因此适配为：

```python
import numpy as np


def obs_to_state(obs, deadline_norm=0.0):
    omega = np.asarray(obs["preference"], dtype=np.float32)
    servers = np.asarray(obs["servers"], dtype=np.float32).T
    mask = np.asarray(obs["mask2"], dtype=np.float32)

    # GMORL server feature 第 3 位是 task_size / 1e6
    task_size_norm = float(servers[0, 3]) if servers.ndim == 2 and servers.shape[1] > 3 else 0.0

    task_pref = np.array(
        [task_size_norm, float(deadline_norm), omega[0], omega[1]],
        dtype=np.float32,
    )

    state = np.concatenate([task_pref, servers.flatten()]).astype(np.float32)
    return state, mask
```

动作维度固定为：

```text
action_dim = MAX_EDGE_NUM + 1 = 11
```

其中：

```text
action 0 = cloud
action 1..E = edge server
```

## 6. Prior 构造：prior_builder.py

第一版只实现三个 prior：

```python
import numpy as np


def normalize_with_mask(p, mask):
    p = np.asarray(p, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    p = np.clip(p, 0.0, None) * mask
    s = float(p.sum())
    if s <= 1e-8:
        p = mask.copy()
        s = float(p.sum())
    return (p / max(s, 1e-8)).astype(np.float32)


def uniform_prior(mask):
    return normalize_with_mask(mask, mask)


def last_action_prior(last_action, mask, action_dim):
    p = np.zeros(action_dim, dtype=np.float32)
    if last_action is not None and 0 <= int(last_action) < action_dim:
        p[int(last_action)] = 1.0
    else:
        p[:] = 1.0
    return normalize_with_mask(p, mask)


def last_probs_prior(last_probs, mask):
    if last_probs is None:
        return uniform_prior(mask)
    return normalize_with_mask(last_probs, mask)
```

训练第一版建议默认使用：

```text
last_probs_prior
```

即：

```text
第一个 step: uniform prior
后续 step: 上一步 actor 输出 probs
```

## 7. Agent 修改：fd_agent.py

可以参考当前项目：

```text
FDEdge-main/mofd_v8.py
FDEdge-main/mofd_v5.py
FDEdge-main/mofd_model.py
FDEdge-main/feedback_diffusion.py
FDEdge-main/helpers.py
```

但第一版不要保留 H-MCSS 三源候选逻辑。

建议新建简化版 agent：

```python
class GMORL_SLAFD_Agent:
    def __init__(...):
        # actor = prior-feedback diffusion
        # critic1/critic2 = vector critic [B, A, 2]
        # target1/target2 = target vector critic
        pass

    def take_action(self, state_np, prior_np, mask_np, stochastic=True):
        state = torch.tensor(state_np[None], dtype=torch.float32, device=self.device)
        prior = torch.tensor(prior_np[None], dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np[None], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probs = self.actor(state, prior, prior=prior)
            probs = probs * mask
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

        probs_np = probs.detach().cpu().numpy()[0]
        probs_np = np.clip(probs_np, 0.0, 1.0)
        probs_np = probs_np / max(float(probs_np.sum()), 1e-8)

        if stochastic:
            action = int(np.random.choice(len(probs_np), p=probs_np))
        else:
            action = int(np.argmax(probs_np))
        return action, probs_np
```

关键点：

```text
latent = prior
prior condition = prior
```

也就是说扩散起点和先验条件都来自同一个 prior。

如果使用当前 `FeedbackDiffusion`，需要确保它的 `p_sample_loop()` 是：

```python
def p_sample_loop(self, state, latent_action_probs, prior=None):
    x = latent_action_probs.clone()
    for i in reversed(range(self.n_timesteps)):
        timesteps = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
        x = self.p_sample(x, timesteps, state, prior=prior)
    return x
```

不要使用：

```python
x = torch.randn_like(latent_action_probs)
```

否则 prior-feedback 退化为普通随机扩散。

## 8. ReplayBuffer

复用 V5/V8 的 vector reward buffer 格式即可：

```text
(s, a, x_lat, prior, mask, r_vec, s_next, x_lat_next, mask_next)
```

在本方案里：

```text
x_lat = prior
x_lat_next = next_prior = probs
```

第一版 `r_vec` 维度为 2：

```python
r_vec = [r_delay_with_sla, r_energy]
```

其中：

```python
r_delay_with_sla = info["r_T"] + info["sla_penalty"]
r_energy = info["r_E"]
```

## 9. 训练入口：train_gmorl_sla_fd.py

不要使用 GMORL 原来的 `train.py`。

训练伪代码：

```python
from env_gmorl_sla import MEC_Env
from gmorl_adapter import obs_to_state
from prior_builder import uniform_prior, last_probs_prior
from fd_agent import GMORL_SLAFD_Agent


def train_one_seed(cfg, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MEC_Env(
        conf_file="config_sla.json",
        conf_name=cfg["conf_name"],
        w=0.5,
    )

    obs = env.reset()
    state, mask = obs_to_state(obs)
    state_dim = len(state)
    action_dim = len(mask)

    agent = GMORL_SLAFD_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=cfg["hidden_dim"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        denoising_steps=cfg["denoising_steps"],
        device=cfg["device"],
    )

    buffer = ReplayBuffer(cfg["buffer_size"])

    train_omegas = cfg["train_omegas"]

    for epoch in range(cfg["epochs"]):
        for omega in train_omegas:
            env.w = float(omega[0])
            obs = env.reset()
            state, mask = obs_to_state(obs)
            prior = uniform_prior(mask)
            done = False

            while not done:
                action, probs = agent.take_action(
                    state_np=state,
                    prior_np=prior,
                    mask_np=mask,
                    stochastic=True,
                )

                next_obs, reward, done, info = env.step(action)
                next_state, next_mask = obs_to_state(next_obs)

                r_vec = np.array([
                    float(info["r_T"]) + float(info["sla_penalty"]),
                    float(info["r_E"]),
                ], dtype=np.float32)

                buffer.add(
                    state,
                    action,
                    prior,
                    prior,
                    mask,
                    r_vec,
                    next_state,
                    probs,
                    next_mask,
                )

                state = next_state
                mask = next_mask
                prior = last_probs_prior(probs, mask)

                if buffer.size() >= cfg["warmup"]:
                    for _ in range(cfg["updates_per_step"]):
                        batch = buffer.sample(cfg["batch_size"])
                        agent.update(batch)
```

## 10. 评估：eval_gmorl_sla_fd.py

固定 omega 网格：

```python
eval_omegas = [
    [0.0, 1.0],
    [0.1, 0.9],
    ...
    [1.0, 0.0],
]
```

每个 omega 跑多个 episode，保存：

```text
omega_T
omega_E
avg_delay
avg_energy
sla_violation_rate
finished_task_count
p95_delay
p99_delay
```

评估时：

```text
stochastic=False
prior 第一步 uniform
后续 prior = 上一步 probs
```

## 11. 指标保存：metrics.py

至少保存三个 CSV：

```text
results_gmorl_sla_fd/training_curve.csv
results_gmorl_sla_fd/pareto_eval.csv
results_gmorl_sla_fd/summary.txt
```

`training_curve.csv` 字段：

```text
epoch
seed
avg_reward
avg_delay
avg_energy
sla_violation_rate
actor_loss
critic_loss
entropy
alpha
```

`pareto_eval.csv` 字段：

```text
seed
omega_T
omega_E
avg_delay
avg_energy
sla_violation_rate
p95_delay
p99_delay
finished_task_count
```

`summary.txt` 至少写：

```text
final HV
final avg delay
final avg energy
final SLA violation rate
per-omega best/worst SLA violation
```

## 12. 真实数据集接入：第二阶段

第一阶段不要接真实 trace。先跑通 synthetic GMORL + SLA + prior-feedback diffusion。

第二阶段再新增：

```text
trace_loader.py
```

统一输出：

```python
{
    "arrival_step": int,
    "user_id": int,
    "task_size": float,
    "cpu_cycles": float,
    "deadline": float,
}
```

然后在环境中替换 `generate_task()`：

```text
use_trace=False: 使用原 Poisson 生成任务
use_trace=True: 从 trace 中按 arrival_step 取任务
```

论文中表述为：

```text
trace-driven simulation
```

不要表述成真实 MEC 系统实验，因为无线信道、用户距离和服务器频率仍然来自仿真。

## 13. 实验对照

至少需要四组：

```text
A. GMORL 原版
B. GMORL + SLA
C. GMORL + prior-feedback diffusion
D. GMORL + SLA + prior-feedback diffusion
```

如果时间允许，再加：

```text
E. SLA + ordinary MLP actor
F. SLA + random-start diffusion actor
G. SLA + prior-start diffusion actor
```

这样可以证明：

```text
收益来自 prior-feedback diffusion，而不是单纯来自 diffusion 或 SLA penalty。
```

## 14. 验收标准

代码修改完成后必须满足：

```text
1. gmorl_sla_fd 不 import tianshou。
2. CPU 上可以跑一个 episode。
3. train_gmorl_sla_fd.py --smoke 可以跑 2 个 epoch。
4. 每一步 action 不越界。
5. prior 概率和始终接近 1。
6. mask 后无效服务器概率为 0。
7. info 中能看到 r_T、r_E、sla_penalty、sla_violations。
8. results 中能看到 delay、energy、SLA violation 曲线。
```

最小 smoke 设置：

```text
epochs = 2
episodes_per_epoch = 5
train_omegas = [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]]
batch_size = 64
warmup = 100
denoising_steps = 3
```

smoke 阶段不要求效果，只要求训练闭环正常、指标正常输出。

## 15. 论文叙事建议

推荐表述：

```text
We build upon the GMORL-style preference-conditioned offloading formulation,
but extend it to SLA-aware edge services and replace the conventional policy
with a prior-feedback diffusion policy.
```

中文含义：

```text
我们不是把 GMORL 本身当创新，而是在 GMORL 这类偏好条件化卸载框架上，
加入 SLA 约束，并用 prior-feedback diffusion policy 改善多目标调度。
```

避免表述：

```text
不要说 feedback diffusion 是完全原创。
不要说真实数据集等于真实 MEC 实验。
不要说 prior 一定提升效果，除非实验已经证明。
不要把 SLA penalty 包装成严格约束，除非实现了 constrained RL 或拉格朗日乘子。
```

## 16. 给 Claude 的一句话执行指令

请基于 GMORL 项目新建 `FDEdge-main/gmorl_sla_fd/`，不要继续使用 Tianshou。保留 GMORL 的 MEC 环境与偏好输入，加入 SLA deadline / violation / penalty 统计；用当前项目的 PyTorch feedback diffusion actor 和 vector critic 接管训练；扩散起点使用上一时刻策略输出的 prior，第一版不加入 H-MCSS、omega buffer、COR、Pareto memory。完成后提供一个 CPU 可运行的 smoke 训练脚本和结果 CSV。
