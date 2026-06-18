# GMORL 接入当前核心模型与 SLA 违约模块 TODO

生成时间: 2026-06-14

## 总判断

这次迁移不要理解成“在 GMORL 上小修小补”，而应该理解成:

```text
保留 GMORL 的 MEC 环境与偏好条件化设定
移除 GMORL 原 Tianshou 训练框架
接入当前项目的 PyTorch 扩散 Actor + Vector Critic
加入 SLA deadline / violation / penalty 统计
```

新项目建议独立放在:

```text
D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd
```

不要直接污染原始 GMORL 目录:

```text
D:/python_project/实验版5/FDEdge-main/Generalizable-Pareto-Optimal-Offloading-with-Reinforcement-Learning-in-Mobile-Edge-Computing-main
```

核心原则:

- 第一版先跑通最小闭环。
- 不引入 H-MCSS、omega buffer、COR、Pareto memory。
- 不一开始就接真实数据集。
- 先证明 `GMORL + SLA + prior-feedback diffusion` 是否有效。

## P0: 项目拆分与可运行底座

### 1. 新建独立 GMORL-SLA-FD 目录

要做什么:

```text
新建 FDEdge-main/gmorl_sla_fd/
```

建议目录:

```text
gmorl_sla_fd/
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

意义:

原始 GMORL 是论文复现代码，当前 V8 项目是另一套自写 SAC / diffusion 框架。直接混改会导致版本边界不清，后面很难判断问题来自 GMORL、SLA、扩散模型还是训练框架。

验收:

- 原始 GMORL 目录不被破坏。
- 新目录可以独立运行 smoke 脚本。
- README 或日志明确写出当前运行的是 `gmorl_sla_fd`。

### 2. 去掉 Tianshou 依赖

要做什么:

从 GMORL 的 `Env.py` 复制出 `env_gmorl_sla.py`，删除这些依赖:

```python
import tianshou as ts
from tianshou.env import DummyVectorEnv
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
```

同时不要再使用原始 `train.py` 里的:

```python
ts.policy.DiscreteSACPolicy
ts.data.Collector
ts.data.VectorReplayBuffer
ts.trainer.offpolicy_trainer
```

意义:

当前项目的扩散 Actor 接口是:

```text
actor(state, latent_or_prior, prior) -> action_probs
```

GMORL / Tianshou 默认 Actor 接口是:

```text
actor(obs) -> logits
```

两者强行适配会增加大量胶水代码。直接砍掉 Tianshou，改用当前项目的 PyTorch 训练循环更稳。

验收:

- `gmorl_sla_fd` 中 `rg "tianshou"` 没有有效 import。
- CPU 环境可以 import `env_gmorl_sla.py`。
- 不再出现 `numba/backports` 依赖报错。

### 3. 去掉 CUDA 强绑定

要做什么:

不要使用 GMORL 原始 `train.py` 里的:

```python
torch.cuda.set_device(0)
torch.set_default_tensor_type("torch.cuda.FloatTensor")
```

改成:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

意义:

原始 GMORL 强制 GPU，不利于 smoke 测试，也不利于 Claude 或另一个 Codex 快速验证。新项目必须先保证 CPU 可跑。

验收:

- 无 GPU 时也能跑 1 个 episode。
- 日志打印 `device=cpu` 或 `device=cuda`。

## P1: SLA 违约模块接入

### 4. 在配置中加入 SLA 参数

要做什么:

在 `config_sla.json` 中加入:

```json
"sla_enable": true,
"sla_deadline_base": 1.0,
"sla_deadline_per_mbit": 0.05,
"sla_penalty_beta": 1.0
```

deadline 计算:

```text
deadline = sla_deadline_base + sla_deadline_per_mbit * task_size_Mbit
```

意义:

不能给所有任务一个固定 deadline。大任务天然更慢，小任务天然更快。使用“基础 deadline + 按任务大小增长”的形式更合理。

验收:

- 每个新任务都有 `deadline` 字段。
- deadline 随任务大小变化。

### 5. 给任务对象增加 SLA 字段

要做什么:

在环境创建 `the_task` 时加入:

```python
task_mbit = float(the_task["size"]) / 1e6
the_task["deadline"] = self.sla_deadline_base + self.sla_deadline_per_mbit * task_mbit
the_task["completion_delay"] = 0.0
the_task["sla_violation"] = 0.0
```

意义:

SLA 必须绑定到单个任务，而不是 episode 结束后粗略估计。否则无法计算 violation rate、p95 delay、p99 delay。

验收:

- 任意进入 offload / execution 队列的任务都包含 `deadline`。
- 不出现 KeyError。

### 6. 修复 finished_task 没有真正记录的问题

要做什么:

GMORL 原始 `step()` 里有:

```python
finished_task = []
```

但任务完成时没有 append。需要在 cloud 和 edge 任务完成处加入:

```python
done_task = deepcopy(task)
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

意义:

如果没有 finished_task，SLA 违约率只能靠估计，不能严肃写进论文。这个是 SLA 模块的基础。

验收:

- 有任务完成时 `finished_count > 0`。
- `finished_delay` 有数值。
- `sla_violations` 可以统计。

### 7. 改造 reward 与 info 返回

要做什么:

把 `get_reward()` 改成:

```python
def get_reward(self, finished_task):
    sla_violations = sum(float(t.get("sla_violation", 0.0)) for t in finished_task)
    sla_penalty = -self.sla_penalty_beta * sla_violations
    reward = self.w * self.rew_t + (1.0 - self.w) * self.rew_e + sla_penalty
    return reward, sla_penalty, sla_violations
```

`step()` 末尾返回:

```python
info = {
    "r_T": float(self.rew_t),
    "r_E": float(self.rew_e),
    "sla_penalty": float(sla_penalty),
    "sla_violations": float(sla_violations),
    "finished_count": int(len(finished_task)),
    "finished_delay": [float(t["completion_delay"]) for t in finished_task],
}
```

意义:

训练需要 reward，论文实验需要分项指标。`info` 必须把 delay reward、energy reward、SLA penalty 拆开，后续才能画曲线和做消融。

验收:

- 每个 step 的 `info` 包含 `r_T/r_E/sla_penalty/sla_violations`。
- 训练日志能统计 episode-level SLA violation rate。

## P2: GMORL 状态接入当前核心模型

### 8. 编写 obs_to_state 适配器

要做什么:

新建 `gmorl_adapter.py`，把 GMORL dict observation 转成当前模型需要的 flat state。

GMORL obs:

```text
obs["preference"] = [omega_T, omega_E]
obs["servers"]    = [67, MAX_EDGE_NUM + 1]
obs["mask2"]      = [MAX_EDGE_NUM + 1]
```

转换后:

```text
state = [task_size_norm, deadline_norm, omega_T, omega_E, servers_flatten]
mask  = obs["mask2"]
```

关键要求:

```text
state[2:4] 必须是 omega
```

意义:

当前 V5/V8 的 actor 和 critic 默认从 `state[2:4]` 读取 omega。如果这里放错，偏好条件化会直接失效。

验收:

- `state.shape[0] = 4 + action_dim * server_feature_dim`。
- `state[2] = omega_T`。
- `state[3] = omega_E`。
- `mask` 长度等于 action_dim。

### 9. 固定 action_dim 为 cloud + edge slots

要做什么:

GMORL 的动作是:

```text
action 0 = cloud
action 1..E = edge server
```

固定:

```text
action_dim = MAX_EDGE_NUM + 1 = 11
```

无效 edge 由 mask 屏蔽。

意义:

扩散 actor 输出的是固定长度动作概率。服务器数量变化不能改变网络输出维度，只能靠 mask 控制有效动作。

验收:

- actor 输出 `[B, 11]`。
- 无效服务器概率为 0。
- action 不会选到无效 edge。

## P3: Prior-Feedback Diffusion Actor 接入

### 10. 复制并精简当前扩散模块

要做什么:

从当前项目复制必要文件到 `gmorl_sla_fd`:

```text
helpers.py
feedback_diffusion.py
```

从 `mofd_model.py` 或 `mofd_v5.py` 中提取:

```text
PCPolicyNet
FeedbackDiffusion
```

不要复制:

```text
H-MCSS
OmegaLatentBuffer
ParetoMemoryBuffer
COR
```

意义:

本次迁移的核心是“扩散 actor + GMORL 环境”，不是复刻整个 V8 工程。精简模块可以减少调试难度。

验收:

- `fd_model.py` 中只包含 diffusion actor 必需组件。
- 没有三源候选逻辑。

### 11. 扩散起点改为 prior

要做什么:

确保 `FeedbackDiffusion.p_sample_loop()` 使用:

```python
x = latent_action_probs.clone()
```

不要使用:

```python
x = torch.randn_like(latent_action_probs)
```

调用时:

```text
latent_action_probs = prior
prior condition = prior
```

意义:

如果从随机噪声开始，那就是普通 diffusion actor；如果从 prior 开始，才是 prior-feedback diffusion。这个是本方案和普通扩散策略的关键区别。

验收:

- 代码中能明确看到 `x = prior.clone()` 或等价逻辑。
- random-start diffusion 只作为消融，不作为主模型。

### 12. prior 来源第一版只用 last_probs

要做什么:

新建 `prior_builder.py`:

```text
第一个 step: uniform prior
后续 step: 上一步 actor 输出 probs
```

不要第一版加入物理启发式 prior 或 buffer prior。

意义:

last_probs 是最简单、最公平、最容易验证的反馈源。先证明“反馈起点 + diffusion refinement”是否有效，再扩展更复杂 prior。

验收:

- 每个 step 的 prior 概率和为 1。
- prior 被 mask 后仍然归一化。
- 第一步 prior 为 uniform。

## P4: Vector Critic 与 SAC 更新

### 13. 接入当前 V8 的 clean vector critic 思想

要做什么:

在 `fd_agent.py` 中实现双 critic:

```text
critic1(state) -> [B, action_dim, 2]
critic2(state) -> [B, action_dim, 2]
```

两个目标:

```text
Q_0 = delay/SLA channel
Q_1 = energy channel
```

第一版 reward vector:

```python
r_vec = [
    info["r_T"] + info["sla_penalty"],
    info["r_E"],
]
```

意义:

这样能保留多目标可解释性。critic 不是只学一个混合 reward，而是分别学 delay/SLA 和 energy 两张价值表。

验收:

- critic 输出形状为 `[B, 11, 2]`。
- replay buffer 中 `r_vec.shape = [2]`。

### 14. actor 更新按 omega 标量化 Q

要做什么:

actor loss 中使用:

```python
q_eff = omega_T * Q_delay_sla + omega_E * Q_energy
```

其中:

```python
omega = state[:, 2:4]
```

意义:

critic 学两个目标，actor 根据当前 omega 选择偏好。这是偏好条件化多目标强化学习的核心。

验收:

- 改变 omega 后，同一状态下动作概率会变化。
- `omega=(1,0)` 更偏向 delay/SLA。
- `omega=(0,1)` 更偏向 energy。

### 15. critic target 使用 V8 clean target

要做什么:

critic target 使用:

```text
target_vec = r_vec + gamma * V_vec
```

不要把 entropy 平摊塞进两个目标通道。

意义:

这样 `Q_delay_sla` 和 `Q_energy` 更干净，后续 Pareto / PCR / 排序分析才不会被 entropy 污染。

验收:

- target vector 里没有 `alpha * entropy / N_OBJ`。
- entropy 只出现在 actor loss 中。

## P5: 自写训练循环

### 16. 新建 train_gmorl_sla_fd.py

要做什么:

训练循环基本结构:

```text
for epoch:
  for omega in train_omegas:
    env.w = omega_T
    obs = env.reset()
    prior = uniform_prior(mask)
    while not done:
      state, mask = obs_to_state(obs)
      action, probs = agent.take_action(state, prior, mask)
      next_obs, reward, done, info = env.step(action)
      next_state, next_mask = obs_to_state(next_obs)
      buffer.add(...)
      prior = probs
      if buffer.size > warmup:
        agent.update(batch)
```

意义:

替代 Tianshou 的 Collector 和 offpolicy_trainer，使当前扩散模型能自然接入。

验收:

- `--smoke` 可以跑 2 epoch。
- buffer size 正常增长。
- actor/critic loss 有输出。

### 17. 训练 omega 先用粗网格

要做什么:

第一版训练:

```text
omega = [0.0, 0.25, 0.5, 0.75, 1.0]
```

意义:

GMORL 原版训练 64 个 omega，泛化难度偏低。粗 omega 训练可以测试 unseen preference interpolation。

验收:

- 日志打印 train omega 数量。
- 评估时可以区分 seen omega 和 unseen omega。

## P6: 评估与指标

### 18. 新建 eval_gmorl_sla_fd.py

要做什么:

测试 dense omega:

```text
omega = 0.00, 0.05, 0.10, ..., 1.00
```

每个 omega 跑多个 episode，保存:

```text
avg_delay
avg_energy
sla_violation_rate
p95_delay
p99_delay
HV
```

意义:

扩散模型的收益不能只看训练 reward。必须看 Pareto 前沿、SLA 违约率和尾部延迟。

验收:

- 输出 `pareto_eval.csv`。
- 输出 `training_curve.csv`。
- 输出 `summary.txt`。

### 19. 增加 prior refinement 诊断

要做什么:

评估时额外比较:

```text
直接用 prior 选动作
扩散 refined 后选动作
```

记录:

```text
prior_action
diffusion_action
prior_score
diffusion_score
changed_action_rate
```

意义:

这能回答一个关键审稿问题: 扩散模型到底有没有纠偏，还是只是照抄 prior。

验收:

- 能统计 `changed_action_rate`。
- 能比较 prior-only 和 diffusion-refined 的 delay/energy/SLA。

## P7: 必要消融

### 20. 四组基础对照

要做什么:

至少实现:

```text
A. GMORL original
B. GMORL + SLA
C. GMORL + prior-feedback diffusion
D. GMORL + SLA + prior-feedback diffusion
```

意义:

分别证明:

```text
SLA 是否有效
扩散是否有效
两者组合是否有效
```

验收:

- 每组都有同样 omega 网格评估。
- 每组都有 delay / energy / SLA violation / HV。

### 21. prior 起点消融

要做什么:

比较:

```text
uniform prior
last-action prior
last-probs prior
random-start diffusion
```

意义:

证明性能不是“扩散模型天然带来的”，而是 prior-feedback 起点有贡献。

验收:

- 一张表说明不同 prior 的 HV、SLA violation、p95 delay。

## P8: 真实数据集第二阶段再接

### 22. 暂缓 trace-driven workload

要做什么:

第一版不要接真实数据集。等 synthetic GMORL + SLA + diffusion 跑通后，再加:

```text
trace_loader.py
```

统一输出:

```python
{
    "arrival_step": int,
    "user_id": int,
    "task_size": float,
    "cpu_cycles": float,
    "deadline": float,
}
```

意义:

真实 trace 会引入任务分布、字段映射、deadline 定义等额外变量。如果和 SLA + diffusion 同时改，调试和归因会很困难。

验收:

- 第一阶段不依赖外部数据集。
- 第二阶段再做 trace-driven simulation。

## P9: 最小验收清单

### 23. Smoke test

要做什么:

提供命令:

```text
python train_gmorl_sla_fd.py --smoke
```

smoke 配置:

```text
epochs = 2
episodes_per_epoch = 5
train_omegas = [[0.0,1.0], [0.5,0.5], [1.0,0.0]]
batch_size = 64
warmup = 100
denoising_steps = 3
```

意义:

先证明流程闭环，不追求效果。

验收:

- 不报错。
- action 不越界。
- prior 概率和为 1。
- 无效 action 概率为 0。
- loss、entropy、alpha 有输出。
- SLA 指标有输出。

### 24. 最小论文结果

要做什么:

至少输出:

```text
Method | HV ↑ | Avg Delay ↓ | Avg Energy ↓ | SLA Violation ↓ | p95 Delay ↓
```

以及一张:

```text
Delay-Energy Pareto frontier
```

意义:

这是判断这个方向是否值得继续投入的最小证据。

验收:

- 可以比较 GMORL、SLA-only、FD-only、SLA+FD。
- 如果 SLA+FD 没有明显改善，及时停止，不继续包装。

## 一句话执行指令

请新建 `FDEdge-main/gmorl_sla_fd/`，以 GMORL 环境为底座，删除 Tianshou 训练框架，接入当前项目的 prior-feedback diffusion actor 和 clean vector critic；在环境中加入 SLA deadline、completion delay、violation 和 penalty；第一版只使用 last-probs prior，不引入 H-MCSS、omega buffer、COR、Pareto memory；完成后提供 CPU 可运行 smoke 训练、dense omega 评估和完整指标 CSV。
