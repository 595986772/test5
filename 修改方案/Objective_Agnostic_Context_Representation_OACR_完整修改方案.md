# Objective-Agnostic Context Representation (OACR) 完整修改方案

> 目标：在当前 v8 clean vector critic 基础上，补充一个不重复的表示学习机制：
> **objective-value decoupling 已由 v8 完成，OACR 进一步做 context-preference decoupling**。
> 本方案只改状态表示和辅助训练，不改变边缘环境物理公式，也不替代现有 vector critic。

---

## 0. 一句话定位

当前 v8 已经把目标价值分开：

```text
Q(s,a) = [Q_delay(s,a), Q_energy(s,a)]
Q_eff = omega_T * Q_delay + omega_E * Q_energy
```

这解决的是 **delay / energy 两个目标值混在一起学** 的问题。

但当前 actor / critic 的输入仍然把环境状态和 preference vector omega 放在同一个 state 里：

```text
state = [task, workload, omega_T, omega_E, server states, queue, channel]
```

因此网络仍可能把“环境理解”和“偏好权衡”纠缠在一起。OACR 要解决的是另一个问题：

```text
先学习不含 omega 的环境表示 z_ctx；
再把 z_ctx 与 preference vector omega 组合起来做动作决策和价值估计。
```

更准确的论文定位：

```text
v8: clean objective-value decoupling
OACR: context-preference decoupling
```

---

## 1. 为什么这个点不是重复 v8

### 1.1 v8 已经完成的事

v8 的核心是 clean vector critic：

- critic target 不再混入 entropy；
- critic loss 不再按 omega 给某个目标降权；
- critic 输出保持逐目标向量值，即 delay channel 和 energy channel 分开学习；
- actor 仍通过 preference vector omega 标量化向量 Q。

这属于 **objective-value decoupling**：

```text
目标通道分开：delay value / energy value
```

### 1.2 OACR 要新增的事

OACR 不讨论目标通道是否分开，而讨论 state representation 是否分开：

```text
环境动态：任务大小、计算量、队列、服务器频率、信道、valid mask
偏好权衡：omega_T / omega_E
```

这两者含义不同。环境动态是客观系统状态，preference vector omega 是用户偏好。一个好的模型应先理解系统上下文，再按 omega 调整决策，而不是把某个上下文和某个 omega 的组合死记住。

OACR 属于 **context-preference decoupling**：

```text
z_ctx = E_ctx(state without omega)
policy = pi(a | z_ctx, omega)
Q = Q(z_ctx, a, omega)
```

---

## 2. 论文故事线

现有 preference-conditioned MORL 方法通常直接拼接系统状态和 preference vector omega。这个设计简单，但会带来一个隐患：网络可能学习到的是特定 context-omega 组合下的动作模式，而不是可复用的边缘环境动态表示。

例如训练集中出现过：

```text
低负载 + 偏 delay
高负载 + 偏 energy
```

但测试时出现：

```text
高负载 + 偏 delay
```

如果模型把 context 和 omega 纠缠编码，它可能无法稳定泛化。OACR 的核心思想是：先从不含 omega 的 context 中学习环境动态表示，再在决策阶段注入 omega，使同一个环境表示可以被不同偏好复用。

建议论文贡献写法：

```text
We introduce an objective-agnostic context representation module that separates edge-context modeling from preference-conditioned decision making. Unlike the clean vector critic, which decouples objective values, OACR decouples system-context representation from the preference vector omega. The context encoder is trained with an auxiliary next-context prediction objective, allowing the policy and critic to reuse preference-independent edge dynamics across unseen preferences and non-stationary contexts.
```

中文表述：

```text
本文提出 objective-agnostic context representation，将边缘环境动态建模与 preference vector omega 的偏好权衡显式解耦。该机制与 v8 的 clean vector critic 正交：v8 分离 delay / energy 的目标价值，OACR 分离环境表示与偏好输入，从而提升未见偏好和上下文漂移下的泛化能力。
```

---

## 3. 方法设计

### 3.1 原始结构

当前 v5/v8 的 actor 和 critic 基本形式：

```text
actor_input  = raw_state
critic_input = raw_state

raw_state = [task, workload, omega_T, omega_E, server_1, ..., server_E]
```

其中 omega 通常位于：

```python
OMEGA_STATE_SLICE = slice(2, 4)
```

### 3.2 OACR 结构

将 raw state 拆为：

```text
ctx   = raw_state without omega
omega = raw_state[:, 2:4]
```

学习 context embedding：

```text
z_ctx = ContextEncoder(ctx)
```

actor / critic 使用：

```text
h = concat(z_ctx, omega)
```

然后：

```text
actor:  pi(a | z_ctx, omega)
critic: Q(z_ctx, a, omega) -> [Q_delay, Q_energy]
```

### 3.3 辅助动态预测任务

只加 ContextEncoder 可能被 reviewer 质疑为“换了个 MLP 名字”。因此 OACR 需要一个辅助任务，让 z_ctx 真正学习环境动态。

推荐第一版预测：

```text
next server queue summary
next server load summary
next channel summary
next valid mask
```

不推荐第一版预测：

```text
next task size
next task compute demand
```

原因：任务到达可能是随机的，强行预测会让辅助 loss 噪声很大。

辅助 loss：

```text
L_ctx = MSE(AuxHead(z_ctx), target_next_ctx)
```

总 critic loss：

```text
L_critic_total = L_critic + lambda_ctx * L_ctx
```

第一版只把 aux loss 加在 critic encoder 上，actor 先不加辅助 loss，避免 SAC actor 被额外目标干扰。

---

## 4. 代码修改方案

### 4.1 新增模块：`context_encoder.py`

新增文件：

```text
D:/python_project/实验版5/FDEdge-main/context_encoder.py
```

包含：

```python
class ContextEncoder(nn.Module):
    def __init__(self, ctx_dim, z_dim=64, hidden_dim=128):
        ...

    def forward(self, ctx):
        return z_ctx


class ContextAuxHead(nn.Module):
    def __init__(self, z_dim, target_dim, hidden_dim=128):
        ...

    def forward(self, z_ctx):
        return pred_next_ctx
```

同时提供 helper：

```python
def split_state_context_omega(s, omega_slice=slice(2, 4)):
    omega = s[:, omega_slice]
    ctx = torch.cat([s[:, :omega_slice.start], s[:, omega_slice.stop:]], dim=-1)
    return ctx, omega
```

### 4.2 修改 critic 网络

当前 `mofd_v5.py` 中的 `QValueNetV5` 直接吃 state：

```python
q_vec = critic(s)
```

新增一个可选版本：

```python
class QValueNetOACR(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, n_obj=2,
                 omega_dim=2, z_dim=64):
        self.ctx_dim = state_dim - omega_dim
        self.ctx_encoder = ContextEncoder(self.ctx_dim, z_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(z_dim + omega_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_obj),
        )

    def forward(self, s):
        ctx, omega = split_state_context_omega(s)
        z = self.ctx_encoder(ctx)
        h = torch.cat([z, omega], dim=-1)
        return self.q_head(h).view(B, action_dim, n_obj)
```

### 4.3 修改 agent 初始化

在 `MOFD_SAC_V5.__init__` 或新建 `MOFD_SAC_V8_OACR` 中加入参数：

```python
use_oacr=False
ctx_z_dim=64
ctx_aux_lambda=0.05
ctx_aux_target="server_queue_channel"
```

初始化 critic 时：

```python
if use_oacr:
    self.critic1 = QValueNetOACR(...)
    self.critic2 = QValueNetOACR(...)
else:
    self.critic1 = QValueNetV5(...)
    self.critic2 = QValueNetV5(...)
```

第一版建议新建子类：

```text
mofd_v8_oacr.py
```

继承 `MOFD_SAC_V8`，只覆写 critic 初始化和 update 中的 aux loss，避免污染 v8 主文件。

### 4.4 修改 replay / update

ReplayBuffer 已有：

```text
s, a, r_vec, s_next
```

因此可以从 `s_next` 构造辅助目标，不需要改 buffer 结构。

在 update 中：

```python
if self.use_oacr and self.ctx_aux_lambda > 0:
    pred1 = self.critic1.predict_next_ctx(s_t)
    pred2 = self.critic2.predict_next_ctx(s_t)
    target = build_ctx_aux_target(sn_t).detach()

    aux1 = F.mse_loss(pred1, target)
    aux2 = F.mse_loss(pred2, target)

    total_c1 = c1_loss_main + cor_loss1 + self.ctx_aux_lambda * aux1
    total_c2 = c2_loss_main + cor_loss2 + self.ctx_aux_lambda * aux2
```

注意：不要把 aux loss 加到 actor loss。第一版只服务 critic representation。

### 4.5 可选第二阶段：actor 也使用 OACR

第一阶段建议只做 critic-only OACR。若有效，再把 actor 输入也改为：

```text
actor_input = concat(z_ctx_actor, omega)
```

但这会牵涉 `FeedbackDiffusion` 内部 policy net 的 state_dim，改动较大，建议作为 OACR-v2。

---

## 5. 实验设计

### 5.1 主对比

至少比较：

```text
V8
V8 + OACR encoder only
V8 + OACR encoder + next-context auxiliary prediction
```

如果资源允许，加：

```text
V8 + same-param MLP
V8 + OACR + shuffled auxiliary target
```

后两个是为了排除“只是参数变多”的质疑。

### 5.2 实验一：普通 Pareto frontier

目的：证明 OACR 不损害基础性能。

设置：

```text
train omega: 21-point uniform grid
eval omega: 21-point uniform grid
```

指标：

```text
HV
IGD
Pareto point count
delay spread
energy spread
Spearman(omega_E, energy)
preference violation rate
```

预期：

```text
OACR 至少不降低 HV；
若有效，应降低 preference violation rate，并让 frontier 更平滑。
```

### 5.3 实验二：unseen preference interpolation

目的：证明 OACR 减少 context-omega 组合记忆，提高未见 preference vector omega 泛化。

训练：

```text
train omega_T = {0.0, 0.25, 0.5, 0.75, 1.0}
```

测试：

```text
eval omega_T = 0.0, 0.05, 0.10, ..., 1.0
```

指标：

```text
seen HV
unseen HV
interpolation gap
frontier smoothness
preference violation rate
```

预期：

```text
V8 + OACR 的 unseen HV 更高，interpolation gap 更小。
```

### 5.4 实验三：context shift

目的：证明 OACR 学到的是可迁移的 edge context 表示。

训练环境：

```text
bit_range=(10, 40)
f_range=(10, 40)
channel=normal
workload=medium
```

测试环境：

```text
bit_range=(30, 70)
f_range=(8, 30)
channel=degraded
workload=heavy
```

指标：

```text
shifted-context HV
relative HV drop
avg scalar cost under fixed omega
delay / energy spread
```

预期：

```text
OACR 相比 V8 在 shifted context 下性能下降更少。
```

### 5.5 实验四：context drift recovery

目的：验证 non-stationary environment 下恢复能力。

漂移类型：

```text
sudden workload drift
gradual channel drift
server frequency degradation
```

对比：

```text
V8
V8 + OACR
V8 + OACR + omega buffer
```

指标：

```text
recovery time
post-drift regret
final post-drift cost
rolling J_omega
```

预期：

```text
OACR 的 recovery time 更短，post-drift regret 更低。
```

### 5.6 辅助任务有效性消融

对比：

```text
OACR encoder only
OACR + next-context prediction
OACR + shuffled next-context target
```

如果 `next-context prediction` 明显好于 `encoder only` 和 `shuffled target`，才能说明辅助动态建模是真贡献。

---

## 6. 论文图表建议

建议放在 "Generalization and Adaptation Analysis" 小节。

图 1：OACR 结构图

```text
raw state -> split -> ctx / omega
ctx -> ContextEncoder -> z_ctx
concat(z_ctx, omega) -> actor / vector critic
z_ctx -> AuxHead -> next-context prediction
```

图 2：unseen preference response curve

```text
omega_T vs delay
omega_T vs energy
V8 vs V8+OACR
```

图 3：context shift 下 Pareto frontier

```text
normal context frontier
shifted context frontier
```

图 4：drift recovery curve

```text
episode vs rolling J_omega
drift point marked
```

表 1：普通 Pareto 指标

```text
HV / IGD / sparsity / violation rate
```

表 2：泛化与漂移指标

```text
unseen HV / interpolation gap / recovery time / post-drift regret
```

---

## 7. 诚实边界

不要写：

```text
OACR fully solves MORL objective conflict.
OACR guarantees Pareto optimality.
OACR replaces vector critic.
```

应该写：

```text
OACR is orthogonal to the clean vector critic.
It separates context representation from preference-conditioned trade-off learning.
It improves generalization under unseen preferences and shifted edge contexts, when supported by experiments.
```

如果实验只在普通 Pareto frontier 上提升不明显，但在 context shift / drift 上有效，就把贡献定位为：

```text
generalization and adaptation module
```

不要硬写成主 Pareto 性能模块。

---

## 8. 风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| 参数变多导致虚假提升 | OACR 增加 encoder 参数 | 加 same-param MLP baseline |
| 辅助任务噪声大 | next task size 不可预测 | 只预测 server queue / channel / valid mask |
| actor 改动导致 SAC 不稳定 | feedback diffusion actor 对输入维度敏感 | 第一版只做 critic-only OACR |
| 简单环境下收益不明显 | 当前 delay-energy 环境较简单 | 主打 unseen preference / context shift / drift |
| 与 vector critic 贡献混淆 | v8 已经做 objective-value decoupling | 明确 OACR 是 context-preference decoupling |

---

## 9. 推荐执行顺序

1. 新建 `context_encoder.py`，实现 `ContextEncoder`、`ContextAuxHead` 和 `split_state_context_omega`。
2. 新建 `mofd_v8_oacr.py`，继承 `MOFD_SAC_V8`。
3. 第一版只替换 critic 为 `QValueNetOACR`，actor 不动。
4. 在 update 中加入 `ctx_aux_loss`，只作用于 critic。
5. monitor CSV 增加：

```text
ctx_aux_loss
ctx_pred_error
ctx_z_norm
```

6. 跑 smoke：

```text
V8
V8 + OACR encoder only
V8 + OACR aux
```

7. 跑 unseen preference interpolation。
8. 跑 context shift。
9. 若有效，再考虑 actor+critic OACR。

---

## 10. 最小验收标准

代码验收：

```text
use_oacr=False 时 v8 行为不变。
use_oacr=True 时 critic 输出 shape 仍为 [B, action_dim, 2]。
ctx_aux_loss 能正常下降或保持稳定。
checkpoint 保存 / 加载不破坏旧 v8。
```

实验验收：

```text
普通 Pareto HV 不显著低于 v8。
unseen preference interpolation gap 小于 v8。
context shift 下 relative HV drop 小于 v8。
drift recovery time 或 post-drift regret 优于 v8。
```

论文验收：

```text
清楚区分 objective-value decoupling 和 context-preference decoupling。
不声称 OACR 保证 Pareto 最优。
把 OACR 定位为提升泛化和 non-stationary environment 适应性的表示学习模块。
```

---

## 11. 与现有方案的关系

与 v8：

```text
v8 是基础，OACR 是 v8 上的可选增强。
```

与 PCR：

```text
PCR 约束不同 omega 下的输出顺序；
OACR 改善 state representation。
二者可以组合，但第一阶段应分开消融。
```

与 Pareto Memory：

```text
Pareto Memory 改候选 prior；
OACR 改 actor / critic 对 context 的理解。
二者作用点不同。
```

与 quality-aware 三目标：

```text
三目标环境更复杂时，OACR 更有必要。
因为 delay / energy / accuracy 下 context 与 omega_q / omega 的纠缠更强。
```

---

## 12. 最终推荐版本

第一版不要做太大：

```text
OACR-v1 = V8 clean vector critic + critic-only context encoder + next server-context auxiliary prediction
```

如果 OACR-v1 在 unseen preference 和 context shift 上有效，再升级：

```text
OACR-v2 = actor + critic both use objective-agnostic context embedding
```

这样最容易归因，也最不容易破坏当前 v8 主模型。
