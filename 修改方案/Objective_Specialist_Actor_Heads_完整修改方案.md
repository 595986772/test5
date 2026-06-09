# Objective Specialist Actor Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 V5 的反馈扩散 actor 中加入轻量目标专家头，使 actor 在偏延迟和偏能耗偏好下能形成更清晰的策略分工，同时保持连续 `omega` 插值能力。

**Architecture:** 保留现有 PC-FDN / feedback diffusion actor 的主体结构，只把 `PCPolicyNet` 最后一层动作输出从单头改成可选双头：`delay_head` 和 `energy_head`。两个 head 共享前面的 server encoder、context encoder、time embedding 和 fusion body，最后按 `omega` 做 soft mixture，输出一个最终动作概率分布。

**Tech Stack:** Python, PyTorch, current `mofd_model.py`, current `mofd_v5.py`, current `mofd_main.py`, pytest, existing FDEdge V5 training loop.

---

## 1. 一句话解释

当前 V5 是：

```text
一个共享网络 + 一个输出头
```

它要同时负责低延迟、低能耗和折中偏好，容易学成平均策略。

本方案改成：

```text
一个共享网络 + 两个很小的输出头
```

两个头分别输出：

```text
delay_head  -> 一份偏低延迟的动作建议
energy_head -> 一份偏低能耗的动作建议
```

最终不是硬选某个头，而是按当前偏好连续混合：

```text
pi = omega_delay * pi_delay + omega_energy * pi_energy
```

这样 `omega` 稍微变化，动作分布也平滑变化，不会破坏 unseen preference interpolation。

---

## 2. 为什么不直接根据 omega 硬选头

硬选规则例如：

```text
omega_delay > 0.5 -> delay head
omega_delay <= 0.5 -> energy head
```

这个规则有三个问题：

```text
1. omega=(0.49,0.51) 和 omega=(0.51,0.49) 只差一点，但动作可能突然跳变。
2. 阈值是人为超参数，reviewer 会问为什么是 0.5 或 0.7。
3. 某些 head 可能长期不用，训练不充分。
```

本方案用 soft mixture：

```text
omega=(1.0,0.0) -> 几乎只听 delay head
omega=(0.0,1.0) -> 几乎只听 energy head
omega=(0.5,0.5) -> 两个 head 各听一半
```

这和论文主线更一致：

```text
偏好是连续变量，策略也应该连续变化。
```

---

## 3. 当前代码对应关系

当前 actor 的核心在：

```text
D:\python_project\实验版5\FDEdge-main\mofd_model.py
```

当前 `PCPolicyNet` 的最后输出层是：

```python
self.fusion = nn.Sequential(
    nn.Linear(fusion_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, Emax),
)
```

这里最后的：

```python
nn.Linear(hidden_dim, Emax)
```

就是“输出头”。它把隐藏特征变成每台服务器的动作分数。

当前 V5 初始化 actor 的位置是：

```text
D:\python_project\实验版5\FDEdge-main\mofd_v5.py
```

当前代码：

```python
backbone = PCPolicyNet(
    Emax=Emax, task_pref_dim=4,
    per_server_dim=per_srv_dim,
    hidden_dim=hidden_dim, t_dim=t_dim,
)
self.actor = FeedbackDiffusion(
    state_dim=state_dim, action_dim=Emax, model=backbone,
    beta_schedule='vp', denoising_steps=denoising_steps,
).to(device)
```

当前训练脚本实例化 V5 的位置是：

```text
D:\python_project\实验版5\FDEdge-main\mofd_main.py
```

需要从 `cfg` 加开关传到 `MOFD_SAC_V5`，再传给 `PCPolicyNet`。

---

## 4. 推荐设计

### 4.1 第一版只做两个 head

只做：

```text
delay_head
energy_head
```

先不做：

```text
balanced head
hard routing
gating network
diversity loss
```

原因：

```text
1. 两个目标对应两个 head，逻辑最直接。
2. 中间偏好可以由两个 head 混合得到。
3. 改动最小，消融最清楚。
4. 不需要解释人为阈值。
```

### 4.2 共享主体不要复制

不是这样：

```text
actor_delay  = 一个完整 actor
actor_energy = 一个完整 actor
```

而是这样：

```text
server_enc / ctx_enc / time_emb / fusion_body 共享
最后只有 delay_head 和 energy_head 分开
```

复杂度增加主要是：

```text
多一个 Linear(hidden_dim, Emax)
```

如果 `hidden_dim=128, Emax=6`，多出来约：

```text
128 * 6 + 6 = 774 个参数
```

相对完整 actor 和 critic 很小。

### 4.3 推荐使用 neutral shared context

当前 state 前 4 维是：

```text
state[0] = task data size
state[1] = compute density * task size
state[2] = omega_delay
state[3] = omega_energy
```

如果 shared encoder 已经吃了 `omega`，最后又用 `omega` 混合两个 head，会出现双重偏好注入：

```text
shared feature 里有 omega
head mixture 里也有 omega
```

这不是不能跑，但专家头会变得不够干净。

推荐第一版在 specialist 模式下使用 neutral shared context：

```python
task_pref_for_shared = task_pref.clone()
task_pref_for_shared[:, 2:4] = 0.5
ctx_emb = self.ctx_enc(task_pref_for_shared)
```

这样：

```text
共享主体负责看任务和服务器状态；
omega 只负责混合 delay/energy 两个专家头。
```

这更符合论文表述：

```text
objective-specialist heads are composed by the preference vector.
```

为了对照，可以保留一个开关：

```python
specialist_neutral_context=True
```

默认 `True`。

---

## 5. 具体模型结构

### 5.1 单头旧结构

旧结构等价于：

```python
h = fusion(fused)
pi = softmax(h)
```

### 5.2 双头新结构

新结构：

```python
h = fusion_body(fused)

logits_delay = delay_head(h)
logits_energy = energy_head(h)

pi_delay = softmax(logits_delay / temperature)
pi_energy = softmax(logits_energy / temperature)

omega = state[:, 2:4]
omega = omega / omega.sum(dim=1, keepdim=True)

pi = omega[:, 0:1] * pi_delay + omega[:, 1:2] * pi_energy
```

最终返回的仍然是：

```text
[B, Emax] 的动作概率分布
```

所以 `mofd_v5.py` 里的 `_masked_actor`、`take_action`、critic 训练、actor loss 都可以基本不动。

---

## 6. 文件修改范围

### Modify: `D:\python_project\实验版5\FDEdge-main\mofd_model.py`

职责：

```text
1. 给 PCPolicyNet 加 use_specialist_heads 开关。
2. 把 fusion 拆成 fusion_body + output_head。
3. specialist 开启时使用 delay_head / energy_head。
4. forward 里按 omega soft mixture 输出最终概率。
5. 保存轻量 head 监控指标，例如 mean_head_l1 / mean_head_kl。
```

### Modify: `D:\python_project\实验版5\FDEdge-main\mofd_v5.py`

职责：

```text
1. 给 MOFD_SAC_V5.__init__ 加 use_specialist_heads 等参数。
2. 初始化 PCPolicyNet 时传入这些参数。
3. update 返回值里可选加入 head_l1 / head_kl，便于训练监控。
```

### Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`

职责：

```text
1. 默认 cfg 中加入 use_specialist_heads=False。
2. run_single_seed 创建 AgentClass 时传入 specialist 参数。
3. monitor CSV 可选记录 head_l1 / head_kl。
```

### Create: `D:\python_project\实验版5\FDEdge-main\tests\test_specialist_heads.py`

职责：

```text
1. 测试单头模式旧行为输出 shape 不变。
2. 测试双头模式输出合法概率分布。
3. 测试 omega=(1,0) 主要听 delay_head，omega=(0,1) 主要听 energy_head。
```

---

## 7. 实现任务

### Task 1: 写 specialist heads 单元测试

**Files:**
- Create: `D:\python_project\实验版5\FDEdge-main\tests\test_specialist_heads.py`
- Modify later: `D:\python_project\实验版5\FDEdge-main\mofd_model.py`

- [ ] **Step 1: 写双头输出合法性测试**

```python
import torch

from mofd_model import PCPolicyNet


def make_state(batch=4, emax=3, per_server_dim=2):
    state_dim = 4 + emax * per_server_dim
    state = torch.zeros(batch, state_dim)
    state[:, 0] = 0.5
    state[:, 1] = 0.5
    state[:, 2] = torch.linspace(0.0, 1.0, batch)
    state[:, 3] = 1.0 - state[:, 2]
    return state


def test_specialist_heads_return_valid_distribution():
    net = PCPolicyNet(
        Emax=3,
        task_pref_dim=4,
        per_server_dim=2,
        hidden_dim=32,
        t_dim=8,
        use_specialist_heads=True,
    )
    x = torch.full((4, 3), 1.0 / 3.0)
    t = torch.zeros(4, dtype=torch.long)
    state = make_state(batch=4, emax=3, per_server_dim=2)

    out = net(x, t, state)

    assert out.shape == (4, 3)
    assert torch.all(out >= 0.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)
```

- [ ] **Step 2: 写 omega soft mixture 测试**

```python
def test_omega_extremes_select_corresponding_head():
    net = PCPolicyNet(
        Emax=3,
        task_pref_dim=4,
        per_server_dim=2,
        hidden_dim=32,
        t_dim=8,
        use_specialist_heads=True,
    )
    with torch.no_grad():
        net.delay_head.weight.zero_()
        net.energy_head.weight.zero_()
        net.delay_head.bias.copy_(torch.tensor([8.0, 0.0, 0.0]))
        net.energy_head.bias.copy_(torch.tensor([0.0, 8.0, 0.0]))

    x = torch.full((3, 3), 1.0 / 3.0)
    t = torch.zeros(3, dtype=torch.long)
    state = make_state(batch=3, emax=3, per_server_dim=2)
    state[0, 2:4] = torch.tensor([1.0, 0.0])
    state[1, 2:4] = torch.tensor([0.0, 1.0])
    state[2, 2:4] = torch.tensor([0.5, 0.5])

    out = net(x, t, state)

    assert out[0, 0] > 0.99
    assert out[1, 1] > 0.99
    assert out[2, 0] > 0.45
    assert out[2, 1] > 0.45
```

- [ ] **Step 3: 写单头兼容性测试**

```python
def test_single_head_mode_still_works():
    net = PCPolicyNet(
        Emax=3,
        task_pref_dim=4,
        per_server_dim=2,
        hidden_dim=32,
        t_dim=8,
        use_specialist_heads=False,
    )
    x = torch.full((2, 3), 1.0 / 3.0)
    t = torch.zeros(2, dtype=torch.long)
    state = make_state(batch=2, emax=3, per_server_dim=2)

    out = net(x, t, state)

    assert out.shape == (2, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 4: 运行测试确认失败**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
pytest tests\test_specialist_heads.py -q
```

Expected:

```text
FAIL because PCPolicyNet does not accept use_specialist_heads
```

### Task 2: 修改 PCPolicyNet 结构

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_model.py`

- [ ] **Step 1: 修改 `PCPolicyNet.__init__` 参数**

把：

```python
def __init__(self, Emax=6, task_pref_dim=4, per_server_dim=3,
             hidden_dim=128, t_dim=16):
```

改成：

```python
def __init__(self, Emax=6, task_pref_dim=4, per_server_dim=3,
             hidden_dim=128, t_dim=16,
             use_specialist_heads=False,
             specialist_neutral_context=True,
             specialist_temperature=1.0):
```

并保存：

```python
self.use_specialist_heads = bool(use_specialist_heads)
self.specialist_neutral_context = bool(specialist_neutral_context)
self.specialist_temperature = float(specialist_temperature)
self.last_head_stats = {}
```

- [ ] **Step 2: 拆分 fusion**

把当前：

```python
self.fusion = nn.Sequential(
    nn.Linear(fusion_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, Emax),
)
```

改成：

```python
self.fusion_body = nn.Sequential(
    nn.Linear(fusion_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
)
self.output_head = nn.Linear(hidden_dim, Emax)
self.delay_head = nn.Linear(hidden_dim, Emax)
self.energy_head = nn.Linear(hidden_dim, Emax)
```

这里保留 `output_head` 是为了单头旧模式兼容。

- [ ] **Step 3: 在 forward 中处理 neutral context**

当前：

```python
task_pref, per_srv = self.split_state(state)
srv_emb = self.server_enc(per_srv.reshape(B * self.Emax, self.per_server_dim))
srv_emb = srv_emb.reshape(B, -1)
ctx_emb = self.ctx_enc(task_pref)
```

改成：

```python
task_pref, per_srv = self.split_state(state)
omega = task_pref[:, 2:4].clamp(min=0.0)
omega = omega / (omega.sum(dim=1, keepdim=True) + 1e-8)

srv_emb = self.server_enc(per_srv.reshape(B * self.Emax, self.per_server_dim))
srv_emb = srv_emb.reshape(B, -1)

if self.use_specialist_heads and self.specialist_neutral_context:
    task_pref_for_shared = task_pref.clone()
    task_pref_for_shared[:, 2:4] = 0.5
else:
    task_pref_for_shared = task_pref
ctx_emb = self.ctx_enc(task_pref_for_shared)
```

- [ ] **Step 4: 在 forward 中实现单头/双头分支**

把：

```python
fused = torch.cat([srv_emb, ctx_emb, t_emb, x], dim=1)
out = self.fusion(fused)
return F.softmax(out, dim=1)
```

改成：

```python
fused = torch.cat([srv_emb, ctx_emb, t_emb, x], dim=1)
h = self.fusion_body(fused)

if not self.use_specialist_heads:
    out = self.output_head(h)
    return F.softmax(out, dim=1)

temp = max(self.specialist_temperature, 1e-6)
logits_delay = self.delay_head(h)
logits_energy = self.energy_head(h)
pi_delay = F.softmax(logits_delay / temp, dim=1)
pi_energy = F.softmax(logits_energy / temp, dim=1)
pi = omega[:, 0:1] * pi_delay + omega[:, 1:2] * pi_energy
pi = pi / (pi.sum(dim=1, keepdim=True) + 1e-8)

with torch.no_grad():
    l1 = torch.mean(torch.abs(pi_delay - pi_energy)).detach()
    entropy_delay = -torch.sum(pi_delay * torch.log(pi_delay + 1e-8), dim=1).mean()
    entropy_energy = -torch.sum(pi_energy * torch.log(pi_energy + 1e-8), dim=1).mean()
    self.last_head_stats = {
        "head_l1": float(l1.cpu()),
        "head_H_delay": float(entropy_delay.detach().cpu()),
        "head_H_energy": float(entropy_energy.detach().cpu()),
    }

return pi
```

- [ ] **Step 5: 运行单元测试**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
pytest tests\test_specialist_heads.py -q
```

Expected:

```text
3 passed
```

### Task 3: 把开关接入 V5

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_v5.py`

- [ ] **Step 1: 修改 `MOFD_SAC_V5.__init__` 参数**

在参数表中加入：

```python
use_specialist_heads=False,
specialist_neutral_context=True,
specialist_temperature=1.0,
```

- [ ] **Step 2: 保存配置**

在 `__init__` 内加入：

```python
self.use_specialist_heads = bool(use_specialist_heads)
self.specialist_neutral_context = bool(specialist_neutral_context)
self.specialist_temperature = float(specialist_temperature)
```

- [ ] **Step 3: 初始化 PCPolicyNet 时传参**

把：

```python
backbone = PCPolicyNet(
    Emax=Emax, task_pref_dim=4,
    per_server_dim=per_srv_dim,
    hidden_dim=hidden_dim, t_dim=t_dim,
)
```

改成：

```python
backbone = PCPolicyNet(
    Emax=Emax, task_pref_dim=4,
    per_server_dim=per_srv_dim,
    hidden_dim=hidden_dim, t_dim=t_dim,
    use_specialist_heads=use_specialist_heads,
    specialist_neutral_context=specialist_neutral_context,
    specialist_temperature=specialist_temperature,
)
```

- [ ] **Step 4: 在 update 返回中加入可选监控项**

在 return dict 前加入：

```python
head_stats = getattr(getattr(self.actor, "model", None), "last_head_stats", {})
```

并在返回 dict 里加入：

```python
head_l1=float(head_stats.get("head_l1", 0.0)),
head_H_delay=float(head_stats.get("head_H_delay", 0.0)),
head_H_energy=float(head_stats.get("head_H_energy", 0.0)),
```

这样训练日志能看出两个 head 是否完全学成一样。

- [ ] **Step 5: 运行 import smoke test**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python -c "from mofd_v5 import MOFD_SAC_V5; print('ok')"
```

Expected:

```text
ok
```

### Task 4: 把配置接入 mofd_main.py

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`

- [ ] **Step 1: 默认 cfg 加参数**

在 cfg 中加入：

```python
use_specialist_heads=False,
specialist_neutral_context=True,
specialist_temperature=1.0,
```

默认 `False`，保证旧实验不受影响。

- [ ] **Step 2: AgentClass 初始化时传参**

在创建 `AgentClass` 的参数字典中加入：

```python
use_specialist_heads=cfg.get("use_specialist_heads", False),
specialist_neutral_context=cfg.get("specialist_neutral_context", True),
specialist_temperature=cfg.get("specialist_temperature", 1.0),
```

注意：如果 `MOFD_SAC_V8` 没有这些参数，有两种处理方式。

推荐处理方式：

```python
agent_kwargs = dict(
    state_dim=env.state_dim,
    Emax=cfg["Emax"],
    hidden_dim=cfg["hidden_dim"],
    actor_lr=cfg["actor_lr"],
    critic_lr=cfg["critic_lr"],
    alpha=cfg["alpha_init"],
    alpha_lr=cfg["alpha_lr"],
    target_entropy=cfg["target_entropy"],
    tau=cfg["tau"],
    gamma=cfg["gamma"],
    denoising_steps=cfg["denoising_steps"],
    alpha_T=cfg["alpha_T"],
    alpha_E=cfg["alpha_E"],
    use_cor=cfg.get("use_cor", True),
    cor_lambda=cfg.get("cor_lambda", 0.1),
    cor_c=cfg.get("cor_c", 0.0),
    use_popart=cfg.get("use_popart", True),
    popart_beta=cfg.get("popart_beta", 0.01),
    device=device,
)
if AgentClass is MOFD_SAC_V5:
    agent_kwargs.update(dict(
        use_specialist_heads=cfg.get("use_specialist_heads", False),
        specialist_neutral_context=cfg.get("specialist_neutral_context", True),
        specialist_temperature=cfg.get("specialist_temperature", 1.0),
    ))
agent = AgentClass(**agent_kwargs)
```

这样不会破坏 V8 加载。

- [ ] **Step 3: 监控日志兼容**

当前每步 update 返回 dict，训练日志已经收集常规字段。可以先不画 head 曲线，只把字段留在 monitor CSV。若已有 monitor CSV 构造固定列，则追加：

```python
head_l1
head_H_delay
head_H_energy
```

如果当前 monitor CSV 固定字段不方便改，第一版可以只在 epoch print 中输出均值：

```python
if ep_losses and "head_l1" in ep_losses[0]:
    mean_head_l1 = np.mean([x.get("head_l1", 0.0) for x in ep_losses])
    print(f"head_l1={mean_head_l1:.4f}")
```

- [ ] **Step 4: 运行主入口 smoke test**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python -c "import mofd_main; print('ok')"
```

Expected:

```text
ok
```

### Task 5: 新建消融脚本

**Files:**
- Create: `D:\python_project\实验版5\FDEdge-main\run_ablation_specialist_heads.py`

- [ ] **Step 1: 写脚本**

```python
import mofd_main


def main():
    variants = [
        ("single_head", dict(use_specialist_heads=False)),
        ("two_head_neutral", dict(
            use_specialist_heads=True,
            specialist_neutral_context=True,
            specialist_temperature=1.0,
        )),
        ("two_head_omega_shared", dict(
            use_specialist_heads=True,
            specialist_neutral_context=False,
            specialist_temperature=1.0,
        )),
    ]

    for name, override in variants:
        cfg_override = dict(
            seeds=[0, 1, 2],
            final_eval_n_pref=21,
            file_prefix=f"abl_specialist_{name}",
        )
        cfg_override.update(override)
        print(f"[specialist ablation] running {name}")
        mofd_main.main(cfg_override=cfg_override)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 先跑单 seed 快速版**

为了快速 smoke，可以临时改成：

```python
seeds=[0],
num_epochs=5,
final_eval_n_epi=1,
```

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python run_ablation_specialist_heads.py
```

Expected:

```text
[specialist ablation] running single_head
[specialist ablation] running two_head_neutral
[specialist ablation] running two_head_omega_shared
```

---

## 8. 训练和梯度如何工作

两个 head 都会学习。

当：

```text
omega = (0.9, 0.1)
```

最终输出：

```text
pi = 0.9 * pi_delay + 0.1 * pi_energy
```

actor loss 的梯度主要流向 `delay_head`，也有少量流向 `energy_head`。

当：

```text
omega = (0.1, 0.9)
```

梯度主要流向 `energy_head`。

中间偏好：

```text
omega = (0.5, 0.5)
```

两个 head 都会学习。

所以训练集必须覆盖两端偏好。如果只训练中间 `omega`，两个 head 会缺少明确分工。

推荐训练偏好至少包括：

```text
(1.0, 0.0)
(0.75, 0.25)
(0.5, 0.5)
(0.25, 0.75)
(0.0, 1.0)
```

这也和 unseen preference interpolation 实验天然匹配。

---

## 9. 是否需要 diversity loss

第一版不加。

原因：

```text
1. diversity loss 会强行让两个 head 不同，可能为了不同而不同。
2. 如果 critic 还不够干净，它可能把 head 推向错误差异。
3. 加了以后不容易归因：提升来自双头，还是来自正则。
```

如果实验发现：

```text
head_l1 长期接近 0
两个 head 输出几乎完全一样
Pareto frontier 没有展开
```

再加很小的正则：

```python
L_div = -lambda_div * mean(abs(pi_delay - pi_energy))
```

或者：

```python
L_sim = lambda_sim * cosine_similarity(pi_delay, pi_energy).mean()
```

建议从：

```text
lambda_div = 0.001
```

开始，不要超过 `0.005`。这个正则属于第二阶段，不进入第一版主方案。

---

## 10. 实验方案

### 10.1 必做主消融

```text
A. V5 single-head actor
B. V5 two-head actor, neutral shared context
C. V5 two-head actor, omega shared context
```

核心指标：

```text
HV mean/std
delay_min mean/std
energy_min mean/std
energy spread
Pareto non-dominated point count
unseen omega interpolation gap
Spearman preference consistency
head_l1 / head entropy
```

预期：

```text
two-head neutral 如果有效，应该提升 Pareto 展开或降低插值 gap。
two-head omega_shared 如果不如 neutral，说明 omega 双重注入确实干扰专家分工。
```

### 10.2 和 clean critic 的组合实验

双头 actor 更依赖 critic 给出干净的偏好反馈，因此后续应该和 clean vector critic 组合：

```text
A. current V5 + single head
B. current V5 + two heads
C. clean vector critic + single head
D. clean vector critic + two heads
```

如果 D 最好，论文故事就是：

```text
clean vector critic 提供干净多目标价值；
objective-specialist heads 缓解 actor 的目标冲突；
PCR 保证偏好间一致性。
```

### 10.3 不建议一开始和 Pareto Memory 混跑

先不要同时打开：

```text
Pareto Memory Candidate
Objective-Specialist Actor Heads
PCR
clean critic
```

否则结果变好也不知道是谁贡献的。

推荐顺序：

```text
1. single head vs two heads
2. clean critic vs clean critic + two heads
3. clean critic + PCR vs clean critic + PCR + two heads
4. 最后再叠 Pareto Memory
```

---

## 11. 论文表述建议

推荐写法：

```text
To reduce interference between objective-specific behaviors, we replace the single action projection layer of the preference-conditioned diffusion actor with lightweight objective-specialist heads. The heads share the entire actor backbone and produce delay-oriented and energy-oriented action distributions, which are continuously composed according to the preference vector.
```

中文意思：

```text
我们不是复制多个 actor，而是只把最后的动作输出层拆成两个目标专家头。
两个头共享 actor 主体，分别给出偏延迟和偏能耗的动作分布，再由 omega 连续混合。
```

不要写成：

```text
delay head is guaranteed to minimize delay
energy head is guaranteed to minimize energy
hard routing selects the optimal expert
```

更稳的说法是：

```text
The specialist heads provide an inductive bias for objective-specific behavior, while the final policy is still trained end-to-end by the preference-conditioned actor-critic objective.
```

也就是：

```text
它提供结构偏置，不提供理论保证。
```

---

## 12. 风险和保护

### 风险 1: 两个 head 学成一样

表现：

```text
head_l1 接近 0
pi_delay 和 pi_energy 几乎一样
Pareto frontier 没展开
```

保护：

```text
训练 omega 覆盖两端。
先看 head_l1 曲线。
第二阶段再加很小 diversity loss。
```

### 风险 2: 两个 head 被强行分开导致性能下降

保护：

```text
第一版不加 diversity loss。
只靠 omega soft mixture 自然分工。
```

### 风险 3: omega 双重注入导致专家不干净

保护：

```text
默认 specialist_neutral_context=True。
保留 omega_shared 作为消融，不作为主方法。
```

### 风险 4: 训练开销被 reviewer 质疑

保护：

```text
强调只复制最后 action projection layer。
报告参数量增加。
报告推理时间或至少报告 head 参数数量。
```

参数量估计：

```text
extra params = hidden_dim * Emax + Emax
hidden_dim=128, Emax=6 -> 774 params
```

这是因为旧模型已有一个 `output_head`；双头模式额外多出一个同尺寸输出头。如果保留旧 `output_head` 但双头模式不用它，实际模型文件里会多两个 head，但有效推理只用两个 specialist heads。为了参数公平，也可以在 specialist 模式下不创建 `output_head`，但第一版为了兼容更简单，保留即可。

---

## 13. 验收标准

功能验收：

```text
pytest tests\test_specialist_heads.py -q 通过。
use_specialist_heads=False 时旧单头输出 shape 和概率性质不变。
use_specialist_heads=True 时输出 shape 仍为 [B, Emax]，sum=1。
omega=(1,0) 时主要使用 delay_head。
omega=(0,1) 时主要使用 energy_head。
```

训练验收：

```text
5 epoch smoke run 不报错。
actor_loss / entropy / alpha 不出现 NaN。
head_l1 有非零值，但不应剧烈爆炸。
```

实验验收：

```text
two-head neutral 的 HV 不低于 single-head。
unseen omega interpolation gap 不劣化。
delay_min 或 energy_min 至少一个方向有改善，或者 Pareto spread 更好。
```

论文验收：

```text
只声称缓解 objective-specific behavior interference。
不声称 head 天然保证对应目标最优。
不使用 hard routing 作为主方法。
```

---

## 14. 推荐执行顺序

```text
1. 先写 tests/test_specialist_heads.py。
2. 修改 mofd_model.py，让单元测试通过。
3. 接入 mofd_v5.py 参数开关。
4. 接入 mofd_main.py 默认配置。
5. 跑 5 epoch smoke。
6. 跑 3 seeds: single_head / two_head_neutral / two_head_omega_shared。
7. 如果 two_head_neutral 有效，再和 clean vector critic、PCR 组合。
```

---

## 15. Self-Review

Spec coverage:

```text
已覆盖动机、模型结构、当前代码入口、具体文件修改、测试、训练逻辑、实验、论文表述和风险。
```

Placeholder scan:

```text
没有留下占位标记或未定义接口；所有新增参数均给出默认值。
```

Type consistency:

```text
PCPolicyNet.forward 输入不变，输出仍为 torch.Tensor[B, Emax]。
MOFD_SAC_V5._masked_actor 不需要改变。
mofd_main.py 通过 cfg 开关传参，默认关闭，不破坏旧实验。
```
