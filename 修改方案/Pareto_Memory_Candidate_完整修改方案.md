# Pareto Memory Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 V5 当前的普通 `prior` 候选升级为 Pareto archive 引导的高质量候选，在不增加候选数量的前提下改善 delay-energy Pareto frontier。

**Architecture:** 新增一个轻量 `ParetoMemoryBuffer`，负责从历史 episode 中保存局部非支配动作先验，并在推理/训练时返回一个 `[Emax]` 的 `pareto_memory` prior。V5 的 `take_action` 仍保持三候选结构 `{feedback, prior, random}`，区别是 `prior` 不再来自普通历史均值，而来自经过 Pareto dominance 过滤的 archive。

**Tech Stack:** Python, NumPy, PyTorch, current FDEdge V5 code path, existing `mofd_main.py` training loop, existing `mofd_v5.py` three-source candidate selector.

---

## 1. 背景判断

当前 `results` 里的候选源结果显示三类源存在互补性：

```text
src_feedback: HV=252.08, energy_min=1.823, delay_min=29.34
src_prior:    HV=248.37, energy_min=2.227, delay_min=31.62
src_random:   HV=228.28, energy_min=2.511, delay_min=27.47
src_full3:    HV=219.48, energy_min=2.393, delay_min=27.79
```

离线合并四类源的非支配点后，oracle HV 约为 `267.05`，说明不同源确实存在可利用的互补性。但 `src_full3` 低于单源，说明简单堆候选源不能保证变好。更稳的方向是提高单个 `prior` 候选的质量，而不是增加候选数量。

Pareto Memory Candidate 的核心想法：

```text
普通 prior: 以前用过的动作。
Pareto memory prior: 历史中没有被明显支配的好动作。
```

它不假设 feedback 天然省能，也不假设 random 天然低延迟。它只利用多目标优化里最朴素、最能站住的规则：如果一个历史动作在 delay 和 energy 上同时不差，才值得进入候选记忆。

---

## 2. 当前代码入口

需要关注三个位置：

```text
D:\python_project\实验版5\FDEdge-main\mofd_main.py
  - OmegaLatentBuffer: 当前普通上下文 buffer
  - run_episode: 每步调用 agent.take_action(state, latent, prior_latent, mask)
  - run_single_seed: 每个 episode 前 retrieve_prior，episode 后 update buffer

D:\python_project\实验版5\FDEdge-main\mofd_v5.py
  - MOFD_SAC_V5.take_action: 当前三源候选选择
  - _scalarize_q: 用 ω 对 vector Q 标量化

D:\python_project\实验版5\FDEdge-main\results
  - abl_mcss_src_*_pareto_aggregated.csv: 候选源消融结果
```

当前 V5 `take_action` 已经是三源候选：

```python
cand = torch.cat([latent, prior, rand_init], dim=0)
```

因此本方案不改变候选数量，只改变第二个候选 `prior` 的来源：

```text
Before: prior = OmegaLatentBuffer.retrieve_prior(omega, env_ctx=env_ctx, action_dim=env.action_dim, current_epoch=epoch)
After:  prior = ParetoMemoryBuffer.retrieve_prior(omega, env_ctx=env_ctx, action_dim=env.action_dim, current_epoch=epoch)
```

---

## 3. 方法设计

### 3.1 存什么

Archive entry 保存 episode 级动作先验，而不是保存整个轨迹的每一步候选。这样内存和检索成本都低。

```python
entry = {
    "key": np.ndarray shape [5],          # [ω_T, ω_E, E_norm, f_E_norm, tran_norm]
    "prior": np.ndarray shape [Emax],     # episode latent_slice 平均后的动作概率
    "delay": float,                       # 该 episode 的总 delay 或平均 delay
    "energy": float,                      # 该 episode 的总 energy 或平均 energy
    "epoch": int,
    "omega_bin": int,
    "ctx_bin": tuple[int, int, int],
}
```

`prior` 的构造沿用当前 `OmegaLatentBuffer.retrieve_prior` 的思想：

```python
prior = latent_slice.reshape(-1, action_dim).mean(axis=0)
prior = np.clip(prior, 0.0, None)
prior = prior / (prior.sum() + 1e-8)
```

这样它仍然是一个合法的动作概率向量，可以直接传给 `agent.take_action(state, latent, prior_latent, mask)`。

### 3.2 怎么判断 Pareto 好动作

本项目里 delay 和 energy 都是越小越好。给定两个 entry：

```python
def dominates(a, b, eps_delay=0.0, eps_energy=0.0):
    no_worse = (
        a.delay <= b.delay + eps_delay and
        a.energy <= b.energy + eps_energy
    )
    strictly_better = (
        a.delay < b.delay - eps_delay or
        a.energy < b.energy - eps_energy
    )
    return no_worse and strictly_better
```

为了避免不同环境上下文之间直接比较造成误删，只在局部 bucket 内做 dominance：

```text
bucket = (omega_bin, E_bin, f_bin, tran_bin)
```

例如：

```python
omega_bin = round(ω_T * 20)        # 21 个偏好桶
E_bin = round(E_norm * 3)          # 4 个服务器数量桶
f_bin = round(f_E_norm * 3)        # 4 个计算能力桶
tran_bin = round(tran_norm * 3)    # 4 个通信能力桶
```

局部比较比全局比较更严谨，因为不同 workload/channel/server context 下的 raw delay/energy 不完全可比。

### 3.3 怎么写入

写入规则：

```text
1. warmup 前不写入，避免早期随机策略污染 archive。
2. 如果新 entry 被同 bucket 内旧 entry 支配，则不写入。
3. 如果新 entry 支配旧 entry，则删除旧 entry。
4. 如果 bucket 超过容量，按 crowding distance + scalar score 保留多样性。
```

最小版本可以先不做复杂 crowding，只保留：

```text
每个 bucket 最多 16 个 entry。
超过容量时，保留：
  - delay 最低的 2 个
  - energy 最低的 2 个
  - 按当前 bucket 中心 ω 标量化得分最好的 12 个
```

这个规则比单纯保留 top score 更适合 Pareto，因为它保住两端极值，不容易把 archive 压成中间点。

### 3.4 怎么检索

检索时使用当前 query：

```python
query = [ω_T, ω_E, E_norm, f_E_norm, tran_norm]
```

流程：

```text
1. 找到 query 附近的 top_k archive entries。
2. 如果最近邻相似度低于 conf_threshold，返回 uniform prior。
3. 在 top_k 中再次做局部非支配过滤。
4. 用当前 ω 对历史 delay/energy 做标量化，选一个 entry 的 prior 返回。
```

标量化用于检索时只起到“从 archive 中取哪个 prior”的作用，最终动作仍由 V5 critic 再次选择。

```python
def archive_score(entry, omega, alpha_T=1.0, alpha_E=0.25, d_scale=1.0, e_scale=1.0):
    d = entry.delay / max(d_scale, 1e-8)
    e = entry.energy / max(e_scale, 1e-8)
    return -(omega[0] * alpha_T * d + omega[1] * alpha_E * e)
```

`d_scale/e_scale` 用 archive 内 running median 或 running RMS，避免 delay 数值量级压过 energy。

### 3.5 和 V5 三候选的关系

最终推理仍然是：

```text
C = {feedback, pareto_memory, random}
```

其中：

```text
feedback: 当前 episode 内上一时刻动作反馈。
pareto_memory: 历史局部非支配动作先验。
random: 随机探索候选。
```

V5 critic 按当前 `ω` 选择最终动作：

```python
q_eff = ω_T * alpha_T * Q_delay + ω_E * alpha_E * Q_energy
expected_q = sum(probs * q_eff)
```

因此 Pareto Memory Candidate 的角色不是替 critic 做决策，而是提供一个更优质的候选起点。

---

## 4. 文件结构

### Create: `D:\python_project\实验版5\FDEdge-main\pareto_memory.py`

职责：

```text
定义 ParetoArchiveEntry。
定义 ParetoMemoryBuffer。
实现局部 dominance pruning。
实现 retrieve_prior / update / save_pickle / load_pickle / stats / save_log。
```

### Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`

职责：

```text
根据 cfg.use_pareto_memory 选择 OmegaLatentBuffer 或 ParetoMemoryBuffer。
episode 前调用 retrieve_prior。
episode 后把 delay/energy 传给 ParetoMemoryBuffer.update。
保存 pmem 日志和 pickle。
```

### Modify: `D:\python_project\实验版5\FDEdge-main\mofd_v5.py`

职责：

```text
默认不改 take_action 的候选数量。
只把日志里的 prior 标签解释为 pareto_memory。
可选增加 source_names，便于输出 feedback/pareto_memory/random pick ratio。
```

### Create: `D:\python_project\实验版5\FDEdge-main\tests\test_pareto_memory.py`

职责：

```text
验证 dominance、bucket pruning、检索 fallback、检索返回合法概率分布。
```

### Create: `D:\python_project\实验版5\FDEdge-main\run_ablation_pareto_memory.py`

职责：

```text
跑 no-buffer / omega-buffer / pareto-memory 三组对照。
保存 HV、delay_min、energy_min、Pareto 点、archive stats、pick ratio。
```

---

## 5. 具体实现任务

### Task 1: 新增 ParetoMemoryBuffer 单元测试

**Files:**
- Create: `D:\python_project\实验版5\FDEdge-main\tests\test_pareto_memory.py`
- Create later: `D:\python_project\实验版5\FDEdge-main\pareto_memory.py`

- [ ] **Step 1: 写 dominance 测试**

```python
import numpy as np

from pareto_memory import ParetoArchiveEntry, ParetoMemoryBuffer


def make_entry(delay, energy, prior=None, omega=(0.5, 0.5), epoch=10):
    if prior is None:
        prior = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    key = np.array([omega[0], omega[1], 0.5, 0.5, 0.5], dtype=np.float32)
    return ParetoArchiveEntry(
        key=key,
        prior=prior.astype(np.float32),
        delay=float(delay),
        energy=float(energy),
        epoch=int(epoch),
        omega_bin=10,
        ctx_bin=(2, 2, 2),
    )


def test_dominates_requires_no_worse_and_one_strictly_better():
    buf = ParetoMemoryBuffer(action_dim=3)
    good = make_entry(delay=10.0, energy=2.0)
    bad = make_entry(delay=12.0, energy=3.0)
    tradeoff = make_entry(delay=8.0, energy=4.0)

    assert buf._dominates(good, bad)
    assert not buf._dominates(bad, good)
    assert not buf._dominates(good, tradeoff)
    assert not buf._dominates(tradeoff, good)
```

- [ ] **Step 2: 写 bucket pruning 测试**

```python
def test_update_prunes_dominated_entries_inside_same_bucket():
    buf = ParetoMemoryBuffer(action_dim=3, warmup_epochs=0, bucket_capacity=16)
    omega = np.array([0.5, 0.5], dtype=np.float32)
    env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    latent_bad = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
    latent_good = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)

    buf.update(omega, latent_bad, env_ctx=env_ctx, epoch=1, delay=12.0, energy=3.0)
    buf.update(omega, latent_good, env_ctx=env_ctx, epoch=2, delay=10.0, energy=2.0)

    assert len(buf.entries) == 1
    assert np.allclose(buf.entries[0].prior, np.array([0.7, 0.2, 0.1], dtype=np.float32))
```

- [ ] **Step 3: 写检索合法性测试**

```python
def test_retrieve_prior_returns_valid_distribution():
    buf = ParetoMemoryBuffer(action_dim=3, warmup_epochs=0, conf_threshold=0.0)
    omega = np.array([0.8, 0.2], dtype=np.float32)
    env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    latent = np.array([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1]], dtype=np.float32)

    buf.update(omega, latent, env_ctx=env_ctx, epoch=1, delay=8.0, energy=4.0)
    prior = buf.retrieve_prior(omega, env_ctx=env_ctx, action_dim=3, current_epoch=2)

    assert prior.shape == (3,)
    assert np.all(prior >= 0.0)
    assert abs(float(prior.sum()) - 1.0) < 1e-6
```

- [ ] **Step 4: 运行测试，确认当前失败**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
pytest tests\test_pareto_memory.py -q
```

Expected:

```text
FAIL because pareto_memory module does not exist
```

### Task 2: 实现 ParetoMemoryBuffer

**Files:**
- Create: `D:\python_project\实验版5\FDEdge-main\pareto_memory.py`

- [ ] **Step 1: 新建 entry dataclass 和 buffer 骨架**

```python
from __future__ import annotations

import collections
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ParetoArchiveEntry:
    key: np.ndarray
    prior: np.ndarray
    delay: float
    energy: float
    epoch: int
    omega_bin: int
    ctx_bin: Tuple[int, int, int]


class ParetoMemoryBuffer:
    def __init__(
        self,
        action_dim: int,
        n_omega_bins: int = 21,
        n_ctx_bins: int = 4,
        max_entries: int = 512,
        bucket_capacity: int = 16,
        top_k: int = 20,
        sigma: float = 0.3,
        conf_threshold: float = 0.4,
        warmup_epochs: int = 5,
        age_decay: float = 0.05,
        reward_gate_k: float = 1.0,
        reward_window: int = 50,
        alpha_T: float = 1.0,
        alpha_E: float = 0.25,
    ):
        self.action_dim = int(action_dim)
        self.n_omega_bins = int(n_omega_bins)
        self.n_ctx_bins = int(n_ctx_bins)
        self.max_entries = int(max_entries)
        self.bucket_capacity = int(bucket_capacity)
        self.top_k = int(top_k)
        self.sigma = float(sigma)
        self.conf_threshold = float(conf_threshold)
        self.warmup_epochs = int(warmup_epochs)
        self.age_decay = float(age_decay)
        self.reward_gate_k = float(reward_gate_k)
        self.reward_window = int(reward_window)
        self.alpha_T = float(alpha_T)
        self.alpha_E = float(alpha_E)

        self.entries: List[ParetoArchiveEntry] = []
        self.log = []
        self.update_count: Dict[Tuple[int, int, int, int], int] = {}
        self._reward_history = collections.deque(maxlen=self.reward_window)
        self._delay_rms = 1.0
        self._energy_rms = 1.0
```

- [ ] **Step 2: 实现 key/bin/prior 工具函数**

```python
    @staticmethod
    def _build_key(omega, env_ctx):
        if env_ctx is None:
            env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return np.concatenate([
            np.asarray(omega, dtype=np.float32),
            np.asarray(env_ctx, dtype=np.float32),
        ]).astype(np.float32)

    def _omega_bin(self, omega):
        b = int(round(float(omega[0]) * (self.n_omega_bins - 1)))
        return max(0, min(self.n_omega_bins - 1, b))

    def _ctx_bin(self, env_ctx):
        if env_ctx is None:
            env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        ctx = np.asarray(env_ctx, dtype=np.float32)
        bins = []
        for x in ctx:
            b = int(round(float(x) * (self.n_ctx_bins - 1)))
            bins.append(max(0, min(self.n_ctx_bins - 1, b)))
        return tuple(bins)

    def _bucket_key(self, omega, env_ctx):
        return (self._omega_bin(omega),) + self._ctx_bin(env_ctx)

    def _latent_to_prior(self, latent_slice):
        lat = np.asarray(latent_slice, dtype=np.float32)
        if lat.shape[-1] != self.action_dim:
            raise ValueError(f"latent action dim {lat.shape[-1]} != action_dim {self.action_dim}")
        prior = lat.reshape(-1, self.action_dim).mean(axis=0)
        prior = np.clip(prior, 0.0, None)
        s = float(prior.sum())
        if s < 1e-8:
            return np.full(self.action_dim, 1.0 / self.action_dim, dtype=np.float32)
        return (prior / s).astype(np.float32)
```

- [ ] **Step 3: 实现 dominance 和 bucket pruning**

```python
    def _dominates(self, a, b, eps_delay=0.0, eps_energy=0.0):
        no_worse = (
            a.delay <= b.delay + eps_delay and
            a.energy <= b.energy + eps_energy
        )
        strictly_better = (
            a.delay < b.delay - eps_delay or
            a.energy < b.energy - eps_energy
        )
        return bool(no_worse and strictly_better)

    def _same_bucket(self, entry, bucket_key):
        return (entry.omega_bin,) + tuple(entry.ctx_bin) == tuple(bucket_key)

    def _bucket_entries(self, bucket_key):
        return [e for e in self.entries if self._same_bucket(e, bucket_key)]

    def _scalar_score(self, entry, omega):
        d = entry.delay / max(self._delay_rms, 1e-8)
        e = entry.energy / max(self._energy_rms, 1e-8)
        return -(float(omega[0]) * self.alpha_T * d + float(omega[1]) * self.alpha_E * e)

    def _prune_bucket_capacity(self, bucket_key, omega):
        bucket = self._bucket_entries(bucket_key)
        if len(bucket) <= self.bucket_capacity:
            return

        keep_ids = set()
        by_delay = sorted(bucket, key=lambda e: e.delay)
        by_energy = sorted(bucket, key=lambda e: e.energy)
        by_score = sorted(bucket, key=lambda e: self._scalar_score(e, omega), reverse=True)

        for e in by_delay[:2] + by_energy[:2]:
            keep_ids.add(id(e))
        for e in by_score:
            if len(keep_ids) >= self.bucket_capacity:
                break
            keep_ids.add(id(e))

        self.entries = [
            e for e in self.entries
            if not self._same_bucket(e, bucket_key) or id(e) in keep_ids
        ]
```

- [ ] **Step 4: 实现 update**

```python
    def update(self, omega, latent_slice, env_ctx=None, epoch=None, reward=None, delay=None, energy=None):
        if delay is None or energy is None:
            return
        if epoch is not None and self.warmup_epochs > 0 and int(epoch) < self.warmup_epochs:
            return

        if reward is not None and self.reward_gate_k > 0:
            r = float(reward)
            self._reward_history.append(r)
            if len(self._reward_history) >= 10:
                hist = np.asarray(self._reward_history, dtype=np.float32)
                if r < float(hist.mean() - self.reward_gate_k * hist.std()):
                    return

        omega_arr = np.asarray(omega, dtype=np.float32)
        key = self._build_key(omega_arr, env_ctx)
        bucket_key = self._bucket_key(omega_arr, env_ctx)
        prior = self._latent_to_prior(latent_slice)
        new_entry = ParetoArchiveEntry(
            key=key,
            prior=prior,
            delay=float(delay),
            energy=float(energy),
            epoch=int(epoch) if epoch is not None else 0,
            omega_bin=int(bucket_key[0]),
            ctx_bin=tuple(bucket_key[1:]),
        )

        self._delay_rms = 0.99 * self._delay_rms + 0.01 * abs(new_entry.delay)
        self._energy_rms = 0.99 * self._energy_rms + 0.01 * abs(new_entry.energy)

        bucket = self._bucket_entries(bucket_key)
        if any(self._dominates(old, new_entry) for old in bucket):
            return

        self.entries = [
            old for old in self.entries
            if not (self._same_bucket(old, bucket_key) and self._dominates(new_entry, old))
        ]
        self.entries.append(new_entry)

        if len(self.entries) > self.max_entries:
            self.entries = sorted(self.entries, key=lambda e: e.epoch)[-self.max_entries:]

        self._prune_bucket_capacity(bucket_key, omega_arr)
        self.update_count[bucket_key] = self.update_count.get(bucket_key, 0) + 1
```

- [ ] **Step 5: 实现 retrieve_prior 和 stats**

```python
    def _age_weights(self, indices, current_epoch):
        if current_epoch is None or self.age_decay <= 0:
            return np.ones(len(indices), dtype=np.float32)
        ages = np.array(
            [max(0, int(current_epoch) - int(self.entries[i].epoch)) for i in indices],
            dtype=np.float32,
        )
        return np.exp(-self.age_decay * ages).astype(np.float32)

    def retrieve_prior(self, omega, env_ctx=None, action_dim=None, current_epoch=None):
        dim = self.action_dim if action_dim is None else int(action_dim)
        uniform = np.full(dim, 1.0 / dim, dtype=np.float32)
        if dim != self.action_dim or not self.entries:
            return uniform.copy()

        omega_arr = np.asarray(omega, dtype=np.float32)
        query = self._build_key(omega_arr, env_ctx)
        keys = np.stack([e.key for e in self.entries]).astype(np.float32)
        dists = np.linalg.norm(keys - query, axis=1)
        k = min(self.top_k, len(self.entries))
        top_idx = np.argpartition(dists, k - 1)[:k]
        sim = np.exp(-(dists[top_idx] ** 2) / max(self.sigma ** 2, 1e-12)).astype(np.float32)

        max_sim = float(sim.max()) if sim.size else 0.0
        if max_sim < self.conf_threshold:
            self.log.append(dict(omega=tuple(omega_arr), hit=0, conf=max_sim, n_entries=len(self.entries), gated=1))
            return uniform.copy()

        local = [self.entries[int(i)] for i in top_idx]
        nondom = []
        for cand in local:
            if not any(other is not cand and self._dominates(other, cand) for other in local):
                nondom.append(cand)
        if not nondom:
            nondom = local

        best = max(nondom, key=lambda e: self._scalar_score(e, omega_arr))
        prior = np.clip(best.prior.astype(np.float32), 0.0, None)
        s = float(prior.sum())
        if s < 1e-8:
            return uniform.copy()
        prior = prior / s
        self.log.append(dict(omega=tuple(omega_arr), hit=1, conf=max_sim, n_entries=len(self.entries), gated=0))
        return prior.astype(np.float32)

    def stats(self):
        if not self.log:
            return dict(hit_rate=0.0, gate_rate=0.0, mean_conf=0.0, n_entries=len(self.entries), n_calls=0)
        hits = np.array([x["hit"] for x in self.log], dtype=np.float32)
        gates = np.array([x["gated"] for x in self.log], dtype=np.float32)
        confs = np.array([x["conf"] for x in self.log], dtype=np.float32)
        return dict(
            hit_rate=float(hits.mean()),
            gate_rate=float(gates.mean()),
            mean_conf=float(confs.mean()),
            n_entries=len(self.entries),
            n_calls=len(self.log),
        )
```

- [ ] **Step 6: 实现保存/加载/日志**

```python
    def save_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(
                entries=self.entries,
                log=self.log,
                update_count=self.update_count,
                config=dict(
                    action_dim=self.action_dim,
                    n_omega_bins=self.n_omega_bins,
                    n_ctx_bins=self.n_ctx_bins,
                    max_entries=self.max_entries,
                    bucket_capacity=self.bucket_capacity,
                    top_k=self.top_k,
                    sigma=self.sigma,
                    conf_threshold=self.conf_threshold,
                    warmup_epochs=self.warmup_epochs,
                    age_decay=self.age_decay,
                    reward_gate_k=self.reward_gate_k,
                    reward_window=self.reward_window,
                    alpha_T=self.alpha_T,
                    alpha_E=self.alpha_E,
                ),
                reward_history=list(self._reward_history),
                delay_rms=self._delay_rms,
                energy_rms=self._energy_rms,
            ), f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.entries = d.get("entries", [])
        self.log = d.get("log", [])
        self.update_count = d.get("update_count", {})
        self._delay_rms = float(d.get("delay_rms", 1.0))
        self._energy_rms = float(d.get("energy_rms", 1.0))
        self._reward_history = collections.deque(d.get("reward_history", []), maxlen=self.reward_window)

    def save_log(self, path):
        if not self.log:
            return
        rows = []
        for x in self.log:
            rows.append([
                float(x["omega"][0]),
                float(x["omega"][1]),
                int(x["hit"]),
                float(x["conf"]),
                int(x["n_entries"]),
                int(x["gated"]),
            ])
        np.savetxt(
            path,
            np.asarray(rows, dtype=float),
            fmt="%.6f",
            header="omega0 omega1 hit conf n_entries gated",
            comments="",
        )

    def save_update_count(self, path):
        if not self.update_count:
            return
        rows = []
        for key, count in sorted(self.update_count.items()):
            rows.append(list(key) + [count])
        np.savetxt(
            path,
            np.asarray(rows, dtype=int),
            fmt="%d",
            header="omega_bin E_bin f_bin tran_bin update_count",
            comments="",
        )

    def save_archive(self, path):
        if not self.entries:
            return
        rows = []
        for e in self.entries:
            rows.append([
                e.omega_bin,
                e.ctx_bin[0],
                e.ctx_bin[1],
                e.ctx_bin[2],
                e.delay,
                e.energy,
                e.epoch,
            ])
        np.savetxt(
            path,
            np.asarray(rows, dtype=float),
            fmt="%.6f",
            header="omega_bin E_bin f_bin tran_bin delay energy epoch",
            comments="",
        )
```

- [ ] **Step 7: 运行单元测试**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
pytest tests\test_pareto_memory.py -q
```

Expected:

```text
3 passed
```

### Task 3: 接入 mofd_main.py

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`

- [ ] **Step 1: 增加 import**

在 import 区域加入：

```python
from pareto_memory import ParetoMemoryBuffer
```

- [ ] **Step 2: 在 run_single_seed 中选择 buffer 类型**

替换当前 `omega_buf` 初始化逻辑为：

```python
    if cfg.get("use_pareto_memory", False):
        omega_buf = ParetoMemoryBuffer(
            action_dim=env.action_dim,
            n_omega_bins=cfg.get("pmem_n_omega_bins", cfg["final_eval_n_pref"]),
            n_ctx_bins=cfg.get("pmem_n_ctx_bins", 4),
            max_entries=cfg.get("pmem_max_entries", 512),
            bucket_capacity=cfg.get("pmem_bucket_capacity", 16),
            top_k=cfg.get("pmem_top_k", 20),
            sigma=cfg.get("pmem_sigma", 0.3),
            conf_threshold=cfg.get("pmem_conf_threshold", 0.4),
            warmup_epochs=cfg.get("pmem_warmup_epochs", 5),
            age_decay=cfg.get("pmem_age_decay", 0.05),
            reward_gate_k=cfg.get("pmem_reward_gate_k", 1.0),
            reward_window=cfg.get("pmem_reward_window", 50),
            alpha_T=cfg["alpha_T"],
            alpha_E=cfg["alpha_E"],
        )
    elif cfg.get("use_omega_buffer", True):
        omega_buf = OmegaLatentBuffer(
            decay=cfg.get("obuf_decay", 0.5),
            retrieve_noise=cfg.get("obuf_noise", 0.05),
            n_bins=cfg["final_eval_n_pref"],
            sigma=cfg.get("obuf_sigma", 0.3),
            conf_threshold=cfg.get("obuf_conf_threshold", 0.4),
            max_entries=cfg.get("obuf_max_entries", 500),
            top_k=cfg.get("obuf_top_k", 20),
            warmup_epochs=cfg.get("obuf_warmup_epochs", 5),
            age_decay=cfg.get("obuf_age_decay", 0.05),
            reward_gate_k=cfg.get("obuf_reward_gate_k", 1.0),
            reward_window=cfg.get("obuf_reward_window", 50),
        )
    else:
        omega_buf = None
```

- [ ] **Step 3: episode 后 update 时传入 delay/energy**

把当前 episode 结束后的 buffer update 改成兼容两种 buffer：

```python
            if omega_buf is not None:
                ep_reward = -(float(omega[0]) * cfg["alpha_T"] * d
                              + float(omega[1]) * cfg["alpha_E"] * e)
                if isinstance(omega_buf, ParetoMemoryBuffer):
                    omega_buf.update(
                        omega,
                        latent_slice,
                        env_ctx=env_ctx,
                        epoch=epoch,
                        reward=ep_reward,
                        delay=d,
                        energy=e,
                    )
                else:
                    omega_buf.update(
                        omega,
                        latent_slice,
                        env_ctx=env_ctx,
                        epoch=epoch,
                        reward=ep_reward,
                    )
```

- [ ] **Step 4: 保存 Pareto memory 专属日志**

在保存 buffer 日志的位置加入：

```python
    if isinstance(omega_buf, ParetoMemoryBuffer):
        omega_buf.save_log(os.path.join(results_dir, f"{prefix}_pmem_log_seed{seed}.csv"))
        omega_buf.save_update_count(os.path.join(results_dir, f"{prefix}_pmem_updates_seed{seed}.csv"))
        omega_buf.save_archive(os.path.join(results_dir, f"{prefix}_pmem_archive_seed{seed}.csv"))
    elif omega_buf is not None:
        omega_buf.save_log(os.path.join(results_dir, f"{prefix}_obuf_log_seed{seed}.csv"))
        omega_buf.save_update_count(os.path.join(results_dir, f"{prefix}_obuf_updates_seed{seed}.csv"))
```

- [ ] **Step 5: checkpoint 保存文件名区分**

在 checkpoint 保存处改成：

```python
    if isinstance(omega_buf, ParetoMemoryBuffer):
        omega_buf.save_pickle(os.path.join(ckpt_dir, "pareto_memory.pkl"))
    elif omega_buf is not None:
        omega_buf.save_pickle(os.path.join(ckpt_dir, "omega_buf.pkl"))
```

- [ ] **Step 6: 运行导入级 smoke test**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python -c "import mofd_main; import pareto_memory; print('ok')"
```

Expected:

```text
ok
```

### Task 4: 接入 checkpoint 加载

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`

- [ ] **Step 1: 修改 load_agent_from_ckpt**

在加载 `omega_buf.pkl` 前先检查 `pareto_memory.pkl`：

```python
    pareto_path = os.path.join(ckpt_dir, "pareto_memory.pkl")
    obuf_path = os.path.join(ckpt_dir, "omega_buf.pkl")
    if os.path.exists(pareto_path):
        omega_buf = ParetoMemoryBuffer(
            action_dim=env.action_dim,
            n_omega_bins=cfg.get("pmem_n_omega_bins", cfg.get("final_eval_n_pref", 21)),
            n_ctx_bins=cfg.get("pmem_n_ctx_bins", 4),
            max_entries=cfg.get("pmem_max_entries", 512),
            bucket_capacity=cfg.get("pmem_bucket_capacity", 16),
            top_k=cfg.get("pmem_top_k", 20),
            sigma=cfg.get("pmem_sigma", 0.3),
            conf_threshold=cfg.get("pmem_conf_threshold", 0.4),
            warmup_epochs=cfg.get("pmem_warmup_epochs", 5),
            age_decay=cfg.get("pmem_age_decay", 0.05),
            reward_gate_k=cfg.get("pmem_reward_gate_k", 1.0),
            reward_window=cfg.get("pmem_reward_window", 50),
            alpha_T=cfg.get("alpha_T", 1.0),
            alpha_E=cfg.get("alpha_E", 0.25),
        )
        omega_buf.load_pickle(pareto_path)
        print(f"[ckpt] pareto_memory restored: {len(omega_buf.entries)} entries")
    elif os.path.exists(obuf_path):
        omega_buf = OmegaLatentBuffer(
            decay=cfg.get("obuf_decay", 0.5),
            retrieve_noise=cfg.get("obuf_noise", 0.05),
            n_bins=cfg.get("final_eval_n_pref", 21),
            sigma=cfg.get("obuf_sigma", 0.3),
            conf_threshold=cfg.get("obuf_conf_threshold", 0.4),
            max_entries=cfg.get("obuf_max_entries", 500),
            top_k=cfg.get("obuf_top_k", 20),
            warmup_epochs=cfg.get("obuf_warmup_epochs", 5),
            age_decay=cfg.get("obuf_age_decay", 0.05),
            reward_gate_k=cfg.get("obuf_reward_gate_k", 1.0),
            reward_window=cfg.get("obuf_reward_window", 50),
        )
        omega_buf.load_pickle(obuf_path)
        print(f"[ckpt] omega_buf restored: {len(omega_buf.entries)} entries")
```

这个分支保留当前原参数块，不改变旧 checkpoint 行为。

- [ ] **Step 2: 运行加载级 smoke test**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python -c "from pareto_memory import ParetoMemoryBuffer; b=ParetoMemoryBuffer(action_dim=6); print(b.stats())"
```

Expected:

```text
{'hit_rate': 0.0, 'gate_rate': 0.0, 'mean_conf': 0.0, 'n_entries': 0, 'n_calls': 0}
```

### Task 5: 配置和运行脚本

**Files:**
- Modify: `D:\python_project\实验版5\FDEdge-main\mofd_main.py`
- Create: `D:\python_project\实验版5\FDEdge-main\run_ablation_pareto_memory.py`

- [ ] **Step 1: 在默认 cfg 中加入 Pareto memory 参数**

加入：

```python
        use_pareto_memory=False,
        pmem_n_omega_bins=21,
        pmem_n_ctx_bins=4,
        pmem_max_entries=512,
        pmem_bucket_capacity=16,
        pmem_top_k=20,
        pmem_sigma=0.3,
        pmem_conf_threshold=0.4,
        pmem_warmup_epochs=5,
        pmem_age_decay=0.05,
        pmem_reward_gate_k=1.0,
        pmem_reward_window=50,
```

- [ ] **Step 2: 新建三组消融运行脚本**

`run_ablation_pareto_memory.py` 的核心配置直接调用当前项目已有的 `mofd_main.main(cfg_override=...)`：

```python
import mofd_main


def main():
    variants = [
        ("no_buffer", dict(use_omega_buffer=False, use_pareto_memory=False)),
        ("omega_buffer", dict(use_omega_buffer=True, use_pareto_memory=False)),
        ("pareto_memory", dict(use_omega_buffer=False, use_pareto_memory=True)),
    ]

    for name, override in variants:
        cfg_override = dict(
            seeds=[0, 1, 2],
            final_eval_n_pref=21,
            file_prefix=f"abl_{name}",
        )
        cfg_override.update(override)
        print(f"[ablation] running {name}")
        mofd_main.main(cfg_override=cfg_override)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行单 seed 快速实验**

Run:

```powershell
cd D:\python_project\实验版5\FDEdge-main
python run_ablation_pareto_memory.py
```

Expected:

```text
[ablation] running no_buffer
[ablation] running omega_buffer
[ablation] running pareto_memory
```

并在 `results` 下生成：

```text
abl_no_buffer_*
abl_omega_buffer_*
abl_pareto_memory_*
```

---

## 6. 实验方案

### 6.1 主实验

对比：

```text
C0: no buffer
C1: current OmegaLatentBuffer
C2: ParetoMemoryCandidate
```

每组至少 `3 seeds`，最终报告：

```text
HV mean/std
delay_min mean/std
energy_min mean/std
energy spread
Pareto non-dominated point count
pick ratio: feedback / pareto_memory / random
archive hit_rate / gate_rate / n_entries
```

判断标准：

```text
ParetoMemoryCandidate 的 HV 不低于 current OmegaLatentBuffer。
energy_min 接近或优于 feedback 单源优势。
delay_min 不明显劣于 random 单源优势。
中间 ω 的 Pareto 点更平滑，energy spread 不塌缩。
```

### 6.2 必做消融

只做 Pareto memory 相关消融，避免与 PCR / clean critic 混在一起看不清贡献。

```text
A. current OmegaLatentBuffer
B. ParetoMemory without dominance pruning
C. ParetoMemory with local dominance pruning
D. ParetoMemory with local dominance pruning + age decay
```

如果 C 明显好于 B，才能说明 Pareto dominance 本身有贡献，而不是普通 memory 容量变化带来的结果。

### 6.3 和 clean vector critic 的组合实验

Pareto memory 本身依赖历史真实 delay/energy 写入，但最终候选选择依赖 critic。因此建议后续组合：

```text
V5-current + ParetoMemory
clean-vector-critic + ParetoMemory
clean-vector-critic + PCR + ParetoMemory
```

这组实验用于论文最终模型，不用于证明 Pareto memory 单模块贡献。

---

## 7. 论文表述建议

推荐写法：

```text
We introduce a Pareto archive-guided candidate retrieval module. Instead of treating all historical actions as useful priors, the archive keeps locally non-dominated action priors under similar preference and system context. The retrieved prior is used only as one candidate initialization, while the final action remains selected by the preference-conditioned vector critic.
```

中文逻辑：

```text
我们不说某个候选源天然适合某个目标。
我们只说历史中真实表现为非支配的动作更值得作为候选先验。
最终动作仍由当前 state、ω 和 vector critic 决定。
```

不要写成：

```text
Pareto memory enables drift recovery.
Pareto memory guarantees optimal delay-energy tradeoff.
Pareto memory replaces critic decision making.
```

更稳的贡献定位：

```text
Pareto memory is a candidate-quality module, not a standalone adaptation mechanism.
```

---

## 8. 风险和保护

### 风险 1: Archive 过时

如果 workload/channel/server context 变化，旧动作可能不再好。

保护：

```text
使用 context bucket。
使用 age decay。
最近邻相似度低时返回 uniform。
最终仍由 critic 对候选再评分。
```

### 风险 2: 单 episode 指标噪声大

某个动作可能只是偶然在某个 episode 表现好。

保护：

```text
warmup 后再写入。
reward gate 过滤明显差 episode。
每个 bucket 保留多个非支配 entry，而不是只保留一个。
多 seed 验证。
```

### 风险 3: Archive 塌缩到少数中间点

如果只按 scalar score 留存，低延迟端和低能耗端会被删掉。

保护：

```text
每个 bucket 强制保留 delay 最低的 2 个和 energy 最低的 2 个。
其余位置再按 scalar score 保留。
```

### 风险 4: 计算复杂度上升

本方案不增加 actor 候选数量，只把普通 `prior` 换成 Pareto archive prior。检索复杂度主要是 archive top-k 距离计算：

```text
O(N_archive * key_dim)
```

`N_archive <= 512`，`key_dim=5`，相对 actor/critic 前向很小。

---

## 9. 验收标准

功能验收：

```text
pytest tests\test_pareto_memory.py -q 通过。
use_pareto_memory=False 时旧 V5 行为不变。
use_pareto_memory=True 时 results 下生成 pmem 日志和 archive 文件。
Pareto memory retrieve_prior 返回合法概率分布。
```

实验验收：

```text
3 seeds 下 ParetoMemoryCandidate 的 HV 平均值不低于 current OmegaLatentBuffer。
energy_min 不明显差于 current best energy-oriented setting。
delay_min 不明显差于 current random-oriented setting。
archive hit_rate 不能长期接近 0，否则说明检索没有发挥作用。
gate_rate 不能长期接近 1，否则说明 context/ω key 太严格。
```

论文验收：

```text
只把 Pareto memory 写成 candidate-quality enhancement。
不把它包装成 drift recovery 或 memory-conditioned policy。
不声称 feedback/random/prior 有固定目标偏置。
```

---

## 10. 推荐执行顺序

```text
1. 先实现 ParetoMemoryBuffer 和单元测试。
2. 接入 mofd_main.py，但保持 use_pareto_memory=False 为默认。
3. 单 seed smoke run，确认不报错且 archive 有写入。
4. 跑 3 seeds 消融：no-buffer / omega-buffer / pareto-memory。
5. 如果 ParetoMemory 单模块有效，再和 clean vector critic、PCR 组合。
```

不要一开始同时改 actor 结构、PCR、critic target 和 Pareto memory。否则结果变好也难以归因。

---

## 11. Self-Review

Spec coverage:

```text
已覆盖动机、数据依据、模块接口、写入规则、检索规则、V5 接入点、测试、实验和论文表述边界。
```

Placeholder scan:

```text
文档没有留下未定义参数；所有建议参数都有默认值；所有核心函数都有明确输入输出。
```

Type consistency:

```text
ParetoArchiveEntry.prior 为 np.ndarray[Emax]。
ParetoMemoryBuffer.retrieve_prior 返回 np.ndarray[Emax]。
mofd_main.py 继续把 prior_latent 传给 agent.take_action，和当前 V5 接口一致。
```
