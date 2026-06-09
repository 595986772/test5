"""
MOFD Main
=========
端到端训练并评估 G-FDEdge (Multi-Objective Feedback Diffusion) 调度方法。

训练流程：
  每个 epoch 采样 n_prefs_per_epoch 个上下文 (E, f_E, ω)，
  每个上下文跑一个 episode (T 个时隙) 并在线更新网络；
  每个 epoch 结束后在固定测试偏好集上评估 Pareto 前沿与超体积 (HV)。

对比基线：
  Random、Round-Robin、Greedy-Min-Delay。

输出：
  results/mofd_*.csv   逐 epoch 的 HV / 延迟 / 能耗曲线
  results/mofd_pareto_*.csv   终态 Pareto 前沿
  results/mofd_pareto_front.png
  results/mofd_training_curve.png
"""
import os
# Windows 上 torch/numpy(MKL) 可能各自链接一份 libiomp5md.dll，
# 引发 "OMP: Error #15" 冲突。必须在 import torch/numpy 之前设置。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# 同时限制 OpenMP 线程数，避免 MKL 与 OMP 线程池互相踩踏
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import shutil
import time
import collections
from datetime import datetime
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mofd_environment import MOFDEnvironment
from mofd_v5 import MOFD_SAC_V5, ReplayBuffer as ReplayBufferV5
from mofd_v8 import MOFD_SAC_V8
from mofd_pareto_memory import ParetoMemoryBuffer
from helpers import build_preference_set, hypervolume_2d, pareto_front_2d, SHARED_EVAL_SEED_OFFSET
from task_generator import make_task_generator, TaskGenerator, RandomTaskGenerator


# ------------------------------------------------------------
# 任务生成 (解耦到 task_generator.py, 支持 random / poisson / trace)
# ------------------------------------------------------------
# 模块级全局 generator: 各 main 脚本在 main() 开头按 cfg 调用
# set_task_generator(...) 覆盖, 基线脚本无需改动 sample_tasks 调用点.
_TASK_GENERATOR: TaskGenerator = RandomTaskGenerator()


def set_task_generator(gen):
    """由各 main 在启动时调用, 切换全局任务到达模式 (random/poisson/trace)."""
    global _TASK_GENERATOR
    assert isinstance(gen, TaskGenerator), f'expected TaskGenerator, got {type(gen)}'
    _TASK_GENERATOR = gen
    print(f'[task_generator] active mode = {gen.name} ({gen})')


def get_task_generator():
    return _TASK_GENERATOR


def sample_tasks(env, rng):
    """向后兼容接口: 委派给当前全局 generator."""
    return _TASK_GENERATOR.sample(env, rng)


# ------------------------------------------------------------
# Helper: episode-level 状态上下文向量 (供 buffer key 使用)
# ------------------------------------------------------------
def make_env_ctx(E, f_E, tran_rate, Emax=6, f_max=40.0, tran_max=40.0):
    """从 episode-level 配置抽取 3 维归一化 context: [E_norm, f_E_norm, tran_norm].

    Why: 不同 (E, f_E, tran_rate) 下策略产出的 latent 结构差异显著, 仅用 ω 做 key
         会让 buffer EMA 把不同 config 的 latent 混在一起, 检索结果失配.
    """
    e_n = float(E) / max(Emax, 1)
    f_arr = np.asarray(f_E, dtype=np.float32)
    f_n = float(f_arr.mean()) / max(f_max, 1e-6) if f_arr.size else 0.5
    t_arr = np.asarray(tran_rate, dtype=np.float32)
    t_n = float(t_arr.mean()) / max(tran_max, 1e-6) if t_arr.size else 0.5
    return np.array([e_n, f_n, t_n], dtype=np.float32)


# ------------------------------------------------------------
# Context-Aware Latent Buffer (R3-B + 置信门)
# ------------------------------------------------------------
class OmegaLatentBuffer:
    """带 episode 上下文的 attention-pooling latent buffer + confidence gate.

    Key:    [ω₀, ω₁, E_norm, f_E_norm, tran_rate_norm]  (5-dim)
    Pool:   top-K Gaussian-kernel attention (避免 N 全量距离计算)
    Gate:   max_w / sum_w < conf_threshold → 退回 randn (避免 buffer 误伤)

    解决问题:
      R3 (state-blind): 现在 key 含 episode config, 不同 config 的 latent 分开存
      slot 0 退化:      置信门让 buffer 在无相似训练样本时静音, 等价于 C0 randn

    向后兼容:
      decay / n_bins 仅作 dummy 参数保留, 不影响接口
    """
    def __init__(self, decay=0.5, retrieve_noise=0.05, n_bins=21,
                 sigma=0.3, conf_threshold=0.4, max_entries=500, top_k=20,
                 warmup_epochs=5, age_decay=0.05,
                 reward_gate_k=1.0, reward_window=50):
        # 主参数
        # entries: list of (key_vec [5], latent ndarray, epoch int)
        # epoch=0 兼容旧 pickle (load_pickle 时 2-tuple → 补 0)
        self.entries = []
        self.retrieve_noise = retrieve_noise
        self.sigma = sigma
        self.conf_threshold = conf_threshold
        self.max_entries = max_entries
        self.top_k = top_k
        # #2 Tier-1: warmup skip + age decay
        # warmup_epochs <= 0 → 关闭 warmup 跳过
        # age_decay <= 0 或 current_epoch=None → 关闭 age decay
        self.warmup_epochs = int(warmup_epochs)
        self.age_decay = float(age_decay)
        # #2 Tier-2: reward-gated write
        # reward_gate_k <= 0 → 关闭 reward gate
        self.reward_gate_k = float(reward_gate_k)
        self.reward_window = int(reward_window)
        self._reward_history = collections.deque(maxlen=self.reward_window)
        self.log = []
        # 兼容字段 (旧外部代码可能还会读)
        self.decay = decay
        self.n_bins = n_bins
        self.update_count = {}   # (b0, b1) -> int, 仅用于 save_update_count 兼容

    @staticmethod
    def _build_key(omega, env_ctx):
        if env_ctx is None:
            env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return np.concatenate([np.asarray(omega, dtype=np.float32),
                                np.asarray(env_ctx, dtype=np.float32)])

    def _bin_omega(self, omega, n_bins=21):
        b = int(round(float(omega[0]) * (n_bins - 1)))
        b = max(0, min(n_bins - 1, b))
        return (b, n_bins - 1 - b)

    def update(self, omega, latent_slice, env_ctx=None,
               epoch=None, reward=None):
        """
        epoch / reward 任一为 None 则对应 gate 关闭.
          - epoch < warmup_epochs (>0)        → 跳过写入 (#2 Tier-1)
          - reward 落入 running mean - k*std  → 跳过写入 (#2 Tier-2)
        entry 写成 3-tuple (key, latent, epoch); epoch=None 时记 0.
        """
        # #2 Tier-1: warmup 期不写
        if (epoch is not None and self.warmup_epochs > 0
                and int(epoch) < self.warmup_epochs):
            return
        # #2 Tier-2: reward gate
        if reward is not None and self.reward_gate_k > 0:
            r = float(reward)
            self._reward_history.append(r)
            if len(self._reward_history) >= 10:
                rh = np.asarray(self._reward_history, dtype=np.float32)
                if r < rh.mean() - self.reward_gate_k * rh.std():
                    return
        key = self._build_key(omega, env_ctx)
        ep_int = int(epoch) if epoch is not None else 0
        self.entries.append((key, latent_slice.astype(np.float32).copy(), ep_int))
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        b = self._bin_omega(omega)
        self.update_count[b] = self.update_count.get(b, 0) + 1

    def save_pickle(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(
                entries=self.entries,
                log=self.log,
                update_count=self.update_count,
                retrieve_noise=self.retrieve_noise,
                sigma=self.sigma,
                conf_threshold=self.conf_threshold,
                max_entries=self.max_entries,
                top_k=self.top_k,
                decay=self.decay, n_bins=self.n_bins,
                warmup_epochs=self.warmup_epochs,
                age_decay=self.age_decay,
                reward_gate_k=self.reward_gate_k,
                reward_window=self.reward_window,
                reward_history=list(self._reward_history),
            ), f)

    def load_pickle(self, path):
        import pickle
        with open(path, 'rb') as f:
            d = pickle.load(f)
        raw_entries = d.get('entries', [])
        # 兼容旧 pickle (2-tuple) → 补 epoch=0
        normalized = []
        for e in raw_entries:
            if len(e) == 2:
                normalized.append((e[0], e[1], 0))
            else:
                normalized.append(tuple(e))
        self.entries = normalized
        self.log = d.get('log', [])
        self.update_count = d.get('update_count', {})
        for k in ('retrieve_noise', 'sigma', 'conf_threshold',
                  'max_entries', 'top_k', 'decay', 'n_bins',
                  'warmup_epochs', 'age_decay',
                  'reward_gate_k', 'reward_window'):
            if k in d:
                setattr(self, k, d[k])
        rh = d.get('reward_history', [])
        self._reward_history = collections.deque(
            rh, maxlen=self.reward_window)

    def _age_weights(self, topk_idx, current_epoch):
        """#2 Tier-1: 年龄衰减权重. current_epoch=None 或 age_decay<=0 → 全 1."""
        if current_epoch is None or self.age_decay <= 0:
            return np.ones(len(topk_idx), dtype=np.float32)
        ages = np.array(
            [max(0, int(current_epoch) - int(self.entries[i][2]))
             for i in topk_idx], dtype=np.float32)
        return np.exp(-self.age_decay * ages).astype(np.float32)

    def retrieve_prior(self, omega, env_ctx=None, action_dim=6,
                       current_epoch=None):
        """返回 [action_dim] 的 episode-level 动作先验.

        与 retrieve 共享 attention pooling + 置信门, 但:
          - 输出聚合到单一 [Emax] (T, N 维取平均)
          - gated/empty 时返回均匀分布 (而非 randn) 作为中性先验
          - 不加 retrieve_noise (要稳定的先验信号)
          - current_epoch 提供时叠加 age decay (训练期; eval 期传 None 不衰减)
        """
        uniform = np.full(action_dim, 1.0 / action_dim, dtype=np.float32)
        if not self.entries:
            return uniform.copy()

        query = self._build_key(omega, env_ctx)
        keys = np.stack([e[0] for e in self.entries])
        dists = np.linalg.norm(keys - query, axis=1)
        K = min(self.top_k, len(self.entries))
        topk_idx = np.argpartition(dists, K - 1)[:K]
        sub_dists = dists[topk_idx]
        w_omega = np.exp(-(sub_dists ** 2) / max(self.sigma ** 2, 1e-12))
        w_age = self._age_weights(topk_idx, current_epoch)
        w = w_omega * w_age
        # 置信度仍按 ω-相似度判 (年龄不应影响门控)
        if w_omega.sum() < 1e-12 or float(w_omega.max()) < self.conf_threshold:
            return uniform.copy()
        if w.sum() < 1e-12:
            return uniform.copy()

        priors, weights = [], []
        for i, idx in enumerate(topk_idx):
            lat = self.entries[idx][1]
            if lat.ndim == 0 or lat.shape[-1] != action_dim:
                continue
            p = lat.reshape(-1, action_dim).mean(axis=0)
            priors.append(p)
            weights.append(w[i])
        if not priors:
            return uniform.copy()
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)
        priors = np.stack(priors).astype(np.float32)
        pooled = np.einsum('i,ij->j', weights, priors)
        pooled = np.clip(pooled, 0.0, None)
        s = pooled.sum()
        if s < 1e-8:
            return uniform.copy()
        return (pooled / s).astype(np.float32)

    def _cold_slice(self, omega, shape, env_ctx=None, current_epoch=None):
        """冷启/门控 fallback: 用 retrieve_prior 得到的 [Emax] 先验广播到 slice 形.

        替代原先的 randn fallback. 动机:
          H-MCSS 的 fb 候选输入若为 randn, 与 rd 候选 (also randn_like) 结构同构,
          导致 denoiser 在两路上输出几乎一致, fb 通道事实退化. 改用 prior 广播,
          fb 起点 = 平滑先验, cell 命中后被 probs 覆写 → 真正的 within-episode 反馈链.
        """
        shape = tuple(shape)
        action_dim = shape[-1]
        prior = self.retrieve_prior(omega, env_ctx=env_ctx,
                                    action_dim=action_dim,
                                    current_epoch=current_epoch)
        return np.broadcast_to(prior, shape).astype(np.float32).copy()

    def retrieve(self, omega, shape, rng, env_ctx=None, current_epoch=None):
        shape = tuple(shape)
        # 空 buffer / 空 attention → prior 广播 fallback (替代 randn)
        if not self.entries:
            self.log.append(dict(omega=tuple(omega), hit=0,
                                 nn_dist=float('nan'),
                                 n_entries=0, conf=0.0, gated=1))
            return self._cold_slice(omega, shape, env_ctx=env_ctx,
                                    current_epoch=current_epoch)

        query = self._build_key(omega, env_ctx)
        keys = np.stack([e[0] for e in self.entries])           # [N, 5]
        dists = np.linalg.norm(keys - query, axis=1)            # [N]

        # top-K 候选 (防 N 大时 attention 退化为均匀分布)
        K = min(self.top_k, len(self.entries))
        topk_idx = np.argpartition(dists, K - 1)[:K]
        sub_dists = dists[topk_idx]

        # Gaussian-kernel attention weights (raw, in (0, 1])
        w = np.exp(-(sub_dists ** 2) / max(self.sigma ** 2, 1e-12))
        w_sum = float(w.sum())
        # 置信度 = 最近邻的绝对相似度 = exp(-d_min² / σ²)
        # 解读: conf=1 表示完美命中, conf=0.4 表示距离 ≈ 0.96σ
        max_w_abs = float(w.max())
        if w_sum < 1e-12:
            self.log.append(dict(omega=tuple(omega), hit=0,
                                 nn_dist=float(sub_dists.min()),
                                 n_entries=len(self.entries),
                                 conf=0.0, gated=1))
            return self._cold_slice(omega, shape, env_ctx=env_ctx,
                                    current_epoch=current_epoch)

        # 置信门: 最近邻相似度太低 → 无足够近的训练样本 → 退回 prior 广播
        if max_w_abs < self.conf_threshold:
            self.log.append(dict(omega=tuple(omega), hit=0,
                                 nn_dist=float(sub_dists.min()),
                                 n_entries=len(self.entries),
                                 conf=max_w_abs, gated=1))
            return self._cold_slice(omega, shape, env_ctx=env_ctx,
                                    current_epoch=current_epoch)

        # 形状过滤 (eval 时 T/N/A 可能与训练略异)
        valid = [i for i in topk_idx.tolist()
                 if self.entries[i][1].shape == shape]
        if not valid:
            self.log.append(dict(omega=tuple(omega), hit=0,
                                 nn_dist=float(sub_dists.min()),
                                 n_entries=len(self.entries),
                                 conf=max_w_abs, gated=1))
            return self._cold_slice(omega, shape, env_ctx=env_ctx,
                                    current_epoch=current_epoch)

        # 在合法子集上重新归一化权重 + einsum 池化 (叠加 age decay)
        v_keys = np.stack([self.entries[i][0] for i in valid])
        v_dists = np.linalg.norm(v_keys - query, axis=1)
        w_v_omega = np.exp(-(v_dists ** 2) / max(self.sigma ** 2, 1e-12)).astype(np.float32)
        w_v_age = self._age_weights(valid, current_epoch)
        w_v = w_v_omega * w_v_age
        s = w_v.sum()
        if s < 1e-12:
            return self._cold_slice(omega, shape, env_ctx=env_ctx,
                                    current_epoch=current_epoch)
        w_v = w_v / s
        v_latents = np.stack([self.entries[i][1] for i in valid]).astype(np.float32)
        pooled = np.einsum('i,i...->...', w_v, v_latents)

        noise = rng.standard_normal(size=shape).astype(np.float32) * self.retrieve_noise
        self.log.append(dict(omega=tuple(omega), hit=1,
                             nn_dist=float(v_dists.min()),
                             n_entries=len(self.entries),
                             conf=max_w_abs, gated=0))
        return (pooled + noise).astype(np.float32)

    def stats(self):
        if not self.log:
            return dict(hit_rate=0.0, mean_nn_dist=0.0, n_entries=0,
                        n_calls=0, gate_rate=0.0, mean_conf=0.0)
        hits = np.array([l['hit'] for l in self.log], dtype=int)
        gates = np.array([l.get('gated', 0) for l in self.log], dtype=int)
        confs = np.array([l.get('conf', 0.0) for l in self.log], dtype=float)
        dists = [l['nn_dist'] for l in self.log
                 if l['hit'] == 1 and not np.isnan(l['nn_dist'])]
        return dict(
            hit_rate=float(hits.mean()),
            mean_nn_dist=float(np.mean(dists)) if dists else 0.0,
            n_entries=len(self.entries),
            n_calls=len(self.log),
            gate_rate=float(gates.mean()),
            mean_conf=float(confs.mean()),
        )

    def save_log(self, path):
        if not self.log:
            return
        rows = []
        for l in self.log:
            rows.append([
                float(l['omega'][0]), float(l['omega'][1]),
                int(l['hit']),
                float(l['nn_dist']) if not np.isnan(l['nn_dist']) else -1.0,
                int(l['n_entries']),
                float(l.get('conf', 0.0)),
                int(l.get('gated', 0)),
            ])
        np.savetxt(path, np.asarray(rows, dtype=float), fmt='%.6f',
                   header='omega0 omega1 hit nn_dist n_entries conf gated',
                   comments='')

    def save_update_count(self, path):
        if not self.update_count:
            return
        rows = []
        for (b0, b1), c in sorted(self.update_count.items()):
            rows.append([b0, b1, c])
        np.savetxt(path, np.asarray(rows, dtype=int), fmt='%d',
                   header='bin0 bin1 update_count', comments='')


# ------------------------------------------------------------
# 一个完整 episode
# (F1: latent_slice 改为 per-episode 创建, 不跨 episode 共享;
#      因此移除原 latent_cache + bucket + build_3candidates 机制.
#      V2/V3 的 FC-MCSS 仍然有效, 由模型内部 _build_candidates 在单一起点上加扰动构造.)
# ------------------------------------------------------------
def run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                latent_slice, stochastic=True, train_buffer=None,
                update_fn=None, alpha_T=1.0, alpha_E=0.25,
                batch_size=64, buffer_warmup=500,
                update_every=1, prior_latent=None,
                use_true_feedback=False):
    """
    latent_slice: 形如 [T, N_max, Emax] 的 ndarray, 就地更新, episode 内部的反馈历史.
    prior_latent: [Emax] 的 ndarray, episode 起点 buffer 给的持久先验, 不被覆盖.
                  None 时回退到均匀分布.
    use_true_feedback: True → 真·within-episode 反馈链, 决策 k 的去噪起点 = 决策 k-1 的
                  输出 probs (首个决策从 prior_latent 起); 修复"每个 (t,n) 读自己未写过的
                  latent_slice 格→反馈从不传递"的老问题. 默认 False = 原行为 (主项目不变).
    单 latent 路径: V1 直接走单链去噪; V2/V3 在 take_action 内部用 _build_candidates
    在该单一起点上加高斯扰动构造多候选, 再 critic 选优 (FC-MCSS).
    """
    env.reset_env(tasks, E, f_E, tran_rate, omega)

    if prior_latent is None:
        prior_latent = np.full(env.action_dim, 1.0 / env.action_dim, dtype=np.float32)
    prior_latent = np.asarray(prior_latent, dtype=np.float32)

    sum_delay, sum_energy, n_tasks = 0.0, 0.0, 0
    losses = []
    step_count = 0
    fb_start = prior_latent.copy()   # 真反馈链的滚动起点; 首个决策 = prior (flag 关时不用)

    for t in range(env.time_slots - 1):
        T_len = len(env.tasks_bit[t])
        for n in range(T_len):
            state = env.get_state(t, n)
            mask = env.get_valid_mask()
            # 去噪起点: 真反馈 → 上一步输出; 否则 → 原 latent_slice 格 (实际恒为初值)
            latent = fb_start.copy() if use_true_feedback else latent_slice[t, n].copy()
            action, probs = agent.take_action(state, latent, prior_latent, mask, stochastic=stochastic)
            # 反馈历史写回 (flag 关时供 buffer next_latent 用; flag 开时下一步起点直接用 probs)
            latent_slice[t, n] = probs.astype(np.float32)
            if use_true_feedback:
                fb_start = probs.astype(np.float32)        # 下一决策的起点 = 本次反馈

            r_T, r_E, delay, energy, real_action = env.step(t, n, action)
            r_scal = float(omega[0] * alpha_T * r_T + omega[1] * alpha_E * r_E)

            # 下一状态 / 下一反馈
            if n == T_len - 1:
                nt, nn = t + 1, 0
            else:
                nt, nn = t, n + 1
            if nt < env.time_slots and nn < len(env.tasks_bit[nt]):
                next_state = env.get_state(nt, nn)
                # 真反馈下, 下一决策起点 = 本次 probs (Bellman 一致); 否则读下一格
                next_latent = (probs.astype(np.float32) if use_true_feedback
                               else latent_slice[nt, nn].copy())
                next_mask = env.get_valid_mask()
            else:
                next_state = state
                next_latent = probs.astype(np.float32) if use_true_feedback else latent
                next_mask = mask

            if train_buffer is not None:
                # V5 ReplayBuffer 需要 r_vec=(r_T, r_E)
                reward_arg = ((float(r_T), float(r_E))
                              if isinstance(train_buffer, ReplayBufferV5)
                              else r_scal)
                train_buffer.add(state, real_action, latent, prior_latent, mask,
                                 reward_arg, next_state, next_latent, next_mask)

            sum_delay += delay
            sum_energy += energy
            n_tasks += 1
            step_count += 1

            if (update_fn is not None
                    and train_buffer is not None
                    and train_buffer.size() > buffer_warmup
                    and step_count % update_every == 0):
                batch = train_buffer.sample(batch_size)
                losses.append(update_fn(batch))

        env.update_proc_queues(t)

    avg_delay = sum_delay / max(n_tasks, 1)
    avg_energy = sum_energy / max(n_tasks, 1)
    return avg_delay, avg_energy, losses


# ------------------------------------------------------------
# 在固定偏好集上扫出 Pareto 前沿
# ------------------------------------------------------------
def evaluate_pareto(env, agent, n_pref=11, n_eval_epi=2,
                    alpha_T=1.0, alpha_E=0.25, seed=42,
                    fix_context=False, omega_buf=None, eval_use_prior=False,
                    gate_log=None, use_true_feedback=False):
    """
    对每个偏好 ω，跑 n_eval_epi 次 (默认随机上下文) 取平均，得到该偏好下 (avg_delay, avg_energy)。
    fix_context=True 时固定一个上下文，便于可视化单个 MEC 系统的 Pareto 前沿。

    eval_use_prior=True (需传 omega_buf): 去噪起点改用**训练同款** retrieve_prior, 消除
      训练/评估起点不一致; 命中失败(gate, 返回≈uniform)时退回 feedback 零起点。
      默认 False = 原行为 (起点全零)。gate_log 给个 list 则逐次记 0/1 命中失败标记。
    """
    prefs = build_preference_set(n_pref)
    points = []
    A = env.action_dim
    uni = np.full(A, 1.0 / A, dtype=np.float32)
    # 预采一个上下文 (只用于 fix_context=True)
    fixed_E, fixed_f_E, fixed_tr, _ = env.sample_context(np.random.default_rng(seed))
    for omega in prefs:
        # 关键: 每个 omega 评估前重置 rng, 让所有 omega 看到完全相同的 n_eval_epi 个环境
        # 否则 21 个 omega 各看不同环境, 环境方差盖过 omega 信号 → Pareto 不单调
        rng = np.random.default_rng(seed)
        delays, energies = [], []
        for _ in range(n_eval_epi):
            if fix_context:
                E, f_E, tran_rate = fixed_E, fixed_f_E, fixed_tr
            else:
                E, f_E, tran_rate, _ = env.sample_context(rng)
            tasks = sample_tasks(env, rng)
            if eval_use_prior and omega_buf is not None:
                # 训练同款 key (make_env_ctx 默认 f_max/tran_max, 与 run_single_seed 一致)
                env_ctx = make_env_ctx(E, f_E, tran_rate, Emax=env.Emax)
                start = omega_buf.retrieve_prior(omega, env_ctx=env_ctx,
                                                 action_dim=A, current_epoch=None)
                gated = bool(np.allclose(start, uni, atol=1e-6))
                if gated:                                   # 命中失败 → 退回 feedback 零起点
                    start = np.zeros(A, dtype=np.float32)
                if gate_log is not None:
                    gate_log.append(1 if gated else 0)
                latent = np.broadcast_to(
                    start, [env.time_slots, env.n_tasks_max, A]).astype(np.float32).copy()
                prior_arg = start.astype(np.float32)
            else:
                # 原行为: 起点全零 (可复现), prior 走 run_episode 默认 uniform
                latent = np.zeros([env.time_slots, env.n_tasks_max, A], dtype=np.float32)
                prior_arg = None
            d, e, _ = run_episode(env, agent, tasks, E, f_E, tran_rate, omega,
                                  latent, stochastic=False,
                                  alpha_T=alpha_T, alpha_E=alpha_E,
                                  prior_latent=prior_arg,
                                  use_true_feedback=use_true_feedback)
            delays.append(d); energies.append(e)
        points.append([float(np.mean(delays)), float(np.mean(energies))])
    return np.array(points)


# ------------------------------------------------------------
# 滑动平均: 前 (window-1) 个位置用累积均值，其余用移动窗口
# (启发式基线 Random / RR / GreedyDelay / GreedyEnergy 已解耦到
#  baselines/Heuristic/heuristic_main.py, 由 compare_hv.py 统一对比)
# ------------------------------------------------------------
def smooth(arr, window):
    arr = np.asarray(arr, dtype=float)
    if window is None or window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    body = np.convolve(arr, kernel, mode='valid')
    front = np.array([arr[:i + 1].mean() for i in range(window - 1)])
    return np.concatenate([front, body])


# ------------------------------------------------------------
# 用 Random 基线的 Pareto 点标定本 seed 的 HV 固定参考点
# (避免每 epoch 参考点漂移导致 HV 曲线不可比; 跨 seed 聚合需要)
# ------------------------------------------------------------
def compute_fixed_ref(env, seed, scale=1.3, n_pref=11):
    """用随机策略的 Pareto 点标定本 seed 的 HV 固定参考点.

    原本依赖已解耦到 baselines/Heuristic 的 _run_heuristic, 这里内联一个
    轻量的 random-action rollout, 不引入跨目录依赖.
    """
    rng = np.random.default_rng(seed)
    prefs = build_preference_set(n_pref)
    pts = []
    for omega in prefs:
        E, f_E, tran_rate, _ = env.sample_context(rng)
        tasks = sample_tasks(env, rng)
        env.reset_env(tasks, E, f_E, tran_rate, omega)
        sum_d, sum_e, n = 0.0, 0.0, 0
        for t in range(env.time_slots - 1):
            T_len = len(env.tasks_bit[t])
            for nn in range(T_len):
                action = int(rng.integers(0, E))
                _, _, d, e, _ = env.step(t, nn, action)
                sum_d += d; sum_e += e; n += 1
            env.update_proc_queues(t)
        pts.append([sum_d / max(n, 1), sum_e / max(n, 1)])
    pts = np.array(pts)
    return (float(pts[:, 0].max() * scale + 1e-3),
            float(pts[:, 1].max() * scale + 1e-3))


# ------------------------------------------------------------
# Checkpoint 加载: drift / shift 实验复用训练好的权重
# ------------------------------------------------------------
def load_agent_from_ckpt(ckpt_dir, cfg, env, device):
    """从 ckpt_dir 加载 actor/critic/log_alpha 到一个新 agent 实例 (V5).

    ckpt_dir 形如 'results/mofd_20260507_143022/ckpt_seed0',
    应包含 actor.pt / critic1.pt / critic2.pt / log_alpha.pt / meta.json.
    """
    meta_path = os.path.join(ckpt_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('state_dim') != env.state_dim:
            raise ValueError(
                f"ckpt state_dim={meta['state_dim']} != env.state_dim={env.state_dim}")
        if meta.get('Emax') != cfg['Emax']:
            raise ValueError(
                f"ckpt Emax={meta['Emax']} != cfg.Emax={cfg['Emax']}")

    AgentClass = MOFD_SAC_V8 if cfg.get('use_v8', False) else MOFD_SAC_V5
    agent = AgentClass(
        state_dim=env.state_dim, Emax=cfg['Emax'],
        hidden_dim=cfg['hidden_dim'],
        actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'],
        alpha=cfg['alpha_init'], alpha_lr=cfg['alpha_lr'],
        target_entropy=cfg['target_entropy'],
        tau=cfg['tau'], gamma=cfg['gamma'],
        denoising_steps=cfg['denoising_steps'],
        alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
        use_cor=cfg.get('use_cor', True),
        cor_lambda=cfg.get('cor_lambda', 0.1),
        cor_c=cfg.get('cor_c', 0.0),
        use_popart=cfg.get('use_popart', True),
        popart_beta=cfg.get('popart_beta', 0.01),
        device=device,
    )
    agent.actor.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'actor.pt'), map_location=device))
    agent.critic1.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'critic1.pt'), map_location=device))
    agent.critic2.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'critic2.pt'), map_location=device))
    agent.target1.load_state_dict(agent.critic1.state_dict())
    agent.target2.load_state_dict(agent.critic2.state_dict())
    la_path = os.path.join(ckpt_dir, 'log_alpha.pt')
    if os.path.exists(la_path):
        la = torch.load(la_path, map_location=device)
        with torch.no_grad():
            agent.log_alpha.copy_(la['log_alpha'].to(device))
    # V6 新增: 恢复 hypernet (若 ckpt 存在 + agent 支持)
    hn_path = os.path.join(ckpt_dir, 'hypernet.pt')
    if (os.path.exists(hn_path)
            and hasattr(agent, 'hypernet') and agent.hypernet is not None):
        agent.hypernet.load_state_dict(
            torch.load(hn_path, map_location=device))
        print(f'[ckpt] hypernet restored from {hn_path}')

    # 加载 ω-buffer (若存在); 没有则返回 None, 调用方按需新建
    omega_buf = None
    obuf_path = os.path.join(ckpt_dir, 'omega_buf.pkl')
    if os.path.exists(obuf_path):
        omega_buf = OmegaLatentBuffer(
            decay=cfg.get('obuf_decay', 0.5),
            retrieve_noise=cfg.get('obuf_noise', 0.05),
            n_bins=cfg.get('final_eval_n_pref', 21),
            sigma=cfg.get('obuf_sigma', 0.3),
            conf_threshold=cfg.get('obuf_conf_threshold', 0.4),
            max_entries=cfg.get('obuf_max_entries', 500),
            top_k=cfg.get('obuf_top_k', 20),
        )
        omega_buf.load_pickle(obuf_path)
        print(f'[ckpt] omega_buf restored: {len(omega_buf.entries)} entries')

    print(f'[ckpt] agent loaded from {ckpt_dir}')
    return agent, omega_buf


def load_run_from_ckpt(ckpt_dir, cfg, seed, device):
    """重建一个与 run_single_seed 返回结构兼容的 dict (agent + env + obuf + 训练曲线).

    用于 drift / shift 实验跳过训练直接评估.
    依赖: ckpt_dir 内有 actor.pt / critic*.pt / log_alpha.pt / logs.npz / (omega_buf.pkl).
    """
    env = MOFDEnvironment(
        Emax=cfg['Emax'], num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'], time_slots=cfg['time_slots'],
        f_range=cfg['f_range'], seed=seed,
    )
    agent, omega_buf = load_agent_from_ckpt(ckpt_dir, cfg, env, device)

    logs_path = os.path.join(ckpt_dir, 'logs.npz')
    if os.path.exists(logs_path):
        d = np.load(logs_path)
        log_hv = d['log_hv']; log_avg_delay = d['log_avg_delay']
        log_avg_energy = d['log_avg_energy']
        fixed_ref = tuple(d['fixed_ref'].tolist())
        pts_gfd = d['pts_gfd']
    else:
        n = int(cfg.get('num_epochs', 0))
        log_hv = np.zeros(n); log_avg_delay = np.zeros(n); log_avg_energy = np.zeros(n)
        fixed_ref = (1.0, 1.0); pts_gfd = np.zeros((0, 2))

    return dict(
        seed=seed,
        log_hv=log_hv, log_avg_delay=log_avg_delay, log_avg_energy=log_avg_energy,
        fixed_ref=fixed_ref, pts_gfd=pts_gfd,
        obuf_stats=omega_buf.stats() if omega_buf is not None else None,
        agent=agent, env=env, omega_buf=omega_buf,
    )


# ------------------------------------------------------------
# 单 seed 完整训练 + 最终 Pareto 评估
# ------------------------------------------------------------
def run_single_seed(cfg, seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MOFDEnvironment(
        Emax=cfg['Emax'],
        num_tasks_max=cfg['num_tasks_max'],
        bit_range=cfg['bit_range'],
        time_slots=cfg['time_slots'],
        f_range=cfg['f_range'],
        delay_scale=cfg.get('delay_scale', 0.05),
        energy_scale=cfg.get('energy_scale', 0.25),
        seed=seed,
    )
    AgentClass = MOFD_SAC_V8 if cfg.get('use_v8', False) else MOFD_SAC_V5
    agent = AgentClass(
        state_dim=env.state_dim, Emax=cfg['Emax'],
        hidden_dim=cfg['hidden_dim'],
        actor_lr=cfg['actor_lr'], critic_lr=cfg['critic_lr'],
        alpha=cfg['alpha_init'], alpha_lr=cfg['alpha_lr'],
        target_entropy=cfg['target_entropy'],
        tau=cfg['tau'], gamma=cfg['gamma'],
        denoising_steps=cfg['denoising_steps'],
        alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
        use_cor=cfg.get('use_cor', True),
        cor_lambda=cfg.get('cor_lambda', 0.1),
        cor_c=cfg.get('cor_c', 0.0),
        use_popart=cfg.get('use_popart', True),
        popart_beta=cfg.get('popart_beta', 0.01),
        device=device,
    )
    buf = ReplayBufferV5(cfg['buf_size'])
    cor_msg = (f"COR(λ={cfg.get('cor_lambda',0.1)},c={cfg.get('cor_c',0.0)})"
               if cfg.get('use_cor', True) else "no-COR")
    pa_msg = "PopArt-lite" if cfg.get('use_popart', True) else "no-PopArt"
    print(f"[seed {seed}] agent = {AgentClass.__name__} (vector-Q + {cor_msg} + {pa_msg})")

    # F1: 取消跨 episode 的 latent_cache, 改为 per-episode 创建 latent_slice
    rng = np.random.default_rng(seed + 1)

    # Path B: ω-NN Latent Warm-start Buffer; 关闭 (cfg.use_omega_buffer=False)
    # 即退化为 F1-only baseline (C0), 用于消融对照.
    # R3-B + 置信门: 接受 sigma / conf_threshold / max_entries / top_k 等新参数
    if cfg.get('use_pareto_memory', False):
        # 最小 Pareto 版: 非支配筛选 + 选单个好 prior (替代加权平均)
        omega_buf = ParetoMemoryBuffer(
            action_dim=env.action_dim,
            n_omega_bins=cfg.get('pmem_n_omega_bins', cfg['final_eval_n_pref']),
            n_ctx_bins=cfg.get('pmem_n_ctx_bins', 4),
            bucket_capacity=cfg.get('pmem_bucket_capacity', 16),
            max_entries=cfg.get('pmem_max_entries', 500),
            top_k=cfg.get('pmem_top_k', 20),
            sigma=cfg.get('pmem_sigma', 0.3),
            conf_threshold=cfg.get('pmem_conf_threshold', 0.4),
            warmup_epochs=cfg.get('pmem_warmup_epochs', 5),
            alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
        )
    elif cfg.get('use_omega_buffer', True):
        omega_buf = OmegaLatentBuffer(
            decay=cfg.get('obuf_decay', 0.5),
            retrieve_noise=cfg.get('obuf_noise', 0.05),
            n_bins=cfg['final_eval_n_pref'],
            sigma=cfg.get('obuf_sigma', 0.3),
            conf_threshold=cfg.get('obuf_conf_threshold', 0.4),
            max_entries=cfg.get('obuf_max_entries', 500),
            top_k=cfg.get('obuf_top_k', 20),
            # #2 Tier-1 + Tier-2 参数 (cfg 缺省时使用类内默认)
            warmup_epochs=cfg.get('obuf_warmup_epochs', 5),
            age_decay=cfg.get('obuf_age_decay', 0.05),
            reward_gate_k=cfg.get('obuf_reward_gate_k', 1.0),
            reward_window=cfg.get('obuf_reward_window', 50),
        )
    else:
        omega_buf = None

    fixed_ref = compute_fixed_ref(env, seed=seed + 7777)
    print(f"[seed {seed}] fixed HV ref = ({fixed_ref[0]:.3f}, {fixed_ref[1]:.3f})")
    print(f"[seed {seed}] omega_buf = {'ON' if omega_buf is not None else 'OFF (C0 baseline)'}")

    log_hv, log_avg_delay, log_avg_energy = [], [], []
    # ---- 训练健康度监控曲线 (per-epoch, 用于诊断 SAC + 通道归一化是否稳定) ----
    # c_loss: critic loss 均值        a_loss: actor loss 均值
    # alpha:  SAC 温度 exp(log_alpha)  H: actor 策略熵
    # sigma_T/sigma_E: PopArt 两通道 RMS 估计 (差异>5×→ PopArt 失效)
    log_c_loss, log_a_loss = [], []
    log_alpha_curve, log_H = [], []
    log_sigma_T, log_sigma_E = [], []
    # V6 新增: hypernet supervised loss + diversity reg signal
    log_hyper_loss, log_div_loss = [], []
    t0 = time.time()

    # 训练 ω 集合 (默认 = 评估 ω 集合, 21 个); 可由 cfg['train_omegas'] 覆盖
    # (例如 ω-shift 实验: 训练用 11 个 even-idx ω, 评估仍 21 个)
    train_omegas = cfg.get('train_omegas', None)
    if train_omegas is None:
        train_omegas = build_preference_set(cfg['final_eval_n_pref'])
    else:
        train_omegas = np.asarray(train_omegas, dtype=np.float32)
    print(f"[seed {seed}] train_omegas count = {len(train_omegas)}")
    latent_shape = [env.time_slots, env.n_tasks_max, env.action_dim]

    for epoch in range(cfg['num_epochs']):
        ep_delays, ep_energies, ep_losses = [], [], []
        for i in range(cfg['n_prefs_per_epoch']):
            # 轮询 21 个评估 ω, 不再随机连续采样 (epoch 间错开起点保证均衡)
            omega_fixed = train_omegas[(epoch * cfg['n_prefs_per_epoch'] + i) % len(train_omegas)]
            E, f_E, tran_rate, omega = env.sample_context(rng, fixed_omega=omega_fixed)
            tasks = sample_tasks(env, rng)
            # episode-level 上下文向量, 用作 buffer key 的状态分量
            env_ctx = make_env_ctx(E, f_E, tran_rate, Emax=cfg['Emax'])
            # Path B: 用 (ω, ctx)-attention buffer 检索 warm-start; 否则退化为 uniform 冷启动
            # #3 Tier-1: 不再用 buffer.retrieve 做 fb 候选; fb 与 pr 都来自 prior 广播,
            # 让 H-MCSS Q-net 学到的是 “fb=pr (光滑) vs rd (噪声)” 的二元结构, 避免
            # fb 自带 retrieve_noise 与 randn rd 同源, 使 pr 候选永远被吃掉.
            if omega_buf is not None:
                prior_latent = omega_buf.retrieve_prior(omega, env_ctx=env_ctx,
                                                        action_dim=env.action_dim,
                                                        current_epoch=epoch)
            else:
                # C0 baseline: 无 buffer, 用 uniform 先验
                prior_latent = np.full(env.action_dim, 1.0 / env.action_dim,
                                        dtype=np.float32)
            latent_slice = np.broadcast_to(
                prior_latent, latent_shape
            ).astype(np.float32).copy()
            d, e, losses = run_episode(
                env, agent, tasks, E, f_E, tran_rate, omega,
                latent_slice,
                stochastic=True, train_buffer=buf,
                update_fn=agent.update,
                alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
                batch_size=cfg['batch_size'], buffer_warmup=cfg['buffer_warmup'],
                update_every=cfg.get('update_every', 1),
                prior_latent=prior_latent,
                use_true_feedback=cfg.get('use_true_feedback', False),
            )
            # episode 结束后写入 buffer (latent_slice 已被 run_episode 就地写入 policy 输出)
            # #2 Tier-2: 传 epoch (warmup 跳过) + proxy reward (低分剧集不入库)
            if omega_buf is not None:
                ep_reward = -(float(omega[0]) * cfg['alpha_T'] * d
                              + float(omega[1]) * cfg['alpha_E'] * e)
                if isinstance(omega_buf, ParetoMemoryBuffer):
                    omega_buf.update(omega, latent_slice, env_ctx=env_ctx,
                                     epoch=epoch, reward=ep_reward, delay=d, energy=e)
                else:
                    omega_buf.update(omega, latent_slice, env_ctx=env_ctx,
                                     epoch=epoch, reward=ep_reward)
            ep_delays.append(d); ep_energies.append(e); ep_losses.extend(losses)

        pts = evaluate_pareto(env, agent,
                              n_pref=cfg['train_eval_n_pref'],
                              n_eval_epi=cfg['train_eval_n_epi'],
                              alpha_T=cfg['alpha_T'], alpha_E=cfg['alpha_E'],
                              seed=seed * 100000 + epoch,
                              omega_buf=omega_buf,
                              eval_use_prior=cfg.get('eval_use_prior', False),
                              use_true_feedback=cfg.get('use_true_feedback', False))
        hv = hypervolume_2d(pts, fixed_ref)
        log_hv.append(hv)
        log_avg_delay.append(float(np.mean(ep_delays)))
        log_avg_energy.append(float(np.mean(ep_energies)))

        # ---- 监控曲线: 聚合本 epoch 内所有 update step 的训练健康指标 ----
        if ep_losses:
            log_c_loss.append(float(np.mean([l['c_loss'] for l in ep_losses])))
            log_a_loss.append(float(np.mean([l['a_loss'] for l in ep_losses])))
            log_alpha_curve.append(float(np.mean([l['alpha'] for l in ep_losses])))
            log_H.append(float(np.mean([l['H'] for l in ep_losses])))
            log_sigma_T.append(float(np.mean([l.get('sigma_T', 1.0) for l in ep_losses])))
            log_sigma_E.append(float(np.mean([l.get('sigma_E', 1.0) for l in ep_losses])))
            log_hyper_loss.append(float(np.mean([l.get('hyper_loss', 0.0) for l in ep_losses])))
            log_div_loss.append(float(np.mean([l.get('div_loss', 0.0) for l in ep_losses])))
        else:
            # warmup 期 buffer 不足时无 update, 记 NaN 占位 (画图时跳过)
            log_c_loss.append(float('nan'))
            log_a_loss.append(float('nan'))
            log_alpha_curve.append(float('nan'))
            log_H.append(float('nan'))
            log_sigma_T.append(float('nan'))
            log_sigma_E.append(float('nan'))
            log_hyper_loss.append(float('nan'))
            log_div_loss.append(float('nan'))

        log_interval = max(1, cfg['num_epochs'] // 20)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            h_mean = float(np.mean([l['H'] for l in ep_losses])) if ep_losses else 0.0
            elapsed = time.time() - t0
            buf_msg = ''
            if omega_buf is not None:
                s = omega_buf.stats()
                buf_msg = (f" | obuf hit={s['hit_rate']:.2f} "
                           f"gate={s['gate_rate']:.2f} "
                           f"conf={s['mean_conf']:.2f} "
                           f"nn_d={s['mean_nn_dist']:.2f} N={s['n_entries']}")
            # 监控曲线: 同步打印 4 条诊断量, 便于实时观察 SAC 健康度
            mon_msg = ''
            if log_c_loss and not np.isnan(log_c_loss[-1]):
                mon_msg = (f" | c_loss={log_c_loss[-1]:.4f} a_loss={log_a_loss[-1]:.4f} "
                           f"α={log_alpha_curve[-1]:.4f} "
                           f"σ_T={log_sigma_T[-1]:.3f} σ_E={log_sigma_E[-1]:.3f}")
            print(f"[seed {seed} epoch {epoch + 1:03d}/{cfg['num_epochs']}] "
                  f"HV={hv:.4f} d={log_avg_delay[-1]:.4f} e={log_avg_energy[-1]:.4f} "
                  f"H(π)={h_mean:.3f} elapsed={elapsed:.1f}s{buf_msg}{mon_msg}")

    # --------- H-MCSS 三源候选胜出统计 (训练期累积) ---------
    if hasattr(agent, '_mcss_pick_count'):
        pc = list(agent._mcss_pick_count)
        tot = sum(pc) or 1
        print(f"[seed {seed}] H-MCSS picks (train) feedback/prior/random = "
              f"{pc[0]}/{pc[1]}/{pc[2]}  "
              f"({pc[0]/tot:.1%} / {pc[1]/tot:.1%} / {pc[2]/tot:.1%})")
        # 重置, 让后续 evaluate_pareto / drift 评估单独计数
        agent._mcss_pick_count = [0, 0, 0]

    # --------- 最终评估: 仅 G-FDEdge (启发式已解耦到 baselines/Heuristic) ---------
    pts_gfd = evaluate_pareto(env, agent, alpha_T=cfg['alpha_T'],
                              alpha_E=cfg['alpha_E'],
                              seed=seed * 100000 + SHARED_EVAL_SEED_OFFSET,
                              n_pref=cfg['final_eval_n_pref'],
                              n_eval_epi=cfg['final_eval_n_epi'],
                              omega_buf=omega_buf,
                              eval_use_prior=cfg.get('eval_use_prior', False),
                              use_true_feedback=cfg.get('use_true_feedback', False))

    if hasattr(agent, '_mcss_pick_count'):
        pc = list(agent._mcss_pick_count)
        tot = sum(pc) or 1
        print(f"[seed {seed}] H-MCSS picks (final eval) feedback/prior/random = "
              f"{pc[0]}/{pc[1]}/{pc[2]}  "
              f"({pc[0]/tot:.1%} / {pc[1]/tot:.1%} / {pc[2]/tot:.1%})")
        agent._mcss_pick_count = [0, 0, 0]

    # --------- 保存 ω-buffer 日志 (用于消融分析) ---------
    results_dir = cfg.get('results_dir', 'results')
    prefix = cfg.get('file_prefix', 'mofd')
    os.makedirs(results_dir, exist_ok=True)
    if omega_buf is not None:
        omega_buf.save_log(os.path.join(results_dir, f'{prefix}_obuf_log_seed{seed}.csv'))
        omega_buf.save_update_count(
            os.path.join(results_dir, f'{prefix}_obuf_updates_seed{seed}.csv'))
        s = omega_buf.stats()
        print(f"[seed {seed}] obuf final: entries={s['n_entries']} "
              f"hit_rate={s['hit_rate']:.3f} mean_nn_dist={s['mean_nn_dist']:.3f} "
              f"n_calls={s['n_calls']}")

    # --------- 保存 actor / critic / log_alpha 权重 ---------
    # drift / shift 实验可用 load_agent_from_ckpt(...) 直接加载, 跳过训练
    ckpt_dir = os.path.join(results_dir, f'ckpt_seed{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(agent.actor.state_dict(),   os.path.join(ckpt_dir, 'actor.pt'))
    torch.save(agent.critic1.state_dict(), os.path.join(ckpt_dir, 'critic1.pt'))
    torch.save(agent.critic2.state_dict(), os.path.join(ckpt_dir, 'critic2.pt'))
    torch.save({'log_alpha': agent.log_alpha.detach().cpu()},
               os.path.join(ckpt_dir, 'log_alpha.pt'))
    # V6 新增: 保存 hypernet (drift / shift 评估时需要)
    if hasattr(agent, 'hypernet') and agent.hypernet is not None:
        torch.save(agent.hypernet.state_dict(),
                   os.path.join(ckpt_dir, 'hypernet.pt'))
    # 保存 ω-buffer 状态 (entries, ω_keys, ctx_keys 等), drift 实验可继承 warm-start
    if omega_buf is not None:
        try:
            omega_buf.save_pickle(os.path.join(ckpt_dir, 'omega_buf.pkl'))
        except Exception as e:
            print(f"[seed {seed}] omega_buf save skipped: {e}")
    # 训练曲线 + 终态 Pareto 点也存进去, drift/shift 复用时可还原 run dict
    np.savez(
        os.path.join(ckpt_dir, 'logs.npz'),
        log_hv=np.asarray(log_hv, dtype=np.float64),
        log_avg_delay=np.asarray(log_avg_delay, dtype=np.float64),
        log_avg_energy=np.asarray(log_avg_energy, dtype=np.float64),
        log_c_loss=np.asarray(log_c_loss, dtype=np.float64),
        log_a_loss=np.asarray(log_a_loss, dtype=np.float64),
        log_alpha_curve=np.asarray(log_alpha_curve, dtype=np.float64),
        log_H=np.asarray(log_H, dtype=np.float64),
        log_sigma_T=np.asarray(log_sigma_T, dtype=np.float64),
        log_sigma_E=np.asarray(log_sigma_E, dtype=np.float64),
        log_hyper_loss=np.asarray(log_hyper_loss, dtype=np.float64),
        log_div_loss=np.asarray(log_div_loss, dtype=np.float64),
        fixed_ref=np.asarray(fixed_ref, dtype=np.float64),
        pts_gfd=np.asarray(pts_gfd, dtype=np.float64),
    )

    # ---- 单独导出监控 CSV (人可读, 调参时直接打开看) ----
    monitor_csv = os.path.join(results_dir, f'{prefix}_monitor_seed{seed}.csv')
    with open(monitor_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,c_loss,a_loss,alpha,H,sigma_T,sigma_E,hv,avg_delay,avg_energy\n')
        for i in range(len(log_hv)):
            f.write(f"{i},{log_c_loss[i]:.6f},{log_a_loss[i]:.6f},"
                    f"{log_alpha_curve[i]:.6f},{log_H[i]:.6f},"
                    f"{log_sigma_T[i]:.6f},{log_sigma_E[i]:.6f},"
                    f"{log_hv[i]:.6f},{log_avg_delay[i]:.6f},{log_avg_energy[i]:.6f}\n")
    print(f"[seed {seed}] monitor CSV: {monitor_csv}")
    # 关键 ckpt 元数据 (用于加载侧验证维度匹配)
    with open(os.path.join(ckpt_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'seed': int(seed),
            'state_dim': int(env.state_dim),
            'Emax': int(cfg['Emax']),
            'hidden_dim': int(cfg['hidden_dim']),
            'denoising_steps': int(cfg['denoising_steps']),
            'gamma': float(cfg['gamma']),
            'tau': float(cfg['tau']),
        }, f, indent=2, ensure_ascii=False)
    print(f"[seed {seed}] ckpt saved to {ckpt_dir}")

    return dict(
        seed=seed,
        log_hv=np.array(log_hv),
        log_avg_delay=np.array(log_avg_delay),
        log_avg_energy=np.array(log_avg_energy),
        log_c_loss=np.array(log_c_loss),
        log_a_loss=np.array(log_a_loss),
        log_alpha_curve=np.array(log_alpha_curve),
        log_H=np.array(log_H),
        log_sigma_T=np.array(log_sigma_T),
        log_sigma_E=np.array(log_sigma_E),
        log_hyper_loss=np.array(log_hyper_loss),
        log_div_loss=np.array(log_div_loss),
        fixed_ref=fixed_ref,
        pts_gfd=pts_gfd,
        obuf_stats=omega_buf.stats() if omega_buf is not None else None,
        # in-memory handles, 供 drift / 在线适应等下游实验复用
        agent=agent,
        env=env,
        omega_buf=omega_buf,
    )


# ------------------------------------------------------------
# 主流程: 多 seed 聚合 + 滑动平均 + 置信带
# ------------------------------------------------------------
def main(cfg_override=None):
    """cfg_override: 可选 dict, 训练前用 cfg.update() 覆盖默认 cfg (mofd_v2_main 等用)."""
    cfg = dict(
        # ---- 规模扩到 FDEdge 原文量级 ----
        Emax=6,
        num_tasks_max=50,        # was 8
        bit_range=(10, 40),
        time_slots=100,          # was 20
        f_range=(10, 40),
        num_epochs=100,           # V4 (envelope): K=4 时每 epoch ~12 min, 50 epochs ~10h
        n_prefs_per_epoch=8,
        # ---- 多 seed / 平滑 ----
        seeds=[0],
        smooth_window=5,
        # ---- 训练中评估 (轻量,每 epoch 一次) ----
        train_eval_n_pref=11,
        train_eval_n_epi=1,
        # ---- 最终评估 (高分辨率) ----
        final_eval_n_pref=21,
        final_eval_n_epi=3,
        # ---- 奖励通道归一化 (env 内对 r_T, r_E 预缩放) ----
        # delay ≈ 20s × 0.05 ≈ 1.0;  energy ≈ 4J × 0.25 ≈ 1.0  →  两通道对称
        delay_scale=0.05,
        energy_scale=0.25,
        # ---- ω 标量化系数 (改造后默认 1.0, 走真凸组合 r = ω·r_T + ω·r_E) ----
        # 旧值 alpha_T=1.0, alpha_E=0.25 会让能耗权重被打 4 折 → C1 自适应分配失效.
        # 保留字段是为了向后兼容 ckpt 加载与对照消融 (回滚: alpha_E=0.25).
        alpha_T=1.0,
        alpha_E=1.0,
        actor_lr=1e-4,
        critic_lr=1e-3,
        alpha_init=0.05,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.95,
        denoising_steps=3,
        hidden_dim=128,
        # 离散动作 |A|=6, 熵 H ∈ [0, log(6)≈1.79]. 旧值 -1.0 是不可达负数,
        # 导致 SAC 永远在压熵 → α 单调指数下降到 0 (epoch ≈ 6-7 就崩).
        # 通道归一化后 reward 量级 -1, α 项与 Q 项可比, α=0 必致 actor 学崩.
        # 新值 0.5 ≈ 0.3 × log(6), 让 α 在 H ≈ 0.5 时进入稳态.
        target_entropy=0.5,
        buf_size=10000,
        batch_size=64,
        buffer_warmup=500,
        update_every=4,
        # ---- ω-NN Latent Warm-start Buffer (Path B / DAFM) ----
        # use_omega_buffer=True → C2-aware (V4 envelope + ω-buffer warm-start)
        # use_omega_buffer=False → C0 (envelope only, 无 ω-buffer)
        use_omega_buffer=True,
        obuf_decay=0.5,        # EMA: new = decay*current + (1-decay)*history
        obuf_noise=0.05,       # 检索后扰动幅度, 防止过度拷贝退化
        # ---- V5: Vector-Q + ω-weighted critic loss + COR + PopArt-lite ----
        # 项目固定使用 V5 (Vector-Q + ω-加权 critic loss + COLA 风格 COR + PopArt 量级归一).
        use_v5=True,
        use_cor=True,          # COR 开关 (Conflict Objective Regularization)
        cor_lambda=0.1,        # COR 系数 (COLA 默认)
        cor_c=0.0,             # COR 触发阈值 (ρ < c 才生效, c=0 即仅梯度冲突时正则)
        use_popart=True,       # V5: PopArt-lite 开关
        popart_beta=0.01,      # V5: σ_k 的 EMA 步长
        # ---- 任务到达模式 ('random' / 'poisson' / 'trace') ----
        task_mode='random',
        task_kwargs=dict(),
    )
    if cfg_override:
        cfg.update(cfg_override)
    print('Config:', cfg)

    # ---- 每次运行写入独立 timestamp 目录, 便于 (1) 不互相覆盖 (2) 复现 (3) drift 复用 ckpt ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = cfg.get('file_prefix', 'mofd')
    base_results = cfg.get('results_root', 'results')
    results_dir = os.path.join(base_results, f'{prefix}_{ts}')
    os.makedirs(results_dir, exist_ok=True)
    cfg['results_dir'] = results_dir  # 透传给 run_single_seed

    # 把本次 run 的 cfg 快照存进去 (cfg 含 tuple/np 类型, 用 default=str)
    with open(os.path.join(results_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    print(f'Run dir: {os.path.abspath(results_dir)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 按 cfg 设置全局任务到达模式, 所有调用 sample_tasks 的地方自动生效
    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))

    # ========== 多 seed 训练 ==========
    all_runs = []
    t_all = time.time()
    for seed in cfg['seeds']:
        print(f"\n========== Seed {seed} ==========")
        run = run_single_seed(cfg, seed, device)
        all_runs.append(run)
    print(f"\n[Total] all {len(cfg['seeds'])} seeds done in {time.time() - t_all:.1f}s")

    # ========== 训练曲线聚合 (mean ± std, 滑动平均) ==========
    hv_all = np.stack([r['log_hv'] for r in all_runs], axis=0)        # [S, E]
    d_all = np.stack([r['log_avg_delay'] for r in all_runs], axis=0)
    e_all = np.stack([r['log_avg_energy'] for r in all_runs], axis=0)

    w = cfg['smooth_window']

    def agg(arr):
        return smooth(arr.mean(0), w), smooth(arr.std(0), w)

    hv_m, hv_s = agg(hv_all)
    d_m, d_s = agg(d_all)
    e_m, e_s = agg(e_all)

    np.savetxt(os.path.join(results_dir, 'mofd_hv_all_seeds.csv'), hv_all, fmt='%.5f')
    np.savetxt(os.path.join(results_dir, 'mofd_hv_mean_std.csv'),
               np.stack([hv_m, hv_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(results_dir, 'mofd_avg_delay_all_seeds.csv'), d_all, fmt='%.5f')
    np.savetxt(os.path.join(results_dir, 'mofd_avg_delay_mean_std.csv'),
               np.stack([d_m, d_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')
    np.savetxt(os.path.join(results_dir, 'mofd_avg_energy_all_seeds.csv'), e_all, fmt='%.5f')
    np.savetxt(os.path.join(results_dir, 'mofd_avg_energy_mean_std.csv'),
               np.stack([e_m, e_s], axis=1), fmt='%.5f',
               header='mean_smooth std_smooth', comments='')

    epochs = np.arange(cfg['num_epochs'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, title, mu, sd, color in zip(
            axes,
            ['Hypervolume', 'Avg Delay (s)', 'Avg Energy (J)'],
            [hv_m, d_m, e_m], [hv_s, d_s, e_s],
            ['tab:blue', 'tab:orange', 'tab:green']):
        ax.plot(epochs, mu, color=color, linewidth=2)
        ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.25, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} (mean ± std, {len(all_runs)} seeds, smooth={w})')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mofd_training_curves.png'), dpi=150)
    plt.close()

    # ========== 训练健康度监控曲线 (调参时直接看这张图) ==========
    monitor_keys = ['log_c_loss', 'log_a_loss', 'log_alpha_curve',
                    'log_H', 'log_sigma_T', 'log_sigma_E']
    if all(k in all_runs[0] for k in monitor_keys):
        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
        norm_label = 'PopArt'
        panels = [
            ('log_c_loss',      'Critic Loss',       'tab:red'),
            ('log_a_loss',      'Actor Loss',        'tab:purple'),
            ('log_alpha_curve', 'SAC α = exp(log_α)','tab:brown'),
            ('log_H',           'Actor Entropy H',   'tab:olive'),
            ('log_sigma_T',     f'{norm_label} σ_T (delay)','tab:cyan'),
            ('log_sigma_E',     f'{norm_label} σ_E (energy)','tab:pink'),
        ]
        for ax, (key, title, color) in zip(axes3.flat, panels):
            curves = np.stack([r[key] for r in all_runs], axis=0)
            mu_c = np.nanmean(curves, axis=0)
            sd_c = np.nanstd(curves, axis=0)
            ax.plot(epochs, mu_c, color=color, linewidth=2)
            ax.fill_between(epochs, mu_c - sd_c, mu_c + sd_c, alpha=0.25, color=color)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'mofd_monitor_curves.png'), dpi=150)
        plt.close()
        print(f'[monitor] saved mofd_monitor_curves.png (6 panels: '
              f'c_loss/a_loss/α/H/σ_T/σ_E)')

    # ========== G-FDEdge Pareto 聚合 (跨方法对比交给 compare_hv.py) ==========
    pts_pool = np.vstack([r['pts_gfd'] for r in all_runs])
    ref_local = (float(pts_pool[:, 0].max() * 1.1 + 1e-6),
                 float(pts_pool[:, 1].max() * 1.1 + 1e-6))

    hvs = [hypervolume_2d(r['pts_gfd'], ref_local) for r in all_runs]
    mu, sd = float(np.mean(hvs)), float(np.std(hvs))
    print(f"\n=== G-FDEdge Final HV (local ref={ref_local}): {mu:.4f} ± {sd:.4f} ===")

    np.savetxt(os.path.join(results_dir, f'{prefix}_pareto_aggregated.csv'),
               pts_pool, fmt='%.5f')
    # 同时在 results/ 根目录写一份 (与 compare_hv.py 默认查找路径一致)
    np.savetxt(os.path.join(base_results, f'{prefix}_pareto_aggregated.csv'),
               pts_pool, fmt='%.5f')

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 5))
    ax2.scatter(pts_pool[:, 0], pts_pool[:, 1], color='red', marker='o',
                s=25, alpha=0.5, label=f'G-FDEdge (HV={mu:.2f}±{sd:.2f})')
    pf = pareto_front_2d(pts_pool)
    if len(pf) > 0:
        ax2.plot(pf[:, 0], pf[:, 1], color='red', alpha=0.6)
    ax2.set_xlabel('Avg Delay (s)')
    ax2.set_ylabel('Avg Energy (J)')
    ax2.set_title(f'G-FDEdge Pareto Front (pooled {len(all_runs)} seeds)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mofd_pareto_aggregated.png'), dpi=150)
    plt.close()

    with open(os.path.join(results_dir, 'mofd_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('=== MOFD (G-FDEdge) Multi-seed Summary ===\n')
        f.write(f'Seeds: {cfg["seeds"]}\n')
        f.write(f'Config: {cfg}\n')
        f.write(f'Local HV ref: {ref_local}\n')
        f.write(f'HV per seed: {hvs}\n')
        f.write(f'HV mean ± std: {mu:.4f} ± {sd:.4f}\n')
        f.write('\n(启发式基线请运行 baselines/Heuristic/heuristic_main.py;\n'
                ' 跨方法统一 HV 对比请运行 compare_hv.py.)\n')

    # ---- 同步一份到 results/latest/, 方便 compare_hv.py / analyze_drift_detrend.py 默认读 ----
    latest_dir = os.path.join(base_results, 'latest')
    try:
        if os.path.lexists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                shutil.rmtree(latest_dir, ignore_errors=True)
        shutil.copytree(results_dir, latest_dir)
        print(f'Mirrored to {os.path.abspath(latest_dir)}')
    except Exception as e:
        print(f'[warn] failed to update latest mirror: {e}')

    print(f'\nAll results saved to {os.path.abspath(results_dir)}')
    print('Key files:')
    print('  - mofd_training_curves.png   (HV / delay / energy with ±std band)')
    print('  - mofd_pareto_aggregated.png (G-FDEdge Pareto front)')
    print('  - mofd_summary.txt           (final HV mean±std)')
    print('  - ckpt_seed*/                (actor/critic weights, drift 实验可加载)')
    print('  Baselines: run baselines/Heuristic/heuristic_main.py + the 3 RL baselines,')
    print('             then python compare_hv.py for unified HV table.')


if __name__ == '__main__':
    main()
