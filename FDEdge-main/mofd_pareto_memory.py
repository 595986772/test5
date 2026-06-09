"""
ParetoMemoryBuffer (最小版)
===========================
OmegaLatentBuffer 的即插即换替代: 接口完全一致 (retrieve_prior / update / stats /
save_log / save_update_count / save_pickle / load_pickle), 由 cfg.use_pareto_memory 切换。

唯一的不同 (这就是"最小 Pareto 版"的全部):
  OmegaLatentBuffer: 把附近历史动作做注意力**加权平均** → 好坏混在一起, prior 偏"糊"。
  ParetoMemoryBuffer: 每个 (ω, 上下文) 桶内只留**非支配** entry (delay/energy 样样被碾压
                      的删掉, 各有专长的留, 强制保住"最快"与"最省"两端);
                      检索时在近邻非支配里**按当前 ω 选单个最好的** prior 返回 (不平均)。

故意不加 (留待验证有效后再考虑): age decay / reward gate / crowding 距离 / 任何新旋钮。
依赖 update 时额外传入该 episode 真实 (delay, energy) —— mofd_main 已在 ParetoMemory 分支传。

key / env_ctx / 概率向量构造均沿用 OmegaLatentBuffer 习惯, 保证可比。
"""
import collections
import pickle
import numpy as np


class ParetoMemoryBuffer:
    def __init__(self, action_dim, n_omega_bins=21, n_ctx_bins=4,
                 bucket_capacity=16, max_entries=500, top_k=20,
                 sigma=0.3, conf_threshold=0.4, warmup_epochs=5,
                 alpha_T=1.0, alpha_E=0.25, **_ignored):
        self.action_dim = int(action_dim)
        self.n_omega_bins = int(n_omega_bins)
        self.n_ctx_bins = int(n_ctx_bins)
        self.bucket_capacity = int(bucket_capacity)
        self.max_entries = int(max_entries)
        self.top_k = int(top_k)
        self.sigma = float(sigma)
        self.conf_threshold = float(conf_threshold)
        self.warmup_epochs = int(warmup_epochs)
        self.alpha_T = float(alpha_T)
        self.alpha_E = float(alpha_E)

        # entry = dict(key[5], prior[A], delay, energy, epoch, bucket)
        self.entries = []
        self.log = []
        self.update_count = {}
        # 标量化用的量级归一 (running RMS), 防 delay 数值压过 energy
        self._d_rms = 1.0
        self._e_rms = 1.0

    # ---------- key / bin / prior 工具 ----------
    @staticmethod
    def _build_key(omega, env_ctx):
        if env_ctx is None:
            env_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return np.concatenate([np.asarray(omega, dtype=np.float32),
                               np.asarray(env_ctx, dtype=np.float32)]).astype(np.float32)

    def _bucket(self, omega, env_ctx):
        ob = int(round(float(omega[0]) * (self.n_omega_bins - 1)))
        ob = max(0, min(self.n_omega_bins - 1, ob))
        ctx = (np.array([0.5, 0.5, 0.5], dtype=np.float32)
               if env_ctx is None else np.asarray(env_ctx, dtype=np.float32))
        cb = tuple(max(0, min(self.n_ctx_bins - 1,
                              int(round(float(x) * (self.n_ctx_bins - 1))))) for x in ctx)
        return (ob,) + cb

    def _latent_to_prior(self, latent_slice):
        lat = np.asarray(latent_slice, dtype=np.float32).reshape(-1, self.action_dim)
        prior = np.clip(lat.mean(axis=0), 0.0, None)
        s = float(prior.sum())
        if s < 1e-8:
            return np.full(self.action_dim, 1.0 / self.action_dim, dtype=np.float32)
        return (prior / s).astype(np.float32)

    @staticmethod
    def _dominates(a, b):
        """a 支配 b: 两目标都不差, 且至少一个严格更优 (delay/energy 均越小越好)."""
        no_worse = a['delay'] <= b['delay'] and a['energy'] <= b['energy']
        strictly = a['delay'] < b['delay'] or a['energy'] < b['energy']
        return no_worse and strictly

    def _score(self, e, omega):
        """当前 ω 下的标量化得分 (越大越好), 量级归一防 delay 压过 energy."""
        d = e['delay'] / max(self._d_rms, 1e-8)
        en = e['energy'] / max(self._e_rms, 1e-8)
        return -(float(omega[0]) * self.alpha_T * d + float(omega[1]) * self.alpha_E * en)

    # ---------- 写入: 非支配剪枝 + 容量保两端 ----------
    def update(self, omega, latent_slice, env_ctx=None, epoch=None,
               reward=None, delay=None, energy=None):
        if delay is None or energy is None:
            return  # 最小版必须有真实 delay/energy 才能做 Pareto 判定
        if epoch is not None and self.warmup_epochs > 0 and int(epoch) < self.warmup_epochs:
            return
        omega = np.asarray(omega, dtype=np.float32)
        bucket = self._bucket(omega, env_ctx)
        new = dict(key=self._build_key(omega, env_ctx),
                   prior=self._latent_to_prior(latent_slice),
                   delay=float(delay), energy=float(energy),
                   epoch=int(epoch) if epoch is not None else 0,
                   bucket=bucket)
        self._d_rms = 0.99 * self._d_rms + 0.01 * abs(new['delay'])
        self._e_rms = 0.99 * self._e_rms + 0.01 * abs(new['energy'])

        same = [e for e in self.entries if e['bucket'] == bucket]
        # 新点被同桶旧点支配 → 不写
        if any(self._dominates(old, new) for old in same):
            return
        # 删掉被新点支配的旧点
        self.entries = [e for e in self.entries
                        if e['bucket'] != bucket or not self._dominates(new, e)]
        self.entries.append(new)
        self.update_count[bucket] = self.update_count.get(bucket, 0) + 1

        self._cap_bucket(bucket, omega)
        if len(self.entries) > self.max_entries:                 # 全局兜底
            self.entries = sorted(self.entries, key=lambda e: e['epoch'])[-self.max_entries:]

    def _cap_bucket(self, bucket, omega):
        same = [e for e in self.entries if e['bucket'] == bucket]
        if len(same) <= self.bucket_capacity:
            return
        keep = set()
        for e in sorted(same, key=lambda x: x['delay'])[:2]:     # 最快 2
            keep.add(id(e))
        for e in sorted(same, key=lambda x: x['energy'])[:2]:    # 最省 2
            keep.add(id(e))
        for e in sorted(same, key=lambda x: self._score(x, omega), reverse=True):
            if len(keep) >= self.bucket_capacity:
                break
            keep.add(id(e))
        self.entries = [e for e in self.entries
                        if e['bucket'] != bucket or id(e) in keep]

    # ---------- 检索: 近邻非支配里选单个最好的 prior ----------
    def retrieve_prior(self, omega, env_ctx=None, action_dim=None, current_epoch=None):
        dim = self.action_dim if action_dim is None else int(action_dim)
        uniform = np.full(dim, 1.0 / dim, dtype=np.float32)
        if dim != self.action_dim or not self.entries:
            return uniform.copy()
        omega = np.asarray(omega, dtype=np.float32)
        query = self._build_key(omega, env_ctx)
        keys = np.stack([e['key'] for e in self.entries])
        dists = np.linalg.norm(keys - query, axis=1)
        k = min(self.top_k, len(self.entries))
        idx = np.argpartition(dists, k - 1)[:k]
        sims = np.exp(-(dists[idx] ** 2) / max(self.sigma ** 2, 1e-12))
        max_sim = float(sims.max()) if sims.size else 0.0
        # 置信门: 最近邻太远 → 退 uniform (评估侧会转成 feedback 零起点)
        if max_sim < self.conf_threshold:
            self.log.append(dict(omega=tuple(omega), hit=0, conf=max_sim,
                                 nn_dist=float(dists[idx].min()),
                                 n_entries=len(self.entries), gated=1))
            return uniform.copy()
        local = [self.entries[int(i)] for i in idx]
        nondom = [c for c in local
                  if not any(o is not c and self._dominates(o, c) for o in local)]
        if not nondom:
            nondom = local
        best = max(nondom, key=lambda e: self._score(e, omega))
        prior = np.clip(best['prior'].astype(np.float32), 0.0, None)
        s = float(prior.sum())
        if s < 1e-8:
            return uniform.copy()
        self.log.append(dict(omega=tuple(omega), hit=1, conf=max_sim,
                             nn_dist=float(dists[idx].min()),
                             n_entries=len(self.entries), gated=0))
        return (prior / s).astype(np.float32)

    # ---------- stats / 存档 (字段对齐 OmegaLatentBuffer, 供 mofd_main 打印) ----------
    def stats(self):
        if not self.log:
            return dict(hit_rate=0.0, gate_rate=0.0, mean_conf=0.0,
                        mean_nn_dist=0.0, n_entries=len(self.entries), n_calls=0)
        hit = np.array([x['hit'] for x in self.log], dtype=float)
        gate = np.array([x['gated'] for x in self.log], dtype=float)
        conf = np.array([x['conf'] for x in self.log], dtype=float)
        dist = [x['nn_dist'] for x in self.log if x['hit'] == 1]
        return dict(hit_rate=float(hit.mean()), gate_rate=float(gate.mean()),
                    mean_conf=float(conf.mean()),
                    mean_nn_dist=float(np.mean(dist)) if dist else 0.0,
                    n_entries=len(self.entries), n_calls=len(self.log))

    def save_log(self, path):
        if not self.log:
            return
        rows = [[float(x['omega'][0]), float(x['omega'][1]), int(x['hit']),
                 float(x['nn_dist']), int(x['n_entries']), float(x['conf']),
                 int(x['gated'])] for x in self.log]
        np.savetxt(path, np.asarray(rows, dtype=float), fmt='%.6f',
                   header='omega0 omega1 hit nn_dist n_entries conf gated', comments='')

    def save_update_count(self, path):
        if not self.update_count:
            return
        rows = [list(b) + [c] for b, c in sorted(self.update_count.items())]
        np.savetxt(path, np.asarray(rows, dtype=int), fmt='%d',
                   header='omega_bin ctx0 ctx1 ctx2 update_count', comments='')

    def save_archive(self, path):
        if not self.entries:
            return
        rows = [list(e['bucket']) + [e['delay'], e['energy'], e['epoch']]
                for e in self.entries]
        np.savetxt(path, np.asarray(rows, dtype=float), fmt='%.6f',
                   header='omega_bin ctx0 ctx1 ctx2 delay energy epoch', comments='')

    def save_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(entries=self.entries, log=self.log,
                             update_count=self.update_count,
                             d_rms=self._d_rms, e_rms=self._e_rms,
                             action_dim=self.action_dim), f)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.entries = d.get('entries', [])
        self.log = d.get('log', [])
        self.update_count = d.get('update_count', {})
        self._d_rms = float(d.get('d_rms', 1.0))
        self._e_rms = float(d.get('e_rms', 1.0))
