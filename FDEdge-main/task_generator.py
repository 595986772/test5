"""
Task Generator
==============
三种可切换的任务到达方式, 供 MOFD 主实验 + 各基线共用:

  * 'random'  — 与原 sample_tasks 等价: 每 slot 均匀随机采 k∈[1, n_max] 个任务,
                大小在 [min_bit, max_bit] 均匀分布.  (默认)
  * 'poisson' — 每 slot 任务数服从 Poisson(λ(t)), λ(t) = λ0 + λ1·sin(2πt/T),
                大小服从截断指数分布 (均值 ~ (min+max)/2).
  * 'trace'   — 从 dataset/ 下的真实轨迹 CSV 映射, 支持 wrap-around 对齐 time_slots.

所有 Generator 输出格式与 env.reset_env 要求一致: list[np.ndarray(k_t,)], 长度 = env.time_slots.

使用方法
--------
    from task_generator import make_task_generator
    gen = make_task_generator('poisson', lam0=5.0, lam1=2.0)
    tasks = gen.sample(env, rng)       # 用同一个 rng 保持可复现

或由主脚本 cfg 驱动:
    cfg['task_mode'] = 'trace'
    cfg['task_kwargs'] = dict(trace_path='dataset/tasks_trace.csv')
    gen = make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {}))

Trace CSV 格式 (header + 逗号分隔):
    slot,bits
    0,25.3
    0,18.4
    1,30.1
    1,22.0
    ...
其中 slot 是从 0 开始的整数, bits 是任务大小 (Mbit). 若 trace 总 slot 数 < env.time_slots,
会自动循环填充; 若 > env.time_slots, 截取前 time_slots 个.
"""
import csv
import os
from collections import defaultdict

import numpy as np


# ============================================================
# 基类
# ============================================================
class TaskGenerator:
    """抽象基类. 子类实现 sample(env, rng) -> list[np.ndarray]."""
    name = 'base'

    def sample(self, env, rng):
        raise NotImplementedError

    def __repr__(self):
        return f'<{self.__class__.__name__}>'


# ============================================================
# 1) Random (均匀分布, 默认, 与原 sample_tasks 等价)
# ============================================================
class RandomTaskGenerator(TaskGenerator):
    """每 slot 随机 k∈[1, n_max] 个任务, 大小均匀分布."""
    name = 'random'

    def sample(self, env, rng):
        tasks = []
        for _ in range(env.time_slots):
            k = int(rng.integers(1, env.n_tasks_max + 1))
            sizes = rng.uniform(env.min_bit, env.max_bit, size=[k])
            tasks.append(sizes.astype(np.float32))
        return tasks


# ============================================================
# 2) Poisson (到达数 Poisson, 大小截断指数; 可选正弦日周期调制)
# ============================================================
class PoissonTaskGenerator(TaskGenerator):
    """每 slot 任务数 k ~ Poisson(λ(t)), λ(t) = λ0 + λ1·sin(2π t / period).

    任务大小服从 [min_bit, max_bit] 截断指数分布, 均值默认为 (min+max)/2.
    用于模拟 IoT / 视频流 等带周期性的真实到达模式.

    参数
    ----
    lam0 : 基础到达率 (每 slot 平均任务数), 推荐 ~ n_max/2
    lam1 : 正弦振幅 (>= 0 且 < lam0, 保证 λ(t) > 0)
    period : 正弦周期 (slots). None 时自动取 env.time_slots
    size_mean : 任务大小指数分布均值; None 时 (min+max)/2
    """
    name = 'poisson'

    def __init__(self, lam0=None, lam1=0.0, period=None, size_mean=None):
        self.lam0 = lam0
        self.lam1 = lam1
        self.period = period
        self.size_mean = size_mean

    def sample(self, env, rng):
        lam0 = self.lam0 if self.lam0 is not None else max(env.n_tasks_max / 2.0, 1.0)
        lam1 = max(0.0, min(self.lam1, lam0 - 1e-6))  # 保证 λ(t) > 0
        T = env.time_slots
        period = self.period if self.period is not None else T
        mean_bit = self.size_mean if self.size_mean is not None else 0.5 * (env.min_bit + env.max_bit)

        tasks = []
        for t in range(T):
            lam_t = lam0 + lam1 * np.sin(2.0 * np.pi * t / max(period, 1))
            k = int(rng.poisson(max(lam_t, 1e-3)))
            k = max(1, min(k, env.n_tasks_max))  # clip 到 [1, n_max] 匹配 comp_density 槽位
            sizes = rng.exponential(mean_bit, size=[k])
            sizes = np.clip(sizes, env.min_bit, env.max_bit).astype(np.float32)
            tasks.append(sizes)
        return tasks


# ============================================================
# 3) Trace (从真实数据集 CSV 映射)
# ============================================================
class TraceTaskGenerator(TaskGenerator):
    """从 CSV 文件加载真实任务到达轨迹.

    CSV 必须含两列: slot (int), bits (float). 多个同 slot 行视为该 slot 内的多个任务.

    参数
    ----
    trace_path : CSV 路径 (相对 / 绝对均可)
    wrap : True 时 trace 短于 time_slots 自动循环; False 时补空 slot
    offset : 起始 slot 偏移, 支持在同一 trace 上滑动采样 (训练增强)
    shuffle_offset : True 时每次 sample 用随机 offset, 用于训练期打散
    """
    name = 'trace'

    def __init__(self, trace_path='dataset/tasks_trace.csv',
                 wrap=True, offset=0, shuffle_offset=False):
        self.trace_path = trace_path
        self.wrap = wrap
        self.offset = offset
        self.shuffle_offset = shuffle_offset
        self._by_slot = None
        self._max_slot = -1

    def _load(self):
        if self._by_slot is not None:
            return
        path = self.trace_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'[TraceTaskGenerator] 找不到轨迹文件: {path}\n'
                f'请在 dataset/ 下放置 CSV, 格式为:\n'
                f'    slot,bits\n    0,25.3\n    0,18.4\n    1,30.1\n    ...\n'
                f'其中 slot 是从 0 开始的整数, bits 是任务大小 (Mbit).'
            )

        by_slot = defaultdict(list)
        max_slot = -1
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'slot' not in reader.fieldnames or 'bits' not in reader.fieldnames:
                raise ValueError(
                    f'[TraceTaskGenerator] CSV 表头必须包含 "slot" 和 "bits" 列, '
                    f'当前表头: {reader.fieldnames}'
                )
            for row in reader:
                s = int(row['slot'])
                b = float(row['bits'])
                by_slot[s].append(b)
                if s > max_slot:
                    max_slot = s

        if max_slot < 0:
            raise ValueError(f'[TraceTaskGenerator] CSV {path} 没有任何有效数据行')

        self._by_slot = {s: np.asarray(v, dtype=np.float32) for s, v in by_slot.items()}
        self._max_slot = max_slot
        print(f'[TraceTaskGenerator] 已加载 {sum(len(v) for v in self._by_slot.values())} '
              f'个任务, 覆盖 {max_slot + 1} 个 slot, 文件={path}')

    def sample(self, env, rng):
        self._load()
        trace_len = self._max_slot + 1
        offset = self.offset
        if self.shuffle_offset and trace_len > 1:
            offset = int(rng.integers(0, trace_len))

        tasks = []
        for t in range(env.time_slots):
            src = (offset + t) % trace_len if self.wrap else offset + t
            sizes = self._by_slot.get(src, np.zeros(0, dtype=np.float32))
            # 截断到 [1, n_tasks_max], 匹配 env.comp_density 的槽位上限
            if len(sizes) == 0:
                sizes = rng.uniform(env.min_bit, env.max_bit, size=[1]).astype(np.float32)
            elif len(sizes) > env.n_tasks_max:
                sizes = sizes[:env.n_tasks_max]
            # 截断到 env 要求的 bit 范围 (避免超出状态归一化假设)
            sizes = np.clip(sizes, env.min_bit, env.max_bit).astype(np.float32)
            tasks.append(sizes)
        return tasks


# ============================================================
# 工厂
# ============================================================
_REGISTRY = {
    'random':  RandomTaskGenerator,
    'poisson': PoissonTaskGenerator,
    'trace':   TraceTaskGenerator,
}


def make_task_generator(mode='random', **kwargs):
    """主入口. 根据 mode 构造对应 generator.

    mode: 'random' / 'poisson' / 'trace'
    kwargs: 对应 generator 的构造参数
    """
    mode = (mode or 'random').lower()
    if mode not in _REGISTRY:
        raise ValueError(f'Unknown task_mode={mode!r}, 可选: {list(_REGISTRY.keys())}')
    return _REGISTRY[mode](**kwargs)


# ============================================================
# 快速自测
# ============================================================
if __name__ == '__main__':
    class _EnvStub:
        time_slots = 10
        n_tasks_max = 8
        min_bit = 10.0
        max_bit = 40.0

    env = _EnvStub()
    rng = np.random.default_rng(0)

    for mode, kw in [('random', {}),
                     ('poisson', dict(lam0=4.0, lam1=2.0))]:
        gen = make_task_generator(mode, **kw)
        tasks = gen.sample(env, rng)
        ks = [len(x) for x in tasks]
        sizes = np.concatenate(tasks) if tasks else np.array([])
        print(f'[{mode:>8}] slots={len(tasks)} k_per_slot={ks} '
              f'size mean={sizes.mean():.2f} min={sizes.min():.2f} max={sizes.max():.2f}')

    # trace 模式: 仅在 dataset/tasks_trace.csv 存在时测试
    try:
        gen = make_task_generator('trace', trace_path='dataset/tasks_trace.csv')
        tasks = gen.sample(env, rng)
        ks = [len(x) for x in tasks]
        print(f'[   trace] slots={len(tasks)} k_per_slot={ks}')
    except FileNotFoundError as e:
        print(f'[   trace] 跳过: {e}'.splitlines()[0])
