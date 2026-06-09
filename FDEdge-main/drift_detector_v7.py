"""
CUSUM-RL Online Drift Detector (V7 新增, 替代 CUSUM)
======================================================
CUSUM (Page 1954) 监测 raw reward 流, 但对 reward 噪声敏感且无法区分
"reward 变小因为 ω 变了" vs "reward 变小因为环境漂移了".

CUSUM-RL (NeurIPS 2025) 改监测 TD error (Bellman residual):
  δ_k = r_k + γ · V_k(s') − Q_k(s,a)

优势:
  1. TD error 在 stationary MDP 下期望为 0 (critic 收敛后), 漂移时均值偏移
  2. 天然捕获 dynamics + reward 双通道变化
  3. SAC critic 在训练/评估中都在算 TD error, 零额外开销

本模块提供 CUSUMRLDetector: 双通道 (delay / energy) 独立 CUSUM, 任一触发即报警.

API (兼容 drift_detector.CUSUMDetector):
  d = CUSUMRLDetector(window=20, threshold=3.0, drift_param=0.5)
  for each step:
      if d.update(td_error_vec):   # td_error_vec shape [2]
          ...trigger drift response

参数:
  window=20        滑动窗口长度 (估计 TD error 基线 μ, σ)
  threshold=3.0    CUSUM 报警阈值 (z-score 累积量上限)
  drift_param=0.5  δ, 允许噪声容限 (高→抗噪但延迟大)

参考:
  - CUSUM-RL, NeurIPS 2025: Testing Stationarity and Change Point Detection in RL
  - RCCDA, arXiv 2505.24149: drift adaptation under constraints
"""
import numpy as np


class _CUSUMChannel:
    """单通道 CUSUM (正负双向)."""

    def __init__(self, window=20, threshold=3.0, drift_param=0.5,
                 reset_on_alert=True):
        self.window = int(window)
        self.threshold = float(threshold)
        self.delta = float(drift_param)
        self.reset_on_alert = bool(reset_on_alert)
        self.reset()

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.values = []
        self.alert_count = 0
        self.last_alert_step = -1
        self.step = 0

    def update(self, value):
        v = float(value)
        self.values.append(v)
        self.step += 1
        if len(self.values) < self.window + 1:
            return False
        ref = self.values[-(self.window + 1):-1]
        mu = float(np.mean(ref))
        sigma = float(np.std(ref) + 1e-6)
        z = (v - mu) / sigma
        self.s_pos = max(0.0, self.s_pos + z - self.delta)
        self.s_neg = min(0.0, self.s_neg + z + self.delta)
        if self.s_pos > self.threshold or self.s_neg < -self.threshold:
            self.alert_count += 1
            self.last_alert_step = self.step
            if self.reset_on_alert:
                self.s_pos = 0.0
                self.s_neg = 0.0
            return True
        return False

    def stats(self):
        return dict(
            step=self.step,
            n_values=len(self.values),
            s_pos=self.s_pos,
            s_neg=self.s_neg,
        )


class CUSUMRLDetector:
    """CUSUM-RL: 双通道 TD-error CUSUM 检测器.

    与 CUSUMDetector 的区别:
      - CUSUMDetector: 输入标量 reward, 监测 reward 流漂移
      - CUSUMRLDetector: 输入 TD error 向量 [δ_T, δ_E], 双通道独立检测

    TD error 应在 episode rollouts 时由 agent 提供 (SAC critic forward).
    """

    def __init__(self, window=20, threshold=3.0, drift_param=0.5,
                 reset_on_alert=True):
        self.window = int(window)
        self.threshold = float(threshold)
        self.drift_param = float(drift_param)
        self.reset_on_alert = bool(reset_on_alert)
        self.ch_T = _CUSUMChannel(window, threshold, drift_param, reset_on_alert)
        self.ch_E = _CUSUMChannel(window, threshold, drift_param, reset_on_alert)

    def reset(self):
        self.ch_T.reset()
        self.ch_E.reset()

    def update(self, td_error_vec):
        """Push TD error vector, 返回 True 若任一通道触发.

        td_error_vec: array-like [2] = [δ_T, δ_E]
        """
        td = np.asarray(td_error_vec, dtype=np.float64).ravel()
        alert_T = self.ch_T.update(float(td[0]))
        alert_E = self.ch_E.update(float(td[1]))
        return alert_T or alert_E

    @property
    def alert_count(self):
        return self.ch_T.alert_count + self.ch_E.alert_count

    @property
    def last_alert_step(self):
        return max(self.ch_T.last_alert_step, self.ch_E.last_alert_step)

    @property
    def step(self):
        return max(self.ch_T.step, self.ch_E.step)

    def stats(self):
        return dict(
            T=self.ch_T.stats(),
            E=self.ch_E.stats(),
            alert_count=self.alert_count,
            last_alert_step=self.last_alert_step,
        )


# ============================================================
# 快速自测
# ============================================================
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    d = CUSUMRLDetector(window=20, threshold=3.0, drift_param=0.5)

    # 模拟: 前 30 步 TD error ~ N(0, 0.5) (stationary),
    #        第 31 步起漂移到 N(2.0, 0.5) (MDP 变化).
    alerts = []
    for t in range(60):
        if t < 30:
            td = rng.normal(0.0, 0.5, size=2)
        else:
            td = rng.normal(2.0, 0.5, size=2)
        if d.update(td):
            alerts.append(t)
    print(f'CUSUM-RL alerts at steps: {alerts}')
    print(f'Stats: {d.stats()}')
    # 期望: 第一次报警出现在 t∈[31, 38] 区间
