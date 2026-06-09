"""
CUSUM-based Online ω-Drift Detector (drift_main 新增)
======================================================
当前 mofd_omega_drift_main.py 的 `drift_aware=True` 是 ORACLE — 直接读
omega_schedule 知道 ω 在变. 部署时不现实 (审稿人必问).

本模块提供 deployable 漂移检测: 监测 reward 流的累积偏离 (CUSUM), 超阈值
就触发 drift signal → C2-cusum 方法在 drift signal 时调用 retrieve_prior.

设计参考:
  - Page (1954) Sequential Detection of a Change in a Mean (CUSUM 原始)
  - RCCDA (arXiv 2505.24149, May 2025): resource-constrained drift adaptation
  - 在 MEC/RL 上下文的应用: 把单步 reward 视为 i.i.d. 流, 不正常偏离即漂移

参数与典型设置:
  window=20         滑动窗口长度 (用于估计基线 μ, σ)
  threshold=3.0     CUSUM 报警阈值 (3 sigma 等价)
  drift_param=0.5   δ, 抗噪声但延迟检测的 trade-off (低则灵敏高假报警)

API:
  d = CUSUMDetector(window=20, threshold=3.0, drift_param=0.5)
  for each slot:
      if d.update(reward):
          ...trigger retrieve_prior(omega_estimated_or_current)
"""
import numpy as np


class CUSUMDetector:
    """单向 / 双向 CUSUM 复合检测器.

    跟踪 reward 流的 (μ, σ) 滑动估计, 计算 z-score 累积上下界, 任一越界即报警.
    报警后内部累积量重置 (`reset_on_alert=True`), 防止单次漂移持续报警.
    """

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
        self.rewards = []
        self.alert_count = 0
        self.last_alert_step = -1
        self.step = 0

    def update(self, reward):
        """Push 一个新 reward, 返回 True/False 是否触发漂移信号."""
        r = float(reward)
        self.rewards.append(r)
        self.step += 1
        if len(self.rewards) < self.window + 1:
            return False
        ref = self.rewards[-(self.window + 1):-1]   # 不含当前点的窗口
        mu = float(np.mean(ref))
        sigma = float(np.std(ref) + 1e-6)
        z = (r - mu) / sigma
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
        """诊断用: 返回当前累积量与报警计数."""
        return dict(
            step=self.step,
            n_rewards=len(self.rewards),
            s_pos=self.s_pos,
            s_neg=self.s_neg,
            alert_count=self.alert_count,
            last_alert_step=self.last_alert_step,
        )


# ============================================================
# 快速自测
# ============================================================
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    d = CUSUMDetector(window=20, threshold=3.0, drift_param=0.5)
    # 模拟: 前 30 步 reward ~ N(-1.0, 0.2), 第 31 步起漂移到 N(-3.0, 0.2)
    alerts = []
    for t in range(60):
        mu_t = -1.0 if t < 30 else -3.0
        r = float(rng.normal(mu_t, 0.2))
        if d.update(r):
            alerts.append(t)
    print(f'Alerts triggered at slots: {alerts}')
    print(f'Stats: {d.stats()}')
    # 期望: 第一次报警出现在 t∈[31, 35] 区间 (短延迟)
