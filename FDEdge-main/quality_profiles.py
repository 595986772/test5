"""
quality_profiles.py — 质量档位的真实标定数据
============================================
两类边缘 AI 负载, 每个"质量档"对应一个真实模型变体, 数值取自公开 benchmark:

  [detection] YOLOv11 n/s/m/l/x  (Ultralytics 官方, COCO 640px)
      https://docs.ultralytics.com/models/yolo11/
      字段: mAP50-95 / params(M) / FLOPs(B) / CPU-ONNX(ms) / T4(ms)

  [llm] Qwen2.5 base 0.5B~72B  (Qwen2.5 Technical Report, Table 5, arXiv:2412.15115)
      字段: MMLU / GSM8K / HumanEval / MATH

诚实边界 (写论文务必照此交代):
  * acc / FLOPs / params / 延迟 = 公开 benchmark 原值, 属"calibrated from published
    benchmarks", **不是我们自己实测部署**. 若要写"实测", 必须真在设备上跑一遍。
  * cycle_mult / energy_mult 由上面的 FLOPs(检测) 或 params(LLM) 归一算出 (compute 代理)。
  * 难度依赖 (acc 随 difficulty 下降) 用"易/难两个基准的差"建模:
      - LLM: 易=MMLU, 难=MATH (或 GSM8K) —— 小模型在难任务上崩得更狠, 是真实现象
        (如 0.5B: MMLU 47.5 → MATH 19.5; 72B: 86.1 → 62.1), 这一项有真实数据支撑;
      - detection: 第一版用建模量 difficulty_sensitivity, 真实化时可换成 COCO 的
        mAP_small / mAP_medium / mAP_large 分箱 (官方 eval 自带)。
"""
import numpy as np


# ============ 原始真实数据 (verbatim from sources) ============
# 每档: (name, compute, acc_easy, acc_hard, latency_ms)
#   compute = FLOPs(B)[detection] 或 params(B)[llm], 仅用于算 compute 比值
#   acc_easy / acc_hard 为 [0,1]; detection 暂用同一 mAP (无官方难度分箱时)

_YOLO11 = [
    # name, FLOPs(B), mAP(=acc_easy=acc_hard 暂), CPU-ONNX ms
    ("yolo11n", 6.5,   0.395, 0.395, 56.1),
    ("yolo11s", 21.5,  0.470, 0.470, 90.0),
    ("yolo11m", 68.0,  0.515, 0.515, 183.2),
    ("yolo11l", 86.9,  0.534, 0.534, 238.6),
    ("yolo11x", 194.9, 0.547, 0.547, 462.8),
]

_QWEN25 = [
    # name, params(B), MMLU(易), MATH(难), GSM8K, HumanEval  (均 /100)
    ("qwen2.5-0.5b", 0.5,  0.475, 0.195, 0.416, 0.305),
    ("qwen2.5-1.5b", 1.5,  0.609, 0.350, 0.685, 0.372),
    ("qwen2.5-3b",   3.0,  0.656, 0.426, 0.791, 0.421),
    ("qwen2.5-7b",   7.0,  0.742, 0.498, 0.854, 0.579),
    ("qwen2.5-14b",  14.0, 0.797, 0.556, 0.902, 0.567),
    ("qwen2.5-32b",  32.0, 0.833, 0.577, 0.929, 0.585),
    ("qwen2.5-72b",  72.0, 0.861, 0.621, 0.915, 0.591),
]

_RAW = {"detection": _YOLO11, "llm": _QWEN25}

# 推荐的档位子集 (索引), 第一版别用全部, 取轻/中/重几档即可
_DEFAULT_LEVELS = {
    "detection": [0, 2, 4],   # n / m / x
    "llm":       [0, 3, 6],   # 0.5B / 7B / 72B
}


def to_env_profiles(domain="detection", level_idx=None):
    """构造环境用的 profile 列表 (list of dict), 字段对齐修改方案:
        name / cycle_mult / energy_mult / acc_base / difficulty_sensitivity

    cycle_mult, energy_mult: 以所选档位里最重的一档为 1.0, 其余按 compute 比例缩放
                             (compute = FLOPs[detection] 或 params[llm])。
    acc_base:               易任务 (difficulty=0) 的精度。
    difficulty_sensitivity: acc 随 difficulty 线性下降的斜率 = acc_easy - acc_hard。
                            => acc(difficulty) = acc_base - difficulty_sensitivity * difficulty
    """
    rows = _RAW[domain]
    idx = level_idx if level_idx is not None else _DEFAULT_LEVELS[domain]
    sel = [rows[i] for i in idx]
    max_compute = max(r[1] for r in sel)

    profiles = []
    for r in sel:
        name, compute, acc_easy, acc_hard = r[0], r[1], r[2], r[3]
        mult = float(compute / max_compute)
        profiles.append({
            "name": name,
            "cycle_mult": mult,
            "energy_mult": mult,                       # 第一版: 能耗用 compute 代理; 真值可换实测
            "acc_base": float(acc_easy),
            "difficulty_sensitivity": float(max(acc_easy - acc_hard, 0.0)),
            # --- 透明起见, 保留原始数据 ---
            "_raw_compute": float(compute),
            "_latency_ms": float(r[4]) if domain == "detection" else None,
        })
    return profiles


def accuracy(profile, difficulty):
    """真实标定的精度模型: acc = acc_base - sensitivity * difficulty, 截断到 [0,1]."""
    acc = profile["acc_base"] - profile["difficulty_sensitivity"] * float(difficulty)
    return float(np.clip(acc, 0.0, 1.0))


if __name__ == "__main__":
    for dom in ("detection", "llm"):
        print(f"\n=== {dom} profiles (default levels) ===")
        for p in to_env_profiles(dom):
            print(f"  {p['name']:<14} cycle_mult={p['cycle_mult']:.3f} "
                  f"acc_base={p['acc_base']:.3f} "
                  f"diff_sens={p['difficulty_sensitivity']:.3f}")
