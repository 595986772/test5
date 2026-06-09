"""
MOFD V2 — Dense workload variant (路径 A: 拉真 ω 权衡空间)
=========================================================
目的: 在 V2 环境 (含 cloud action) 上让 ω-响应曲线呈现单调趋势.

V1 (0515 ckpt) 单调的关键不是网络, 而是 50 任务负载让 ω 真实"必须选";
V2 默认 10 任务负载下, delay 跨度只有几秒, ω 信号被淹没. 本脚本仅调环境
负载 + 训练长度 + popart 节奏, 不动 V5 网络架构, 不动训练算法.

与 mofd_v2_main.py 的差异 (只改 3 个数, 控制单次训练 ~3h):
  num_tasks_max:   10 → 20      (任务负载 ×2, ω 权衡空间真实化)
  time_slots:     100 → 120     (累计 reward 跨度更大但训练量可控)
  popart_beta:    0.01 → 0.001  (对齐 0515 V1, normalize 更稳)
  num_epochs:     100 (不变, 跟 0515 一致)

输出:
  results/mofd_v2_dense_<ts>/mofd_v2_dense_pareto_aggregated.csv
  results/mofd_v2_dense_pareto_aggregated.csv

跑完用 draw_omega_response_from_pareto.py 直接出图:
  python draw_omega_response_from_pareto.py \
      --csv results/mofd_v2_dense_pareto_aggregated.csv \
      --out results/mofd_v2_dense_fig_omega_response
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mofd_environment_v2 import MOFDEnvironmentV2
import mofd_main as _mm


class _MOFDEnvAdapterV2(MOFDEnvironmentV2):
    """让 mofd_main 透明用 V2 env: cfg['Emax'] 表示 action_dim (含 cloud)."""

    def __init__(self, Emax=7,
                 num_tasks_max=20,
                 bit_range=(10, 40),
                 time_slots=120,
                 f_range=(10, 40),
                 delay_scale=0.05,
                 energy_scale=0.25,
                 seed=0,
                 cloud_f_range=(50, 70),
                 cloud_tran_rate_range=(80, 120),
                 cloud_kappa=1e-4,
                 kappa=1e-3,
                 **kwargs):
        n_edges = max(1, int(Emax) - 1)
        super().__init__(
            Emax=n_edges,
            num_tasks_max=num_tasks_max,
            bit_range=bit_range,
            time_slots=time_slots,
            f_range=f_range,
            cloud_f_range=cloud_f_range,
            cloud_tran_rate_range=cloud_tran_rate_range,
            kappa=kappa, cloud_kappa=cloud_kappa,
            delay_scale=delay_scale, energy_scale=energy_scale,
            seed=seed,
        )


_mm.MOFDEnvironment = _MOFDEnvAdapterV2


def main():
    cfg_override = dict(
        # ---- env: dense workload ----
        Emax=7,                       # 6 edges + 1 cloud
        num_tasks_max=20,             # ↑ 从 10 提到 20 (核心改动 1)
        bit_range=(10, 40),
        time_slots=120,               # ↑ 从 100 提到 120 (核心改动 2)
        f_range=(10, 40),

        # ---- 训练规模 ----
        num_epochs=100,               # 跟 0515 一致, 控制总训练时长
        n_prefs_per_epoch=8,
        seeds=[0],
        smooth_window=5,
        train_eval_n_pref=11,
        train_eval_n_epi=1,
        final_eval_n_pref=21,
        final_eval_n_epi=3,

        # ---- 奖励通道归一化 ----
        delay_scale=0.05,
        energy_scale=0.25,
        alpha_T=1.0,
        alpha_E=1.0,

        # ---- 训练超参 ----
        actor_lr=1e-4,
        critic_lr=1e-3,
        alpha_init=0.05,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.95,
        denoising_steps=3,
        hidden_dim=128,
        target_entropy=0.5,
        buf_size=10000,
        batch_size=64,
        buffer_warmup=500,
        update_every=4,

        # ---- ω-buffer / V5 ----
        use_omega_buffer=True,
        obuf_decay=0.5,
        obuf_noise=0.05,
        use_v7=False,
        use_nap=True,
        nap_beta=0.01,
        use_v6=False, use_v5=True, use_envelope=False,
        use_hypernet=True, hyper_lr=1e-4, hyper_hidden=64, div_lambda=0.1,
        use_cor=True, cor_lambda=0.1, cor_c=0.0,
        use_popart=True, popart_beta=0.001,   # ↓ 从 0.01 降到 0.001 (核心改动 4)
        n_relabel_omegas=4,

        task_mode='random', task_kwargs=dict(),

        # ---- 输出 ----
        file_prefix='mofd_v2_dense',
        results_root='results',
    )
    print('[mofd_v2_dense] launching V2 training with dense workload')
    print(f'  num_tasks_max={cfg_override["num_tasks_max"]}, '
          f'time_slots={cfg_override["time_slots"]}, '
          f'num_epochs={cfg_override["num_epochs"]}, '
          f'popart_beta={cfg_override["popart_beta"]}')
    print(f'  estimated time: 2-3 hours on GPU')
    _mm.main(cfg_override=cfg_override)


if __name__ == '__main__':
    main()
