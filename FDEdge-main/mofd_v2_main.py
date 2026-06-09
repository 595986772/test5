"""
MOFD (G-FDEdge) V2 training entrypoint
======================================
在 MOFDEnvironmentV2 (含 cloud 动作) 上重新训练 G-FDEdge.

设计:
  - 直接复用 mofd_main.main(cfg_override=...) 完成训练 + 评估
  - 通过 monkey-patch 把 mofd_main 内部的 MOFDEnvironment 调用换成
    V2 适配器: 调用约定 Emax = action_dim 含 cloud (= N_edges + 1)
  - 输出文件名 prefix='mofd_v2', 自动写到
    `results/mofd_v2_<ts>/omega_resp_seed0云端已输出.csv` 和
    `results/omega_resp_seed0云端已输出.csv` (compare_hv_v2.py 默认读后者)

注意:
  - 这个脚本不动 mofd_main.py 内核 (只用 cfg_override 钩子和 monkey-patch)
  - V1 训练不受影响, mofd_main.py 单独跑仍然走 V1 env
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


# ============================================================
# 适配器: 把 mofd_main 的 V1 调用约定翻译成 V2
# ------------------------------------------------------------
# mofd_main 用 `MOFDEnvironment(Emax=cfg['Emax'], ...)`. 在 V1 里 Emax = 边缘数
# = action_dim. 在 V2 里 Emax_edges = 边缘数, action_dim = Emax_edges + 1
# (位 0 是 cloud). 让 model/buffer 看到 action_dim 维度正确, 我们这样约定:
#   cfg['Emax'] (传给适配器) = 总 action_dim, 含 cloud
#   适配器内部 super().__init__(Emax=cfg['Emax']-1)  → V2 base 类拿到 N_edges
#   env.action_dim 自动 = Emax_edges + 1 = cfg['Emax']
class _MOFDEnvAdapterV2(MOFDEnvironmentV2):
    """让 mofd_main 透明用 V2 env: cfg['Emax'] 表示 action_dim (含 cloud)."""

    def __init__(self, Emax=7,
                 num_tasks_max=10,
                 bit_range=(10, 40),
                 time_slots=100,
                 f_range=(10, 40),
                 delay_scale=0.05,
                 energy_scale=0.25,
                 seed=0,
                 # V2 专属 (默认值跟 _smoke_env_v2_strat 验证过的工作配置一致)
                 cloud_f_range=(50, 70),
                 cloud_tran_rate_range=(80, 120),
                 cloud_kappa=1e-4,
                 kappa=1e-3,
                 **kwargs):
        n_edges = max(1, int(Emax) - 1)             # 留一位给 cloud
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


# 替换 mofd_main 命名空间里的 MOFDEnvironment, 所有内部 new env 调用走 V2
_mm.MOFDEnvironment = _MOFDEnvAdapterV2


# ============================================================
# V2 训练入口
# ============================================================
def main():
    cfg_override = dict(
        # ---- env (V2 默认) ----
        # 注意: 这里 Emax 是 action_dim 含 cloud (= 6 edges + 1 cloud = 7)
        Emax=7,
        num_tasks_max=10,           # 跟 V2 env 默认对齐 (避免队列饱和)
        bit_range=(10, 40),
        time_slots=100,
        f_range=(10, 40),

        # ---- 训练规模 ----
        num_epochs=100,
        n_prefs_per_epoch=8,
        seeds=[0],
        smooth_window=5,
        train_eval_n_pref=11,
        train_eval_n_epi=1,
        final_eval_n_pref=21,
        final_eval_n_epi=3,

        # ---- 奖励通道归一化 (跟 V1 一致, 不需要重调) ----
        delay_scale=0.05,
        energy_scale=0.25,
        alpha_T=1.0,
        alpha_E=1.0,

        # ---- 训练超参 (沿用 V1) ----
        actor_lr=1e-4,
        critic_lr=1e-3,
        alpha_init=0.05,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.95,
        denoising_steps=3,
        hidden_dim=128,
        # 离散动作 |A|=7, H ∈ [0, log(7)≈1.95]; target_entropy 沿用 0.5
        target_entropy=0.5,
        buf_size=10000,
        batch_size=64,
        buffer_warmup=500,
        update_every=4,

        # ---- ω-buffer / V7 (沿用 V1) ----
        use_omega_buffer=True,
        obuf_decay=0.5,
        obuf_noise=0.05,
        use_v7=False,
        use_nap=True,
        nap_beta=0.01,
        use_v6=False, use_v5=True, use_envelope=False,
        use_hypernet=True, hyper_lr=1e-4, hyper_hidden=64, div_lambda=0.1,
        use_cor=True, cor_lambda=0.1, cor_c=0.0,
        use_popart=True, popart_beta=0.01,
        n_relabel_omegas=4,

        task_mode='random', task_kwargs=dict(),

        # ---- 输出 ----
        file_prefix='mofd_v2',      # 决定 results_dir 名 + csv 文件名
        results_root='results',
    )
    print('[mofd_v2] launching training with V2 env (cloud action enabled)')
    _mm.main(cfg_override=cfg_override)


if __name__ == '__main__':
    main()
