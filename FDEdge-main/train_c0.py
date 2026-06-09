"""单独训练 C0 (no ω-buffer) ckpt, 用于 drift / shift 实验对比.

与 mofd_main.py 的 cfg 完全一致, 唯一区别: use_omega_buffer=False.
跑完后 ckpt 路径形如 results/mofd_c0_<ts>/ckpt_seed0, 丢给 drift_main 即可.

用法:
    python train_c0.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
from datetime import datetime
import torch

from mofd_main import run_single_seed, set_task_generator
from task_generator import make_task_generator


def main():
    cfg = dict(
        # ---- 与 mofd_main.py 默认 cfg 严格一致, 只改 use_omega_buffer ----
        Emax=6,
        num_tasks_max=50,
        bit_range=(10, 40),
        time_slots=100,
        f_range=(10, 40),
        num_epochs=100,
        n_prefs_per_epoch=8,
        seeds=[0],
        smooth_window=5,
        train_eval_n_pref=11,
        train_eval_n_epi=1,
        final_eval_n_pref=21,
        final_eval_n_epi=3,
        delay_scale=0.05,
        energy_scale=0.25,
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
        target_entropy=0.5,
        buf_size=10000,
        batch_size=64,
        buffer_warmup=500,
        update_every=4,
        use_omega_buffer=False,   # ← 与 mofd_main 唯一区别: C0 不用 buffer
        obuf_decay=0.5,
        obuf_noise=0.05,
        use_v5=True,
        use_cor=True,
        cor_lambda=0.1,
        cor_c=0.0,
        use_popart=True,
        popart_beta=0.001,
        use_envelope=False,
        n_relabel_omegas=4,
        task_mode='random',
        task_kwargs=dict(),
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'mofd_c0_{ts}')
    os.makedirs(results_dir, exist_ok=True)
    cfg['results_dir'] = results_dir
    cfg['file_prefix'] = 'mofd_c0'

    with open(os.path.join(results_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    set_task_generator(make_task_generator(cfg['task_mode'], **cfg.get('task_kwargs', {})))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_c0] device = {device}')
    print(f'[train_c0] run dir = {os.path.abspath(results_dir)}')
    print(f'[train_c0] cfg.use_omega_buffer = {cfg["use_omega_buffer"]} (C0 baseline)')

    r = run_single_seed(cfg, seed=0, device=device)

    ckpt_dir = os.path.join(results_dir, 'ckpt_seed0')
    print(f'\n[train_c0] DONE. ckpt saved to: {os.path.abspath(ckpt_dir)}')
    print(f'[train_c0] 下一步: python mofd_omega_drift_main.py \\')
    print(f'             --ckpt-c0 {ckpt_dir} \\')
    print(f'             --ckpt-c2 results/mofd_20260515_175014/ckpt_seed0')


if __name__ == '__main__':
    main()
