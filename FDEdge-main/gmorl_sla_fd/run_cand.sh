#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_cand_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_cd_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
say "=== M-候选 critic 选择: 扩散 vs 高斯 (都 sparsemax, 同协议) ==="
# 重训存 critic
step seq_sp_diff  python train_frac_seq.py --actor diffusion --tag seq_sp_diff --warmstart direct_sp_diff --episodes 250 --T 20 --sparsemax --randn_prior
step seq_sp_gauss python train_frac_seq.py --actor mlp --tag seq_sp_gauss --warmstart direct_sp_gauss --episodes 250 --sparsemax
# M 扫: 1(基线单采) / 16 / 32
step eval_m1  python eval_frac.py --tags seq_sp_diff,seq_sp_gauss --actors diffusion,mlp --k 10 --n_cand 1
cpv cand_m1
step eval_m16 python eval_frac.py --tags seq_sp_diff,seq_sp_gauss --actors diffusion,mlp --k 10 --n_cand 16
cpv cand_m16
step eval_m32 python eval_frac.py --tags seq_sp_diff,seq_sp_gauss --actors diffusion,mlp --k 10 --n_cand 32
cpv cand_m32
say "=== DONE ==="
