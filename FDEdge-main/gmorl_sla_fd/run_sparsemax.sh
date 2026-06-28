#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_sparsemax_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_sm_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
say "=== SPARSEMAX 主对比: 显式z开关, 多峰留扩散 (PopArt默认开) ==="
step direct_sp_diff  python train_frac_direct.py --actor diffusion --tag direct_sp_diff --iters 4000 --T 20 --sparsemax --randn_prior
step direct_sp_gauss python train_frac_direct.py --actor mlp       --tag direct_sp_gauss --iters 4000 --sparsemax
step seq_sp_diff  python train_frac_seq.py --actor diffusion --tag seq_sp_diff --warmstart direct_sp_diff --episodes 250 --T 20 --sparsemax --randn_prior
step seq_sp_gauss python train_frac_seq.py --actor mlp       --tag seq_sp_gauss --warmstart direct_sp_gauss --episodes 250 --sparsemax
step eval_sp      python eval_frac.py --tags seq_sp_diff,seq_sp_gauss --actors diffusion,mlp --k 12
for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/sp_$f 2>/dev/null; done
say "=== SPARSEMAX DONE ==="
