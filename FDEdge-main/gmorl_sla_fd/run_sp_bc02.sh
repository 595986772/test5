#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_sp_bc02_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_s2_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
say "=== sparsemax 扩散 bc_eta=0.02 (稳定与激进的甜点) ==="
step seq_sp_diff_bc02 python train_frac_seq.py --actor diffusion --tag seq_sp_diff_bc02 --warmstart direct_sp_diff --episodes 250 --T 20 --sparsemax --randn_prior --bc_eta 0.02
step eval_sp_bc02     python eval_frac.py --tags seq_sp_diff_bc02,seq_sp_gauss --actors diffusion,mlp --k 12
for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/sp_bc02_$f 2>/dev/null; done
say "=== DONE ==="
