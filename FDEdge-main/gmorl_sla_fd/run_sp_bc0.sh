#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_sp_bc0_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_sb_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
say "=== sparsemax 扩散 bc_eta=0 (解保守锚, 看能否压到 K=2.5) ==="
step seq_sp_diff_bc0 python train_frac_seq.py --actor diffusion --tag seq_sp_diff_bc0 --warmstart direct_sp_diff --episodes 250 --T 20 --sparsemax --randn_prior --bc_eta 0
step eval_sp_bc0     python eval_frac.py --tags seq_sp_diff_bc0,seq_sp_gauss --actors diffusion,mlp --k 12
for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/sp_bc0_$f 2>/dev/null; done
say "=== DONE ==="
