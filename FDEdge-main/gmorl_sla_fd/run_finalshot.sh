#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_finalshot_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_fs_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
say "=== FINAL SHOT: randn prior + T=20 (解 feedback-prior 自锁) ==="
step direct_randn  python train_frac_direct.py --actor diffusion --tag direct_randn --iters 4000 --T 20 --randn_prior
step seq_diff_randn python train_frac_seq.py --actor diffusion --tag seq_diff_randn --warmstart direct_randn --episodes 250 --T 20 --randn_prior
step eval_fs       python eval_frac.py --tags seq_diff_randn,seq_gauss_pa --actors diffusion,mlp --k 12
for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/fs_$f 2>/dev/null; done
say "=== FINAL SHOT DONE ==="
