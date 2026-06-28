#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_v3_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_v3_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
say "=== v3 PopArt main compare (仅改 PopArt, 其余同 v2) ==="
step seq_diff_pa  python train_frac_seq.py --actor diffusion --tag seq_diff_pa  --warmstart diff_v2  --episodes 250
step seq_gauss_pa python train_frac_seq.py --actor mlp       --tag seq_gauss_pa --warmstart gauss_v2 --episodes 250
step eval_pa      python eval_frac.py --tags seq_diff_pa,seq_gauss_pa --actors diffusion,mlp --k 12
for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/pa_$f 2>/dev/null; done
say "=== DONE ==="
