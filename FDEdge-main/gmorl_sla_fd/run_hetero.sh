#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_hetero_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_ht_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=11; DT=6; EP=300
say "=== 异构 env: 扩散(sparsemax+候选) vs 高斯, 同协议 ==="
step seq_het_diff  python train_frac_seq.py --env hetero --actor diffusion --tag seq_het_diff --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT
step seq_het_gauss python train_frac_seq.py --env hetero --actor mlp --tag seq_het_gauss --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT
step eval_m1  python eval_frac.py --tags seq_het_diff,seq_het_gauss --actors diffusion,mlp --k 10 --n_cand 1
cpv het_m1
step eval_m16 python eval_frac.py --tags seq_het_diff,seq_het_gauss --actors diffusion,mlp --k 10 --n_cand 16
cpv het_m16
step eval_m32 python eval_frac.py --tags seq_het_diff,seq_het_gauss --actors diffusion,mlp --k 10 --n_cand 32
cpv het_m32
say "=== DONE ==="
