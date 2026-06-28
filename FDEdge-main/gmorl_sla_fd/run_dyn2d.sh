#!/usr/bin/env bash
# dyn2 @ deadline=14: 协调开销把绝对延迟抬了~2-3s, deadline 随之从11->14 匹配延迟尺度
# (coord=0.10 dl=0.15 不变). 目标: diffusion 高ω点可行(p95<14), gauss 处处不可行 -> 更宽SLA可行前沿。
# 新 tag dyn2_*14; 旧 dyn2_* / dyn_* 全保留。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn2d_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn2d_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=14; DT=6; EP=300; CR=0.10; DLR=0.15
DF=dyn2_diff14; GF=dyn2_gauss14

say "=== dyn2 @ deadline=$DL (coord=$CR dl=$DLR): diffusion vs gauss, seed0 ==="
step ${DF} python train_frac_seq.py --env dyn --actor diffusion --tag ${DF} --seed 0 \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 8 --div_target 0.05 --dl_ratio $DLR --coord_ratio $CR
step ${GF} python train_frac_seq.py --env dyn --actor mlp --tag ${GF} --seed 0 \
     --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT \
     --dl_ratio $DLR --coord_ratio $CR
step eval_m1  python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
cpv dyn2d_m1
say "=== DONE ==="
