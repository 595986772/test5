#!/usr/bin/env bash
# dyn2: 升级 env (coord_ratio=0.10 协调开销 + dl_ratio=0.15 结果回传 -> 切分非免费, 延迟角非平凡)。
# seed0 controlled protocol: diffusion(T20+sparse+div0.05, 无FiLM) vs gauss, 单动作确定性评测。
# 旧 dyn_diff/dyn_gauss 保留不动; 新 tag dyn2_*。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn2_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn2_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=11; DT=6; EP=300; CR=0.10; DLR=0.15
DF=dyn2_diff; GF=dyn2_gauss

say "=== dyn2 (coord=$CR dl=$DLR): diffusion vs gauss, seed0, 单动作评测 ==="
step ${DF} python train_frac_seq.py --env dyn --actor diffusion --tag ${DF} --seed 0 \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 8 --div_target 0.05 --dl_ratio $DLR --coord_ratio $CR
step ${GF} python train_frac_seq.py --env dyn --actor mlp --tag ${GF} --seed 0 \
     --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT \
     --dl_ratio $DLR --coord_ratio $CR
step eval_m1  python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
cpv dyn2_m1
step eval_m32 python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 32
cpv dyn2_m32
say "=== DONE ==="
