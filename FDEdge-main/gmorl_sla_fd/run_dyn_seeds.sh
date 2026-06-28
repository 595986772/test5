#!/usr/bin/env bash
# 多seed确认 dyn go/no-go 的单动作扩散胜势 (seed 0 已有 dyn_diff/dyn_gauss; 这里补 1/2/3)。
# 每 seed: 训 diff + gauss, 单动作评测(n_cand=1), HV 落 cmp/seedS_*。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn_seeds.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
DL=11; DT=6; EP=300
for S in 1 2 3; do
  DF=dyn_diff_s${S}; GF=dyn_gauss_s${S}
  say "=== seed ${S} ==="
  step ${DF} python train_frac_seq.py --env dyn --actor diffusion --tag ${DF} --seed ${S} \
       --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
       --m_div 8 --div_target 0.05
  step ${GF} python train_frac_seq.py --env dyn --actor mlp --tag ${GF} --seed ${S} \
       --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT
  step eval_s${S} python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
  for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/seed${S}_$f 2>/dev/null; done
done
say "=== ALL SEEDS DONE ==="
