#!/usr/bin/env bash
# fair-gauss 决定性检验: diffusion 拿了自动调温的 diversity-exploration, 而 gauss 只有 α=0.005 微熵。
# 给 gauss 更高熵(α=0.05)做公平探索, 看 seed0/3 的崩溃是否消失。
#   崩溃消失+gauss竞争 -> 扩散"胜"是gauss调参假象, 否定头条。
#   仍崩 -> "扩散更鲁棒"叙事成立。
# gauss训练~1.5min/个, 4seed便宜。eval 对各seed已存的 diffusion ckpt。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn_fairgauss.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
DL=11; DT=6; EP=300
for S in 0 1 2 3; do
  GF=dyn_gaussA05_s${S}
  if [ "$S" = "0" ]; then DF=dyn_diff; else DF=dyn_diff_s${S}; fi
  say "=== fair-gauss seed ${S} (α=0.05) vs ${DF} ==="
  step ${GF} python train_frac_seq.py --env dyn --actor mlp --tag ${GF} --seed ${S} \
       --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT --alpha 0.05
  step evalA05_s${S} python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
  for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/fairA05_s${S}_$f 2>/dev/null; done
done
say "=== FAIR-GAUSS DONE ==="
