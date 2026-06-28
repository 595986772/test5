#!/usr/bin/env bash
# 公平性决定性检验 (dyn2 @ deadline=14, coord=0.10 dl=0.15):
#  ①扩散去掉diversity (m_div=1): 还赢? -> 赢=架构胜(diversity只是bonus); 也塌=diversity在干活.
#  ②gauss低熵 (α=0.001) best-shot 防塌: 还塌成GPU-only? -> 仍塌=单峰真限制.
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn2_fair.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn2f_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
DL=14; DT=6; EP=300; CR=0.10; DLR=0.15

say "=== ①扩散 m_div=1 (无diversity) ==="
step diffM1 python train_frac_seq.py --env dyn --actor diffusion --tag dyn2_diffM1 --seed 0 \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 1 --dl_ratio $DLR --coord_ratio $CR
say "=== ②gauss α=0.001 (低熵 best-shot) ==="
step gaussA001 python train_frac_seq.py --env dyn --actor mlp --tag dyn2_gaussA001 --seed 0 \
     --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT \
     --dl_ratio $DLR --coord_ratio $CR --alpha 0.001
step eval_diffM1_vs_gauss python eval_frac.py --tags dyn2_diffM1,dyn2_gauss14 --actors diffusion,mlp --k 10 --n_cand 1
step eval_diff_vs_gaussA001 python eval_frac.py --tags dyn2_diff14,dyn2_gaussA001 --actors diffusion,mlp --k 10 --n_cand 1
say "=== DONE ==="
