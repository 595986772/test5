#!/usr/bin/env bash
# 路②正式全跑: div_target=0.05 (扫描胜出) 300ep + 完整 eval(HV vs 高斯, m1/m32) + 诊断。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_sac05_master.log; : > "$LOG"
SCRATCH="/c/Users/陈亮/AppData/Local/Temp/claude/D--python-project----5/cc14f2bb-1e73-40ce-9aff-f5ab40b725d6/scratchpad"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_sac_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=11; DT=6; EP=300; TAG=seq_het_sac05

say "=== 路②正式: div=0.05 抗塌缩扩散-SAC ==="
step ${TAG} python train_frac_seq.py --env hetero --actor diffusion --tag ${TAG} \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 8 --div_target 0.05
step eval_${TAG}_m1  python eval_frac.py --tags ${TAG},seq_het_gauss --actors diffusion,mlp --k 10 --n_cand 1
cpv ${TAG}_m1
step eval_${TAG}_m32 python eval_frac.py --tags ${TAG},seq_het_gauss --actors diffusion,mlp --k 10 --n_cand 32
cpv ${TAG}_m32
step diag_${TAG}_base   python "$SCRATCH/diag_base_diffusion.py" ${TAG} seq_het_gauss
step diag_${TAG}_critic python "$SCRATCH/diag_critic_vs_actor.py" ${TAG}
say "=== DONE ==="
