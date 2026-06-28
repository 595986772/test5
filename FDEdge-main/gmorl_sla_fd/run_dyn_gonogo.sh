#!/usr/bin/env bash
# go/no-go: dyn env (Shannon时变速率+随机任务类型+增强异构) 上, 单动作确定性评测下
# diffusion(T20+sparsemax+弱diversity, 无FiLM, 无m32) vs gauss, 公平同协议。
# 主看能耗/平衡区 (w<=0.5) 扩散是否赢 (探针预言赢点在此); m32 仅作 ablation。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_dyn_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_dyn_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=11; DT=6; EP=300
DF=dyn_diff; GF=dyn_gauss

say "=== dyn go/no-go: diffusion(T20+sparse+div0.05, 无FiLM) vs gauss, 单动作评测 ==="
step ${DF} python train_frac_seq.py --env dyn --actor diffusion --tag ${DF} \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 8 --div_target 0.05
step ${GF} python train_frac_seq.py --env dyn --actor mlp --tag ${GF} \
     --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT
step eval_m1  python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
cpv dyn_m1
step eval_m32 python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 32
cpv dyn_m32
say "=== DONE ==="
