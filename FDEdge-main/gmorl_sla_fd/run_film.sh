#!/usr/bin/env bash
# ω-FiLM 条件化: diffusion(film+diversity) vs gauss(film) 同条件公平对比, 抢纯延迟角 + 硬化 ω-自适应。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_film_master.log; : > "$LOG"
SCRATCH="/c/Users/陈亮/AppData/Local/Temp/claude/D--python-project----5/cc14f2bb-1e73-40ce-9aff-f5ab40b725d6/scratchpad"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_film_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
DL=11; DT=6; EP=300
DF=seq_het_sacfilm; GF=seq_het_gaussfilm

say "=== ω-FiLM: diffusion(film+div0.05) vs gauss(film) ==="
step ${DF} python train_frac_seq.py --env hetero --actor diffusion --tag ${DF} \
     --episodes $EP --warm_eps 30 --T 20 --sparsemax --randn_prior --deadline $DL --arrival_dt $DT \
     --m_div 8 --div_target 0.05 --omega_film
step ${GF} python train_frac_seq.py --env hetero --actor mlp --tag ${GF} \
     --episodes $EP --warm_eps 30 --sparsemax --deadline $DL --arrival_dt $DT --omega_film
step eval_m1  python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 1
cpv film_m1
step eval_m32 python eval_frac.py --tags ${DF},${GF} --actors diffusion,mlp --k 10 --n_cand 32
cpv film_m32
step diag_base   python "$SCRATCH/diag_base_diffusion.py" ${DF} ${GF}
step diag_critic python "$SCRATCH/diag_critic_vs_actor.py" ${DF}
say "=== DONE ==="
