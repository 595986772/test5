#!/usr/bin/env bash
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
LOG=result2/frac/run_diag_master.log; : > "$LOG"
ts(){ date '+%H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local n="$1"; shift; say ">>> $n"; if "$@" >"result2/frac/_diag_${n}.log" 2>&1; then say "<<< $n DONE"; else say "!!! $n FAILED"; fi; }
cpv(){ for f in frac_eval_points.csv frac_eval_summary.csv; do cp result2/frac/cmp/$f result2/frac/cmp/${1}_$f 2>/dev/null; done; }
say "=== DIAG: BC锁死 vs env单峰 ==="
# 诊断1: bc_eta=0 能否逃出铺开盆地 (BC-anchoring 假设)
step diff_bc0   python train_frac_seq.py --actor diffusion --tag seq_diff_bc0 --warmstart diff_v2 --episodes 250 --bc_eta 0
step eval_bc0   python eval_frac.py --tags seq_diff_bc0,seq_diff_pa,seq_gauss_pa --actors diffusion,diffusion,mlp --k 12
cpv diag_bc0
# 诊断2: 异构 env "用哪k台" 是否真多峰 (扩散 vs 高斯, 都 hetero)
step diff_het   python train_frac_seq.py --actor diffusion --tag seq_diff_het --warmstart diff_v2  --episodes 250 --hetero
step gauss_het  python train_frac_seq.py --actor mlp       --tag seq_gauss_het --warmstart gauss_v2 --episodes 250 --hetero
step eval_het   python eval_frac.py --tags seq_diff_het,seq_gauss_het --actors diffusion,mlp --k 12
cpv diag_het
say "=== DIAG DONE ==="
