#!/usr/bin/env bash
# v2 全量重训 driver (当前码, 16维 warm 通道)。串行 (8路并行 GPU 争用会崩)。
# 主对比: seq_diff_v2 vs seq_gauss_v2 (带 warmstart)。
# ablation: seq_*_nows (无 warmstart) — 回应 review 的 warm-start 敏感性。
set +e
cd "D:/python_project/实验版5/FDEdge-main/gmorl_sla_fd" || exit 1
mkdir -p result2/frac/cmp
LOG=result2/frac/run_v2_master.log
: > "$LOG"
DIR_ITERS=4000     # direct warm-start 源 (与旧 diff_l3 同尺度)
SEQ_EPS=250        # 序贯主训练 (与旧 seq_diff 同尺度)
KEVAL=12

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
step(){ local name="$1"; shift; say ">>> $name : $*";
        if "$@" > "result2/frac/_v2_${name}.log" 2>&1; then say "<<< $name DONE";
        else say "!!! $name FAILED -> result2/frac/_v2_${name}.log"; fi; }
copy_eval(){ local lab="$1"; for f in frac_eval_points.csv frac_eval_summary.csv frac_eval_meta.json; do
        [ -f "result2/frac/cmp/$f" ] && cp "result2/frac/cmp/$f" "result2/frac/cmp/${lab}_$f"; done;
        say "    [eval 数据另存为 cmp/${lab}_*]"; }

say "=== START v2 retrain (direct iters=$DIR_ITERS, seq eps=$SEQ_EPS, eval k=$KEVAL) ==="

# 1-2: warm-start 源 (direct, 可微 surrogate, bandit)
step direct_diff  python train_frac_direct.py --actor diffusion --tag diff_v2  --iters $DIR_ITERS
step direct_gauss python train_frac_direct.py --actor mlp       --tag gauss_v2 --iters $DIR_ITERS

# 3-4: 序贯主训练 (硬 env + critic + γ Bellman + feedback prior), 带 warmstart
step seq_diff_ws  python train_frac_seq.py --actor diffusion --tag seq_diff_v2  --warmstart diff_v2  --episodes $SEQ_EPS
step seq_gauss_ws python train_frac_seq.py --actor mlp       --tag seq_gauss_v2 --warmstart gauss_v2 --episodes $SEQ_EPS

# 5: 主对比评估 (硬 env, 读 meta 还原, 落 CSV)
step eval_main    python eval_frac.py --tags seq_diff_v2,seq_gauss_v2 --actors diffusion,mlp --k $KEVAL
copy_eval main

# 6-7: ablation — 无 warmstart (测 warm-start 敏感性; 可能撞鸡生蛋, 那也是诚实结果)
step seq_diff_nows  python train_frac_seq.py --actor diffusion --tag seq_diff_nows  --episodes $SEQ_EPS
step seq_gauss_nows python train_frac_seq.py --actor mlp       --tag seq_gauss_nows --episodes $SEQ_EPS

# 8-9: ablation 评估 (同 actor, 有 vs 无 warmstart)
step eval_abl_diff  python eval_frac.py --tags seq_diff_v2,seq_diff_nows   --actors diffusion,diffusion --k $KEVAL
copy_eval abl_diff
step eval_abl_gauss python eval_frac.py --tags seq_gauss_v2,seq_gauss_nows --actors mlp,mlp --k $KEVAL
copy_eval abl_gauss

say "=== ALL DONE ==="
